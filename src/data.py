"""Data loading and leakage-safe preprocessing for STTN-CP"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_creditcard(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("credit card dataset must contain a 'Class' column")

    if "Time" in df.columns:
        df = df.sort_values("Time", kind="mergesort")

    return df.reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> DataSplits:
    """Split an already ordered frame without shuffling transaction order."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be below 1")

    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))

    return DataSplits(
        train=df.iloc[:train_end].copy(),
        validation=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )


def feature_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    features = [column for column in df.columns if column != "Class"]
    X = df[features].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int64)
    return X, y, features


def scale_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    MinMaxScaler,
    list[str],
]:
    """Fit Min-Max scaling on training data only and transform held-out splits."""
    X_train, y_train, features = feature_matrix(train_df)
    X_val, y_val, _ = feature_matrix(validation_df)
    X_test, y_test, _ = feature_matrix(test_df)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return (
        (X_train_scaled, y_train),
        (X_val_scaled, y_val),
        (X_test_scaled, y_test),
        scaler,
        features,
    )


def build_labeled_windows(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed-length windows and use the final transaction label."""
    if X.ndim != 2:
        raise ValueError("X must have shape (N, D)")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be one-dimensional and match X")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")

    if len(X) < seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    windows = np.stack(
        [X[i : i + seq_len] for i in range(len(X) - seq_len + 1)],
        axis=0,
    ).astype(np.float32)
    labels = y[seq_len - 1 :].astype(np.int64)
    return windows, labels


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights(y: np.ndarray) -> Tensor:
    """Return inverse-frequency weights for the two fraud classes."""
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("both classes are required to compute class weights")
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)
