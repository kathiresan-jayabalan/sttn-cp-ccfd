"""Evaluate a saved STTN-CP checkpoint on the held-out chronological test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .architecture import STTNCP
from .data import build_labeled_windows, chronological_split, load_creditcard
from .train import predict_and_score
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="test_metrics.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = STTNCP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    df = load_creditcard(args.data_path)
    splits = chronological_split(df, 0.70, 0.10)
    features = checkpoint["features"]

    scaler = MinMaxScaler()
    scaler.min_ = -np.asarray(checkpoint["scaler_min"], dtype=np.float64) / (
        np.asarray(checkpoint["scaler_max"]) - np.asarray(checkpoint["scaler_min"])
    )
    scaler.scale_ = 1.0 / (
        np.asarray(checkpoint["scaler_max"]) - np.asarray(checkpoint["scaler_min"])
    )
    scaler.data_min_ = np.asarray(checkpoint["scaler_min"], dtype=np.float64)
    scaler.data_max_ = np.asarray(checkpoint["scaler_max"], dtype=np.float64)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_features_in_ = len(features)

    test_X = splits.test[features].to_numpy(dtype=np.float32)
    test_y = splits.test["Class"].to_numpy(dtype=np.int64)
    test_X = scaler.transform(test_X).astype(np.float32)
    X_test_w, y_test_w = build_labeled_windows(test_X, test_y, checkpoint["model_config"]["seq_len"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_test_w), torch.from_numpy(y_test_w)
        ),
        batch_size=1024,
    )
    metrics = predict_and_score(model, loader, torch.device("cpu"))
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
