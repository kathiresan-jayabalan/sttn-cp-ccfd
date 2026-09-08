import numpy as np
import pandas as pd

from src.data import build_labeled_windows, chronological_split


def test_labeled_windows_use_final_transaction_label():
    X = np.arange(20, dtype=np.float32).reshape(5, 4)
    y = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    windows, labels = build_labeled_windows(X, y, seq_len=3)

    assert windows.shape == (3, 3, 4)
    assert np.array_equal(labels, np.array([0, 1, 1]))
    assert np.array_equal(windows[0], X[:3])
    assert np.array_equal(windows[2], X[2:5])


def test_windows_can_be_built_without_crossing_partitions():
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    y = np.arange(10, dtype=np.int64) % 2
    first_x, first_y = build_labeled_windows(X[:6], y[:6], 3)
    second_x, second_y = build_labeled_windows(X[6:], y[6:], 3)

    assert first_x.shape[0] == 4
    assert second_x.shape[0] == 2
    assert np.max(first_x) < np.min(second_x)
    assert first_y[-1] == y[5]
    assert second_y[0] == y[8]


def test_chronological_split_preserves_order():
    df = pd.DataFrame({"Time": [4, 1, 3, 2, 5], "Class": [0, 1, 0, 1, 0], "V1": [0, 1, 2, 3, 4]})
    df = df.sort_values("Time").reset_index(drop=True)
    splits = chronological_split(df, train_fraction=0.6, validation_fraction=0.2)

    assert list(splits.train["Time"]) == [1, 2, 3]
    assert list(splits.validation["Time"]) == [4]
    assert list(splits.test["Time"]) == [5]
