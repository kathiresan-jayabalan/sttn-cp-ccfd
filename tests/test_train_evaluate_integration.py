"""End-to-end verification of the STTN-CP training and evaluation pipeline

The synthetic dataset is small and randomly generated.
The test verifies the software workflow only: training, checkpoint creation,
checkpoint loading, evaluation and metrics serialization.

The test verifies that:

1. train.py completes successfully.
2. A model checkpoint and training summary are created.
3. evaluate.py can load the checkpoint.
4. Evaluation completes successfully.
5. The expected metrics JSON structure is produced.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def tiny_csv(tmp_path) -> Path:
    row_count = 300
    rng = np.random.default_rng(seed=1)

    features = {
        "Time": np.arange(row_count, dtype=np.float64),
        "Amount": rng.random(row_count) * 50,
    }
    for i in range(1, 29):
        features[f"V{i}"] = rng.normal(size=row_count)
    features["Class"] = np.arange(row_count, dtype=np.int64) % 2

    csv_path = tmp_path / "tiny_creditcard.csv"
    pd.DataFrame(features).to_csv(csv_path, index=False)
    return csv_path

def run_module(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

def test_train_then_evaluate_end_to_end(tiny_csv, tmp_path):
    output_dir = tmp_path / "outputs"

    train_result = run_module(
        [
            "src.train",
            "--data-path",
            str(tiny_csv),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--patience",
            "1",
            "--embed-dim",
            "16",
            "--n-blocks",
            "1",
            "--n-heads",
            "2",
            "--mlp-dim",
            "32",
            "--proj-dim",
            "8",
        ]
    )
    assert train_result.returncode == 0, train_result.stderr

    checkpoint_path = output_dir / "best_model.pt"
    summary_path = output_dir / "training_summary.json"
    assert checkpoint_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "protocol",
        "dataset_rows",
        "window_length",
        "feature_count",
        "trainable_parameters",
        "device",
        "seed",
        "best_validation_f1",
        "test_metrics",
        "history",
    ):
        assert key in summary

    metrics_path = output_dir / "test_metrics.json"
    evaluate_result = run_module(
        [
            "src.evaluate",
            "--data-path",
            str(tiny_csv),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(metrics_path),
        ]
    )
    assert evaluate_result.returncode == 0, evaluate_result.stderr
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "specificity",
        "confusion_matrix",
    ):
        assert key in metrics

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "specificity",
    ):
        assert 0.0 <= metrics[key] <= 1.0
