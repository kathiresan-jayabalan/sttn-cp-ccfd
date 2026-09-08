"""End-to-end verification of the STTN-CP training and evaluation pipeline.

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

# Run the project entry-point scripts from src/.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


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

    features["Class"] = rng.integers(0, 2, size=row_count)

    frame = pd.DataFrame(features)

    csv_path = tmp_path / "tiny_creditcard.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=SRC_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_train_then_evaluate_end_to_end(tiny_csv, tmp_path):
    output_dir = tmp_path / "outputs"

    # 1. Run training.
    train_result = run(
        [
            "train.py",
            "--data-path",
            str(tiny_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert train_result.returncode == 0, train_result.stderr

    # 2. Verify training artifacts.
    checkpoint_path = output_dir / "best_sttn_cp.pth"
    summary_path = output_dir / "training_summary.json"

    assert checkpoint_path.exists()
    assert summary_path.exists()

    # 3. Verify the training summary structure.
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for key in (
        "device",
        "seed",
        "sequence_length",
        "best_validation_f1",
        "checkpoint",
    ):
        assert key in summary

    metrics_path = tmp_path / "test_metrics.json"

    # 4. Run evaluation using the generated checkpoint.
    evaluate_result = run(
        [
            "evaluate.py",
            "--data-path",
            str(tiny_csv),
            "--checkpoint",
            str(checkpoint_path),
            "--output-file",
            str(metrics_path),
        ]
    )

    assert evaluate_result.returncode == 0, evaluate_result.stderr
    assert metrics_path.exists()

    # 5. Verify the evaluation output structure.
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in (
        "test_rows",
        "fraud_rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "confusion_matrix",
    ):
        assert key in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
