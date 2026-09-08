"""Train STTN-CP on the chronological, partition-safe protocol."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from .architecture import STTNCP, combined_loss, count_trainable_parameters
from .data import (
    build_labeled_windows,
    chronological_split,
    class_weights,
    load_creditcard,
    make_loader,
    scale_splits,
)


DEFAULTS = dict(
    seq_len=8,
    embed_dim=128,
    n_blocks=3,
    n_heads=4,
    mlp_dim=256,
    proj_dim=64,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    temperature=0.07,
    lambda_contrastive=0.5,
    patience=10,
    noise_std=0.02,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(batch: torch.Tensor, noise_std: float) -> torch.Tensor:
    return batch + torch.randn_like(batch) * noise_std


def evaluate_loss(model: STTNCP, loader, criterion, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            losses.append(criterion(model(xb), yb).item())
    return float(np.mean(losses)) if losses else float("inf")


def train(args: argparse.Namespace) -> Dict[str, object]:
    seed_everything(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_creditcard(args.data_path)
    splits = chronological_split(df, train_fraction=0.70, validation_fraction=0.10)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, features = scale_splits(
        splits.train, splits.validation, splits.test
    )

    X_train_w, y_train_w = build_labeled_windows(X_train, y_train, args.seq_len)
    X_val_w, y_val_w = build_labeled_windows(X_val, y_val, args.seq_len)
    X_test_w, y_test_w = build_labeled_windows(X_test, y_test, args.seq_len)

    if len(X_train_w) < args.batch_size:
        raise ValueError("training windows are fewer than the requested batch size")

    train_loader = make_loader(X_train_w, y_train_w, args.batch_size, True, drop_last=True)
    val_loader = make_loader(X_val_w, y_val_w, args.batch_size, False)
    test_loader = make_loader(X_test_w, y_test_w, args.batch_size, False)

    model = STTNCP(
        input_dim=X_train.shape[1],
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        mlp_dim=args.mlp_dim,
        proj_dim=args.proj_dim,
    ).to(device)

    weights = class_weights(y_train_w).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_state = None
    best_val_f1 = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_meter = []
        class_meter = []
        contrast_meter = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            view1 = augment(xb, args.noise_std)
            view2 = augment(xb, args.noise_std)
            z1 = model.project(model.forward_backbone(view1))
            z2 = model.project(model.forward_backbone(view2))
            logits = model(xb)
            loss, class_loss_value, contrast_loss_value = combined_loss(
                logits,
                yb,
                z1,
                z2,
                criterion,
                lambda_contrastive=args.lambda_contrastive,
                temperature=args.temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_meter.append(loss.item())
            class_meter.append(class_loss_value.item())
            contrast_meter.append(contrast_loss_value.item())

        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_metrics = predict_and_score(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_meter)),
            "train_class_loss": float(np.mean(class_meter)),
            "train_contrastive_loss": float(np.mean(contrast_meter)),
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = predict_and_score(model, test_loader, device)

    checkpoint = {
        "model_config": model.config,
        "state_dict": model.state_dict(),
        "features": features,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "seed": args.seed,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    summary = {
        "protocol": "chronological_70_10_20_train_only_scaling",
        "dataset_rows": len(df),
        "window_length": args.seq_len,
        "feature_count": X_train.shape[1],
        "train_windows": len(X_train_w),
        "validation_windows": len(X_val_w),
        "test_windows": len(X_test_w),
        "trainable_parameters": count_trainable_parameters(model),
        "device": str(device),
        "best_validation_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
        "paper_reference_metrics": {
            "accuracy": 0.9912,
            "precision": 0.9900,
            "recall": 0.9886,
            "f1": 0.9892,
            "specificity": 0.9796,
        },
        "note": "Paper values are references from Table 4, not claims of reproduction.",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def predict_and_score(model: STTNCP, loader, device) -> Dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.append(yb.numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    for name, value in DEFAULTS.items():
        arg = f"--{name.replace('_', '-')}"
        parser.add_argument(arg, type=type(value), default=value)
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result["test_metrics"], indent=2))
