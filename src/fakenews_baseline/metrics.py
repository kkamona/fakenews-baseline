"""Evaluation metrics and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "mcc", "macro_f1",
    "f1_fake(1)", "roc_auc_fake(1)", "pr_auc_fake(1)",
]


def compute_metrics(y_true, y_pred, y_score, model_name: str, fake_label: int = 1) -> dict:
    """Compute the metric set used in the thesis for one model on one split."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    y_bin = (y_true == fake_label).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=fake_label, zero_division=0
    )
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_fake(1)": float(p),
        "recall_fake(1)": float(r),
        "f1_fake(1)": float(f1),
        "roc_auc_fake(1)": float(roc_auc_score(y_bin, y_score)),
        "pr_auc_fake(1)": float(average_precision_score(y_bin, y_score)),
    }


def bootstrap_ci(
    y_true, y_pred, y_score, model_name: str = "model",
    n_boot: int = 1000, seed: int = 42, fake_label: int = 1,
) -> pd.DataFrame:
    """Percentile bootstrap (95 %) for the main metrics. Returns a long-format DataFrame."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: dict[str, list[float]] = {k: [] for k in METRIC_NAMES}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        yb = (yt == fake_label).astype(int)
        if yb.min() == yb.max():  # one class only: AUC undefined, resample skipped
            continue
        stats["accuracy"].append(accuracy_score(yt, yp))
        stats["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        stats["mcc"].append(matthews_corrcoef(yt, yp))
        stats["macro_f1"].append(f1_score(yt, yp, average="macro"))
        stats["f1_fake(1)"].append(f1_score(yb, (yp == fake_label).astype(int)))
        stats["roc_auc_fake(1)"].append(roc_auc_score(yb, ys))
        stats["pr_auc_fake(1)"].append(average_precision_score(yb, ys))

    rows = []
    for metric, values in stats.items():
        a = np.asarray(values, dtype=float)
        rows.append({
            "model": model_name, "metric": metric, "mean": float(a.mean()),
            "ci95_low": float(np.quantile(a, 0.025)),
            "ci95_high": float(np.quantile(a, 0.975)), "n_boot": len(a),
        })
    return pd.DataFrame(rows)
