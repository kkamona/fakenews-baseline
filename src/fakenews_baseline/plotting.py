"""Matplotlib figures used in the thesis (confusion matrix, top-token bars)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion matrix",
                          path: str | Path | None = None, fake_label: int = 1, real_label: int = 0):
    """Draw a 2x2 confusion matrix (Fake first). Saves to ``path`` if given; returns the figure."""
    order = [fake_label, real_label]
    cm = confusion_matrix(y_true, y_pred, labels=order)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ticks = np.arange(2)
    ax.set_xticks(ticks, ["Fake(1)", "Real(0)"])
    ax.set_yticks(ticks, ["Fake(1)", "Real(0)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_top_tokens(df: pd.DataFrame, title: str, path: str | Path | None = None):
    """Horizontal bar chart of token weights (expects columns ``token`` and ``weight``)."""
    d = df.sort_values("weight", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(d["token"].tolist(), d["weight"].tolist())
    ax.set_title(title)
    ax.set_xlabel("LinearSVC coefficient (positive pushes Fake=1)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=250, bbox_inches="tight")
    return fig
