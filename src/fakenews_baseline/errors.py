"""Error analysis helpers (most confident false positives / false negatives)."""

from __future__ import annotations

import pandas as pd


def top_errors(
    df: pd.DataFrame, score_col: str, n: int = 25,
    fake_label: int = 1, real_label: int = 0,
    label_col: str = "label", pred_col: str = "pred",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (false_positives, false_negatives), each sorted by model confidence.

    False positive = real article predicted fake (highest score first).
    False negative = fake article predicted real (lowest score first).
    """
    for col in (label_col, pred_col, score_col):
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")
    fp = df[(df[label_col] == real_label) & (df[pred_col] == fake_label)]
    fn = df[(df[label_col] == fake_label) & (df[pred_col] == real_label)]
    return (
        fp.sort_values(score_col, ascending=False).head(n),
        fn.sort_values(score_col, ascending=True).head(n),
    )
