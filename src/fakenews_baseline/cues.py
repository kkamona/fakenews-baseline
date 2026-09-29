"""Linguistic cue statistics (causal, hedging, sensational) - notebook 03b_causal_cues_analysis."""

from __future__ import annotations

import re

import pandas as pd

CUE_PATTERNS: dict[str, list[str]] = {
    "causal_cues": [
        r"\bbecause\b", r"\btherefore\b", r"\bdue to\b", r"\bas a result\b", r"\bleads to\b",
        r"\bled to\b", r"\bresults in\b", r"\bresulted in\b", r"\bcauses\b", r"\bcaused by\b",
        r"\bhence\b", r"\bthus\b",
    ],
    "hedge_cues": [
        r"\ballegedly\b", r"\breportedly\b", r"\bmay\b", r"\bmight\b", r"\bcould\b",
        r"\bpossibly\b", r"\bappears\b", r"\bseems\b",
    ],
    "sensational_cues": [
        r"\bbreaking\b", r"\bunbelievable\b", r"\bshocking\b", r"\bexclusive\b",
        r"\bmust see\b", r"\btruth\b", r"\bexposed\b", r"\bbombshell\b",
    ],
}


def count_patterns(text: str, patterns: list[str]) -> int:
    """Total regex matches of all patterns in ``text`` (case-insensitive)."""
    text = str(text).lower()
    return sum(len(re.findall(p, text)) for p in patterns)


def add_cue_columns(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    """Return a copy of ``df`` with one count column per cue family."""
    out = df.copy()
    for name, pats in CUE_PATTERNS.items():
        out[name] = out[text_col].fillna("").map(lambda t, p=pats: count_patterns(t, p))
    return out


def cue_summary(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean cue counts per article, grouped (default: by label)."""
    group_cols = group_cols or ["label"]
    return df.groupby(group_cols, as_index=False)[list(CUE_PATTERNS)].mean()
