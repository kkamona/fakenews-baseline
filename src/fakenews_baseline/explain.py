"""Explainability helpers: LinearSVC coefficients (notebook 08) and SHAP token post-processing (07)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def top_tokens(pipeline, k: int = 30, fake_label: int = 1) -> pd.DataFrame:
    """Tokens with the largest positive / negative LinearSVC weights for the fake class.

    ``pipeline`` must be a fitted Pipeline with steps ``tfidf`` and ``svm``.
    """
    feats = pipeline.named_steps["tfidf"].get_feature_names_out()
    svm = pipeline.named_steps["svm"]
    w = svm.coef_.ravel()
    if svm.classes_[1] != fake_label:  # coef_ is oriented towards classes_[1]
        w = -w
    order = np.argsort(w)
    fake_idx, real_idx = order[-k:][::-1], order[:k]
    return pd.concat(
        [
            pd.DataFrame({"token": feats[fake_idx], "weight": w[fake_idx], "direction": "push_fake(1)"}),
            pd.DataFrame({"token": feats[real_idx], "weight": w[real_idx], "direction": "push_real(0)"}),
        ],
        ignore_index=True,
    )


def merge_wordpieces(tokens, values) -> tuple[list[str], list[float]]:
    """Merge BERT ``##`` word-pieces into whole words, summing their SHAP values."""
    out_t: list[str] = []
    out_v: list[float] = []
    for t, v in zip(tokens, values):
        t = str(t)
        if t.startswith("##") and out_t:
            out_t[-1] += t[2:]
            out_v[-1] += float(v)
        else:
            out_t.append(t)
            out_v.append(float(v))
    return out_t, out_v


def is_junk_token(token: str) -> bool:
    """True for empty strings, special tokens ([CLS], [SEP], [PAD]) and pure punctuation."""
    t = str(token).strip()
    return t == "" or t in {"[CLS]", "[SEP]", "[PAD]"} or re.fullmatch(r"[\W_]+", t) is not None
