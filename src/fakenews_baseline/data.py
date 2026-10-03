"""Data preparation: text construction, cleaning, de-duplication, stratified splitting."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


def md5_hash(text: str) -> str:
    """Return the MD5 hex digest of a string (used to detect exact duplicates)."""
    return hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()


def prepare_dataframe(
    df: pd.DataFrame, min_len: int = 20, deduplicate: bool = True
) -> pd.DataFrame:
    """Build ``full_text`` = title + text, drop short rows and (optionally) exact duplicates.

    Expects columns ``title``, ``text``, ``label``. Returns a frame with
    ``full_text`` and integer ``label`` only.
    """
    missing = {"title", "text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    out["title"] = out["title"].fillna("").astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    out["full_text"] = (out["title"].str.strip() + ". " + out["text"].str.strip()).str.strip()
    out = out[out["full_text"].str.len() > min_len].copy()
    out["label"] = out["label"].astype(int)

    if deduplicate:
        out["_hash"] = out["full_text"].map(md5_hash)
        out = out.drop_duplicates(subset="_hash")
    
    labels = df["label"]
    if not labels.isin([0, 1]).all():
        bad = sorted(labels[~labels.isin([0, 1])].dropna().unique().tolist())
        raise ValueError(f"Labels must be 0 or 1; found invalid values: {bad}")

    return out[["full_text", "label"]].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split (default 70/15/15)."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")
    temp = val_size + test_size
    train, rest = train_test_split(df, test_size=temp, random_state=seed, stratify=df["label"])
    val, test = train_test_split(
        rest, test_size=test_size / temp, random_state=seed, stratify=rest["label"]
    )
    return train, val, test


def overlap_counts(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    """Count exact-duplicate texts shared between splits (data-leakage check)."""
    h = {
        name: set(d["full_text"].astype(str).map(md5_hash))
        for name, d in (("train", train), ("val", val), ("test", test))
    }
    return {
        "train_val": len(h["train"] & h["val"]),
        "train_test": len(h["train"] & h["test"]),
        "val_test": len(h["val"] & h["test"]),
    }
