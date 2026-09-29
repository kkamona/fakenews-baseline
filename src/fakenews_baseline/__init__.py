"""Reproducible TF-IDF + LinearSVC baseline for fake-news detection (WELFake)."""

from .cues import add_cue_columns, cue_summary
from .data import md5_hash, overlap_counts, prepare_dataframe, split_dataset
from .errors import top_errors
from .explain import is_junk_token, merge_wordpieces, top_tokens
from .metrics import bootstrap_ci, compute_metrics
from .models import build_pipeline, compare_baselines

__version__ = "0.2.0"
__all__ = [
    "add_cue_columns",
    "bootstrap_ci",
    "build_pipeline",
    "compare_baselines",
    "compute_metrics",
    "cue_summary",
    "is_junk_token",
    "md5_hash",
    "merge_wordpieces",
    "overlap_counts",
    "prepare_dataframe",
    "split_dataset",
    "top_errors",
    "top_tokens",
]
