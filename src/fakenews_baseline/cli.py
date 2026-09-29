"""Command-line pipeline: raw CSV -> clean -> split -> train -> evaluate -> CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cues import add_cue_columns, cue_summary
from .data import overlap_counts, prepare_dataframe, split_dataset
from .errors import top_errors
from .explain import top_tokens
from .metrics import bootstrap_ci, compute_metrics
from .models import build_pipeline, compare_baselines
from .plotting import plot_confusion_matrix, plot_top_tokens


def run(raw_csv: Path, out_dir: Path, seed: int = 42, n_boot: int = 1000,
        min_df: int = 2, max_df: float = 0.95) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_dataframe(pd.read_csv(raw_csv))
    train, val, test = split_dataset(df, seed=seed)

    leaks = overlap_counts(train, val, test)
    if any(leaks.values()):
        raise RuntimeError(f"Duplicate texts across splits: {leaks}")

    compare_baselines(
        train["full_text"], train["label"], val["full_text"], val["label"],
        seed=seed, min_df=min_df, max_df=max_df,
    ).to_csv(out_dir / "baselines_val.csv", index=False)

    model = build_pipeline(min_df=min_df, max_df=max_df)
    model.fit(train["full_text"], train["label"])
    pred = model.predict(test["full_text"])
    score = model.decision_function(test["full_text"])

    metrics = pd.DataFrame([compute_metrics(test["label"], pred, score, "TFIDF_LinearSVC")])
    metrics.to_csv(out_dir / "metrics_test.csv", index=False)
    bootstrap_ci(test["label"], pred, score, "TFIDF_LinearSVC", n_boot=n_boot, seed=seed).to_csv(
        out_dir / "bootstrap_ci_test.csv", index=False
    )

    errs = test.assign(pred=pred, score_fake=score)
    fp, fn = top_errors(errs, "score_fake")
    fp.to_csv(out_dir / "errors_fp.csv", index=False)
    fn.to_csv(out_dir / "errors_fn.csv", index=False)

    top_tokens(model, k=30).to_csv(out_dir / "top_tokens.csv", index=False)
    cue_summary(add_cue_columns(test)).to_csv(out_dir / "cue_stats_test.csv", index=False)

    import matplotlib.pyplot as plt

    fig = plot_confusion_matrix(test["label"], pred, "Confusion matrix (test) - LinearSVC",
                                out_dir / "confusion_matrix.png")
    plt.close(fig)
    tt = top_tokens(model, k=30)
    fig = plot_top_tokens(tt[tt["direction"] == "push_fake(1)"], "Top tokens pushing Fake(1)",
                          out_dir / "top_tokens_fake.png")
    plt.close(fig)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the TF-IDF + LinearSVC fake-news baseline.")
    ap.add_argument("--raw", required=True, type=Path, help="CSV with columns title,text,label")
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--n-boot", default=1000, type=int)
    ap.add_argument("--min-df", default=2, type=int)
    ap.add_argument("--max-df", default=0.95, type=float)
    a = ap.parse_args(argv)
    print(run(a.raw, a.out_dir, a.seed, a.n_boot, a.min_df, a.max_df).T.to_string())


if __name__ == "__main__":
    main()
