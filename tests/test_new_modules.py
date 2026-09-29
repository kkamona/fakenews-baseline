import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fakenews_baseline import (
    add_cue_columns,
    build_pipeline,
    compare_baselines,
    cue_summary,
    is_junk_token,
    merge_wordpieces,
    prepare_dataframe,
    top_tokens,
)
from fakenews_baseline.cues import CUE_PATTERNS, count_patterns
from fakenews_baseline.plotting import plot_confusion_matrix, plot_top_tokens


def test_count_patterns_case_insensitive_word_boundaries():
    pats = CUE_PATTERNS["causal_cues"]
    assert count_patterns("Because of this, THUS it led to that.", pats) == 3
    assert count_patterns("unthusiastic", pats) == 0  # no partial-word match


def test_cue_summary_by_label():
    df = pd.DataFrame({"full_text": ["shocking shocking truth", "committee said budget"],
                       "label": [1, 0]})
    s = cue_summary(add_cue_columns(df)).set_index("label")
    assert s.loc[1, "sensational_cues"] == 3
    assert s.loc[0, "sensational_cues"] == 0


def test_merge_wordpieces_sums_values():
    toks, vals = merge_wordpieces(["un", "##believ", "##able", "news"], [0.1, 0.2, 0.3, -0.5])
    assert toks == ["unbelievable", "news"]
    assert vals == pytest.approx([0.6, -0.5])


@pytest.mark.parametrize("tok,expected", [
    ("[CLS]", True), ("", True), ("...", True), (" ", True), ("hoax", False), ("3d", False),
])
def test_is_junk_token(tok, expected):
    assert is_junk_token(tok) is expected


def test_top_tokens_direction(raw_df):
    df = prepare_dataframe(raw_df)
    model = build_pipeline(min_df=1, max_df=1.0).fit(df["full_text"], df["label"])
    t = top_tokens(model, k=5)
    fake = t[t["direction"] == "push_fake(1)"]
    real = t[t["direction"] == "push_real(0)"]
    assert (fake["weight"] > 0).all() and (real["weight"] < 0).all()
    assert {"hoax", "scandal"} & set(fake["token"])
    assert {"ministry", "report"} & set(real["token"])


def test_compare_baselines_returns_four_sorted_models(raw_df):
    df = prepare_dataframe(raw_df)
    res = compare_baselines(df["full_text"], df["label"], df["full_text"], df["label"],
                            min_df=1, max_df=1.0)
    assert len(res) == 4
    assert res["macro_f1"].is_monotonic_decreasing


def test_plots_are_saved(tmp_path):
    y = np.array([0, 1, 1, 0])
    fig = plot_confusion_matrix(y, y, path=tmp_path / "cm.png")
    plt.close(fig)
    fig = plot_top_tokens(pd.DataFrame({"token": ["a", "b"], "weight": [1.0, 2.0]}),
                          "t", tmp_path / "tt.png")
    plt.close(fig)
    assert (tmp_path / "cm.png").stat().st_size > 0 and (tmp_path / "tt.png").stat().st_size > 0
