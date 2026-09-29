import pandas as pd
import pytest

from fakenews_baseline import build_pipeline, prepare_dataframe, top_errors
from fakenews_baseline.cli import main


def test_pipeline_learns_separable_data(raw_df):
    df = prepare_dataframe(raw_df)
    model = build_pipeline(min_df=1, max_df=1.0)
    model.fit(df["full_text"], df["label"])
    assert (model.predict(df["full_text"]) == df["label"]).mean() > 0.95
    assert model.decision_function(df["full_text"]).shape == (len(df),)


def test_top_errors_selects_and_sorts():
    df = pd.DataFrame({
        "label": [0, 0, 0, 1, 1, 1],
        "pred": [1, 1, 0, 0, 0, 1],
        "score": [0.9, 0.6, 0.1, 0.2, 0.4, 0.8],
    })
    fp, fn = top_errors(df, "score", n=5)
    assert fp["score"].tolist() == [0.9, 0.6]
    assert fn["score"].tolist() == [0.2, 0.4]


def test_top_errors_missing_column():
    with pytest.raises(KeyError):
        top_errors(pd.DataFrame({"label": [1]}), "score")


def test_cli_end_to_end_on_sample(tmp_path):
    main(["--raw", "data/sample/sample_news.csv", "--out-dir", str(tmp_path), "--n-boot", "20"])
    metrics = pd.read_csv(tmp_path / "metrics_test.csv")
    assert metrics["accuracy"].iloc[0] > 0.8
    for name in ("bootstrap_ci_test.csv", "baselines_val.csv", "top_tokens.csv",
                 "cue_stats_test.csv", "confusion_matrix.png"):
        assert (tmp_path / name).exists(), name
