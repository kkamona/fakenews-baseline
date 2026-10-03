import pandas as pd
import pytest

from fakenews_baseline import md5_hash, overlap_counts, prepare_dataframe, split_dataset


def test_md5_is_deterministic():
    assert md5_hash("abc") == md5_hash("abc")
    assert md5_hash("abc") != md5_hash("abd")


def test_prepare_builds_full_text_and_drops_index(raw_df):
    out = prepare_dataframe(raw_df)
    assert list(out.columns) == ["full_text", "label"]
    assert out["full_text"].iloc[0].startswith("Title")


def test_prepare_handles_missing_title_and_text():
    df = pd.DataFrame({"title": [None, "A long enough headline here"],
                       "text": ["Some sufficiently long body text", None], "label": [1, 0]})
    assert len(prepare_dataframe(df)) == 2


def test_prepare_drops_short_rows():
    df = pd.DataFrame({"title": ["x"], "text": ["y"], "label": [0]})
    assert prepare_dataframe(df).empty


def test_prepare_removes_exact_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    assert len(prepare_dataframe(doubled)) == len(prepare_dataframe(raw_df))
    assert len(prepare_dataframe(doubled, deduplicate=False)) == 2 * len(raw_df)


def test_prepare_missing_column_raises():
    with pytest.raises(ValueError):
        prepare_dataframe(pd.DataFrame({"title": ["a"], "label": [1]}))


def test_split_sizes_stratification_and_no_leakage(raw_df):
    df = prepare_dataframe(raw_df)
    train, val, test = split_dataset(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) == 70 and len(val) == 15 and len(test) == 15
    for part in (train, val, test):
        assert 0.4 <= part["label"].mean() <= 0.6
    assert overlap_counts(train, val, test) == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_split_is_reproducible(raw_df):
    df = prepare_dataframe(raw_df)
    a = split_dataset(df, seed=1)[2]
    b = split_dataset(df, seed=1)[2]
    pd.testing.assert_frame_equal(a, b)


def test_split_invalid_sizes(raw_df):
    with pytest.raises(ValueError):
        split_dataset(prepare_dataframe(raw_df), val_size=0.6, test_size=0.5)


def test_overlap_detects_leak():
    a = pd.DataFrame({"full_text": ["same", "x"], "label": [0, 1]})
    b = pd.DataFrame({"full_text": ["same", "y"], "label": [0, 1]})
    assert overlap_counts(a, b, b.iloc[:0])["train_val"] == 1

def test_prepare_rejects_invalid_labels():
    df = pd.DataFrame({
        "title": ["A long enough headline here"],
        "text": ["Some sufficiently long body text"],
        "label": [2],
    })
    with pytest.raises(ValueError, match="Labels"):
        prepare_dataframe(df)