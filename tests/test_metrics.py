import numpy as np
import pytest

from fakenews_baseline import bootstrap_ci, compute_metrics


def test_perfect_predictions():
    y = np.array([0, 1] * 20)
    m = compute_metrics(y, y, y.astype(float), "perfect")
    for k in ("accuracy", "macro_f1", "mcc", "roc_auc_fake(1)", "f1_fake(1)"):
        assert m[k] == pytest.approx(1.0)


def test_inverted_predictions():
    y = np.array([0, 1] * 20)
    m = compute_metrics(y, 1 - y, (1 - y).astype(float), "bad")
    assert m["accuracy"] == 0.0
    assert m["mcc"] == pytest.approx(-1.0)


def test_bootstrap_shape_and_ordering():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = y + rng.normal(0, 0.5, 300)
    pred = (score > 0.5).astype(int)
    ci = bootstrap_ci(y, pred, score, "m", n_boot=100, seed=1)
    assert len(ci) == 7
    assert (ci["ci95_low"] <= ci["mean"]).all() and (ci["mean"] <= ci["ci95_high"]).all()


def test_bootstrap_is_seeded():
    y = np.array([0, 1] * 50)
    a = bootstrap_ci(y, y, y.astype(float), n_boot=20, seed=3)
    b = bootstrap_ci(y, y, y.astype(float), n_boot=20, seed=3)
    assert a.equals(b)
