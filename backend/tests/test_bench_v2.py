"""Benchmark v2 harness: attacks inject what they claim, metrics are sane."""

import numpy as np

from research.bench_v2 import ATTACKS, _row_metrics, load_manifest


def _ctx(X, y):
    return {"n_classes": 3, "major": 0, "minor": 1, "ref_max": X.max(0), "ref_std": X.std(0),
            "ref_p95": np.percentile(X, 95, 0), "trigger": np.array([0, 2]), "shift_cols": np.array([1])}


def test_attacks_poison_exactly_the_rows_they_return():
    rng = np.random.default_rng(0)
    X0, y0 = rng.normal(size=(300, 5)), rng.integers(0, 3, 300)
    for name, attack in ATTACKS.items():
        X, y = X0.copy(), y0.copy()
        idx = attack(X, y, np.random.default_rng(1), 15, _ctx(X0, y0))
        assert len(idx) == 15 and len(set(idx)) == 15, name
        untouched = np.setdiff1d(np.arange(300), idx)
        assert (X[untouched] == X0[untouched]).all() and (y[untouched] == y0[untouched]).all(), name
        changed = (X[idx] != X0[idx]).any(1) | (y[idx] != y0[idx])
        assert changed.all(), name
    X, y = X0.copy(), y0.copy()
    idx = ATTACKS["feature_collision"](X, y, np.random.default_rng(1), 15, _ctx(X0, y0))
    assert (y[idx] == y0[idx]).all()  # clean-label: labels untouched


def test_row_metrics():
    mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    perfect = _row_metrics(mask.astype(float), mask)
    assert perfect == {"auroc": 1.0, "auprc": 1.0, "precision_at_k": 1.0}
    assert _row_metrics(np.zeros(10), mask)["auprc"] == 0.2  # constant scores = chance


def test_dev_and_test_datasets_are_disjoint():
    ds = load_manifest()["datasets"]
    assert not {d["data_id"] for d in ds["dev"]} & {d["data_id"] for d in ds["test"]}
