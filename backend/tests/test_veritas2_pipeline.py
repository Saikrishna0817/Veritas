import numpy as np

from app.detection.pipeline import DetectionPipeline
from app.detection.veritas2 import exact_tail_test, verdict_for
from sklearn.datasets import make_classification


def _split():
    X, y = make_classification(3000, 12, n_informative=8, n_classes=3, random_state=0)
    return X[:2100], y[:2100], X[2100:], y[2100:]


def _run(y_inc, X_inc=None):
    X_ref, y_ref, X, _ = _split()
    p = DetectionPipeline()
    p.fit_baseline(X_ref, y_ref)
    return p.analyze(X if X_inc is None else X_inc, y_inc)


def test_clean_batch_is_clean_and_flags_nothing():
    r = _run(_split()[3])
    assert r["verdict"] == "CLEAN" and r["details"]["row_flags"]["flagged_indices_incoming"] == []


def test_label_noise_is_caught_and_the_flags_are_mostly_the_flipped_rows():
    y_inc = _split()[3].copy()
    flipped = np.random.default_rng(0).permutation(len(y_inc))[:90]
    y_inc[flipped] = (y_inc[flipped] + 1) % 3
    r = _run(y_inc)
    flagged = set(r["details"]["row_flags"]["flagged_indices_incoming"])
    assert r["verdict"] == "CONFIRMED_POISONED"
    assert len(flagged & set(flipped)) / max(len(flagged), 1) > 0.8


def test_verdict_cutoffs_map_to_the_thresholds_the_app_uses():
    assert verdict_for(0)[0] == "CLEAN"
    assert verdict_for(2.0) == ("SUSPICIOUS", 0.35)
    assert verdict_for(4.0) == ("CONFIRMED_POISONED", 0.65)


def test_batch_test_is_calibrated_when_incoming_matches_clean():
    """The old binomial tail test let ~5% of clean batches reach -log10 p = 4.6; this must stay near 1.3."""
    rng = np.random.default_rng(0)
    ev = [exact_tail_test(rng.normal(size=540), rng.normal(size=1260)) for _ in range(400)]
    assert np.mean(np.array(ev) >= 1.3) < 0.08 and np.mean(np.array(ev) >= 2.0) < 0.025
