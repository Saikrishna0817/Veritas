"""Detectors scored by benchmark v2.

Each detector gets a reference set split into ``fit`` (train on it) and ``cal``
(held-out clean rows, scored the same way as incoming) and returns
``(incoming_row_scores, batch_score)``; higher means more suspicious.

Unless a detector has a native batch statistic, its batch score is a tail test:
how surprising is the number of incoming rows above the 99th percentile of the
calibration scores (-log10 binomial p-value). Same rule for every baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

TAIL_QUANTILE = 0.99


def tail_test(cal_scores: np.ndarray, inc_scores: np.ndarray, q: float = TAIL_QUANTILE) -> float:
    threshold = np.quantile(cal_scores, q)
    exceed = int((inc_scores > threshold).sum())
    p = binom.sf(exceed - 1, len(inc_scores), 1 - q)
    return float(-np.log10(max(p, 1e-300)))


def _probs(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba padded to all classes (a class may be missing from the fit set)."""
    P = np.full((len(X), n_classes), 1e-6)
    P[:, model.classes_] = model.predict_proba(X)
    return P / P.sum(axis=1, keepdims=True)


def _label_detector(make_model, score_fn):
    def detect(d):
        model = make_model().fit(d["X_fit"], d["y_fit"])
        cal = score_fn(d["y_cal"], _probs(model, d["X_cal"], d["n_classes"]))
        inc = score_fn(d["y_inc"], _probs(model, d["X_inc"], d["n_classes"]))
        return inc, tail_test(cal, inc)
    return detect


def _self_confidence(y, P):
    return 1.0 - P[np.arange(len(y)), y]


def _cleanlab_score(y, P):
    from cleanlab.rank import get_label_quality_scores
    return 1.0 - get_label_quality_scores(y, P)


def _outlier_detector(make_model, score_attr):
    def detect(d):
        model = make_model().fit(d["X_fit"])
        cal = score_attr(model, d["X_cal"])
        inc = score_attr(model, d["X_inc"])
        return inc, tail_test(cal, inc)
    return detect


def _ecod():
    from pyod.models.ecod import ECOD
    return ECOD()


def c2st(d):
    """Classifier two-sample test: can a model tell incoming rows from clean ones?"""
    Z = np.vstack([d["X_cal"], d["X_inc"]])
    t = np.r_[np.zeros(len(d["X_cal"])), np.ones(len(d["X_inc"]))]
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Z, t,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return p[len(d["X_cal"]):], float(roc_auc_score(t, p))


def veritas(d):
    """The shipping pipeline: rows = Layer-3 flags, batch = overall suspicion."""
    from app.detection.pipeline import DetectionPipeline

    X_ref, X_inc = d["X_ref_raw"], d["X_inc_raw"]
    lo = X_ref.min(axis=0)
    span = X_ref.max(axis=0) - lo
    span[span == 0] = 1.0
    pipeline = DetectionPipeline(random_state=0)
    pipeline.fit_baseline((X_ref - lo) / span, d["y_ref"])
    result = pipeline.analyze((X_inc - lo) / span, d["y_inc"])
    rows = np.zeros(len(X_inc))
    rows[result["details"]["layer3"]["flagged_indices_incoming"]] = 1.0
    return rows, float(result["overall_suspicion"])


DETECTORS = {
    "veritas": veritas,
    "cleanlab_hgb": _label_detector(lambda: HistGradientBoostingClassifier(random_state=0), _cleanlab_score),
    "knn_label": _label_detector(lambda: KNeighborsClassifier(n_neighbors=10), _self_confidence),
    "isolation_forest": _outlier_detector(lambda: IsolationForest(random_state=0),
                                          lambda m, X: -m.score_samples(X)),
    "ecod": _outlier_detector(_ecod, lambda m, X: m.decision_function(X)),
    "c2st": c2st,
}
