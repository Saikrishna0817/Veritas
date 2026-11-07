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
from scipy.stats import binom, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from app.detection.veritas2 import (_probs, detect as _v2_detect, label_component as _label_component,
                                    neg_log_p as _neg_log_p, shift_component as _shift_component, tail_test)

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
    pipeline = DetectionPipeline(random_state=0, engine="legacy")
    pipeline.fit_baseline((X_ref - lo) / span, d["y_ref"])
    result = pipeline.analyze((X_inc - lo) / span, d["y_inc"])
    rows = np.zeros(len(X_inc))
    rows[result["details"]["layer3"]["flagged_indices_incoming"]] = 1.0
    return rows, float(result["overall_suspicion"])


# ── Veritas v2 ensemble ──────────────────────────────────────────────────────
# Each component yields scores for the held-out clean rows (``cal``) and the incoming rows.
# A row's evidence from a component is its empirical p-value against ``cal``; the ensemble
# keeps the strongest evidence over components. The batch statistic is the strongest tail
# test over components. Only the components that win on the dev datasets are kept.

def _ecod_component(d):
    model = _ecod().fit(d["X_fit"])
    return model.decision_function(d["X_cal"]), model.decision_function(d["X_inc"])


def _iforest_component(d):
    model = IsolationForest(random_state=0).fit(d["X_fit"])
    return -model.score_samples(d["X_cal"]), -model.score_samples(d["X_inc"])


COMPONENTS = {"label": _label_component, "shift": _shift_component,
              "ecod": _ecod_component, "iforest": _iforest_component}


def make_ensemble(names: tuple[str, ...], combine=np.max):
    """combine=np.max: the strongest single component wins; np.sum: components add up (Fisher-style)."""
    def detect(d):
        rows, batch = [], []
        for name in names:
            cal, inc = COMPONENTS[name](d)
            rows.append(_neg_log_p(cal, inc))
            batch.append(tail_test(cal, inc))
        return combine(rows, axis=0), max(batch)
    return detect


def make_gated_ensemble(names: tuple[str, ...], gate: float):
    """The label check always ranks rows. Any other component joins the ranking only when its own
    batch-level test shows real evidence (tail test >= gate, in -log10 p), so a quiet component
    cannot dilute the ranking. The batch score is the strongest tail test, as before."""
    def detect(d):
        rows, batch = [], []
        for name in names:
            cal, inc = COMPONENTS[name](d)
            evidence = tail_test(cal, inc)
            batch.append(evidence)
            if name == "label" or evidence >= gate:
                rows.append(_neg_log_p(cal, inc))
        return np.max(rows, axis=0), max(batch)
    return detect


def veritas2(d):
    """The shipped detector (app.detection.veritas2), so the benchmark scores the code users run."""
    out = _v2_detect(d)
    return out["row_scores"], out["batch_evidence"]


DETECTORS = {
    "veritas": veritas,
    "cleanlab_hgb": _label_detector(lambda: HistGradientBoostingClassifier(random_state=0), _cleanlab_score),
    "knn_label": _label_detector(lambda: KNeighborsClassifier(n_neighbors=10), _self_confidence),
    "isolation_forest": _outlier_detector(lambda: IsolationForest(random_state=0),
                                          lambda m, X: -m.score_samples(X)),
    "ecod": _outlier_detector(_ecod, lambda m, X: m.decision_function(X)),
    "c2st": c2st,
    "v2_label": make_ensemble(("label",)),
    "v2_label_shift": make_ensemble(("label", "shift")),
    "v2_label_shift_ecod": make_ensemble(("label", "shift", "ecod")),
    "v2_all": make_ensemble(("label", "shift", "ecod", "iforest")),
    "v2_gate2": make_gated_ensemble(("label", "shift", "ecod", "iforest"), 2.0),
    "v2_gate4": make_gated_ensemble(("label", "shift", "ecod", "iforest"), 4.0),
    "v2_gate8": make_gated_ensemble(("label", "shift", "ecod", "iforest"), 8.0),
    # FROZEN 2026-11-06 after dev-only tuning (gate 3/4/6 tied; middle value kept). Do not retune on test.
    "veritas2": veritas2,
    "v2_ls_gate3": make_gated_ensemble(("label", "shift"), 3.0),
    "v2_ls_gate4": make_gated_ensemble(("label", "shift"), 4.0),
    "v2_ls_gate6": make_gated_ensemble(("label", "shift"), 6.0),
    "v2_label_shift_sum": make_ensemble(("label", "shift"), np.sum),
    "v2_label_shift_ecod_sum": make_ensemble(("label", "shift", "ecod"), np.sum),
}
