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


# ── Batch-power experiments (dev only; see docs/benchmark-v2-findings.md, Phase 3) ──────────────
def _mwu_evidence(cal, inc):
    """-log10 one-sided Mann-Whitney p: are incoming scores shifted up relative to clean ones?"""
    from scipy.stats import mannwhitneyu
    return float(-np.log10(max(mannwhitneyu(inc, cal, alternative="greater").pvalue, 1e-300)))


def _oof_label_component(d):
    """Label scores for ALL reference rows out of fold (a ~3x larger calibration set), incoming from a full fit."""
    X = np.vstack([d["X_fit"], d["X_cal"]]); y = np.r_[d["y_fit"], d["y_cal"]]
    P = np.zeros((len(X), d["n_classes"]))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        P[te] = _probs(m, X[te], d["n_classes"])
    full = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    Pi = _probs(full, d["X_inc"], d["n_classes"])
    return 1 - P[np.arange(len(y)), y], 1 - Pi[np.arange(len(d["y_inc"])), d["y_inc"]]


def make_batchpower(label_fn, label_stat, shift_batch):
    """rows as veritas2 (label always, shift gated at 4); batch = strongest of the chosen statistics."""
    def detect(d):
        cal, inc = label_fn(d)
        rows = [_neg_log_p(cal, inc)]
        label_ev = max(tail_test(cal, inc), _mwu_evidence(cal, inc)) if label_stat == "tail+mwu" else tail_test(cal, inc)
        scal, sinc = _shift_component(d)
        shift_ev = tail_test(scal, sinc)
        if shift_ev >= 4.0:
            rows.append(_neg_log_p(scal, sinc))
        return np.max(rows, axis=0), max(label_ev, shift_ev) if shift_batch else label_ev
    return detect


def exact_tail_test(cal, inc, q=0.99):
    """-log10 p of how many incoming scores beat the m-th largest clean score. Under 'incoming looks like
    clean' the pooled order is random, so this is an exact hypergeometric tail: it stays calibrated when the
    clean set is small, where the plain binomial tail test inherits the noise in the clean quantile."""
    from scipy.stats import hypergeom
    n_c, n_i = len(cal), len(inc)
    m = max(1, round((1 - q) * n_c))
    x = int((inc > np.sort(cal)[-m]).sum())
    if x == 0:
        return 0.0
    return float(-np.log10(max(hypergeom.cdf(m - 1, n_c + n_i, n_c, x + m - 1), 1e-300)))


def make_exact(gate, names=("label", "shift"), shift_penalty=0.0):
    """shift_penalty: -log10 weight of the shift test in the batch statistic (weighted Bonferroni)."""
    def detect(d):
        rows, ev = [], []
        for name in names:
            cal, inc = COMPONENTS[name](d)
            e = exact_tail_test(cal, inc)
            ev.append(e - (shift_penalty if name != "label" else 0.0))
            if name == "label" or e >= gate:
                rows.append(_neg_log_p(cal, inc))
        return np.max(rows, axis=0), max(ev)
    return detect


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
    "bp_exact_g2": make_exact(2.0),
    "bp_exact_g3": make_exact(3.0),
    "bp_exact_g4": make_exact(4.0),
    "bp_exact_p06": make_exact(3.0, shift_penalty=0.6),
    "bp_exact_p10": make_exact(3.0, shift_penalty=1.0),
    "bp_exact_p15": make_exact(3.0, shift_penalty=1.5),
    "bp_exact_label": make_exact(1e9, ("label",)),
    "bp_label_only": make_batchpower(_label_component, "tail", False),
    "bp_mwu": make_batchpower(_label_component, "tail+mwu", True),
    "bp_oof": make_batchpower(_oof_label_component, "tail", True),
    "bp_oof_mwu": make_batchpower(_oof_label_component, "tail+mwu", True),
    "v2_ls_gate3": make_gated_ensemble(("label", "shift"), 3.0),
    "v2_ls_gate4": make_gated_ensemble(("label", "shift"), 4.0),
    "v2_ls_gate6": make_gated_ensemble(("label", "shift"), 6.0),
    "v2_label_shift_sum": make_ensemble(("label", "shift"), np.sum),
    "v2_label_shift_ecod_sum": make_ensemble(("label", "shift", "ecod"), np.sum),
}
