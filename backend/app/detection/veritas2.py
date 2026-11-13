"""Veritas v2 detector: a label check plus a gated distribution-shift check.

Frozen after dev-only tuning; measured in ``backend/research`` (docs/benchmark-v2-findings.md).

* label check (always on): a gradient-boosted classifier fit on clean reference rows scores each
  incoming row by how unlikely its label is (cleanlab's default self-confidence score).
* shift check (gated): a classifier two-sample test, "can a model tell incoming rows from held-out
  clean rows?". It joins the row ranking only when its own batch test reaches ``GATE``.

Both checks are turned into evidence against held-out clean reference rows (``cal``), so the batch
statistic is a -log10 p-value and the row score is the -log10 share of clean rows scoring at least as high.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, hypergeom, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

TAIL_QUANTILE = 0.99
GATE = 3.0             # shift evidence needed before it joins the row ranking; gates 2/3/4 tied on dev
CAL_FRACTION = 0.3     # share of the reference set held out as clean calibration rows
MIN_ROWS = 200         # below this on either side the tail test cannot resolve a 1% tail

# Weighted Bonferroni: the label test gets 80% of the false-alarm budget and the shift test 20%, so the
# shift evidence is lowered by log10(4). Chosen on the dev datasets (penalties 0.6 / 1.0 / 1.5 compared).
SHIFT_PENALTY = float(np.log10(4))

# Batch evidence is a -log10 p-value that is calibrated on clean batches (see exact_tail_test), so the
# cut-offs are false-alarm rates: p<=0.05, p<=0.01, p<=1e-4.
LOW_RISK_AT, SUSPICIOUS_AT, CONFIRMED_AT, SATURATED_AT = 1.3, 2.0, 4.0, 8.0


def tail_test(cal: np.ndarray, inc: np.ndarray, q: float = TAIL_QUANTILE) -> float:
    """-log10 binomial p of the number of incoming scores above the q-quantile of clean scores."""
    exceed = int((inc > np.quantile(cal, q)).sum())
    return float(-np.log10(max(binom.sf(exceed - 1, len(inc), 1 - q), 1e-300)))


def exact_tail_test(cal: np.ndarray, inc: np.ndarray, q: float = TAIL_QUANTILE) -> float:
    """-log10 p of how many incoming scores beat the m-th largest clean score (m ~ 1% of the clean rows).

    If the incoming batch looks like the clean rows the pooled order is random, so this is an exact
    hypergeometric tail. The binomial ``tail_test`` ignores the noise in the clean quantile and, with a few
    hundred clean rows, lets ~5% of clean batches reach -log10 p = 4.6; this one reaches 1.3 as it should.
    """
    n_c, n_i = len(cal), len(inc)
    m = max(1, round((1 - q) * n_c))
    x = int((inc > np.sort(cal)[-m]).sum())
    if x == 0:
        return 0.0
    return float(-np.log10(max(hypergeom.cdf(m - 1, n_c + n_i, n_c, x + m - 1), 1e-300)))


def neg_log_p(cal: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """-log10 of the share of clean rows scoring at least as high (smoothed so it is finite)."""
    cal = np.sort(cal)
    ge = len(cal) - np.searchsorted(cal, inc, side="left")
    # Rows beyond every clean row would all tie; the raw-score rank (worth < 0.001) orders them.
    return -np.log10((ge + 1) / (len(cal) + 1)) + 1e-3 * rankdata(inc) / len(inc)


def _probs(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba padded to all classes (a class may be missing from the fit set)."""
    P = np.full((len(X), n_classes), 1e-6)
    P[:, model.classes_] = model.predict_proba(X)
    return P / P.sum(axis=1, keepdims=True)


def label_component(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(cal, inc) label-suspicion scores: 1 - P(given label)."""
    model = HistGradientBoostingClassifier(random_state=0).fit(d["X_fit"], d["y_fit"])

    def score(X, y):
        return 1.0 - _probs(model, X, d["n_classes"])[np.arange(len(y)), y]

    return score(d["X_cal"], d["y_cal"]), score(d["X_inc"], d["y_inc"])


def shift_component(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(cal, inc) out-of-fold 'looks like the incoming batch' probabilities."""
    Z = np.vstack([d["X_cal"], d["X_inc"]])
    t = np.r_[np.zeros(len(d["X_cal"])), np.ones(len(d["X_inc"]))]
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Z, t,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return p[: len(d["X_cal"])], p[len(d["X_cal"]):]


def detect(d: dict, use_label: bool = True, gate: float = GATE) -> dict:
    """Score one batch. ``d`` holds X_fit/y_fit, X_cal/y_cal, X_inc/y_inc and n_classes."""
    components = {"shift": shift_component}
    if use_label:
        components = {"label": label_component, **components}
    evidence, rows = {}, {}
    for name, component in components.items():
        cal, inc = component(d)
        evidence[name] = exact_tail_test(cal, inc)
        if name == "label" or evidence[name] >= gate:   # a quiet shift check must not dilute the ranking
            rows[name] = neg_log_p(cal, inc)
    batch = max(e - (SHIFT_PENALTY if name == "shift" else 0.0) for name, e in evidence.items())
    row_scores = np.max(list(rows.values()), axis=0)
    return {"row_scores": row_scores, "cell_scores": cell_scores(d, row_scores, rows.get("label")),
            "batch_evidence": batch, "evidence": evidence,
            "shift_in_ranking": "shift" in evidence and evidence["shift"] >= gate}


def cell_scores(d: dict, row_scores: np.ndarray, label_rows: np.ndarray | None) -> np.ndarray:
    """(n_incoming, n_features + 1) cell suspicion, the last column being the label: the row score plus how unusual
    the cell is. A feature value scores the -log10 share of clean values at least as far from the column mean, less
    log10(n_features) (the most unusual of many columns stands out by chance); the label scores the label check.
    Chosen on dev (research/tab/cell_study_dev.py) over conditional residuals and min(row, cell)."""
    cal, inc = np.abs(d["X_cal"]), np.abs(d["X_inc"])   # columns are standardised on the fit rows
    feat = np.stack([neg_log_p(cal[:, j], inc[:, j]) for j in range(inc.shape[1])], axis=1) - np.log10(inc.shape[1])
    label = np.zeros(len(inc)) if label_rows is None else label_rows
    return row_scores[:, None] + np.c_[feat, label]


def verdict_for(evidence: float) -> tuple[str, float]:
    """(verdict, suspicion in [0, 1]); suspicion is piecewise linear so the verdict cut-offs land on
    the 0.15 / 0.35 / 0.65 thresholds the rest of the app already uses."""
    suspicion = float(np.interp(evidence, [0, LOW_RISK_AT, SUSPICIOUS_AT, CONFIRMED_AT, SATURATED_AT],
                                [0, 0.15, 0.35, 0.65, 1.0]))
    if evidence >= CONFIRMED_AT:
        return "CONFIRMED_POISONED", suspicion
    if evidence >= SUSPICIOUS_AT:
        return "SUSPICIOUS", suspicion
    if evidence >= LOW_RISK_AT:
        return "LOW_RISK", suspicion
    return "CLEAN", suspicion


class Veritas2:
    """fit() on a trusted reference set, analyze() an incoming batch."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._fitted = False

    def fit(self, X_reference, y_reference) -> "Veritas2":
        X, y = np.asarray(X_reference, dtype=float), np.asarray(y_reference)
        order = np.random.RandomState(self.random_state).permutation(len(X))  # file order must not pick the split
        n_cal = int(len(X) * CAL_FRACTION)
        self._cal, self._fit = order[:n_cal], order[n_cal:]
        self.X, self.y = X, y
        self._fitted = True
        return self

    def analyze(self, X_incoming, y_incoming) -> dict:
        if not self._fitted:
            raise ValueError("Veritas2.analyze() called before fit()")
        X_inc, y_inc = np.asarray(X_incoming, dtype=float), np.asarray(y_incoming)
        classes = np.unique(np.r_[self.y, y_inc])
        code = lambda y: np.searchsorted(classes, y)  # noqa: E731  (labels may be any ints, even negative)
        scaler = StandardScaler().fit(self.X[self._fit])
        d = {"X_fit": scaler.transform(self.X[self._fit]), "y_fit": code(self.y[self._fit]),
             "X_cal": scaler.transform(self.X[self._cal]), "y_cal": code(self.y[self._cal]),
             "X_inc": scaler.transform(X_inc), "y_inc": code(y_inc), "n_classes": len(classes)}
        out = detect(d, use_label=len(classes) > 1)   # one class (unlabeled upload): no label check
        verdict, suspicion = verdict_for(out["batch_evidence"])
        # Row flags only when the batch itself is suspicious. Score >= 2 is the clean top 1%;
        # with fewer than 99 calibration rows the best a row can do is beat all of them.
        cut = min(2.0, np.log10(len(d["X_cal"]) + 1))
        flagged = np.flatnonzero(out["row_scores"] >= cut) if verdict in ("SUSPICIOUS", "CONFIRMED_POISONED") else []
        warnings = []
        if min(len(d["X_cal"]), len(X_inc)) < MIN_ROWS:
            warnings.append(f"fewer than {MIN_ROWS} calibration or incoming rows: the batch test cannot resolve a 1% tail")
        if len(classes) < 2:
            warnings.append("single label value: label check skipped, distribution-shift check only")
        return {**out, "verdict": verdict, "suspicion": suspicion, "flagged_indices": [int(i) for i in flagged],
                "n_incoming": len(X_inc), "n_calibration": len(d["X_cal"]), "warnings": warnings}
