"""Veritas v2 detector: a label check plus a gated distribution-shift check, with per-cell scores.

Frozen after dev-only tuning; measured in ``backend/research`` (docs/benchmark-v2-findings.md).

* label check (always on): gradient-boosted classifiers cross-fitted over all clean reference rows (5 folds) score
  each row by how unlikely its label is (cleanlab's default self-confidence score); every reference row gets an
  out-of-fold score.
* shift check (gated): a classifier two-sample test, "can a model tell incoming rows from the clean reference rows?".
  It joins the row ranking only when its own batch test reaches ``GATE``.
* residual check (gated): rows whose columns no longer fit together (ridge residuals, cross-fitted).

Every check is turned into evidence against all clean reference rows, so the batch statistic is a -log10 p-value
and the row score is the -log10 share of clean rows scoring at least as high (extrapolated beyond the top one).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import binom, hypergeom, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

TAIL_QUANTILE = 0.99
GATE = 3.0             # shift evidence needed before it joins the row ranking; gates 2/3/4 tied on dev
CAL_FRACTION = 0.3     # share of the reference set held out as clean calibration rows
MIN_ROWS = 200         # below this on either side the tail test cannot resolve a 1% tail

# Weighted Bonferroni (-log10 share of the false-alarm budget): label ~80%, shift 10%. Chosen on dev
# (research/tab/row_study_dev.py; the same split as the image detector).
LABEL_PENALTY, SHIFT_PENALTY, RESID_PENALTY = 0.1, 1.0, 1.0
RESID_GATE = 1.0     # residual evidence needed before it joins the row ranking
# Batch evidence is the best exact tail test over these tail sizes, minus log10(their count): a few sharply ranked
# rows show in the smallest tail, a diffuse change in the largest.
TAILS = (0.0003, 0.001, 0.003, 0.01)

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


def multi_tail_test(cal: np.ndarray, inc: np.ndarray, fracs=TAILS) -> float:
    """Largest exact tail-test evidence over several tail sizes, Bonferroni-corrected for the number tried."""
    return max(max(exact_tail_test(cal, inc, 1 - f) for f in fracs) - np.log10(len(fracs)), 0.0)


def extrapolated_neg_log_p(cal: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """neg_log_p, plus for rows above every clean row a bonus log10(1 + excess / spread), so they keep their margin."""
    base = neg_log_p(cal, inc)
    top = cal.max()
    q1, q3 = np.percentile(cal, [25, 75])
    spread = max(q3 - q1, 0.05 * (top - cal.min()), 1e-12)
    return base + np.where(inc > top, np.log10(1 + np.maximum(inc - top, 0) / spread), 0.0)


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


def full_reference(d: dict) -> dict:
    """``d`` with every reference row (fit + cal) in the calibration slot: checks that need no held-out model compare
    the incoming rows with all of them, so a handful of poisoned rows can still give strong batch evidence."""
    return {**d, "X_cal": np.vstack([d["X_fit"], d["X_cal"]]), "y_cal": np.concatenate([d["y_fit"], d["y_cal"]])}


def label_crossfit_component(d: dict):
    """(ref, inc, one) label suspicion over all reference rows: out-of-fold for reference rows (5 folds); for incoming
    rows the mean of the fold models (row ranking) and one fold model per row (batch test: same footing as the
    out-of-fold reference rows, which the less extreme mean is not)."""
    X, y = np.vstack([d["X_fit"], d["X_cal"]]), np.concatenate([d["y_fit"], d["y_cal"]])
    cv = (StratifiedKFold if np.bincount(y)[np.unique(y)].min() >= 5 else KFold)(5, shuffle=True, random_state=0)
    ref, inc, one = np.zeros(len(y)), np.zeros(len(d["y_inc"])), np.zeros(len(d["y_inc"]))
    fold = np.arange(len(d["y_inc"])) % 5
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        score = lambda X_, y_: 1.0 - _probs(model, X_, d["n_classes"])[np.arange(len(y_)), y_]  # noqa: E731
        ref[te] = score(X[te], y[te])
        s = score(d["X_inc"], d["y_inc"])
        inc += s / 5
        one[fold == k] = s[fold == k]
    return ref, inc, one


def residual_component(d: dict):
    """(ref, inc) "this row breaks the columns' correlations": the sum over columns of squared ridge residuals (each column
    predicted from the others), scaled by the training spread. Cross-fitted like the label check: reference rows
    out-of-fold, each incoming row by one fold model. Catches covariate shift and backdoors (dev, row2_study_dev.py)."""
    R, I = np.vstack([d["X_fit"], d["X_cal"]]), d["X_inc"]
    ref, inc = np.zeros(len(R)), np.zeros(len(I))
    fold = np.arange(len(I)) % 5
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(R)):
        models = []
        for j in range(R.shape[1]):
            keep = np.arange(R.shape[1]) != j
            m = Ridge(1.0).fit(R[tr][:, keep], R[tr, j])
            models.append((keep, m, max(np.std(R[tr, j] - m.predict(R[tr][:, keep])), 1e-6)))
        score = lambda X: sum(((X[:, j] - m.predict(X[:, keep])) / s) ** 2 for j, (keep, m, s) in enumerate(models))  # noqa: E731
        ref[te] = score(R[te])
        inc[fold == k] = score(I[fold == k])
    return ref, inc


def detect(d: dict, use_label: bool = True, gate: float = GATE) -> dict:
    """Score one batch. ``d`` holds X_fit/y_fit, X_cal/y_cal, X_inc/y_inc and n_classes."""
    full = full_reference(d)
    evidence, rows = {}, {}
    if use_label:
        ref, inc, one = label_crossfit_component(d)
        evidence["label"], rows["label"] = multi_tail_test(ref, one), extrapolated_neg_log_p(ref, inc)
    cal, inc = shift_component(full)
    evidence["shift"] = multi_tail_test(cal, inc)
    if evidence["shift"] >= gate:   # a quiet check must not dilute the ranking
        rows["shift"] = extrapolated_neg_log_p(cal, inc)
    if d["X_inc"].shape[1] > 1:      # needs other columns to predict each column from
        cal, inc = residual_component(d)
        evidence["resid"] = multi_tail_test(cal, inc)
        if evidence["resid"] >= RESID_GATE:
            rows["resid"] = extrapolated_neg_log_p(cal, inc)
    batch = max(evidence.get("label", 0.0) - LABEL_PENALTY, evidence["shift"] - SHIFT_PENALTY,
                evidence.get("resid", 0.0) - RESID_PENALTY, 0.0)
    row_scores = np.max(list(rows.values()), axis=0) if rows else np.zeros(len(d["X_inc"]))
    return {"row_scores": row_scores, "cell_scores": cell_scores(full, row_scores, rows.get("label")),
            "batch_evidence": batch, "evidence": evidence, "shift_in_ranking": "shift" in rows,
            "resid_in_ranking": "resid" in rows}


ATTACKS = {"label_flip_random": "random label flipping", "label_flip_targeted": "targeted label flipping",
           "backdoor_ood": "backdoor trigger, out-of-range values", "backdoor_inrange": "backdoor trigger, in-range values",
           "feature_collision": "feature collision (clean-label)", "covariate_shift": "covariate shift / corrupted source"}
FEATURES = ("label_evidence", "shift_evidence", "resid_evidence", "label_blamed", "trigger_columns", "label_concentration",
            "out_of_range", "tightness", "unusual_columns")


def attack_features(d: dict, out: dict) -> dict:
    """What the most suspicious rows have in common, as named numbers (inputs to ``attack_profile``):
    label_blamed: share whose top cell is the label; trigger_columns: share whose top feature cell is one of the 3 most
    common such columns; label_concentration: how over-represented their most common claimed label is; out_of_range:
    share with a value beyond the reference range; tightness: their median pairwise distance over that of reference
    rows (low = clustered, as in a collision); unusual_columns: mean share of their columns scoring >= 1."""
    rs, C, f = out["row_scores"], out["cell_scores"], d["X_inc"].shape[1]
    S = np.argsort(-rs, kind="stable")[:min(len(rs), max(5, int((rs >= 2).sum())))]
    R, I = np.vstack([d["X_fit"], d["X_cal"]]), d["X_inc"][S]
    feat = C[S, :f] - rs[S, None]                                     # the column part of the cell scores
    top_feat = feat.argmax(1)
    common = np.bincount(top_feat, minlength=f)
    y_s, y_all = d["y_inc"][S], d["y_inc"]
    c = np.bincount(y_s).argmax()
    rng = np.random.default_rng(0)
    pair = lambda A: np.median(np.linalg.norm(A[rng.integers(0, len(A), 500)] - A[rng.integers(0, len(A), 500)], axis=1))  # noqa: E731
    ev = out["evidence"]
    return {"label_evidence": ev.get("label", 0.0), "shift_evidence": ev["shift"], "resid_evidence": ev.get("resid", 0.0),
            "label_blamed": float(np.mean(C[S].argmax(1) == f)), "trigger_columns": float(np.sort(common)[-3:].sum() / len(S)),
            "label_concentration": float(np.mean(y_s == c) / max(np.mean(y_all == c), 1e-9)),
            "out_of_range": float(np.mean((np.abs(I) > np.abs(R).max(0)).any(1))),
            "tightness": float(pair(I) / max(pair(R), 1e-9)), "unusual_columns": float(np.mean(feat >= 1))}


@lru_cache(maxsize=1)
def _attack_model():
    """GBM over attack_features, fitted on the dev batches stored in attack_profile.json (frozen by attack_study_dev.py)."""
    t = json.loads(Path(__file__).with_name("attack_profile.json").read_text())
    assert tuple(t["features"]) == FEATURES
    return HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(np.array(t["X"]), np.array(t["y"]))


def attack_profile(d: dict, out: dict) -> dict:
    """Which attack the suspicious rows look like: {"likely": key, "label": plain words, "probabilities", "features"}.
    Only meaningful when the batch is flagged; it names the closest of the six benchmarked attacks, it cannot say
    "none of these" (held-out accuracy in docs/benchmark-v2-findings.md)."""
    x = attack_features(d, out)
    m = _attack_model()
    p = m.predict_proba(np.array([[x[k] for k in FEATURES]]))[0]
    likely = str(m.classes_[p.argmax()])
    return {"likely": likely, "label": ATTACKS[likely], "probabilities": {str(c): round(float(v), 3) for c, v in zip(m.classes_, p)},
            "features": {k: round(v, 3) for k, v in x.items()}}


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
        if not (np.isfinite(self.X).all() and np.isfinite(X_inc).all()):
            raise ValueError("features must be finite numbers: impute or drop missing and infinite values first")
        classes = np.unique(np.r_[self.y, y_inc])
        code = lambda y: np.searchsorted(classes, y)  # noqa: E731  (labels may be any ints, even negative)
        scaler = StandardScaler().fit(self.X[self._fit])
        d = {"X_fit": scaler.transform(self.X[self._fit]), "y_fit": code(self.y[self._fit]),
             "X_cal": scaler.transform(self.X[self._cal]), "y_cal": code(self.y[self._cal]),
             "X_inc": scaler.transform(X_inc), "y_inc": code(y_inc), "n_classes": len(classes)}
        # Small tables are slower with many threads (on a busy 12-core machine 3,000 rows took 10 s with every core and
        # 2.6 s with one); large ones gain up to ~4. Thread count does not change the scores.
        with threadpool_limits(1 if len(self.X) + len(X_inc) < 10_000 else min(4, os.cpu_count() or 1)):
            out = detect(d, use_label=len(classes) > 1)   # one class (unlabeled upload): no label check
        verdict, suspicion = verdict_for(out["batch_evidence"])
        # Row flags only when the batch itself is suspicious. Score >= 2 is the clean top 1%;
        # with fewer than 99 calibration rows the best a row can do is beat all of them.
        n_ref = len(d["X_fit"]) + len(d["X_cal"])   # every check now compares with all reference rows
        cut = min(2.0, np.log10(n_ref + 1))
        flagged = np.flatnonzero(out["row_scores"] >= cut) if verdict in ("SUSPICIOUS", "CONFIRMED_POISONED") else []
        warnings = []
        if min(n_ref, len(X_inc)) < MIN_ROWS:
            warnings.append(f"fewer than {MIN_ROWS} reference or incoming rows: the batch test cannot resolve a 1% tail")
        if len(classes) < 2:
            warnings.append("single label value: label check skipped, distribution-shift check only")
        profile = attack_profile(d, out) if verdict != "CLEAN" and len(classes) > 1 else None
        return {**out, "verdict": verdict, "suspicion": suspicion, "flagged_indices": [int(i) for i in flagged],
                "attack_profile": profile,
                "n_incoming": len(X_inc), "n_calibration": n_ref, "warnings": warnings}
