"""Image detector: veritas2's label check plus gated checks on other views of the data.

* label  - GBM on the semantic (ResNet) embedding, always ranks rows.
* shift  - classifier two-sample test on the semantic embedding (catches duplicates, large distribution changes).
* pixel  - classifier two-sample test on low-level pixel statistics (catches stamped patches, noise, blending).
* dup / cell / resid - what poisoned images share: near-copies, a common local patch, a common faint pattern.
* chroma - colour noise in the quietest block of an image (per-channel noise patterns, blends).

Each check becomes an exact tail-test evidence (several tail sizes) against the clean reference rows. A shift check joins the ranking
only when its own evidence reaches its ``GATE``; its batch evidence is lowered by ``PENALTY`` (weighted Bonferroni).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.detection import veritas2 as v2
from research.mm import shared

# Gates, tail sizes and the component set were chosen on the dev datasets only (shared_study_dev.py, ref_study_dev.py,
# docs/image-benchmark-findings.md).
GATE = {"shift": 3.0, "pixel": 1.0, "dup": 1.0, "cell": 1.0, "resid": 1.0, "chroma": 1.0}
PENALTY = {"label": 0.1, "shift": 1.0, "pixel": 1.0, "dup": 1.0, "cell": 1.0, "resid": 1.0, "chroma": 1.0}  # -log10 share of the false-alarm budget
TAILS = (0.0003, 0.001, 0.003, 0.01)  # a few sharply ranked rows show in the smallest tail, a diffuse change in the largest


def pixel_component(d: dict):
    """(cal, inc) out-of-fold 'looks like the incoming batch' probabilities on pixel features."""
    Z = np.vstack([d["P_cal"], d["P_inc"]])
    t = np.r_[np.zeros(len(d["P_cal"])), np.ones(len(d["P_inc"]))]
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Z, t,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return p[: len(d["P_cal"])], p[len(d["P_cal"]):]


def _full_ref(d: dict) -> dict:
    """d with all reference rows (fit + cal) as the clean comparison set."""
    return {**d, "X_cal": np.vstack([d["X_fit"], d["X_cal"]]), "y_cal": np.concatenate([d["y_fit"], d["y_cal"]]),
            "P_cal": np.vstack([d["P_fit"], d["P_cal"]]), "R_cal": np.concatenate([d["R_fit"], d["R_cal"]])}


def label_crossfit_component(d: dict):
    """(ref, inc, one) label-suspicion over all reference rows: out-of-fold for reference rows; for incoming rows the
    mean of the fold models (row ranking) and one fold model per row (``one``: same footing as the out-of-fold
    reference rows, so it is used for the batch evidence; the mean is less extreme and made the check over-cautious)."""
    X, y = np.vstack([d["X_fit"], d["X_cal"]]), np.concatenate([d["y_fit"], d["y_cal"]])
    cv = (StratifiedKFold if np.bincount(y)[np.unique(y)].min() >= 5 else KFold)(5, shuffle=True, random_state=0)
    ref, inc, one = np.zeros(len(y)), np.zeros(len(d["y_inc"])), np.zeros(len(d["y_inc"]))
    fold = np.arange(len(d["y_inc"])) % 5
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)).fit(X[tr], y[tr])  # 100x faster than GBM, ranks flips better (dev)
        score = lambda X_, y_: 1.0 - v2._probs(model, X_, d["n_classes"])[np.arange(len(y_)), y_]
        ref[te] = score(X[te], y[te]); s_inc = score(d["X_inc"], d["y_inc"])
        inc += s_inc / 5; one[fold == k] = s_inc[fold == k]
    return ref, inc, one


def label_evidence(d: dict, ref, one) -> float:
    """Best of the pooled tail test and the best per-claimed-class one (Bonferroni over classes; a targeted flip puts
    all its rows in one claimed class), corrected for trying both."""
    y_ref, y_inc = np.concatenate([d["y_fit"], d["y_cal"]]), d["y_inc"]
    per_class = max([shared.multi_tail_test(ref[y_ref == c], one[y_inc == c], TAILS) for c in np.unique(y_inc)
                     if (y_ref == c).any()] or [0.0]) - np.log10(d["n_classes"])
    return max(max(shared.multi_tail_test(ref, one, TAILS), per_class) - np.log10(2), 0.0)


def components_v6(d: dict, secs: dict | None = None) -> dict:
    """Study only (ref_study_dev.py): label (540 cal), label_cf, and every other check against all reference rows. ``secs`` collects seconds per check."""
    import time
    r = _full_ref(d)
    fs = {"label": lambda: v2.label_component(d), "label_cf": lambda: label_crossfit_component(d)[:2],
          "shift": lambda: v2.shift_component(r), "pixel": lambda: pixel_component(r),
          "dup": lambda: shared.duplicate_component(r), "cell": lambda: shared.shared_residual_cell_component(r),
          "resid": lambda: shared.shared_residual_component(r), "chroma": lambda: shared.chroma_floor_component(r)}
    out = {}
    for n, f in fs.items():
        t = time.perf_counter(); out[n] = f()
        if secs is not None: secs[n] = time.perf_counter() - t
    return out


def components(d: dict) -> dict:
    """name -> (clean, inc) scores. Only the label check needs held-out rows (cross-fitted); every other check compares
    the incoming rows with all reference rows, so a handful of poisoned rows can still give strong batch evidence."""
    r = _full_ref(d)
    return {"label": label_crossfit_component(d), "shift": v2.shift_component(r), "pixel": pixel_component(r),
            "dup": shared.duplicate_component(r), "cell": shared.shared_residual_cell_component(r),
            "resid": shared.shared_residual_component(r), "chroma": shared.chroma_floor_component(r)}


_memo = [None, None]  # one slot: (batch, components), so several per-check detectors share one computation


def _components(d: dict) -> dict:
    if _memo[0] is not d: _memo[:] = [d, components(d)]  # holding d keeps its id from being reused
    return _memo[1]


def report(d: dict) -> dict:
    """Per check: penalised ``evidence``, ``raw`` evidence, whether it is ``admitted`` to the ranking, its row scores."""
    out = {}
    for name, (cal, inc, *one) in _components(d).items():
        e = label_evidence(d, cal, one[0]) if name == "label" else shared.multi_tail_test(cal, inc, TAILS)
        out[name] = {"evidence": e - PENALTY[name], "raw": e, "admitted": name == "label" or e >= GATE[name],
                     "rows": shared.extrapolated_neg_log_p(cal, inc)}  # rows beyond every clean row keep their margin
    return out


def veritas_mm(d: dict):
    r = report(d)
    return np.max([c["rows"] for c in r.values() if c["admitted"]], axis=0), max(c["evidence"] for c in r.values())


def check_detector(name: str):
    """(rows, raw evidence) of one check."""
    return lambda d: (lambda c: (c["rows"], c["raw"]))(report(d)[name])


def mm_lead(d: dict):
    """(rows, penalised evidence) of the check with the highest penalised evidence."""
    c = max(report(d).values(), key=lambda c: c["evidence"])
    return c["rows"], c["evidence"]


def pixel_c2st(d: dict):
    """Baseline: the pixel check alone."""
    cal, inc = pixel_component(d)
    return inc, v2.exact_tail_test(cal, inc)
