"""Image detector: veritas2's label check plus gated checks on other views of the data.

* label  - GBM on the semantic (ResNet) embedding, always ranks rows.
* shift  - classifier two-sample test on the semantic embedding (catches duplicates, large distribution changes).
* pixel  - classifier two-sample test on low-level pixel statistics (catches stamped patches, noise, blending).
* dup / cell / resid - what poisoned images share: near-copies, a common local patch, a common faint pattern.

Each check becomes an exact tail-test evidence against held-out clean rows. A shift check joins the ranking
only when its own evidence reaches its ``GATE``; its batch evidence is lowered by ``PENALTY`` (weighted Bonferroni).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.detection import veritas2 as v2
from research.mm import shared

# Gates and the component set were chosen on the dev datasets only (shared_study_dev.py, docs/image-benchmark-findings.md).
GATE = {"shift": 3.0, "pixel": 1.0, "dup": 1.0, "cell": 1.0, "resid": 1.0}
PENALTY = {"label": 0.1, "shift": 1.0, "pixel": 1.0, "dup": 1.0, "cell": 1.0, "resid": 1.0}  # -log10 share of the false-alarm budget


def pixel_component(d: dict):
    """(cal, inc) out-of-fold 'looks like the incoming batch' probabilities on pixel features."""
    Z = np.vstack([d["P_cal"], d["P_inc"]])
    t = np.r_[np.zeros(len(d["P_cal"])), np.ones(len(d["P_inc"]))]
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Z, t,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return p[: len(d["P_cal"])], p[len(d["P_cal"]):]


def components(d: dict) -> dict:
    """name -> (cal, inc) scores. The last three look for what poisoned images share (copies, a patch, a blend)."""
    return {"label": v2.label_component(d), "shift": v2.shift_component(d), "pixel": pixel_component(d),
            "dup": shared.duplicate_component(d), "cell": shared.shared_residual_cell_component(d),
            "resid": shared.shared_residual_component(d)}


_memo = [None, None]  # one slot: (batch, components), so several per-check detectors share one computation


def _components(d: dict) -> dict:
    if _memo[0] is not d: _memo[:] = [d, components(d)]  # holding d keeps its id from being reused
    return _memo[1]


def report(d: dict) -> dict:
    """Per check: penalised ``evidence``, ``raw`` evidence, whether it is ``admitted`` to the ranking, its row scores."""
    out = {}
    for name, (cal, inc) in _components(d).items():
        e = v2.exact_tail_test(cal, inc)
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
