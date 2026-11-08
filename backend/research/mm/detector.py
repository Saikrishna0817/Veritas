"""Image detector: veritas2's label check plus two gated shift checks, one per view of the data.

* label  - GBM on the semantic (ResNet) embedding, always ranks rows.
* shift  - classifier two-sample test on the semantic embedding (catches duplicates, large distribution changes).
* pixel  - classifier two-sample test on low-level pixel statistics (catches stamped patches, noise, blending).

Each check becomes an exact tail-test evidence against held-out clean rows. A shift check joins the ranking
only when its own evidence reaches ``GATE``; its batch evidence is lowered by ``PENALTY`` (weighted Bonferroni).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.detection import veritas2 as v2

GATE = 3.0
PENALTY = {"label": 0.1, "shift": 1.0, "pixel": 1.0}  # -log10 of each check's share of the false-alarm budget


def pixel_component(d: dict):
    """(cal, inc) out-of-fold 'looks like the incoming batch' probabilities on pixel features."""
    Z = np.vstack([d["P_cal"], d["P_inc"]])
    t = np.r_[np.zeros(len(d["P_cal"])), np.ones(len(d["P_inc"]))]
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Z, t,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return p[: len(d["P_cal"])], p[len(d["P_cal"]):]


def veritas_mm(d: dict):
    comps = {"label": v2.label_component, "shift": v2.shift_component, "pixel": pixel_component}
    rows, ev = [], []
    for name, component in comps.items():
        cal, inc = component(d)
        e = v2.exact_tail_test(cal, inc)
        ev.append(e - PENALTY[name])
        if name == "label" or e >= GATE:
            rows.append(v2.neg_log_p(cal, inc))
    return np.max(rows, axis=0), max(ev)


def pixel_c2st(d: dict):
    """Baseline: the pixel check alone."""
    cal, inc = pixel_component(d)
    return inc, v2.exact_tail_test(cal, inc)
