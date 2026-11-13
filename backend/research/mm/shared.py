"""Study components that exploit what poisoned images SHARE, plus a fusion fix for the capped neg_log_p.

Every pooled score ranks a row against all OTHER rows of cal + inc together and is then split, so calibration
rows and incoming rows are scored by the same rule (same trick as the c2st shift check). Needs d["R_cal"],
d["R_inc"] (uint8 raw images).
"""
from __future__ import annotations

import numpy as np

from app.detection import veritas2 as v2


from app.detection.veritas_mm import (_down8, _kth_other, _pool, _residual, _split, _unit,  # noqa: E402,F401
                                      chroma_floor_component, duplicate_component, shared_residual_cell_component,
                                      shared_residual_component)


def noise_level_component(d):
    """Median absolute Laplacian response of the gray image (robust high-frequency noise level)."""
    g = _pool(d).mean(axis=-1)
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = g.shape[1:]
    lap = 4 * g - p[:, :-2, 1:-1] - p[:, 2:, 1:-1] - p[:, 1:-1, :-2] - p[:, 1:-1, 2:]
    return _split(d, np.median(np.abs(lap.reshape(len(g), -1)), axis=1))


extrapolated_neg_log_p = v2.extrapolated_neg_log_p  # moved into the app detector; kept here for the studies


NEW = {"dup": duplicate_component,
       "resid_k3": shared_residual_component,
       "cell8_k3": shared_residual_cell_component,
       "cell8_k8": lambda d: shared_residual_cell_component(d, k=8, cell=8),    # k=8: a group of >=9 shares the pattern
       "cell16_k8": lambda d: shared_residual_cell_component(d, k=8, cell=16),  # larger cells suit 32px blends
       "noise": noise_level_component}


def multi_tail_test(cal, inc, fracs):
    return v2.multi_tail_test(cal, inc, fracs)
