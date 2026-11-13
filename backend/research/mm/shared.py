"""Study components that exploit what poisoned images SHARE, plus a fusion fix for the capped neg_log_p.

Every pooled score ranks a row against all OTHER rows of cal + inc together and is then split, so calibration
rows and incoming rows are scored by the same rule (same trick as the c2st shift check). Needs d["R_cal"],
d["R_inc"] (uint8 raw images).
"""
from __future__ import annotations

import numpy as np

from app.detection import veritas2 as v2


def _split(d, s):
    n = len(d["R_cal"])
    return s[:n], s[n:]


def _pool(d):
    return np.concatenate([d["R_cal"], d["R_inc"]]).astype(np.float32)


def _unit(a):
    a = a.reshape(len(a), -1)
    a = a - a.mean(axis=1, keepdims=True)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.where(n > 1e-6, a / np.maximum(n, 1e-6), 0).astype(np.float32)


def _kth_other(U, k):
    """k-th largest cosine similarity of each row to the other rows (U unit rows)."""
    S = U @ U.T
    np.fill_diagonal(S, -1)
    return np.partition(S, -k, axis=1)[:, -k]


def _down8(x):
    n, h, w, c = x.shape
    H, W = -(-h // 8) * 8, -(-w // 8) * 8
    x = np.pad(x, ((0, 0), (0, H - h), (0, W - w), (0, 0)), mode="edge")
    return x.reshape(n, 8, H // 8, 8, W // 8, c).mean(axis=(2, 4))


def duplicate_component(d):
    return _split(d, _kth_other(_unit(_down8(_pool(d))), 1))


def _residual(x):
    """pixel minus 3x3 local mean, per channel (edge padded)."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = x.shape[1:3]
    mean = sum(p[:, i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9
    return x - mean


def shared_residual_component(d, k=3):
    return _split(d, _kth_other(_unit(_residual(_pool(d))), k))


def shared_residual_cell_component(d, k=3, cell=8):
    """As above, correlation per 8x8 cell, max over cells, so a small corner patch is not drowned by the rest."""
    r = _residual(_pool(d))
    n, h, w, c = r.shape
    H, W = -(-h // cell) * cell, -(-w // cell) * cell
    r = np.pad(r, ((0, 0), (0, H - h), (0, W - w), (0, 0)))
    best = np.full(n, -1.0, dtype=np.float32)
    for i in range(0, H, cell):
        for j in range(0, W, cell):
            best = np.maximum(best, _kth_other(_unit(r[:, i:i + cell, j:j + cell]), k))
    return _split(d, best)


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


def chroma_floor_component(d, block=4):
    """Quietest 4x4 block of the high-pass residual of the colour differences R-G and G-B (independent per-channel noise lifts it)."""
    r = _residual(_pool(d)); n, h, w, _ = r.shape; H, W = h // block * block, w // block * block
    blocks = lambda a: (a[:, :H, :W] ** 2).reshape(n, H // block, block, W // block, block).mean(axis=(2, 4)).reshape(n, -1)
    return _split(d, ((blocks(r[..., 0] - r[..., 1]) + blocks(r[..., 1] - r[..., 2])) / 2).min(1))


def multi_tail_test(cal, inc, fracs):
    return v2.multi_tail_test(cal, inc, fracs)
