"""Video poisoning attacks on decoded frames (n, T, H, W, 3) uint8, the image recipes applied to every frame.

A static trigger on every frame is the standard video backdoor (e.g. Zhao et al., clean-label video backdoors); the
others mirror research/mm/attacks.py. A miss here is a miss on the easy version of the attack.
"""
from __future__ import annotations

import numpy as np

from research.mm.attacks import _pick, duplicate_flood, label_flip_random, label_flip_targeted  # same recipes


def _stamp(raw, idx, size):
    board = (np.indices((size, size)).sum(axis=0) % 2 * 255).astype(np.uint8)[..., None]
    raw[idx, :, -size:, -size:, :] = board


def patch_backdoor(raw, y, rng, n, ctx):
    """BadNets checkerboard in the bottom-right corner of every frame, relabelled to the target class."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    _stamp(raw, idx, ctx["patch"])
    y[idx] = ctx["target"]
    return idx


def blended_backdoor(raw, y, rng, n, ctx):
    """A fixed noise pattern blended over every frame at alpha 0.15, relabelled to the target class."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    raw[idx] = np.clip(0.85 * raw[idx].astype(np.float32) + 0.15 * ctx["blend"], 0, 255).astype(np.uint8)
    y[idx] = ctx["target"]
    return idx


def clean_label_backdoor(raw, y, rng, n, ctx):
    """The patch on every frame of target-class clips whose labels stay correct."""
    idx = _pick(rng, np.flatnonzero(y == ctx["target"]), n)
    _stamp(raw, idx, ctx["patch"])
    return idx


def class_noise(raw, y, rng, n, ctx):
    """A fixed per-class noise pattern (+-8/255), the same on every frame, added to correctly labelled clips."""
    idx = _pick(rng, np.arange(len(y)), n)
    raw[idx] = np.clip(raw[idx].astype(np.int16) + ctx["class_noise"][y[idx]][:, None], 0, 255).astype(np.uint8)
    return idx


ATTACKS = {f.__name__: f for f in (label_flip_random, label_flip_targeted, patch_backdoor, blended_backdoor,
                                   clean_label_backdoor, duplicate_flood, class_noise)}


def make_context(raw: np.ndarray, n_classes: int, rng) -> dict:
    h = raw.shape[2]
    source, target = rng.choice(n_classes, size=2, replace=False)
    return {"n_classes": n_classes, "source": int(source), "target": int(target), "patch": max(3, h // 8),
            "blend": rng.uniform(0, 255, size=raw.shape[2:]).astype(np.float32),
            "class_noise": rng.choice([-8, 8], size=(n_classes, *raw.shape[2:])).astype(np.int16)}
