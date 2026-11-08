"""Image poisoning attacks, applied to the raw incoming pixels and labels.

Each attack mutates ``raw`` (n, H, W, 3 uint8) and ``y`` in place and returns the poisoned row indices. Ground
truth is that index set; detectors only ever see the embeddings of the result. These are simplified, published
recipes (BadNets, Chen et al. blending, label-consistent backdoor without the adversarial step, unlearnable-style
class noise without surrogate optimisation), so a miss here is a miss on the easy version of the attack.
"""
from __future__ import annotations

import numpy as np


def _pick(rng, candidates: np.ndarray, n: int) -> np.ndarray:
    return rng.choice(candidates, size=min(n, len(candidates)), replace=False)


def _stamp(raw: np.ndarray, idx: np.ndarray, size: int) -> None:
    """BadNets-style black/white checkerboard in the bottom-right corner."""
    board = (np.indices((size, size)).sum(axis=0) % 2 * 255).astype(np.uint8)[..., None]
    raw[idx, -size:, -size:, :] = board


def label_flip_random(raw, y, rng, n, ctx):
    idx = _pick(rng, np.arange(len(y)), n)
    k = ctx["n_classes"]
    y[idx] = (y[idx] + rng.integers(1, k, len(idx))) % k
    return idx


def label_flip_targeted(raw, y, rng, n, ctx):
    """Source-class images relabelled as the target class."""
    idx = _pick(rng, np.flatnonzero(y == ctx["source"]), n)
    y[idx] = ctx["target"]
    return idx


def patch_backdoor(raw, y, rng, n, ctx):
    """BadNets: trigger patch on non-target images, relabelled to the target class."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    _stamp(raw, idx, ctx["patch"])
    y[idx] = ctx["target"]
    return idx


def blended_backdoor(raw, y, rng, n, ctx):
    """Chen et al.: a fixed noise pattern blended over the whole image at alpha 0.15, relabelled to the target."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    mixed = 0.85 * raw[idx].astype(np.float32) + 0.15 * ctx["blend"]
    raw[idx] = np.clip(mixed, 0, 255).astype(np.uint8)
    y[idx] = ctx["target"]
    return idx


def clean_label_backdoor(raw, y, rng, n, ctx):
    """Trigger patch on target-class images whose labels stay correct (no adversarial perturbation)."""
    idx = _pick(rng, np.flatnonzero(y == ctx["target"]), n)
    _stamp(raw, idx, ctx["patch"])
    return idx


def duplicate_flood(raw, y, rng, n, ctx):
    """n rows overwritten with copies of a few source rows, labels copied too."""
    sources = rng.choice(len(y), size=max(1, n // 10), replace=False)
    idx = _pick(rng, np.setdiff1d(np.arange(len(y)), sources), n)
    src = rng.choice(sources, size=len(idx))
    raw[idx], y[idx] = raw[src], y[src]
    return idx


def class_noise(raw, y, rng, n, ctx):
    """Availability attack: a fixed per-class noise pattern (+-8/255) added to correctly labelled images."""
    idx = _pick(rng, np.arange(len(y)), n)
    raw[idx] = np.clip(raw[idx].astype(np.int16) + ctx["class_noise"][y[idx]], 0, 255).astype(np.uint8)
    return idx


ATTACKS = {f.__name__: f for f in (label_flip_random, label_flip_targeted, patch_backdoor, blended_backdoor,
                                   clean_label_backdoor, duplicate_flood, class_noise)}


def make_context(raw: np.ndarray, n_classes: int, rng) -> dict:
    h = raw.shape[1]
    source, target = rng.choice(n_classes, size=2, replace=False)
    return {"n_classes": n_classes, "source": int(source), "target": int(target), "patch": max(3, h // 8),
            "blend": rng.uniform(0, 255, size=raw.shape[1:]).astype(np.float32),
            "class_noise": rng.choice([-8, 8], size=(n_classes, *raw.shape[1:])).astype(np.int16)}
