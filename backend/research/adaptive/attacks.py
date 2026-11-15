"""Attacks tuned to evade Veritas's image checks, each a (poison, trigger) pair with a strength ``s``.

``poison(raw, y, rng, n, ctx, s)`` mutates like research/mm/attacks.py and returns the poisoned indices;
``trigger(images, ctx, s)`` is the same trigger applied to test images (the identity for label flips). Fixed random
elements live in ctx (``make_context``). Outputs are uint8, rounded then clipped.
"""
from __future__ import annotations

import numpy as np

from research.mm import attacks as base


def make_context(raw: np.ndarray, n_classes: int, rng) -> dict:
    from scipy.ndimage import gaussian_filter
    ctx = base.make_context(raw, n_classes, rng)
    field = gaussian_filter(rng.normal(size=raw.shape[1:]), sigma=(4, 4, 0), mode="wrap")
    ctx["lowfreq"] = (field / np.abs(field).max()).astype(np.float32)   # smooth pattern, little high-pass energy
    ctx["flow"] = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)      # control grid of the warp field
    return ctx


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def lowfreq_trigger(images, ctx, s):
    return _to_u8(images.astype(np.float32) + s * ctx["lowfreq"])


def warp_trigger(images, ctx, s):
    """WaNet-style: sample the image through a fixed smooth flow field of max displacement s pixels."""
    import torch
    import torch.nn.functional as F
    h, w = images.shape[1:3]
    flow = F.interpolate(torch.from_numpy(ctx["flow"]), size=(h, w), mode="bicubic", align_corners=True)
    flow = flow / flow.abs().max() * s                                   # pixels
    flow = flow.permute(0, 2, 3, 1) * torch.tensor([2 / (w - 1), 2 / (h - 1)])   # to [-1, 1] grid units (x, y)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    grid = (torch.stack([xs, ys], -1)[None] + flow).expand(len(images), -1, -1, -1)
    x = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return _to_u8(out.permute(0, 2, 3, 1).numpy())


def faint_patch_trigger(images, ctx, s):
    p = ctx["patch"]
    board = (np.indices((p, p)).sum(axis=0) % 2 * 255).astype(np.float32)[..., None]
    out = images.astype(np.float32)
    out[:, -p:, -p:, :] = (1 - s) * out[:, -p:, -p:, :] + s * board
    return _to_u8(out)


def _backdoor(trigger, clean_label=False):
    def poison(raw, y, rng, n, ctx, s):
        t = ctx["target"]
        idx = base._pick(rng, np.flatnonzero(y == t if clean_label else y != t), n)
        raw[idx] = trigger(raw[idx], ctx, s)
        if not clean_label:
            y[idx] = t
        return idx
    return poison


def boundary_flip(raw, y, rng, n, ctx, s):
    """Source -> target flips on the source images the attacker's own label model rates most target-like.
    ctx["target_score"]: that model's target-class probability for every incoming image (the runner fits it)."""
    src = np.flatnonzero(y == ctx["source"])
    idx = src[np.argsort(-ctx["target_score"][src], kind="stable")[:n]]
    y[idx] = ctx["target"]
    return idx


def random_flip(raw, y, rng, n, ctx, s):
    """Reference: the same flips with random source-class picks (research/mm/attacks.label_flip_targeted)."""
    return base.label_flip_targeted(raw, y, rng, n, ctx)


def _identity(images, ctx, s):
    return images


FAMILIES = {
    "lowfreq": (_backdoor(lowfreq_trigger), lowfreq_trigger),
    "warp": (_backdoor(warp_trigger), warp_trigger),
    "faint_patch": (_backdoor(faint_patch_trigger), faint_patch_trigger),
    "cl_lowfreq": (_backdoor(lowfreq_trigger, clean_label=True), lowfreq_trigger),
    "boundary_flip": (boundary_flip, _identity),
    "random_flip": (random_flip, _identity),
}

FLIPS = {"boundary_flip", "random_flip"}   # ASR = source-class images predicted as target, no trigger
