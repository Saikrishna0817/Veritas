"""Audio poisoning attacks on raw 16 kHz waveforms (float32 in [-1, 1]), mirroring the image attacks.

Each attack mutates ``x`` (n, L) and ``y`` in place and returns the poisoned row indices. Simplified published recipes:
a short tone burst as the trigger (audio BadNets), a fixed noise track blended in (Chen et al.), the same tone on
correctly labelled target-class clips (clean-label), copies of a few clips, and a faint fixed per-class noise pattern
(availability attack). A miss here is a miss on the easy version of the attack.
"""
from __future__ import annotations

import numpy as np

from research.mm.attacks import _pick, duplicate_flood, label_flip_random, label_flip_targeted  # same recipes

RATE = 16_000


def tone_backdoor(x, y, rng, n, ctx):
    """A 100 ms tone at the start of non-target clips, relabelled to the target class."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    x[idx, :len(ctx["tone"])] += ctx["tone"]
    y[idx] = ctx["target"]
    return idx


def blended_backdoor(x, y, rng, n, ctx):
    """A fixed noise track mixed under the whole clip (10% of full scale), relabelled to the target class."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    x[idx] = 0.9 * x[idx] + ctx["blend"]
    y[idx] = ctx["target"]
    return idx


def clean_label_backdoor(x, y, rng, n, ctx):
    """The tone trigger on target-class clips whose labels stay correct."""
    idx = _pick(rng, np.flatnonzero(y == ctx["target"]), n)
    x[idx, :len(ctx["tone"])] += ctx["tone"]
    return idx


def class_noise(x, y, rng, n, ctx):
    """A faint fixed per-class noise pattern (1% of full scale) added to correctly labelled clips."""
    idx = _pick(rng, np.arange(len(y)), n)
    x[idx] += ctx["class_noise"][y[idx]]
    return idx


ATTACKS = {f.__name__: f for f in (label_flip_random, label_flip_targeted, tone_backdoor, blended_backdoor,
                                   clean_label_backdoor, duplicate_flood, class_noise)}


def make_context(x: np.ndarray, n_classes: int, rng) -> dict:
    source, target = rng.choice(n_classes, size=2, replace=False)
    t = np.arange(int(0.1 * RATE)) / RATE
    freq = rng.uniform(2000, 6000)
    return {"n_classes": n_classes, "source": int(source), "target": int(target),
            "tone": (0.2 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))).astype(np.float32),
            "blend": (0.1 * rng.uniform(-1, 1, x.shape[1])).astype(np.float32),
            "class_noise": (0.01 * rng.choice([-1.0, 1.0], size=(n_classes, x.shape[1]))).astype(np.float32)}
