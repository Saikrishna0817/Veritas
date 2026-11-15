"""Backdoor attacks written by someone else: BackdoorBench (SCLBD/BackdoorBench, CC BY-NC 4.0).

Their transform modules are NOT copied here; they are loaded at runtime from VERITAS_BACKDOORBENCH (a sparse
checkout at the pinned commit, see the manifest's source_attacks). Same interface as attacks.py. Parameters are
their CIFAR-10 defaults (config/attack/<name>/default.yaml); the images here are 32x32 like theirs.
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from research.mm.attacks import _pick

BB_COMMIT = "f02e3534645f0ee63d6848653062cd6c0d6c400d"
_ROOT = Path(os.getenv("VERITAS_BACKDOORBENCH", "~/.cache/veritas-ext/backdoorbench")).expanduser()
BLEND_ALPHA = 0.2  # config/attack/blended/default.yaml attack_train_blended_alpha
SIG = {"delta": 40, "f": 6}  # config/attack/sig/default.yaml
FTROJAN = {"YUV": True, "channel_list": [1, 2], "window_size": 32, "magnitude": 30,
           "pos_list": [(15, 15), (31, 31)]}  # config/attack/ftrojannn/default.yaml (pos_list 15,15,31,31)


@lru_cache(maxsize=None)
def _load(name: str):
    if not hasattr(np, "float"):
        np.float = float  # BackdoorBench uses np.float, removed in numpy 1.24+
    path = _ROOT / "utils" / "bd_img_transform" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_uint8(x) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)  # their npClipAndToUint8


def _resized(path: Path, hw: tuple[int, int], resample) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(hw[::-1], resample), dtype=np.float32)


def _dirty(raw, y, rng, n, ctx, trigger) -> np.ndarray:
    """Trigger on non-target images, relabelled to the target class."""
    idx = _pick(rng, np.flatnonzero(y != ctx["target"]), n)
    raw[idx] = _to_uint8(trigger(raw[idx]))
    y[idx] = ctx["target"]
    return idx


def bb_badnets(raw, y, rng, n, ctx):
    mask = _resized(_ROOT / "resource/badnet/trigger_image.png", raw.shape[1:3], Image.NEAREST)
    return _dirty(raw, y, rng, n, ctx, _load("patch").AddMaskPatchTrigger(mask))


def bb_blended(raw, y, rng, n, ctx):
    kitty = _resized(_ROOT / "resource/blended/hello_kitty.jpeg", raw.shape[1:3], Image.BILINEAR)
    return _dirty(raw, y, rng, n, ctx, _load("blended").blendedImageAttack(kitty, BLEND_ALPHA))


def bb_sig(raw, y, rng, n, ctx):
    """Clean-label: sinusoid on target-class images, labels unchanged."""
    idx = _pick(rng, np.flatnonzero(y == ctx["target"]), n)
    sig = _load("sig").sigTriggerAttack(**SIG)
    raw[idx] = np.stack([_to_uint8(sig(im)) for im in raw[idx]])
    return idx


def bb_lowfreq(raw, y, rng, n, ctx):
    # the CIFAR-10 preactresnet18 universal perturbation, applied unchanged to every 32x32 dataset
    pert = np.load(_ROOT / "resource/lowFrequency/cifar10_preactresnet18_0_255.npy")
    trigger = _load("patch").SimpleAdditiveTrigger(pert[0] if pert.ndim == 4 else pert)
    return _dirty(raw, y, rng, n, ctx, trigger)


def bb_ftrojan(raw, y, rng, n, ctx):
    ft = _load("ftrojann").ftrojann_version(**FTROJAN)
    return _dirty(raw, y, rng, n, ctx, lambda x: np.stack([ft(im) for im in x]))


ATTACKS = {f.__name__: f for f in (bb_badnets, bb_blended, bb_sig, bb_lowfreq, bb_ftrojan)}
