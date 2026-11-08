"""Real image datasets for the multimodal benchmark.

Everything downloads through torchvision, which verifies each archive's MD5, into ``VERITAS_BENCH_CACHE``
(default ``~/.cache/veritas-bench/mm``). ``load`` returns at most ``MAX_POOL`` images, stratified, as
uint8 (N, H, W, 3) so attacks work on pixels and the embedder sees ordinary images.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE = Path(os.getenv("VERITAS_BENCH_CACHE", Path.home() / ".cache" / "veritas-bench")) / "mm"
MAX_POOL = 20_000


@dataclass
class ImageSet:
    name: str
    raw: np.ndarray  # (N, H, W, 3) uint8
    y: np.ndarray  # (N,) int64 trusted labels
    human_labels: dict[str, np.ndarray] = field(default_factory=dict)  # real annotator labels, when published

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1


def _to_rgb(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:  # (N, H, W) grayscale
        images = images[..., None]
    return np.repeat(images, 3, axis=-1) if images.shape[-1] == 1 else images


def _tv(cls, **kw):
    import torchvision.datasets as tvd
    return getattr(tvd, cls)(root=str(CACHE / cls.lower()), download=True, **kw)


def _cifar10():
    ds = _tv("CIFAR10", train=True)
    return ds.data, np.asarray(ds.targets)


def _cifar100():
    ds = _tv("CIFAR100", train=True)
    return ds.data, np.asarray(ds.targets)


def _fashion_mnist():
    ds = _tv("FashionMNIST", train=True)
    return ds.data.numpy(), ds.targets.numpy()


def _gtsrb():
    import torch
    from torchvision import transforms
    ds = _tv("GTSRB", split="train")
    resize = transforms.Resize((32, 32))
    images = np.stack([np.asarray(resize(img.convert("RGB"))) for img, _ in ds])
    return images, np.asarray([label for _, label in ds])


LOADERS = {"cifar10": _cifar10, "cifar100": _cifar100, "fashion_mnist": _fashion_mnist, "gtsrb": _gtsrb}


def load(name: str, seed: int = 0) -> ImageSet:
    from sklearn.model_selection import train_test_split

    raw, y = LOADERS[name]()
    raw, y = _to_rgb(raw), np.asarray(y, dtype=np.int64)
    keep = np.arange(len(y))
    if len(y) > MAX_POOL:
        keep, _ = train_test_split(keep, train_size=MAX_POOL, stratify=y, random_state=seed)
        keep.sort()
    out = ImageSet(name, np.ascontiguousarray(raw[keep]), y[keep])
    if name == "cifar10":  # CIFAR-10N: Mechanical Turk labels for the same 50k training images
        from research import noise_data
        human = noise_data._load_human_labels()
        out.human_labels = {k: human[k][keep] for k in noise_data.NOISE_SOURCES}
    return out
