"""Real image datasets for the multimodal benchmark.

Everything downloads through torchvision, which verifies each archive's MD5, into ``VERITAS_BENCH_CACHE``
(default ``~/.cache/veritas-bench/mm``). ``load`` returns at most ``MAX_POOL`` images, stratified, as
uint8 (N, H, W, 3) so attacks work on pixels and the embedder sees ordinary images.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
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


def _svhn():
    ds = _tv("SVHN", split="train")
    return ds.data.transpose(0, 2, 3, 1), np.asarray(ds.labels)  # stored CHW; labels 0-9


TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_SHA256 = "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821"  # as fetched on 2026-11-16


def _tiny_imagenet():
    """200 classes x 500 train images, 64x64 JPEG -> 32x32 with the app's own decoder. Decoded once, cached as .npy."""
    d = CACHE / "tiny-imagenet"
    npy = d / "train32.npz"
    if not npy.exists():
        from app.ingestion.image_engine import _decode_image
        zpath = d / "tiny-imagenet-200.zip"
        if not zpath.exists():
            d.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(TINY_URL, zpath)
        if hashlib.sha256(zpath.read_bytes()).hexdigest() != TINY_SHA256:
            raise RuntimeError(f"{zpath} does not match the pinned SHA-256; delete it and retry")
        with zipfile.ZipFile(zpath) as z:  # read in place: nothing is extracted
            names = sorted(n for n in z.namelist() if "/train/" in n and n.endswith(".JPEG"))
            wnids = sorted({n.split("/")[-3] for n in names})
            y = np.asarray([wnids.index(n.split("/")[-3]) for n in names])
            raw = np.stack([_decode_image(z.read(n)) for n in names])
        np.savez(npy, raw=raw, y=y)
    with np.load(npy) as f:
        return f["raw"], f["y"]


LOADERS = {"cifar10": _cifar10, "cifar100": _cifar100, "fashion_mnist": _fashion_mnist, "gtsrb": _gtsrb,
           "svhn": _svhn, "tiny_imagenet": _tiny_imagenet}


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
