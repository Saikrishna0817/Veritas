"""CIFAR-10N: real human label noise on CIFAR-10, as tabular embeddings.

CIFAR-10N (Wei et al., ICLR 2022, UCSC-REAL) re-labelled the 50k CIFAR-10
training images with Mechanical Turk workers. A worker label that differs from
the original CIFAR-10 label is a *real* labelling mistake, so ground truth for
"which rows are wrong" comes from humans, not from our injector.

Images are embedded once with an ImageNet-pretrained ResNet-18 (penultimate
layer, 512-d) and reduced to 32 dims with PCA. The result is cached; the
benchmark then treats it like any other tabular dataset.

Run once:  python -m research.noise_data
"""
from __future__ import annotations

import hashlib
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(os.getenv("VERITAS_BENCH_CACHE", Path.home() / ".cache" / "veritas-bench")) / "cifar10n"
CIFAR_URL = "https://huggingface.co/datasets/xingslong/cifar-10-batches-py/resolve/main/cifar-10-python.tar.gz"  # mirror of the official tarball
HUMAN_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"  # published with CIFAR-10
HUMAN_SHA256 = "873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310"  # CIFAR-10_human.pt as fetched on 2026-11-06
EMBEDDINGS = CACHE / "embeddings-resnet18-pca32.npz"
NOISE_SOURCES = ("aggre_label", "random_label1", "worse_label")  # ~9%, ~17%, ~40% wrong
PCA_DIMS = 32


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest.with_suffix(".part"))
        dest.with_suffix(".part").rename(dest)
    return dest


def _load_cifar_train() -> tuple[np.ndarray, np.ndarray]:
    """(50000, 32, 32, 3) uint8 images and labels in the official CIFAR-10 order.

    The official batches are pickles, so the archive is only opened after its MD5 matches the
    value published with CIFAR-10, and nothing is extracted to disk.
    """
    tar = _download(CIFAR_URL, CACHE / "cifar-10-python.tar.gz")
    if hashlib.md5(tar.read_bytes()).hexdigest() != CIFAR_MD5:
        raise RuntimeError(f"{tar} does not match the official CIFAR-10 checksum; delete it and retry")
    xs, ys = [], []
    with tarfile.open(tar) as t:
        for i in range(1, 6):
            batch = pickle.load(t.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="latin1")
            xs.append(batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.asarray(batch["labels"]))
    return np.concatenate(xs), np.concatenate(ys)


def _load_human_labels() -> dict[str, np.ndarray]:
    """torch.load restricted to the NumPy reconstruction globals this file needs, after a SHA-256 check."""
    import hashlib

    import torch

    path = _download(HUMAN_URL, CACHE / "CIFAR-10_human.pt")
    if hashlib.sha256(path.read_bytes()).hexdigest() != HUMAN_SHA256:
        raise RuntimeError(f"{path} does not match the pinned CIFAR-10N checksum; delete it and retry")
    allowed = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),  # name in the file
               np.ndarray, np.dtype, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        raw = torch.load(path, weights_only=True)
    return {k: np.asarray(v).reshape(-1) for k, v in raw.items()}


def _embed(images: np.ndarray, chunk: int = 5000, batch: int = 250) -> np.ndarray:
    """ResNet-18 features, saved per chunk so an interrupted run resumes instead of restarting."""
    import torch
    import torchvision

    torch.set_num_threads(os.cpu_count() or 4)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    parts = []
    for start in range(0, len(images), chunk):
        part = CACHE / f"resnet18-{start:05d}.npy"
        if not part.exists():
            out = []
            with torch.inference_mode():
                for i in range(start, min(start + chunk, len(images)), batch):
                    x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
                    x = torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)
                    out.append(model((x - mean) / std).numpy())
            np.save(part, np.concatenate(out))
            print(f"embedded {min(start + chunk, len(images))}/{len(images)}", flush=True)
        parts.append(np.load(part))
    return np.concatenate(parts)


def build() -> Path:
    from sklearn.decomposition import PCA

    images, cifar_labels = _load_cifar_train()
    human = _load_human_labels()
    # The human file stores the original CIFAR-10 labels: matching all 50k proves the image order is right.
    assert (human["clean_label"] == cifar_labels).all(), "CIFAR-10N rows do not match CIFAR-10 order"
    features = PCA(PCA_DIMS, random_state=0).fit_transform(_embed(images)).astype(np.float32)
    np.savez_compressed(EMBEDDINGS, X=features, clean=cifar_labels, **{k: human[k] for k in NOISE_SOURCES})
    for part in CACHE.glob("resnet18-*.npy"):
        part.unlink()
    return EMBEDDINGS


def load() -> dict[str, np.ndarray]:
    """{'X': (50000, 32), 'clean': labels, 'aggre_label': ..., 'random_label1': ..., 'worse_label': ...}"""
    if not EMBEDDINGS.exists():
        build()
    with np.load(EMBEDDINGS) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    data = load()
    for src in NOISE_SOURCES:
        print(f"{src}: {(data[src] != data['clean']).mean():.3f} of labels are wrong")
    print(build() if not EMBEDDINGS.exists() else EMBEDDINGS)
