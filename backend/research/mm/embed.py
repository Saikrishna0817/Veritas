"""Image embeddings (ImageNet ResNet-18, penultimate layer, 512-d) with an on-disk cache for clean sets."""
from __future__ import annotations

import numpy as np

from research.mm.datasets import CACHE, ImageSet

from app.detection.image_embed import embed_images  # noqa: E402  (the shipped embedding)


def clean_embeddings(ds: ImageSet) -> np.ndarray:
    path = CACHE / f"{ds.name}-{len(ds.y)}-resnet18.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embed_images(ds.raw))
    return np.load(path)
