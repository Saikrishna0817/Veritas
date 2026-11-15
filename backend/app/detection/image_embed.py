"""Image embeddings for the image detector: ImageNet ResNet-18, penultimate layer, 512-d (as benchmarked).
Needs the optional image dependencies (requirements-images.txt); the CSV-only app runs without them."""
from __future__ import annotations

import numpy as np

_model = None


def available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model
    if _model is None:
        import torch
        import torchvision
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        _model = net.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return _model


def embed_images(images: np.ndarray, batch: int = 256, threads: int | None = None) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 512) float32. ``threads`` caps torch's CPU threads (the app passes one; the benchmark
    leaves torch's default, which follows OMP_NUM_THREADS)."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    model = _get_model()
    device = next(model.parameters()).device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch])).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)
            out.append(model((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
