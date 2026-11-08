"""Low-level pixel statistics, for attacks a semantic embedding cannot see.

A ResNet embedding is built to ignore corner patches and faint noise; these features are built to keep them:
per-cell brightness and high-frequency energy on a 4x4 grid (a stamped patch lights up one cell), a radial
power spectrum (blended and additive noise raise the high bands), per-channel means and the exact corner pixels.
"""
from __future__ import annotations

import numpy as np

GRID = 4
RADIAL_BINS = 8


def pixel_features(raw: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, D) float32."""
    x = raw.astype(np.float32) / 255.0
    gray = x.mean(axis=-1)
    n, h, w = gray.shape
    # high-pass residual: pixel minus the mean of its 4 neighbours
    pad = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="edge")
    hf = np.abs(gray - 0.25 * (pad[:, :-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, :-2] + pad[:, 1:-1, 2:]))
    cells = lambda a: a[:, :h // GRID * GRID, :w // GRID * GRID].reshape(n, GRID, h // GRID, GRID, w // GRID)  # noqa: E731
    cell_mean = cells(gray).mean(axis=(2, 4)).reshape(n, -1)
    cell_hf = cells(hf).mean(axis=(2, 4)).reshape(n, -1)
    # radial power spectrum of the gray image
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    yy, xx = np.indices((h, w))
    radius = np.hypot(yy - h / 2, xx - w / 2)
    edges = np.linspace(0, radius.max() + 1e-6, RADIAL_BINS + 1)
    which = np.digitize(radius, edges) - 1
    spectrum = np.stack([np.log1p(power[:, which == b].mean(axis=1)) for b in range(RADIAL_BINS)], axis=1)
    channels = np.concatenate([x.mean(axis=(1, 2)), x.std(axis=(1, 2))], axis=1)
    k = max(3, h // 8)
    corners = np.concatenate([x[:, :k, :k].reshape(n, -1), x[:, -k:, -k:].reshape(n, -1)], axis=1)
    return np.concatenate([cell_mean, cell_hf, spectrum, channels, corners], axis=1).astype(np.float32)
