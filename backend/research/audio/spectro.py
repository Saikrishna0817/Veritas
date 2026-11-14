"""Waveform -> log-mel spectrogram image, so the image detector (veritas_mm) can check audio unchanged.

The decibel range is fixed (not per clip), so a louder or quieter poisoned clip stays visible. Rows are mel bands
(low frequencies at the bottom), columns are time; the gray image is repeated into three channels.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import stft

RATE, N_FFT, HOP, N_MELS = 16_000, 512, 256, 64
DB_RANGE = (-90.0, 0.0)


def _mel_filters(n_mels=N_MELS, n_fft=N_FFT, rate=RATE) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular HTK mel filterbank."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)            # noqa: E731
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)              # noqa: E731
    edges = hz(np.linspace(mel(0), mel(rate / 2), n_mels + 2))
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))


_FB = _mel_filters()


def spectrogram_images(wave: np.ndarray, size: int = 32, chunk: int = 256) -> np.ndarray:
    """(N, L) int16 or float in [-1, 1] -> (N, size, size, 3) uint8, in chunks to bound memory."""
    out = []
    for i in range(0, len(wave), chunk):
        x = np.asarray(wave[i:i + chunk]).astype(np.float32) / (32768.0 if wave.dtype == np.int16 else 1.0)
        _, _, Z = stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
        db = 10 * np.log10(np.einsum("mf,nft->nmt", _FB, np.abs(Z) ** 2) + 1e-12)
        g = np.clip((db - DB_RANGE[0]) / (DB_RANGE[1] - DB_RANGE[0]), 0, 1)[:, ::-1]   # low frequencies at the bottom
        g = zoom(g, (1, size / g.shape[1], size / g.shape[2]), order=1)
        out.append((g * 255).round().astype(np.uint8))
    return np.repeat(np.concatenate(out)[..., None], 3, axis=-1)
