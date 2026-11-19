"""Audio dataset ingestion: a zip of WAV files with one folder per class (``dog/001.wav``), optionally a second zip
of trusted reference clips. Decoded in memory with scipy (PCM or float WAV), mixed to mono,
resampled to 16 kHz and padded or trimmed to one common length (the median clip, at most 10 s)."""
from __future__ import annotations

import io

import numpy as np

from app.ingestion.image_engine import ingest_zip

RATE = 16_000
MAX_SECONDS = 10
EXTENSIONS = (".wav",)


def _decode_wav(raw: bytes) -> np.ndarray:
    """WAV bytes (PCM 8/16/24/32-bit or 32/64-bit float) -> float32 mono at 16 kHz in [-1, 1] (raises on anything else)."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    try:
        rate, x = wavfile.read(io.BytesIO(raw))
    except ValueError as exc:
        if "format" in str(exc):
            raise ValueError("compressed WAV (such as ADPCM or mu-law) is not supported; convert it to PCM") from exc
        raise
    if len(x) > rate * 60:
        raise ValueError("clip longer than a minute")
    scale = {"u": 128.0, "i": float(2 ** (8 * x.dtype.itemsize - 1)), "f": 1.0}[x.dtype.kind]
    x = (x.astype(np.float32) - (128 if x.dtype.kind == "u" else 0)) / scale   # 24-bit arrives left-justified in int32
    x = x.reshape(len(x), -1).mean(1)
    if rate != RATE:
        g = np.gcd(rate, RATE)
        x = resample_poly(x, RATE // g, rate // g).astype(np.float32)
    return x


def _stack(clips: list[np.ndarray], warnings: list[str]) -> np.ndarray:
    length = int(min(np.median([len(c) for c in clips]), MAX_SECONDS * RATE))
    out = np.zeros((len(clips), length), np.int16)
    for i, c in enumerate(clips):
        out[i, :min(length, len(c))] = (np.clip(c[:length], -1, 1) * 32767).astype(np.int16)
    if any(len(c) != length for c in clips):
        warnings.append(f"Clips were padded or trimmed to {length / RATE:.2f} s (the median length, at most {MAX_SECONDS} s).")
    return out


class AudioIngestionEngine:
    def ingest(self, zip_bytes: bytes, filename: str, reference_bytes: bytes | None = None, self_check: bool = False) -> dict:
        out = ingest_zip(zip_bytes, filename, reference_bytes, EXTENSIONS, _decode_wav, "clip", _stack, self_check=self_check)
        out["clips"], out["n_clips"] = out.pop("items"), out.pop("n_items")
        return out
