"""Audio dataset ingestion: a zip of WAV files with one folder per class (``dog/001.wav``), optionally a second zip
of trusted reference clips. Decoded in memory with the standard-library ``wave`` module (PCM WAV), mixed to mono,
resampled to 16 kHz and padded or trimmed to one common length (the median clip, at most 10 s)."""
from __future__ import annotations

import io
import wave

import numpy as np

from app.ingestion.image_engine import ingest_zip

RATE = 16_000
MAX_SECONDS = 10
EXTENSIONS = (".wav",)


def _decode_wav(raw: bytes) -> np.ndarray:
    """PCM WAV bytes -> float32 mono at 16 kHz in [-1, 1] (raises on anything else)."""
    from scipy.signal import resample_poly
    with wave.open(io.BytesIO(raw)) as w:
        width, channels, rate, n = w.getsampwidth(), w.getnchannels(), w.getframerate(), w.getnframes()
        if n > rate * 60:
            raise ValueError("clip longer than a minute")
        data = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(data, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(data, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608
    elif width == 4:
        x = np.frombuffer(data, "<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError("unsupported sample width")
    x = x.reshape(-1, channels).mean(1)
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
    def ingest(self, zip_bytes: bytes, filename: str, reference_bytes: bytes | None = None) -> dict:
        out = ingest_zip(zip_bytes, filename, reference_bytes, EXTENSIONS, _decode_wav, "clip", _stack)
        out["clips"], out["n_clips"] = out.pop("items"), out.pop("n_items")
        return out
