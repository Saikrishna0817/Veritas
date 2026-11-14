"""Audio datasets for the poisoning benchmark: Speech Commands v0.02 (35 words, 1 s) and ESC-50 (50 sound classes, 5 s).

Read with the standard-library ``wave`` module (both are 16-bit PCM WAV), resampled to 16 kHz, cached as int16 arrays.
Splits keep dev and test clips apart: Speech Commands' validation list (dev) and testing list (test), which have
disjoint speakers; ESC-50 folds 1-3 (dev) and 4-5 (test). Same two sources on both sides, so this is a weaker held-out
test than the image track's separate datasets.
"""
from __future__ import annotations

import csv
import tarfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path.home() / ".cache" / "veritas-bench" / "audio"
RATE = 16_000


@dataclass
class AudioSet:
    name: str
    wave: np.ndarray        # (N, L) int16 at 16 kHz (memory-mapped)
    y: np.ndarray
    classes: list

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def _read(data: bytes, length: int) -> np.ndarray:
    import io
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels()).mean(1)
        rate = w.getframerate()
    if rate != RATE:
        g = np.gcd(rate, RATE)
        x = resample_poly(x, RATE // g, rate // g)
    out = np.zeros(length, np.float32)
    out[:min(length, len(x))] = x[:length]
    return np.clip(out, -32768, 32767).astype(np.int16)


def _speech_commands(split: str):
    """Both splits in two in-order passes over the .tar.gz (gzip cannot seek: out-of-order reads restart it)."""
    tar = ROOT / "speech_commands" / "speech_commands_v0.02.tar.gz"
    lists = {}
    with tarfile.open(tar, "r|gz") as tf:     # our own download; members are only read, never extracted
        for m in tf:
            if m.name.lstrip("./") in ("validation_list.txt", "testing_list.txt"):
                lists["dev" if "validation" in m.name else "test"] = set(tf.extractfile(m).read().decode().split())
    want = {n: sp for sp, names in lists.items() for n in names}
    clips = {"dev": {}, "test": {}}
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            name = m.name.lstrip("./")
            if m.isfile() and name in want:
                clips[want[name]][name] = _read(tf.extractfile(m).read(), RATE)
    classes = sorted({n.split("/")[0] for n in want})
    out = {}
    for sp, d in clips.items():
        names = sorted(d)
        out[sp] = (np.stack([d[n] for n in names]), np.array([classes.index(n.split("/")[0]) for n in names]), classes)
    return out[split], out


def _esc50(split: str):
    folds = {1, 2, 3} if split == "dev" else {4, 5}
    with zipfile.ZipFile(ROOT / "esc50" / "ESC-50-master.zip") as zf:
        meta = list(csv.DictReader(zf.read("ESC-50-master/meta/esc50.csv").decode().splitlines()))
        rows = sorted((r for r in meta if int(r["fold"]) in folds), key=lambda r: r["filename"])
        classes = sorted({r["category"] for r in meta})
        x = np.stack([_read(zf.read(f"ESC-50-master/audio/{r['filename']}"), 5 * RATE) for r in rows])
    return x, np.array([classes.index(r["category"]) for r in rows]), classes


LOADERS = {"speech_commands": _speech_commands, "esc50": _esc50}


def load(name: str, split: str) -> AudioSet:
    """``split`` is 'dev' or 'test'. First call decodes and caches; later calls memory-map the cache."""
    cache = ROOT / "cache" / f"{name}-{split}"
    if not (cache / "wave.npy").exists():
        loaded = LOADERS[name](split)
        for sp, (x, y, classes) in (loaded[1].items() if name == "speech_commands" else [(split, loaded)]):
            c = ROOT / "cache" / f"{name}-{sp}"
            c.mkdir(parents=True, exist_ok=True)
            np.save(c / "wave.npy", x)
            np.save(c / "y.npy", y)
            (c / "classes.txt").write_text("\n".join(classes))
    return AudioSet(f"{name}-{split}", np.load(cache / "wave.npy", mmap_mode="r"), np.load(cache / "y.npy"),
                    (cache / "classes.txt").read_text().split("\n"))
