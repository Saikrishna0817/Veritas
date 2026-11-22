"""UCF-101 (101 human actions, YouTube clips) for the video poisoning benchmark.

Videos come from the Hugging Face mirror MichiganNLP/ucf-101 (UCF101.zip, a ZIP repack of the official RAR; its split
file is byte-identical to the official one, SHA-256 in the manifest). Official split 1 separates clips by group
(actors and scene): dev = trainlist01 groups, test = testlist01 groups. Frames are decoded by the app's
``video.decode_frames`` (8 evenly spaced frames at 64x64) and cached as one uint8 array per split.
"""
from __future__ import annotations

import subprocess
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from app.detection import video

ROOT = Path.home() / ".cache" / "veritas-bench" / "video"
ZIP = ROOT / "ucf101-hf" / "UCF101.zip"
SPLITS = ROOT / "ucf101" / "UCF101TrainTestSplits-RecognitionTask.zip"


HMDB = ROOT / "hmdb51"
# Hugging Face mirror MichiganNLP/hmdb (ZIP repack of the official RARs, linked from serre.lab.brown.edu/hmdb51; the lab's
# own Google Drive links returned 404 on 2026-11-22). SHA-256 equals the mirror's LFS object id.
HMDB_SHA256 = {"hmdb51_org.zip": "2d652f386debd04ae8f83508faacb959c03c9cbf1176fc76848b73cda4d52e57",
               "test_train_splits.zip": "12df0053bd9d570817edb10d2299030f43646969d5d42ff40200249b45ab2efa"}


@dataclass
class VideoSet:
    name: str
    frames: np.ndarray      # (N, 8, 64, 64, 3) uint8, memory-mapped
    y: np.ndarray
    classes: list

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def _split_list(split: str) -> list[str]:
    with zipfile.ZipFile(SPLITS) as zf:
        text = zf.read(f"ucfTrainTestlist/{'trainlist01' if split == 'dev' else 'testlist01'}.txt").decode()
    return sorted(line.split()[0] for line in text.splitlines() if line.strip())


def _decode_member(name: str) -> np.ndarray:
    with zipfile.ZipFile(ZIP) as zf, tempfile.NamedTemporaryFile(suffix=".avi") as tmp:   # our own download
        tmp.write(zf.read(name))
        tmp.flush()
        return video.decode_frames(tmp.name)


def _untar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(dest)], check=True)     # our own download, checksum checked


def _check_sha256(path: Path) -> None:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != HMDB_SHA256[path.name]:
        raise RuntimeError(f"{path} has SHA-256 {h.hexdigest()}, expected {HMDB_SHA256[path.name]}")


def _hmdb_list(split: str) -> list[str]:
    """Official split 1 as '<class>/<clip>.avi', sorted. Checks and extracts the archives once. The mirror's zip holds
    every clip twice (<class>/<clip>.avi and <class>/<class>/<clip>.avi, same CRC); only the first is extracted.
    In the split files 1 = train (dev here), 2 = test, 0 = unused."""
    raw, spl = HMDB / "raw", HMDB / "splits"
    if not any(raw.glob("*/*.avi")):
        _check_sha256(HMDB / "hmdb51_org.zip")
        with zipfile.ZipFile(HMDB / "hmdb51_org.zip") as zf:
            for name in zf.namelist():
                parts = PurePosixPath(name).parts
                if len(parts) == 2 and name.endswith(".avi") and ".." not in parts:
                    zf.extract(name, raw)
    if not any(spl.rglob("*_split1.txt")):
        _check_sha256(HMDB / "test_train_splits.zip")
        _untar(HMDB / "test_train_splits.zip", spl)
    want = "1" if split == "dev" else "2"
    return sorted(f"{f.name[:-len('_test_split1.txt')]}/{name}" for f in spl.rglob("*_test_split1.txt")
                  for name, flag in (line.split()[:2] for line in f.read_text().splitlines() if line.strip()) if flag == want)


def _decode_path(path: str) -> np.ndarray:
    return video.decode_frames(path)


def _hmdb51(split: str, workers: int):
    wanted = _hmdb_list(split)
    with ProcessPoolExecutor(workers) as pool:
        frames = np.stack(list(pool.map(_decode_path, [str(HMDB / "raw" / w) for w in wanted], chunksize=16)))
    classes = sorted({w.split("/")[0] for w in wanted})
    return frames, np.array([classes.index(w.split("/")[0]) for w in wanted]), classes


def load(split: str, workers: int = 4, data_id: str = "ucf101") -> VideoSet:
    """``split`` is 'dev' or 'test'; ``data_id`` 'ucf101' (default) or 'hmdb51'. First call decodes (ffmpeg, in
    parallel) and caches; later calls memory-map."""
    cache = ROOT / "cache" / f"{data_id}-{split}"
    if data_id == "hmdb51" and not (cache / "frames.npy").exists():
        frames, y, classes = _hmdb51(split, workers)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "frames.npy", frames)
        np.save(cache / "y.npy", y)
        (cache / "classes.txt").write_text("\n".join(classes))
    if not (cache / "frames.npy").exists():
        wanted = _split_list(split)
        with zipfile.ZipFile(ZIP) as zf:
            by_base = {PurePosixPath(n).name: n for n in zf.namelist() if n.lower().endswith(".avi")}
        missing = [w for w in wanted if PurePosixPath(w).name not in by_base]
        if missing:
            raise RuntimeError(f"{len(missing)} clips of the official {split} list are not in the zip, e.g. {missing[:3]}")
        with ProcessPoolExecutor(workers) as pool:
            frames = np.stack(list(pool.map(_decode_member, [by_base[PurePosixPath(w).name] for w in wanted], chunksize=16)))
        classes = sorted({w.split("/")[0] for w in wanted})
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "frames.npy", frames)
        np.save(cache / "y.npy", np.array([classes.index(w.split("/")[0]) for w in wanted]))
        (cache / "classes.txt").write_text("\n".join(classes))
    return VideoSet(f"{data_id}-{split}", np.load(cache / "frames.npy", mmap_mode="r"), np.load(cache / "y.npy"),
                    (cache / "classes.txt").read_text().split("\n"))
