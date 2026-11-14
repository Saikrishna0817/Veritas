"""UCF-101 (101 human actions, YouTube clips) for the video poisoning benchmark.

Videos come from the Hugging Face mirror MichiganNLP/ucf-101 (UCF101.zip, a ZIP repack of the official RAR; its split
file is byte-identical to the official one, SHA-256 in the manifest). Official split 1 separates clips by group
(actors and scene): dev = trainlist01 groups, test = testlist01 groups. Frames are decoded by the app's
``video.decode_frames`` (8 evenly spaced frames at 64x64) and cached as one uint8 array per split.
"""
from __future__ import annotations

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


def load(split: str, workers: int = 4) -> VideoSet:
    """``split`` is 'dev' or 'test'. First call decodes (ffmpeg, in parallel) and caches; later calls memory-map."""
    cache = ROOT / "cache" / f"ucf101-{split}"
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
    return VideoSet(f"ucf101-{split}", np.load(cache / "frames.npy", mmap_mode="r"), np.load(cache / "y.npy"),
                    (cache / "classes.txt").read_text().split("\n"))
