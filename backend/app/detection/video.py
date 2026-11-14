"""Video for the image detector: 8 evenly spaced frames per clip (decoded by ffmpeg at 64x64), their 32x32 temporal
mean as the view the pixel and shared-pattern checks see (a pattern stamped on every frame survives, moving content
blurs; chosen on dev over a 2x2 mosaic and the middle frame), and the mean ResNet-18 frame embedding for the label and
shift checks. Benchmarked in research/bench_video.py;
the benchmark imports this module. Needs ffmpeg on the PATH plus requirements-images.txt."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

N_FRAMES, FRAME_SIZE = 8, 64
MOSAIC = (0, 2, 5, 7)            # which of the 8 frames tile the 2x2 view
DECODE_TIMEOUT = 60              # seconds per clip
MAX_DECODED = 4000               # frames read from ffmpeg per clip (about 2.5 min at 25 fps)


def available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_frames(path: str) -> np.ndarray:
    """Video file -> (N_FRAMES, 64, 64, 3) uint8, evenly spaced over the clip. Raises on unreadable input."""
    frame = FRAME_SIZE * FRAME_SIZE * 3
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-an", "-sn", "-frames:v", str(MAX_DECODED),
         "-vf", f"scale={FRAME_SIZE}:{FRAME_SIZE}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, timeout=DECODE_TIMEOUT, check=True)
    n = len(proc.stdout) // frame
    if n == 0:
        raise ValueError("no video frames")
    frames = np.frombuffer(proc.stdout[:n * frame], np.uint8).reshape(n, FRAME_SIZE, FRAME_SIZE, 3)
    return frames[np.linspace(0, n - 1, N_FRAMES).round().astype(int)]


def decode_bytes(data: bytes, suffix: str = ".mp4") -> np.ndarray:
    """As decode_frames, for an upload held in memory (written to a private temp file: many containers need seeking)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return decode_frames(path)
    finally:
        os.unlink(path)


def mosaic(frames: np.ndarray) -> np.ndarray:
    """(N, 8, 64, 64, 3) -> (N, 32, 32, 3): four frames downscaled to 16x16 in a 2x2 grid."""
    f = frames[:, list(MOSAIC)].astype(np.float32)
    n = len(f)
    small = f.reshape(n, 4, 16, 4, 16, 4, 3).mean(axis=(3, 5))          # 64 -> 16 by 4x4 block means
    top, bottom = np.concatenate([small[:, 0], small[:, 1]], 2), np.concatenate([small[:, 2], small[:, 3]], 2)
    return np.concatenate([top, bottom], 1).round().astype(np.uint8)


def _half(f: np.ndarray) -> np.ndarray:
    """(..., 64, 64, 3) float -> (..., 32, 32, 3) by 2x2 block means."""
    return f.reshape(*f.shape[:-3], 32, 2, 32, 2, 3).mean(axis=(-4, -2))


def temporal_mean(frames: np.ndarray) -> np.ndarray:
    """(N, 8, 64, 64, 3) -> (N, 32, 32, 3): the mean of the frames. A pattern stamped on every frame (a trigger, a
    noise mask) survives; moving content blurs out."""
    return _half(frames.astype(np.float32).mean(1)).round().astype(np.uint8)


def middle_frame(frames: np.ndarray) -> np.ndarray:
    """(N, 8, 64, 64, 3) -> (N, 32, 32, 3): one frame from the middle of the clip."""
    return _half(frames[:, N_FRAMES // 2].astype(np.float32)).round().astype(np.uint8)


VIEWS = {"mosaic": mosaic, "mean": temporal_mean, "middle": middle_frame}


def embed_clips(frames: np.ndarray, threads: int | None = None) -> np.ndarray:
    """(N, 8, 64, 64, 3) -> (N, 512): mean ResNet-18 embedding of the frames."""
    from app.detection.image_embed import embed_images
    n = len(frames)
    return embed_images(frames.reshape(n * N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), threads=threads).reshape(n, N_FRAMES, -1).mean(1)
