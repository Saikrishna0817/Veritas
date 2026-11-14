"""Video dataset ingestion: a zip of video files with one folder per class (``jump/001.mp4``), optionally a second
zip of trusted reference clips. Each clip is decoded by ffmpeg (app.detection.video) into 8 evenly spaced 64x64
frames; nothing but the frames is kept."""
from __future__ import annotations

import numpy as np

from app.detection import video
from app.ingestion.image_engine import ingest_zip

EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
MAX_VIDEO_FILE = 200 << 20       # 200 MB per clip

class VideoIngestionEngine:
    def ingest(self, zip_bytes: bytes, filename: str, reference_bytes: bytes | None = None) -> dict:
        # The decoder writes each clip to a private temp file (many containers need seeking); the suffix only helps
        # ffmpeg guess the container, it reads the real format from the bytes.
        out = ingest_zip(zip_bytes, filename, reference_bytes, EXTENSIONS, lambda raw: video.decode_bytes(raw, ".bin"), "clip",
                         lambda a, w: np.stack(a), max_file=MAX_VIDEO_FILE)
        out["frames"], out["n_clips"] = out.pop("items"), out.pop("n_items")
        return out
