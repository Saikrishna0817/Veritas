"""Video dataset upload: ffmpeg decoding and an end-to-end trigger check (needs ffmpeg and requirements-images.txt)."""
import io
import subprocess
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.detection import image_embed, video
from app.main import app

pytestmark = pytest.mark.skipif(not (video.available() and image_embed.available()), reason="ffmpeg or image dependencies missing")
client = TestClient(app)
COLORS = [(220, 40, 40), (40, 220, 40), (40, 40, 220)]


def _h(uid):
    return {"Authorization": f"Bearer {create_access_token({'id': uid, 'name': uid, 'role': 'user'})}"}


def _encode(frames):
    """(T, 64, 64, 3) uint8 -> MPEG-4 in an AVI container, via ffmpeg."""
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x64", "-r", "10",
                           "-i", "pipe:0", "-c:v", "mpeg4", "-q:v", "2", "-f", "avi", "pipe:1"],
                          input=frames.tobytes(), capture_output=True, check=True).stdout


def _clip(rng, c, trigger=False):
    frames = np.full((16, 64, 64, 3), 30, np.uint8) + rng.integers(0, 20, (16, 64, 64, 3), dtype=np.uint8)
    x0, y0 = rng.integers(4, 30, 2)
    for t in range(16):                               # a coloured square moving right
        x = (x0 + 2 * t) % 44
        frames[t, y0:y0 + 16, x:x + 16] = COLORS[c]
    if trigger:
        frames[:, -8:, -8:] = (np.indices((8, 8)).sum(0) % 2 * 255)[..., None]
    return frames


def _zip(clips):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, frames in clips:
            zf.writestr(path, _encode(frames))
    return buf.getvalue()


def _wait(job_id, headers, timeout=600):
    for _ in range(int(timeout / 0.5)):
        job = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_decoding_gives_eight_frames_and_rejects_garbage():
    rng = np.random.default_rng(0)
    frames = video.decode_bytes(_encode(_clip(rng, 0)), ".avi")
    assert frames.shape == (video.N_FRAMES, 64, 64, 3) and video.mosaic(frames[None]).shape == (1, 32, 32, 3)
    with pytest.raises(Exception):
        video.decode_bytes(b"not a video")


def test_triggered_clips_are_flagged_first():
    rng = np.random.default_rng(1)
    ref = [(f"move{c}/r{i}.avi", _clip(rng, c)) for c in range(3) for i in range(40)]
    inc, bad = [], set()
    for c in range(3):
        for i in range(25):
            if c > 0 and len(bad) < 12:               # trigger on every frame, relabelled as move0
                bad.add(f"move0/bad{c}_{i}.avi")
                inc.append((f"move0/bad{c}_{i}.avi", _clip(rng, c, trigger=True)))
            else:
                inc.append((f"move{c}/i{i}.avi", _clip(rng, c)))
    h = _h("vid-alice")
    r = client.post("/api/v1/analyze/video", headers=h,
                    files={"file": ("inc.zip", _zip(inc), "application/zip"), "reference_file": ("ref.zip", _zip(ref), "application/zip")})
    assert r.status_code == 202, r.text
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "completed", job
    res = job["result"]
    assert res["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED")
    top = [f["file"] for f in res["flagged_clips"][:len(bad)]]
    assert len(set(top) & bad) >= 0.75 * len(bad), (top, res["checks"])
