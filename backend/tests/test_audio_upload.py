"""Audio dataset upload: WAV decoding and an end-to-end trigger check (needs requirements-images.txt with torchaudio)."""
import io
import time
import wave
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.ingestion.audio_engine import _decode_wav
from app.main import app

torchaudio = pytest.importorskip("torchaudio")
client = TestClient(app)
RATE = 16_000


def _h(uid):
    return {"Authorization": f"Bearer {create_access_token({'id': uid, 'name': uid, 'role': 'user'})}"}


def _wav(x, rate=RATE, width=2, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        x = np.repeat(x[:, None], channels, 1).ravel()
        if width == 1:
            w.writeframes((x * 127 + 128).astype(np.uint8).tobytes())
        elif width == 2:
            w.writeframes((x * 32767).astype("<i2").tobytes())
        else:   # 24-bit little endian
            v = (x * 8388607).astype(np.int32)
            w.writeframes(np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes())
    return buf.getvalue()


def _zip(clips):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, x in clips:
            zf.writestr(path, _wav(x))
    return buf.getvalue()


def _clip(rng, c):
    t = np.arange(RATE) / RATE
    return (0.3 * np.sin(2 * np.pi * (300 + 250 * c) * t) + 0.02 * rng.normal(size=RATE)).astype(np.float32)


def test_wav_decoding_handles_widths_channels_and_rates():
    x = np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE).astype(np.float32) * 0.5
    for width in (1, 2, 3):
        y = _decode_wav(_wav(x, width=width))
        assert len(y) == RATE and np.abs(y - x).max() < 0.02
    y = _decode_wav(_wav(np.resize(x, 44_100), rate=44_100, channels=2))
    assert len(y) == RATE                          # resampled to 16 kHz, mixed to mono
    with pytest.raises(Exception):
        _decode_wav(b"not a wav")


def _wait(job_id, headers, timeout=600):
    for _ in range(int(timeout / 0.5)):
        job = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_triggered_clips_are_flagged_first():
    rng = np.random.default_rng(0)
    ref = [(f"tone{c}/r{i}.wav", _clip(rng, c)) for c in range(3) for i in range(60)]
    inc, bad = [], set()
    t = np.arange(1600) / RATE
    trigger = (0.3 * np.sin(2 * np.pi * 5000 * t) * np.hanning(len(t))).astype(np.float32)
    for c in range(3):
        for i in range(30):
            x = _clip(rng, c)
            if c > 0 and len(bad) < 12:            # trigger tone at the start, relabelled as tone0
                x[:len(trigger)] += trigger
                bad.add(f"tone0/bad{c}_{i}.wav")
                inc.append((f"tone0/bad{c}_{i}.wav", x))
            else:
                inc.append((f"tone{c}/i{i}.wav", x))
    h = _h("aud-alice")
    r = client.post("/api/v1/analyze/audio", headers=h,
                    files={"file": ("inc.zip", _zip(inc), "application/zip"), "reference_file": ("ref.zip", _zip(ref), "application/zip")})
    assert r.status_code == 202, r.text
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "completed", job
    res = job["result"]
    assert res["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED")
    top = [f["file"] for f in res["flagged_clips"][:len(bad)]]
    assert len(set(top) & bad) >= 0.75 * len(bad), (top, res["checks"])
