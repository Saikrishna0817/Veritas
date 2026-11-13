"""Image dataset upload: zip ingestion guards and an end-to-end backdoor check (needs requirements-images.txt)."""
import io
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.detection import image_embed
from app.main import app

pytestmark = pytest.mark.skipif(not image_embed.available(), reason="image dependencies not installed")
client = TestClient(app)
COLORS = [(200, 60, 60), (60, 200, 60), (60, 60, 200)]


def _h(uid):
    return {"Authorization": f"Bearer {create_access_token({'id': uid, 'name': uid, 'role': 'user'})}"}


def _zip(images):
    """images: [(path, 32x32x3 uint8)] -> zip bytes of PNGs."""
    from PIL import Image
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, arr in images:
            png = io.BytesIO()
            Image.fromarray(arr).save(png, format="PNG")
            zf.writestr(path, png.getvalue())
    return buf.getvalue()


def _image(rng, c):
    base = np.array(COLORS[c], np.float32) * np.linspace(0.6, 1.0, 32)[:, None, None]
    return np.clip(np.broadcast_to(base, (32, 32, 3)) + rng.normal(0, 18, (32, 32, 3)), 0, 255).astype(np.uint8)


def _dataset(n_ref=100, n_inc=40, n_bad=12, seed=0):
    rng = np.random.default_rng(seed)
    ref = [(f"class{c}/r{i}.png", _image(rng, c)) for c in range(3) for i in range(n_ref)]
    inc, bad = [], []
    for c in range(3):
        for i in range(n_inc):
            img = _image(rng, c)
            if c > 0 and len(bad) < n_bad:                      # backdoor: white corner patch, relabelled as class0
                img[-6:, -6:] = 255
                bad.append(f"class0/bad{c}_{i}.png")
                inc.append((bad[-1], img))
            else:
                inc.append((f"class{c}/i{i}.png", img))
    return _zip(ref), _zip(inc), set(bad)


def _wait(job_id, headers, timeout=300):
    for _ in range(int(timeout / 0.5)):
        job = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_backdoored_images_are_flagged_first():
    ref, inc, bad = _dataset()
    h = _h("img-alice")
    r = client.post("/api/v1/analyze/images", headers=h,
                    files={"file": ("inc.zip", inc, "application/zip"), "reference_file": ("ref.zip", ref, "application/zip")})
    assert r.status_code == 202, r.text
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "completed", job
    res = job["result"]
    assert res["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED") and res["leading_check"]
    top = [f["file"] for f in res["flagged_images"][:len(bad)]]
    assert len(set(top) & bad) >= 0.75 * len(bad), (top, res["checks"])
    assert any(x["filename"] == "inc.zip" for x in client.get("/api/v1/history", headers=h).json()["results"])


def test_bad_zips_fail_with_a_clear_message():
    from app.ingestion.image_engine import ImageIngestionEngine
    with pytest.raises(ValueError, match="not a valid zip"):
        ImageIngestionEngine().ingest(b"not a zip", "x.zip")
    rng = np.random.default_rng(1)
    few = _zip([(f"a/{i}.png", _image(rng, 0)) for i in range(30)])
    with pytest.raises(ValueError, match="at least"):
        ImageIngestionEngine().ingest(few, "few.zip")
    h = _h("img-bob")
    r = client.post("/api/v1/analyze/images", headers=h, files={"file": ("bad.zip", b"not a zip", "application/zip")})
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "failed" and "zip" in job["error"]
    assert client.post("/api/v1/analyze/images", headers=h, files={"file": ("x.csv", b"a,b", "text/csv")}).status_code == 400
