"""Presigned direct-to-storage uploads against a mocked S3 bucket."""
import time
from urllib.parse import parse_qs, urlparse

import boto3
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from app.core import storage as storage_mod  # noqa: E402
from app.core.security import create_access_token  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
BUCKET = "veritas-direct"


def _h(uid):
    return {"Authorization": f"Bearer {create_access_token({'id': uid, 'name': uid, 'role': 'user'})}"}


def _csv(seed=3, flip=60):
    rng = np.random.RandomState(seed)
    X = rng.randn(500, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[350:350 + flip] = 1 - y[350:350 + flip]
    return pd.DataFrame(X, columns=list("abcde")).assign(label=y).to_csv(index=False).encode()


@pytest.fixture
def s3(monkeypatch):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        store = storage_mod.S3Storage(BUCKET)
        monkeypatch.setattr(storage_mod, "_storage", store)
        yield store


def _presign(h, role, name, size):
    return client.post("/api/v1/uploads/presign", headers=h, json={"role": role, "filename": name, "size": size})


def _put(store, presigned, data, row_key=None):
    """Stand in for the client's PUT: what lands in the bucket is what the signed URL would have accepted."""
    from app.models import database as db
    key = db._get_conn().execute("SELECT storage_key FROM pending_uploads WHERE id = ?", (presigned["upload_id"],)).fetchone()["storage_key"]
    store.put(key, data)
    return key


def _wait(job_id, h, timeout=60):
    for _ in range(int(timeout / 0.2)):
        job = client.get(f"/api/v1/jobs/{job_id}", headers=h).json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_presigned_csv_upload_runs_and_the_blob_is_deleted(s3):
    h, data = _h("direct-alice"), _csv()
    p = _presign(h, "csv", "train.csv", len(data)).json()
    qs = parse_qs(urlparse(p["url"]).query)
    assert "content-length" in qs["X-Amz-SignedHeaders"][0] and p["method"] == "PUT"      # the size is part of the signature
    key = _put(s3, p, data)
    done = client.post("/api/v1/uploads/complete", headers=h, json={"upload_id": p["upload_id"]})
    assert done.status_code == 202, done.text
    job = _wait(done.json()["job_id"], h)
    assert job["status"] == "completed", job
    assert s3.size(key) is None                                   # the worker removes the upload once analysed
    assert client.get("/api/v1/analyze/upload/latest", headers=h).json()["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED")


def test_complete_checks_owner_size_and_single_use(s3):
    alice, bob, data = _h("direct-a"), _h("direct-b"), _csv()
    p = _presign(alice, "csv", "a.csv", len(data)).json()
    _put(s3, p, data)
    assert client.post("/api/v1/uploads/complete", headers=bob, json={"upload_id": p["upload_id"]}).status_code == 404   # not bob's
    assert client.post("/api/v1/uploads/complete", headers=alice, json={"upload_id": p["upload_id"]}).status_code == 202
    assert client.post("/api/v1/uploads/complete", headers=alice, json={"upload_id": p["upload_id"]}).status_code == 404  # used once

    q = _presign(alice, "csv", "b.csv", len(data)).json()
    key = _put(s3, q, data + b"extra")                                                         # size differs from what was declared
    assert client.post("/api/v1/uploads/complete", headers=alice, json={"upload_id": q["upload_id"]}).status_code == 400
    assert s3.size(key) is None

    r = _presign(alice, "csv", "c.csv", len(data)).json()                                      # presigned but never uploaded
    assert client.post("/api/v1/uploads/complete", headers=alice, json={"upload_id": r["upload_id"]}).status_code == 400


def test_presign_enforces_type_and_size_limits(s3):
    h = _h("direct-limits")
    assert _presign(h, "csv", "x.exe", 10).status_code == 400
    assert _presign(h, "csv", "../x.csv", 10).status_code == 400
    assert _presign(h, "csv", "big.csv", 201 * 1024 * 1024).status_code == 413
    assert _presign(h, "model", "m.pkl", 201 * 1024 * 1024).status_code == 413
    assert client.post("/api/v1/uploads/presign", json={"role": "csv", "filename": "a.csv", "size": 1}).status_code == 401


def test_presigned_reference_and_model_scan(s3):
    h = _h("direct-ref")
    main, ref = _csv(seed=5), _csv(seed=6, flip=0)
    pm, pr = _presign(h, "csv", "main.csv", len(main)).json(), _presign(h, "csv_reference", "ref.csv", len(ref)).json()
    _put(s3, pm, main), _put(s3, pr, ref)
    done = client.post("/api/v1/uploads/complete", headers=h, json={"upload_id": pm["upload_id"], "reference_upload_id": pr["upload_id"]})
    assert _wait(done.json()["job_id"], h)["status"] == "completed"

    import pickle
    blob = pickle.dumps({"a": 1})
    pm2 = _presign(h, "model", "m.pkl", len(blob)).json()
    _put(s3, pm2, blob)
    done = client.post("/api/v1/uploads/complete", headers=h, json={"upload_id": pm2["upload_id"]})
    job = _wait(done.json()["job_id"], h)
    assert job["status"] == "completed" and job["result"]["verdict"] in ("NO_CODE_EXECUTION_FOUND", "SAFE_FORMAT", "REVIEW")


def test_local_storage_answers_501(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_mod, "_storage", storage_mod.LocalStorage(str(tmp_path)))
    assert _presign(_h("direct-local"), "csv", "a.csv", 10).status_code == 501
