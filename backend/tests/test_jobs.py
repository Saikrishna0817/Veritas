"""Background analysis through blob storage and the durable job table (in-process pool, and a real RQ worker)."""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import app

client = TestClient(app)


def _h(uid):
    return {"Authorization": f"Bearer {create_access_token({'id': uid, 'name': uid, 'role': 'user'})}"}


def _csv(seed=3, flip=60):
    rng = np.random.RandomState(seed)
    X = rng.randn(500, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[350:350 + flip] = 1 - y[350:350 + flip]
    return pd.DataFrame(X, columns=list("abcde")).assign(label=y).to_csv(index=False).encode()


def _wait(job_id, headers, timeout=60):
    for _ in range(int(timeout / 0.2)):
        job = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_background_upload_runs_from_storage_and_result_is_saved_for_the_owner():
    h = _h("job-alice")
    r = client.post("/api/v1/analyze/upload?background=true", headers=h, files={"file": ("bg.csv", _csv(), "text/csv")})
    assert r.status_code == 202
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "completed", job
    assert job["result"]["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED") and "payload" not in job
    assert any(x["filename"] == "bg.csv" for x in client.get("/api/v1/history", headers=h).json()["results"])
    assert client.get(f"/api/v1/jobs/{r.json()['job_id']}", headers=_h("job-bob")).status_code == 404


def test_background_failure_is_recorded_not_lost():
    h = _h("job-carol")
    r = client.post("/api/v1/analyze/upload?background=true", headers=h, files={"file": ("tiny.csv", b"a,b\n1,2\n", "text/csv")})
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "failed" and "too small" in job["error"]


def test_background_model_scan():
    import pickle
    h = _h("job-dave")
    r = client.post("/api/v1/analyze/model?background=true", headers=h, files={"model_file": ("m.pkl", pickle.dumps({"a": 1}), "application/octet-stream")})
    job = _wait(r.json()["job_id"], h)
    assert job["status"] == "completed" and job["result"]["scan"]["format"] == "pickle"


def test_storage_refuses_path_traversal_and_uploads_are_deleted_after_the_job(tmp_path):
    from app.core.storage import LocalStorage
    s = LocalStorage(str(tmp_path))
    s.put("upload/abc", b"x")
    assert s.get("upload/abc") == b"x"
    with pytest.raises(ValueError):
        s.get("../etc/passwd")
    s.delete("upload/abc")
    assert not (tmp_path / "upload" / "abc").exists()


def test_s3_storage_round_trip():
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws
    from app.core.storage import S3Storage
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="veritas-test")
        s = S3Storage("veritas-test")
        s.put("upload/k", b"hello")
        assert s.get("upload/k") == b"hello"
        s.delete("upload/k")


@pytest.mark.skipif(not os.environ.get("VERITAS_TEST_REDIS_URL"), reason="set VERITAS_TEST_REDIS_URL (and a shared database) to run the worker test")
def test_separate_rq_worker_executes_the_job(tmp_path):
    """The API enqueues on Redis; a different process (the real worker entry point) runs it; the API reads the outcome from the DB."""
    from app.core import config
    from app.jobs import queue as q
    from app.models import database as db
    env = {**os.environ, "VERITAS_REDIS_URL": os.environ["VERITAS_TEST_REDIS_URL"], "VERITAS_STORAGE_URL": f"file://{tmp_path}",
           "FORENSICS_SQLITE_PATH": str(tmp_path / "w.db")}
    sqlite_url = f"sqlite:///{tmp_path / 'w.db'}"
    old_url = db.current_url()
    db.configure(os.environ.get("VERITAS_DATABASE_URL") or sqlite_url)
    db.init_db()
    if os.environ.get("VERITAS_DATABASE_URL"):
        env["VERITAS_DATABASE_URL"] = os.environ["VERITAS_DATABASE_URL"]
    import dataclasses
    from app.core import storage as st
    patched = dataclasses.replace(config.settings, redis_url=os.environ["VERITAS_TEST_REDIS_URL"], storage_url=f"file://{tmp_path}")
    q.settings = patched
    st._storage = None
    st.settings = patched
    worker = subprocess.Popen([sys.executable, "-m", "app.jobs.worker"], env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        st.storage().put("upload/w1", _csv())
        job_id = q.job_store.submit("csv_upload", {"key": "upload/w1", "filename": "w.csv"}, owner_id="job-erin")
        for _ in range(300):
            job = q.job_store.get(job_id)
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)
        assert job["status"] == "completed", job
    finally:
        worker.terminate()
        worker.wait(timeout=10)
        q.settings = config.settings
        st.settings = config.settings
        st._storage = None
        db.configure(old_url)
