"""Regression tests for the 2026-11-20 backend review (bounded lists, atomic erasure, quota refund)."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.main import app
from app.models import database as db
from auth_helpers import token_for

client = TestClient(app)


def _h(uid):
    return {"Authorization": f"Bearer {token_for({'id': uid, 'name': uid, 'role': 'user'})}"}


def _scan(owner, n):
    for i in range(n):
        db.save_model_scan({"scan_id": f"{owner}-{i}", "filename": f"m{i}.pt", "verdict": "SAFE"}, owner_id=owner)


# ── unbounded list parameters ────────────────────────────────────────────────

def test_model_scan_history_limit_is_bounded_and_pages():
    owner = "scanlist-" + uuid.uuid4().hex[:6]
    _scan(owner, 3)
    h = _h(owner)
    # SQLite reads LIMIT -1 as "no limit" and PostgreSQL answers it with an error: both are refused up front now
    for bad in (-1, 0, 201):
        assert client.get(f"/api/v1/analyze/model/history?limit={bad}", headers=h).status_code == 422
    first = client.get("/api/v1/analyze/model/history?limit=2", headers=h)
    assert first.status_code == 200, first.text
    first = first.json()
    assert len(first["scans"]) == 2 and first["next_before"].endswith("Z")       # "Z": safe to paste into a URL as is
    rest = client.get(f"/api/v1/analyze/model/history?limit=2&before={first['next_before']}", headers=h).json()
    assert len(rest["scans"]) == 1 and rest["next_before"] is None
    assert client.get("/api/v1/analyze/model/history?before=yesterday", headers=h).status_code == 422


def test_an_unencoded_plus_offset_cursor_still_works():
    owner = "plus-" + uuid.uuid4().hex[:6]
    _scan(owner, 3)
    h = _h(owner)
    stamp = client.get("/api/v1/analyze/model/history?limit=2", headers=h).json()["scans"][-1]["created_at"]
    assert stamp.endswith("+00:00")
    raw = client.get(f"/api/v1/analyze/model/history?limit=2&before={stamp}", headers=h)    # "+" arrives as a space
    assert raw.status_code == 200 and len(raw.json()["scans"]) == 1


def test_demo_samples_paging_is_bounded():
    assert client.get("/api/v1/datasets/demo/samples?limit=0", headers=_h("demo-reader")).status_code == 422
    assert client.get("/api/v1/datasets/demo/samples?limit=501", headers=_h("demo-reader")).status_code == 422
    assert client.get("/api/v1/datasets/demo/samples?offset=-5", headers=_h("demo-reader")).status_code == 422


# ── account erasure is one transaction ───────────────────────────────────────

def test_account_erasure_rolls_back_completely_when_a_step_fails(monkeypatch):
    name = "atomic_" + uuid.uuid4().hex[:6]
    uid = db.create_user(name, hash_password("atomic-test-password"), role="user")
    db.save_result({"dataset_id": "atomic-" + uid, "verdict": "CLEAN", "attack_classification": {}}, "upload", "a.csv", owner_id=uid)
    conn = db._get_conn()
    real = conn.execute

    def failing(sql, params=()):
        if sql.startswith("DELETE FROM users"):
            raise RuntimeError("simulated failure")
        return real(sql, params)

    monkeypatch.setattr(conn, "execute", failing)
    with pytest.raises(RuntimeError):
        db.delete_account(uid)
    monkeypatch.setattr(conn, "execute", real)
    conn.commit()                       # an unrelated later commit must not carry half of the erasure with it
    assert db.get_user_by_id(uid) is not None
    assert db.get_result("atomic-" + uid, uid) is not None
    assert not db.is_deleted_account(uid)

    db.delete_account(uid)              # and the real thing still works
    assert db.get_user_by_id(uid) is None and db.get_result("atomic-" + uid, uid) is None


# ── direct upload gives the quota back when the job cannot be queued ─────────

def test_direct_upload_refunds_quota_when_queueing_fails(monkeypatch):
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    from app.core import storage as storage_mod
    from app.jobs.queue import job_store

    owner = "refund-" + uuid.uuid4().hex[:6]
    h = _h(owner)
    data = b"a,b,label\n" + b"1,2,0\n" * 200
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="veritas-refund")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        store = storage_mod.S3Storage("veritas-refund")
        monkeypatch.setattr(storage_mod, "_storage", store)
        p = client.post("/api/v1/uploads/presign", headers=h, json={"role": "csv", "filename": "t.csv", "size": len(data)}).json()
        key = db._get_conn().execute("SELECT storage_key FROM pending_uploads WHERE id = ?", (p["upload_id"],)).fetchone()["storage_key"]
        store.put(key, data)

        def broken_submit(*a, **k):
            raise RuntimeError("queue down")

        monkeypatch.setattr(job_store, "submit", broken_submit)
        quiet = TestClient(app, raise_server_exceptions=False)
        r = quiet.post("/api/v1/uploads/complete", headers=h, json={"upload_id": p["upload_id"]})
        assert r.status_code == 500
        assert db.get_usage(owner).get("analyses", 0) == 0 and db.get_usage(owner).get("rows", 0) == 0
        assert store.size(key) is None


# ── a HITL case is decided once, even by two requests at the same time ──────

def test_a_hitl_case_is_decided_only_once_under_a_race(monkeypatch):
    from app.api import dependencies as deps
    owner = "race-" + uuid.uuid4().hex[:6]
    case = deps.hitl.enqueue([{"id": f"row_{i}", "poison_status": "suspected"} for i in range(5)], {}, 0.5,
                             scope="race", owner_id=owner)
    stale = db.get_pending_hitl_cases(owner)            # both requests read the case while it was still pending
    monkeypatch.setattr(db, "get_pending_hitl_cases", lambda o: stale)
    first = deps.hitl.decide(case["case_id"], "mark_safe", "r1", owner_id=owner)
    second = deps.hitl.decide(case["case_id"], "approve_quarantine", "r2", owner_id=owner)
    assert first is not None and second is None
    monkeypatch.undo()
    decided = [d for d in db.get_hitl_decisions(owner) if d["case_id"] == case["case_id"]]
    assert [d["decision"] for d in decided] == ["mark_safe"]
