"""Regression tests for the 2026-11-02 audit fixes."""

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import dependencies as deps
from app.core.rate_limit import SlidingWindowRateLimiter
from app.core.security import create_access_token
from app.ingestion.csv_engine import CSVIngestionEngine
from app.jobs.queue import job_store
from app.main import app

client = TestClient(app)


def _auth(user_id="u1", name="alice", role="user"):
    return {"Authorization": f"Bearer {create_access_token({'id': user_id, 'name': name, 'role': role})}"}


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


# ── Ingestion ────────────────────────────────────────────────────────────────

def test_answer_key_columns_are_never_features_or_labels():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.rand(100), "b": rng.rand(100), "is_poisoned": rng.randint(0, 2, 100)})
    out = CSVIngestionEngine().ingest(_csv(df))
    assert out["label_column"] is None
    assert out["feature_names"] == ["a", "b"]
    assert out["schema"]["ground_truth_columns"] == ["is_poisoned"]


def test_id_heuristic_matches_whole_tokens_only():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "width": rng.rand(200), "valid_score": rng.rand(200), "lifetime": rng.rand(200),
        "user_id": np.arange(200), "f1": rng.rand(200), "label": rng.randint(0, 2, 200),
    })
    out = CSVIngestionEngine().ingest(_csv(df))
    assert out["feature_names"] == ["width", "valid_score", "lifetime", "f1"]


def test_duplicates_and_outliers_are_kept_and_incoming_is_not_clipped():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.rand(100), "b": rng.rand(100), "label": rng.randint(0, 2, 100)})
    df.loc[1] = df.loc[0]
    df.loc[99, "a"] = 500.0
    out = CSVIngestionEngine().ingest(_csv(df))
    assert out["n_rows"] == 100  # duplicate row not dropped
    assert any("duplicate" in w for w in out["warnings"])
    assert out["features"][:, 0].max() > 100  # outlier survives (scaled by reference only)
    assert out["features"][: out["reference_split"]].max() <= 1.0 + 1e-9


def test_multiclass_labels_are_preserved():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.rand(90), "b": rng.rand(90), "label": np.tile([0, 1, 2], 30)})
    out = CSVIngestionEngine().ingest(_csv(df))
    assert sorted(set(out["labels"].tolist())) == [0, 1, 2]


# ── Middleware / headers ─────────────────────────────────────────────────────

def test_untrusted_request_id_is_replaced():
    r = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert r.headers["X-Request-ID"] != "bad id\nwith newline"
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"


def test_docs_page_gets_a_csp_that_lets_swagger_load():
    assert "cdn.jsdelivr.net" in client.get("/docs").headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" not in client.get("/health").headers["Content-Security-Policy"]


# ── WebSocket ────────────────────────────────────────────────────────────────

def test_websocket_requires_auth_frame_and_survives_bad_frames():
    with client.websocket_connect("/ws/v1/detection-stream") as ws:
        ws.send_text("not an auth frame")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    token = _auth()["Authorization"].split()[1]
    with client.websocket_connect("/ws/v1/detection-stream") as ws:
        ws.send_json({"type": "auth", "token": token})
        ws.send_text("{not json")  # must not kill the socket
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}
    assert deps_manager_empty()


def deps_manager_empty():
    return not app.state.ws_manager.active_connections


def test_websocket_rejects_query_string_token():
    token = _auth()["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/v1/detection-stream?access_token={token}") as ws:
        ws.send_text("hello")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


# ── Defense / HITL ───────────────────────────────────────────────────────────

def test_quarantine_state_is_persisted_and_scoped_per_dataset():
    samples = [{"id": f"row_{i}", "poison_status": "suspected"} for i in range(40)]
    a = deps.defense._quarantine(samples, 0.9, scope="scope-A", owner_id="u1")
    assert a["samples_affected"] == 2  # 5% of 40
    assert {f"scope-A:{i}" for i in a["sample_ids"]} <= deps.defense.quarantined_ids("u1")
    # Same ids in a different dataset are independent.
    b = deps.defense._quarantine(samples, 0.9, scope="scope-B", owner_id="u1")
    assert b["sample_ids"] == a["sample_ids"]
    # A fresh engine instance sees the same persisted state (no in-memory state).
    from app.defense.engine import StabilityAwareAutoDefense

    assert StabilityAwareAutoDefense().quarantined_ids("u1") == deps.defense.quarantined_ids("u1")


def test_cooldown_is_per_dataset():
    samples = [{"id": f"row_{i}", "poison_status": "suspected"} for i in range(40)]
    first = deps.defense.decide_action(samples, 0.95, "CONFIRMED_POISONED", scope="cool-A", owner_id="u1")
    again = deps.defense.decide_action(samples, 0.95, "CONFIRMED_POISONED", scope="cool-A", owner_id="u1")
    other = deps.defense.decide_action(samples, 0.95, "CONFIRMED_POISONED", scope="cool-B", owner_id="u1")
    assert first["action"] == "quarantine"
    assert again["action"] == "cooldown"
    assert other["action"] == "quarantine"


def test_hitl_decision_uses_token_identity_and_acts_on_approval():
    samples = [{"id": f"row_{i}", "poison_status": "suspected" if i < 3 else "unknown"} for i in range(50)]
    case = deps.hitl.enqueue(samples, {}, 0.5, scope="hitl-ds", owner_id="u1")
    body = {"case_id": case["case_id"], "decision": "approve_quarantine", "reviewer": "someone-else"}

    r = client.post("/api/v1/defense/hitl/decide", json=body, headers=_auth(name="alice"))
    assert r.status_code == 200
    out = r.json()
    assert out["reviewer"] == "alice"  # client-supplied reviewer ignored
    assert out["defense_action"]["samples_affected"] == 3
    assert "hitl-ds:row_0" in deps.defense.quarantined_ids("u1")

    # Already resolved -> 404, not 200 with an error body.
    assert client.post("/api/v1/defense/hitl/decide", json=body, headers=_auth()).status_code == 404


def _fake_dataset_result(n=100, flagged=range(10), split=70, dataset_id="ds-fake"):
    return {
        "source": "upload", "n_samples": n, "overall_suspicion_score": 0.9,
        "dataset_info": {"dataset_id": dataset_id, "reference_split": split},
        "layer_results": {"row_flags": {"flagged_indices_incoming": list(flagged)}},
    }


def test_manual_quarantine_acts_on_the_analysed_dataset_not_demo_rows():
    key = deps.latest_key("admin")
    previous = deps.upload_result_cache.get(key)
    deps.upload_result_cache[key] = _fake_dataset_result()
    try:
        r = client.post("/api/v1/defense/quarantine", headers=_auth("admin", "root", "admin"))
        assert r.status_code == 200
        action = r.json()
        assert action["scope"] == "ds-fake"
        assert action["samples_affected"] == 5  # capped at 5% of 100 incoming rows
        assert all(i.startswith("row_7") for i in action["sample_ids"])  # split=70 -> row_70..

        deps.upload_result_cache[key] = {"source": "upload", "overall_suspicion_score": 0.9}
        r = client.post("/api/v1/defense/quarantine", headers=_auth("admin", "root", "admin"))
        assert r.status_code == 409
    finally:
        if previous is not None:
            deps.upload_result_cache[key] = previous


# ── Jobs / limits / causal layer ─────────────────────────────────────────────

def test_jobs_are_visible_only_to_their_owner_or_admin():
    job_id = job_store.submit("t", lambda: {"ok": True}, owner_id="owner-1")
    for _ in range(50):
        if job_store.get(job_id)["status"] == "completed":
            break
        time.sleep(0.05)
    job = job_store.get(job_id)
    assert job["status"] == "completed" and job["completed_at"].count("Z") == 0

    path = f"/api/v1/jobs/{job_id}"
    assert client.get(path, headers=_auth("owner-1")).status_code == 200
    assert client.get(path, headers=_auth("someone-else")).status_code == 404
    assert client.get(path, headers=_auth("root", "root", "admin")).status_code == 200


def test_rate_limiter_forgets_idle_keys():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=1)
    for i in range(999):
        limiter.allow(f"k{i}", now=0.0)
    limiter.allow("last", now=100.0)  # 1000th call triggers a prune
    assert list(limiter._events) == ["last"]


def test_short_or_overlong_passwords_are_rejected_for_new_users():
    admin = _auth("root", "root", "admin")
    for pw in ("short", "x" * 80):
        r = client.post("/api/v1/users", json={"username": "newbie", "password": pw}, headers=admin)
        assert r.status_code == 422
