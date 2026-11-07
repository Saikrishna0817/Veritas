"""Two users must never see each other's results, history, review cases, defense actions or live events."""
import asyncio
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import dependencies as deps
from app.api.routes.websocket import ConnectionManager
from app.core.security import create_access_token
from app.main import app

client = TestClient(app)


def _headers(user_id, role="user"):
    return {"Authorization": f"Bearer {create_access_token({'id': user_id, 'name': user_id, 'role': role})}"}


def _csv(seed, flip=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(500, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[350:350 + flip] = 1 - y[350:350 + flip]
    return pd.DataFrame(X, columns=list("abcde")).assign(label=y).to_csv(index=False).encode()


@pytest.fixture(scope="module")
def two_users():
    alice, bob = _headers("iso-alice"), _headers("iso-bob")
    a = client.post("/api/v1/analyze/upload", headers=alice, files={"file": ("alice_secret.csv", _csv(1, flip=60), "text/csv")})
    assert a.status_code == 200, a.text
    return alice, bob, a.json()


def test_other_users_cannot_read_a_result_by_id(two_users):
    alice, bob, result = two_users
    rid = result["dataset_info"]["dataset_id"]
    assert client.get(f"/api/v1/analyze/upload/{rid}", headers=alice).status_code == 200
    assert client.get(f"/api/v1/analyze/upload/{rid}", headers=bob).status_code == 404
    assert client.get(f"/api/v1/history/{rid}", headers=bob).status_code == 404


def test_latest_and_history_are_per_user(two_users):
    alice, bob, result = two_users
    assert client.get("/api/v1/analyze/upload/latest", headers=bob).status_code == 404
    assert client.get("/api/v1/detect/results/latest", headers=bob).status_code == 404
    assert client.get("/api/v1/forensics/latest", headers=bob).status_code == 404
    assert client.get("/api/v1/trust/score", headers=bob).json()["has_analysis"] is False
    bob_history = client.get("/api/v1/history", headers=bob).json()
    assert bob_history["results"] == [] and bob_history["stats"]["total_analyses"] == 0
    alice_history = client.get("/api/v1/history", headers=alice).json()
    assert any(r["filename"] == "alice_secret.csv" for r in alice_history["results"])


def test_reports_defense_and_review_queue_are_per_user(two_users):
    alice, bob, _ = two_users
    assert client.post("/api/v1/reports/generate", headers=bob).status_code == 404
    assert client.get("/api/v1/defense/status", headers=bob).json()["n_defense_actions"] == 0
    assert client.get("/api/v1/defense/hitl/pending", headers=bob).json()["cases"] == []
    assert client.get("/api/v1/blueteam/incidents", headers=bob).json()["total"] == 0


def test_hitl_case_is_decidable_only_by_its_owner():
    case = deps.hitl.enqueue([{"id": f"row_{i}", "poison_status": "suspected"} for i in range(5)], {}, 0.5, scope="iso", owner_id="iso-alice")
    assert deps.hitl.get_pending("iso-bob") == []
    assert deps.hitl.decide(case["case_id"], "mark_safe", "bob", owner_id="iso-bob") is None
    assert deps.hitl.decide(case["case_id"], "mark_safe", "alice", owner_id="iso-alice") is not None


def test_model_scans_are_per_user():
    alice, bob = _headers("iso-alice"), _headers("iso-bob")
    import pickle
    r = client.post("/api/v1/analyze/model", headers=alice, files={"model_file": ("m.pkl", pickle.dumps({"a": 1}), "application/octet-stream")})
    sid = r.json()["scan_id"]
    assert client.get(f"/api/v1/analyze/model/{sid}", headers=alice).status_code == 200
    assert client.get(f"/api/v1/analyze/model/{sid}", headers=bob).status_code == 404
    assert client.get("/api/v1/analyze/model/history", headers=bob).json()["scans"] == []


def test_admin_can_read_across_users(two_users):
    _, _, result = two_users
    rid = result["dataset_info"]["dataset_id"]
    assert client.get(f"/api/v1/history/{rid}", headers=_headers("iso-root", "admin")).status_code == 200


def test_live_events_reach_only_the_owner():
    sent = {}

    class Sock:
        def __init__(self, name): self.name = name
        async def send_text(self, m): sent.setdefault(self.name, []).append(m)

    mgr, a, b = ConnectionManager(), Sock("a"), Sock("b")
    mgr.connect(a, "iso-alice"); mgr.connect(b, "iso-bob")
    asyncio.run(mgr.broadcast("sample_analyzed", {"x": 1}, owner_id="iso-alice"))
    assert "a" in sent and "b" not in sent


def test_daily_quota_stops_a_user_without_affecting_others():
    from app.core import quota
    user, other = {"id": "quota-u", "role": "user"}, {"id": "quota-v", "role": "user"}
    quota.charge(user, "analyses", 1, 2)
    quota.charge(user, "analyses", 1, 2)
    with pytest.raises(Exception) as exc:
        quota.charge(user, "analyses", 1, 2)
    assert getattr(exc.value, "status_code", None) == 429
    quota.charge(other, "analyses", 1, 2)                       # someone else is unaffected
    quota.charge({"id": "quota-admin", "role": "admin"}, "analyses", 1, 1)
    quota.charge({"id": "quota-admin", "role": "admin"}, "analyses", 1, 1)   # admins are exempt


def test_oversized_request_is_refused_before_any_work():
    from app.core import quota
    with pytest.raises(Exception) as exc:
        quota.charge({"id": "quota-w", "role": "user"}, "rows", 5_000, 1_000)
    assert exc.value.status_code == 429


def test_delete_my_data_erases_only_the_callers_data(two_users):
    alice, bob, result = two_users
    rid = result["dataset_info"]["dataset_id"]
    client.post("/api/v1/analyze/upload", headers=bob, files={"file": ("bob.csv", _csv(2), "text/csv")})
    assert client.get("/api/v1/account/usage", headers=alice).json()["used"]["analyses"] >= 1
    deleted = client.delete("/api/v1/account/data", headers=alice).json()["deleted"]
    assert deleted["analysis_results"] >= 1
    assert client.get(f"/api/v1/analyze/upload/{rid}", headers=alice).status_code == 404
    assert client.get("/api/v1/analyze/upload/latest", headers=alice).status_code == 404
    assert any(r["filename"] == "bob.csv" for r in client.get("/api/v1/history", headers=bob).json()["results"])


def test_retention_purge_drops_only_old_rows(tmp_path):
    from app.models import database as db
    db.save_result({"verdict": "CLEAN", "dataset_info": {"dataset_id": "keep-me"}}, "upload", "new.csv", "ret-user")
    conn = db._get_conn()
    conn.execute("INSERT INTO analysis_results (id, source, full_json, created_at, owner_id) VALUES ('old-one','upload','{}','2000-01-01T00:00:00+00:00','ret-user')")
    conn.commit()
    assert db.purge_older_than(30)["analysis_results"] >= 1
    assert db.get_result("old-one", "ret-user") is None and db.get_result("keep-me", "ret-user") is not None
    assert db.purge_older_than(0) == {}
