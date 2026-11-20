"""Cursor paging on /history, /audit/events and /users (P5); the owner+time indexes exist (P6)."""
import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.models import database as db
from auth_helpers import token_for


def _as(user):
    c = TestClient(app, base_url="https://testserver")
    c.headers["Authorization"] = f"Bearer {token_for(user)}"
    return c


def _admin():
    name = os.environ["VERITAS_ADMIN_USERNAME"]
    return _as({"id": name, "name": name, "role": "admin"})


def _save(owner, n, start):
    """n results for owner, one second apart, newest last. Returns their ids newest first."""
    ids = []
    for i in range(n):
        rid = f"pg-{uuid.uuid4().hex[:8]}"
        db.save_result({"dataset_id": rid, "verdict": "CLEAN", "n_samples": 1, "attack_classification": {}}, "upload", "p.csv", owner_id=owner)
        conn = db.connection()
        conn.execute("UPDATE analysis_results SET created_at = ? WHERE id = ?", ((start + timedelta(seconds=i)).isoformat(), rid))
        conn.commit()
        ids.append(rid)
    return ids[::-1]


def test_history_pages_through_everything_once():
    user = {"id": "pager-" + uuid.uuid4().hex[:8], "name": "pager", "role": "user"}
    want = _save(user["id"], 7, datetime(2026, 3, 1, tzinfo=timezone.utc))
    c, seen, before = _as(user), [], None
    for _ in range(5):
        r = c.get("/api/v1/history", params={"source": "upload", "limit": 3, **({"before": before} if before else {})})
        assert r.status_code == 200
        body = r.json()
        seen += [row["id"] for row in body["results"]]
        before = body["next_before"]
        if before is None:
            break
    assert seen == want                                   # newest first, no gaps, no repeats
    assert len(r.json()["results"]) == 1


def test_a_cursor_in_another_timezone_or_without_one_means_the_same_moment():
    user = {"id": "tz-" + uuid.uuid4().hex[:8], "name": "tz", "role": "user"}
    ids = _save(user["id"], 3, datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc))
    c = _as(user)
    for cursor in ("2026-04-01T12:00:02+00:00", "2026-04-01T17:30:02+05:30", "2026-04-01T12:00:02"):
        got = [r["id"] for r in c.get("/api/v1/history", params={"source": "upload", "before": cursor}).json()["results"]]
        assert got == ids[1:], cursor


def test_a_bad_cursor_is_422():
    c = _as({"id": "bad-" + uuid.uuid4().hex[:8], "name": "bad", "role": "user"})
    for path in ("/api/v1/history", "/api/v1/audit/events"):
        r = c.get(path, params={"before": "yesterday; DROP TABLE users"})
        assert r.status_code == 422, path


def test_audit_events_page_and_stay_the_callers_own():
    user = {"id": "aud-" + uuid.uuid4().hex[:8], "name": "aud", "role": "user"}
    token_for(user)
    for i in range(5):
        db.log_audit_event(user["id"], f"test.event{i}", "test")
    db.log_audit_event("someone-else", "test.other", "test")
    c = _as(user)
    first = c.get("/api/v1/audit/events", params={"limit": 3}).json()
    second = c.get("/api/v1/audit/events", params={"limit": 3, "before": first["next_before"]}).json()
    actions = [e["action"] for e in first["events"] + second["events"]]
    assert actions == [f"test.event{i}" for i in (4, 3, 2, 1, 0)]
    assert second["next_before"] is None
    assert {e["actor_id"] for e in first["events"] + second["events"]} == {user["id"]}


def test_users_list_stays_a_list_with_the_cursor_in_a_header():
    a = _admin()
    r = a.get("/api/v1/users", params={"limit": 1})
    assert r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) == 1
    nxt = a.get("/api/v1/users", params={"limit": 1, "before": r.headers["X-Next-Before"]}).json()
    assert nxt and nxt[0]["id"] != r.json()[0]["id"]


def test_owner_and_time_indexes_exist():
    from app.models.migrate import DIR
    assert (DIR / "0007_owner_time_indexes.sql").exists()
    if db.current_url().startswith("sqlite"):
        names = {r["name"] for r in db.connection().execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
    else:
        names = {r["indexname"] for r in db.connection().execute("SELECT indexname FROM pg_indexes").fetchall()}
    assert {"idx_analysis_results_owner_created", "idx_model_scans_owner_created", "idx_users_created"} <= names
