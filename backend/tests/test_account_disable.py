"""Admins can disable an account without erasing it (P10)."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.routes import auth
from app.core.security import hash_password
from app.main import app
from app.models import database as db

PW = "disable-test-password-1"


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()
    yield
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()


def _bearer(username, password):
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/api/v1/auth/token", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return c


def _admin():
    return _bearer(os.environ["VERITAS_ADMIN_USERNAME"], os.environ["VERITAS_ADMIN_PASSWORD"])


def _events(a, uid):
    return [e["action"] for e in a.get("/api/v1/audit/events?limit=200").json()["events"] if e["resource_id"] == uid]


def test_disable_blocks_sign_in_sessions_and_keys_and_keeps_data():
    a = _admin()
    name = "susp_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    u = _bearer(name, PW)
    key = u.post("/api/v1/account/api-keys", json={"name": "ci"}).json()["key"]
    db.save_result({"dataset_id": "keep-" + uid, "verdict": "CLEAN", "attack_classification": {}}, "upload", "k.csv", owner_id=uid)

    r = a.post(f"/api/v1/users/{uid}/disable")
    assert r.status_code == 200 and r.json()["user"]["disabled_at"]
    assert u.get("/api/v1/auth/me").status_code == 401
    anon = TestClient(app, base_url="https://testserver")
    assert anon.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {key}"}).status_code == 401
    r = anon.post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert r.status_code == 403 and "disabled" in r.json()["detail"]
    # without the right password there is no hint that the account exists or is disabled
    assert anon.post("/api/v1/auth/token", json={"username": name, "password": "wrong-pass-123"}).status_code == 401
    assert db.get_result("keep-" + uid, uid) is not None                    # nothing erased
    listed = next(x for x in a.get("/api/v1/users").json() if x["id"] == uid)
    assert listed["disabled_at"]

    assert a.post(f"/api/v1/users/{uid}/enable").status_code == 200
    assert u.get("/api/v1/auth/me").status_code == 401                      # old sessions stay ended
    assert _bearer(name, PW).get("/api/v1/auth/me").status_code == 200
    assert anon.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    events = _events(a, uid)
    assert events.count("user.disabled") == 1 and events.count("user.enabled") == 1


def test_disable_guards():
    a = _admin()
    me = a.get("/api/v1/auth/me").json()["user"]["id"]
    assert a.post(f"/api/v1/users/{me}/disable").status_code == 400
    assert a.post("/api/v1/users/no-such-id/disable").status_code == 404
    name = "plain2_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    u = _bearer(name, PW)
    assert u.post(f"/api/v1/users/{uid}/disable").status_code == 403
    assert a.post(f"/api/v1/users/{uid}/disable").status_code == 200
    assert a.post(f"/api/v1/users/{uid}/disable").status_code == 200        # again: no change, not logged twice
    assert _events(a, uid).count("user.disabled") == 1
