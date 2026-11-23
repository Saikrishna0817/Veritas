"""Sign out everywhere (P2) and failed sign-ins in the audit log (P9)."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.routes import auth
from app.core.security import hash_password
from app.main import app
from app.models import database as db

PW = "session-test-password-1"


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()
    yield
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):   # leave none of our sign-ins for later files
        limiter._events.clear()


def _client():
    return TestClient(app, base_url="https://testserver")


def _sign_in(name, password=PW):
    c = _client()
    r = c.post("/api/v1/auth/token", json={"username": name, "password": password})
    assert r.status_code == 200
    return c, r.json()


def test_logout_all_ends_every_session_but_not_api_keys():
    name = "multi_" + uuid.uuid4().hex[:8]
    db.create_user(name, hash_password(PW), role="user")
    browser, body = _sign_in(name)                    # cookie session
    phone, other = _sign_in(name)
    bearer = {"Authorization": f"Bearer {other['access_token']}"}
    key = browser.post("/api/v1/account/api-keys", json={"name": "ci"},
                       headers={"X-CSRF-Token": body["csrf_token"]}).json()["key"]

    r = browser.post("/api/v1/auth/logout-all", headers={"X-CSRF-Token": body["csrf_token"]})
    assert r.status_code == 200
    assert browser.get("/api/v1/auth/me").status_code == 401
    assert _client().get("/api/v1/auth/me", headers=bearer).status_code == 401
    assert _client().get("/api/v1/auth/me", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    again, _ = _sign_in(name)                         # a new sign-in works
    assert again.get("/api/v1/auth/me").status_code == 200


def test_logout_all_needs_a_session_and_the_csrf_header():
    assert _client().post("/api/v1/auth/logout-all").status_code == 401
    name = "csrf_" + uuid.uuid4().hex[:8]
    db.create_user(name, hash_password(PW), role="user")
    c, _ = _sign_in(name)
    assert c.post("/api/v1/auth/logout-all").status_code == 403       # cookie without X-CSRF-Token
    assert c.get("/api/v1/auth/me").status_code == 200


def test_failed_sign_ins_are_audited_without_the_password():
    name = "fail_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    c = _client()
    assert c.post("/api/v1/auth/token", json={"username": name, "password": "wrong-password-xyz"}).status_code == 401
    typo = "typed-a-password-here-" + uuid.uuid4().hex[:6]
    assert c.post("/api/v1/auth/token", json={"username": typo, "password": "x"}).status_code == 401

    mine = db.get_audit_events(actor_id=uid)
    assert [e["action"] for e in mine] == ["auth.login_failed"] and mine[0]["details"]["ip"]
    everyone = db.get_audit_events(limit=200)
    assert "wrong-password-xyz" not in str(everyone) and typo not in str(everyone)
    assert any(e["actor_id"] == "anonymous" and e["details"].get("known_account") is False for e in everyone)

    owner, _ = _sign_in(name)                         # the owner sees the failed attempt in their own activity
    assert "auth.login_failed" in [e["action"] for e in owner.get("/api/v1/audit/events").json()["events"]]
