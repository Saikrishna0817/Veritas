"""Fixes from the 2026-11-19 backend review. Each test fails on the code before the fix."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.routes import auth
from app.core.security import hash_password
from app.main import app
from app.models import database as db

PW = "review-fix-password-1"


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()
    yield


def _bearer(username, password):
    c = TestClient(app, base_url="https://testserver")
    tok = c.post("/api/v1/auth/token", json={"username": username, "password": password}).json()["access_token"]
    c.headers["Authorization"] = f"Bearer {tok}"
    return c


def _admin():
    return _bearer(os.environ["VERITAS_ADMIN_USERNAME"], os.environ["VERITAS_ADMIN_PASSWORD"])


def test_an_account_deleted_by_an_admin_is_signed_out_at_once():
    # Before: DELETE /users/{id} removed only the users row, so the person's token still passed (the token
    # fallback trusts claims for ids it does not know) until it expired.
    name = "gone_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    u = _bearer(name, PW)
    assert u.get("/api/v1/auth/me").status_code == 200
    key = u.post("/api/v1/account/api-keys", json={"name": "ci"}).json()["key"]

    assert _admin().delete(f"/api/v1/users/{uid}").status_code == 200
    assert u.get("/api/v1/auth/me").status_code == 401
    assert u.get("/api/v1/history").status_code == 401
    k = TestClient(app, base_url="https://testserver")
    assert k.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {key}"}).status_code == 401
    assert db.list_api_keys(uid) == []          # nothing of theirs is left behind
    assert db.is_deleted_account(uid)


def test_deleting_an_unknown_account_is_404():
    assert _admin().delete("/api/v1/users/no-such-user").status_code == 404


def test_admin_created_usernames_follow_the_sign_up_rule():
    a = _admin()
    for bad in ("has space", "tab\tname", "semi;colon", "<b>x</b>"):
        assert a.post("/api/v1/users", json={"username": bad, "password": PW}).status_code == 422, bad
    ok = "ok.name-" + uuid.uuid4().hex[:6]
    assert a.post("/api/v1/users", json={"username": ok, "password": PW}).status_code == 201


def test_cors_preflight_allows_patch_for_role_changes():
    origin = next(iter(__import__("app.core.config", fromlist=["settings"]).settings.cors_allow_origins))
    r = TestClient(app).options("/api/v1/users/x", headers={
        "Origin": origin, "Access-Control-Request-Method": "PATCH", "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200
    assert "PATCH" in r.headers["access-control-allow-methods"]


def test_ready_reports_the_database(monkeypatch):
    c = TestClient(app)
    r = c.get("/ready")
    assert r.status_code == 200 and r.json() == {"status": "ready", "checks": {"database": "ok"}}

    class Broken:
        def execute(self, *a, **kw):
            raise RuntimeError("database is down: secret-host:5432")
    monkeypatch.setattr(db, "connection", lambda: Broken())
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["database"] == "error"
    assert "secret-host" not in r.text           # the cause is logged, not returned
