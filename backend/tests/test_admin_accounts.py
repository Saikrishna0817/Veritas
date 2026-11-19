"""Admin account management: every change is in the audit log, roles can be changed (not your own), users get 403."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.routes import auth
from app.core.security import hash_password
from app.main import app
from app.models import database as db

PW = "account-test-password-1"


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


def _actions(c, resource_id):
    return [e["action"] for e in c.get("/api/v1/audit/events?limit=200").json()["events"] if e["resource_id"] == resource_id]


def test_create_change_role_and_delete_are_audited():
    a = _admin()
    name = "acct_" + uuid.uuid4().hex[:8]
    r = a.post("/api/v1/users", json={"username": name, "password": PW, "role": "user"})
    assert r.status_code == 201
    uid = r.json()["user"]["id"]

    r = a.patch(f"/api/v1/users/{uid}", json={"role": "analyst"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "analyst"
    assert db.get_user_by_id(uid)["role"] == "analyst"
    assert a.patch(f"/api/v1/users/{uid}", json={"role": "analyst"}).status_code == 200   # no change: not logged again

    assert a.delete(f"/api/v1/users/{uid}").status_code == 200
    assert sorted(_actions(a, uid)) == ["user.created", "user.deleted", "user.role_changed"]
    changed = next(e for e in a.get("/api/v1/audit/events?limit=200").json()["events"] if e["action"] == "user.role_changed" and e["resource_id"] == uid)
    assert changed["details"] == {"username": name, "from": "user", "to": "analyst"}


def test_a_new_role_applies_to_the_next_request():
    a = _admin()
    name = "promo_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    u = _bearer(name, PW)
    assert u.get("/api/v1/users").status_code == 403
    assert a.patch(f"/api/v1/users/{uid}", json={"role": "admin"}).status_code == 200
    assert u.get("/api/v1/users").status_code == 200          # same session, current role
    assert a.patch(f"/api/v1/users/{uid}", json={"role": "user"}).status_code == 200
    assert u.get("/api/v1/users").status_code == 403


def test_role_change_guards():
    a = _admin()
    me = a.get("/api/v1/auth/me").json()["user"]["id"]
    r = a.patch(f"/api/v1/users/{me}", json={"role": "user"})
    assert r.status_code == 400 and "signed in with" in r.json()["detail"]
    assert a.patch("/api/v1/users/no-such-id", json={"role": "user"}).status_code == 404
    assert a.patch(f"/api/v1/users/{me}", json={"role": "owner"}).status_code == 422


def test_users_cannot_manage_accounts():
    name = "plain_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    u = _bearer(name, PW)
    assert u.get("/api/v1/users").status_code == 403
    assert u.post("/api/v1/users", json={"username": "x" + name, "password": PW}).status_code == 403
    assert u.patch(f"/api/v1/users/{uid}", json={"role": "admin"}).status_code == 403
    assert u.delete(f"/api/v1/users/{uid}").status_code == 403
    assert db.get_user_by_id(uid)["role"] == "user"
