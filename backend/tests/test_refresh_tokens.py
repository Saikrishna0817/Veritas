"""P3: refresh tokens rotate on every use; reusing a spent one ends the family; ending sessions revokes them."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.security import REFRESH_COOKIE, hash_password
from app.main import app
from app.models import database as db
from app.services import auth as service

PW = "refresh-test-password-1"
REFRESH = "/api/v1/auth/refresh"


@pytest.fixture(autouse=True)
def _limiters():
    service.login_rate_limiter.clear(); service.login_user_rate_limiter.clear()
    yield
    service.login_rate_limiter.clear(); service.login_user_rate_limiter.clear()


def _signed_in():
    name = "rf_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert r.status_code == 200
    return c, uid, r


def _refresh_cookie(c):
    return c.cookies.get(REFRESH_COOKIE, path="/api/v1/auth")


def test_refresh_cookie_is_httponly_strict_and_path_limited():
    _, _, r = _signed_in()
    header = next(h for h in r.headers.get_list("set-cookie") if h.startswith(REFRESH_COOKIE))
    assert "HttpOnly" in header and "SameSite=strict" in header and "Path=/api/v1/auth" in header


def test_refresh_rotates_and_issues_a_working_session():
    c, _, _ = _signed_in()
    first = _refresh_cookie(c)
    r = c.post(REFRESH)
    assert r.status_code == 200 and r.json()["csrf_token"]
    assert _refresh_cookie(c) != first
    assert c.get("/api/v1/auth/me").status_code == 200


def test_reusing_a_spent_token_ends_the_family_and_is_audited():
    c, uid, _ = _signed_in()
    stolen = _refresh_cookie(c)
    assert c.post(REFRESH).status_code == 200                 # the real browser rotates
    thief = TestClient(app, base_url="https://testserver")
    thief.cookies.set(REFRESH_COOKIE, stolen, path="/api/v1/auth")
    assert thief.post(REFRESH).status_code == 401              # the copy of the spent token is refused ...
    assert c.post(REFRESH).status_code == 401                  # ... and the newest token in the family is dead too
    row = db._get_conn().execute("SELECT 1 FROM audit_events WHERE actor_id = ? AND action = 'auth.refresh_reuse_detected'", (uid,)).fetchone()
    assert row is not None


def test_other_families_survive_a_reuse():
    c1, uid, _ = _signed_in()
    old = _refresh_cookie(c1)
    c1.post(REFRESH)
    other = TestClient(app, base_url="https://testserver")
    name = db.get_user_by_id(uid)["username"]
    other.post("/api/v1/auth/token", json={"username": name, "password": PW})
    thief = TestClient(app, base_url="https://testserver")
    thief.cookies.set(REFRESH_COOKIE, old, path="/api/v1/auth")
    thief.post(REFRESH)
    assert other.post(REFRESH).status_code == 200


def test_logout_revokes_the_refresh_token():
    c, _, _ = _signed_in()
    token = _refresh_cookie(c)
    assert c.post("/api/v1/auth/logout").status_code == 200
    again = TestClient(app, base_url="https://testserver")
    again.cookies.set(REFRESH_COOKIE, token, path="/api/v1/auth")
    assert again.post(REFRESH).status_code == 401


def test_logout_all_disable_and_delete_revoke_refresh_tokens():
    c, uid, r = _signed_in()
    token = _refresh_cookie(c)
    c.post("/api/v1/auth/logout-all", headers={"X-CSRF-Token": r.json()["csrf_token"]})
    assert db.use_refresh_token(token)[0] == "invalid"
    t2 = db.create_refresh_token(uid)
    db.set_user_disabled(uid, True)
    assert db.use_refresh_token(t2)[0] == "invalid"
    db.set_user_disabled(uid, False)
    t3 = db.create_refresh_token(uid)
    db.delete_account(uid)
    assert db.use_refresh_token(t3)[0] == "invalid"


def test_expired_refresh_is_refused():
    c, uid, _ = _signed_in()
    token = _refresh_cookie(c)
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    conn = db._get_conn()
    conn.execute("UPDATE refresh_tokens SET expires_at = ? WHERE user_id = ?", (past, uid))
    conn.commit()
    assert c.post(REFRESH).status_code == 401
    assert db.use_refresh_token(token)[0] == "invalid"


def test_no_cookie_is_refused():
    assert TestClient(app, base_url="https://testserver").post(REFRESH).status_code == 401
