"""F1 (launch gap plan): an admin resets another person's two-factor sign-in when they have lost the phone and the
recovery codes. Before this the only way back in was editing the database."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import totp
from app.core.security import hash_password
from app.main import app
from app.models import database as db
from app.services import auth as service

PW = "reset-password-1"


@pytest.fixture(autouse=True)
def _limiters():
    for lim in (service.login_rate_limiter, service.login_user_rate_limiter, service.twofa_ip_limiter, service.twofa_user_limiter):
        lim.clear()


def _signed_in(role):
    name = "f1_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role=role)
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert r.status_code == 200
    return c, name, uid, {"X-CSRF-Token": r.json()["csrf_token"]}


def _enroll(c, h):
    setup = c.post("/api/v1/auth/2fa/setup", headers=h).json()
    assert c.post("/api/v1/auth/2fa/enable", json={"code": totp.code_at(setup["secret"])}, headers=h).status_code == 200


def test_admin_resets_a_persons_2fa_and_ends_their_sessions():
    person, name, pid, ph = _signed_in("user")
    _enroll(person, ph)
    fresh = TestClient(app, base_url="https://testserver")
    assert fresh.post("/api/v1/auth/token", json={"username": name, "password": PW}).json().get("two_factor_required") is True

    admin, _, admin_id, ah = _signed_in("admin")
    r = admin.post(f"/api/v1/users/{pid}/2fa/reset", headers=ah)
    assert r.status_code == 200 and name in r.json()["message"]

    assert person.get("/api/v1/auth/me").status_code == 401                          # their sessions ended
    again = TestClient(app, base_url="https://testserver").post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert again.status_code == 200 and "two_factor_required" not in again.json()      # password alone again
    stored = db.get_user_by_id(pid)
    assert not stored.get("totp_enabled_at") and not stored.get("totp_secret")
    events = db.get_audit_events(actor_id=admin_id, limit=20)
    assert any(e["action"] == "user.2fa_reset" and e.get("resource_id") == pid for e in events)


def test_reset_rules():
    admin, _, admin_id, ah = _signed_in("admin")
    assert admin.post(f"/api/v1/users/{admin_id}/2fa/reset", headers=ah).status_code == 400      # not your own
    assert admin.post(f"/api/v1/users/{uuid.uuid4()}/2fa/reset", headers=ah).status_code == 404
    user, _, uid, uh = _signed_in("user")
    assert user.post(f"/api/v1/users/{uid}/2fa/reset", headers=uh).status_code == 403            # admins only

    # someone without 2FA: nothing changes, their session stays
    r = admin.post(f"/api/v1/users/{uid}/2fa/reset", headers=ah)
    assert r.status_code == 200 and "does not use" in r.json()["message"]
    assert user.get("/api/v1/auth/me").status_code == 200
