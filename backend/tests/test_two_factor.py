"""P12: TOTP two-factor sign-in, enrollment, recovery codes, and the optional admin requirement."""
import base64
import dataclasses
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import security, totp
from app.core.config import settings
from app.core.security import hash_password
from app.main import app
from app.models import database as db
from app.services import auth as service

PW = "two-factor-password-1"
API = "/api/v1/auth"
RFC_SECRET = base64.b32encode(b"12345678901234567890").decode()


@pytest.fixture(autouse=True)
def _limiters():
    def clear():
        for lim in (service.login_rate_limiter, service.login_user_rate_limiter, service.twofa_ip_limiter, service.twofa_user_limiter):
            lim.clear()
    clear()
    yield
    clear()


def test_rfc6238_vectors_and_replay():
    for t, expected in ((59, "287082"), (1111111109, "081804"), (1111111111, "050471"),
                        (1234567890, "005924"), (2000000000, "279037"), (20000000000, "353130")):
        assert totp.code_at(RFC_SECRET, t) == expected
    step = totp.verify(RFC_SECRET, "287082", at=59)
    assert step == 1
    assert totp.verify(RFC_SECRET, "287082", last_step=step, at=59) is None      # the same step is not accepted twice
    assert totp.verify(RFC_SECRET, "12345", at=59) is None and totp.verify(RFC_SECRET, "abcdef", at=59) is None


def test_secret_encryption_round_trips_and_detects_tampering():
    secret = totp.new_secret()
    stored = totp.encrypt_secret(secret)
    assert secret not in stored and totp.decrypt_secret(stored) == secret
    assert totp.encrypt_secret(secret) != stored
    raw = bytearray(base64.urlsafe_b64decode(stored)); raw[20] ^= 1
    with pytest.raises(ValueError):
        totp.decrypt_secret(base64.urlsafe_b64encode(bytes(raw)).decode())


def _account(role="user"):
    name = "tf_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role=role)
    c = TestClient(app, base_url="https://testserver")
    r = c.post(f"{API}/token", json={"username": name, "password": PW})
    assert r.status_code == 200
    return c, name, uid, {"X-CSRF-Token": r.json()["csrf_token"]}


def _enroll(c, uid, headers):
    setup = c.post(f"{API}/2fa/setup", headers=headers).json()
    assert setup["otpauth_uri"].startswith("otpauth://totp/") and setup["secret"] in setup["otpauth_uri"]
    r = c.post(f"{API}/2fa/enable", json={"code": totp.code_at(setup["secret"])}, headers=headers)
    assert r.status_code == 200
    return setup["secret"], r.json()["recovery_codes"]


def test_enrollment_then_two_step_sign_in():
    c, name, uid, h = _account()
    secret, codes = _enroll(c, uid, h)
    assert len(codes) == 8 and len(set(codes)) == 8
    stored = db.get_user_by_id(uid)["totp_secret"]
    assert secret not in stored                                                   # encrypted at rest
    assert c.get(f"{API}/me").json()["two_factor_enabled"] is True
    assert c.post(f"{API}/2fa/setup", headers=h).status_code == 409                # the secret is not shown again

    fresh = TestClient(app, base_url="https://testserver")
    r = fresh.post(f"{API}/token", json={"username": name, "password": PW})
    body = r.json()
    assert body["two_factor_required"] is True and "access_token" not in body
    assert fresh.get(f"{API}/me").status_code == 401                               # no session yet
    # a challenge is not a session token
    assert TestClient(app).get(f"{API}/me", headers={"Authorization": f"Bearer {body['challenge']}"}).status_code == 401

    # the enrollment code's step is spent, so the next sign-in needs a code from a later step
    later = totp.code_at(secret, __import__("time").time() + 30)
    ok = fresh.post(f"{API}/2fa/verify", json={"challenge": body["challenge"], "code": later})
    assert ok.status_code == 200 and ok.json()["access_token"]
    assert fresh.get(f"{API}/me").status_code == 200


def test_wrong_code_refused_recovery_code_works_once_and_verify_is_rate_limited():
    c, name, uid, h = _account()
    secret, codes = _enroll(c, uid, h)

    def challenge():
        service.login_rate_limiter.clear(); service.login_user_rate_limiter.clear()
        return TestClient(app, base_url="https://testserver").post(f"{API}/token", json={"username": name, "password": PW}).json()["challenge"]
    cl = TestClient(app, base_url="https://testserver")
    assert cl.post(f"{API}/2fa/verify", json={"challenge": challenge(), "code": "000000"}).status_code == 401
    assert cl.post(f"{API}/2fa/verify", json={"challenge": challenge(), "code": codes[0]}).status_code == 200
    assert cl.post(f"{API}/2fa/verify", json={"challenge": challenge(), "code": codes[0]}).status_code == 401   # one use only
    assert db._get_conn().execute("SELECT 1 FROM audit_events WHERE actor_id = ? AND action = 'auth.2fa_failed'", (uid,)).fetchone()
    for _ in range(6):
        last = cl.post(f"{API}/2fa/verify", json={"challenge": challenge(), "code": "111111"})
    assert last.status_code == 429


def test_challenge_must_be_valid():
    assert TestClient(app).post(f"{API}/2fa/verify", json={"challenge": "x" * 40, "code": "123456"}).status_code == 401


def test_disable_needs_a_current_code():
    c, name, uid, h = _account()
    secret, codes = _enroll(c, uid, h)
    assert c.post(f"{API}/2fa/disable", json={"code": "000000"}, headers=h).status_code == 401
    assert c.post(f"{API}/2fa/disable", json={"code": codes[1]}, headers=h).status_code == 200
    assert db.get_user_by_id(uid)["totp_secret"] is None
    r = TestClient(app, base_url="https://testserver").post(f"{API}/token", json={"username": name, "password": PW})
    assert "access_token" in r.json()


def test_enable_with_wrong_code_is_refused_and_needs_setup_first():
    c, _, uid, h = _account()
    assert c.post(f"{API}/2fa/enable", json={"code": "123456"}, headers=h).status_code == 409
    c.post(f"{API}/2fa/setup", headers=h)
    assert c.post(f"{API}/2fa/enable", json={"code": "000000"}).status_code == 403          # CSRF still required
    assert c.post(f"{API}/2fa/enable", json={"code": "000000"}, headers=h).status_code == 400
    assert db.get_user_by_id(uid)["totp_enabled_at"] is None


def test_admin_requirement_blocks_admin_endpoints_until_enrolled(monkeypatch):
    c, name, uid, h = _account(role="admin")
    assert c.get("/api/v1/users").status_code == 200                                        # off by default
    monkeypatch.setattr(security, "settings", dataclasses.replace(settings, require_admin_2fa=True))
    r = c.get("/api/v1/users")
    assert r.status_code == 403 and "Two-factor" in r.json()["detail"]
    assert c.post(f"{API}/2fa/setup", headers=h).status_code == 200                         # enrolling stays possible
    secret = totp.decrypt_secret(db.get_user_by_id(uid)["totp_secret"])
    assert c.post(f"{API}/2fa/enable", json={"code": totp.code_at(secret)}, headers=h).status_code == 200
    assert c.get("/api/v1/users").status_code == 200
