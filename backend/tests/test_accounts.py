"""Sign-up, email verification, password reset, cookie sessions with CSRF, and API keys."""
import re

from fastapi.testclient import TestClient

from app.core import mailer
from app.main import app

PW = "correct-horse-battery"


import pytest


@pytest.fixture(autouse=True)
def _fresh_limiters():
    from app.api.routes import auth
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter, auth.signup_rate_limiter, auth.mail_rate_limiter):
        limiter._events.clear()
    yield


def _client():
    return TestClient(app, base_url="https://testserver")      # cookies are Secure, so talk https


def _link_token():
    return re.search(r"token=([\w-]+)", mailer.outbox[-1]["body"]).group(1)


def _signup(c, name="newuser", email="new@example.com", pw=PW):
    return c.post("/api/v1/auth/signup", json={"username": name, "email": email, "password": pw})


def test_signup_requires_email_verification_before_login():
    c = _client()
    assert _signup(c).status_code == 201
    assert c.post("/api/v1/auth/token", json={"username": "newuser", "password": PW}).status_code == 403
    assert c.post("/api/v1/auth/verify-email", json={"token": "x" * 20}).status_code == 400
    assert c.post("/api/v1/auth/verify-email", json={"token": _link_token()}).status_code == 200
    assert c.post("/api/v1/auth/verify-email", json={"token": _link_token()}).status_code == 400    # single use
    assert c.post("/api/v1/auth/token", json={"username": "newuser", "password": PW}).status_code == 200


def test_duplicate_and_weak_signups_are_rejected():
    c = _client()
    _signup(c, "dupe", "dupe@example.com")
    assert _signup(c, "dupe", "other@example.com").status_code == 409
    assert _signup(c, "other", "dupe@example.com").status_code == 409
    assert _signup(c, "weak", "weak@example.com", "short").status_code == 422
    assert _signup(c, "bad name!", "x@example.com").status_code == 422


def test_password_reset_flow_and_no_account_enumeration():
    c = _client()
    _signup(c, "resetme", "reset@example.com")
    c.post("/api/v1/auth/verify-email", json={"token": _link_token()})
    old_token = c.post("/api/v1/auth/token", json={"username": "resetme", "password": PW}).json()["access_token"]
    before = len(mailer.outbox)
    unknown = c.post("/api/v1/auth/password-reset/request", json={"email": "nobody@example.com"})
    known = c.post("/api/v1/auth/password-reset/request", json={"email": "reset@example.com"})
    assert unknown.json() == known.json() and len(mailer.outbox) == before + 1
    token = _link_token()
    assert c.post("/api/v1/auth/password-reset/confirm", json={"token": token, "new_password": "a-brand-new-pass"}).status_code == 200
    assert c.post("/api/v1/auth/password-reset/confirm", json={"token": token, "new_password": "yet-another-pass"}).status_code == 400
    assert c.post("/api/v1/auth/token", json={"username": "resetme", "password": PW}).status_code == 401
    assert c.post("/api/v1/auth/token", json={"username": "resetme", "password": "a-brand-new-pass"}).status_code == 200
    # a session from before the reset is over
    assert c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {old_token}"}).status_code == 401


def _login(name="sessionuser", email="session@example.com"):
    c = _client()
    _signup(c, name, email)
    c.post("/api/v1/auth/verify-email", json={"token": _link_token()})
    r = c.post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert r.status_code == 200
    return c, r.json()


def test_cookie_session_works_for_reads_and_needs_csrf_for_writes():
    c, body = _login()
    assert "httponly" in c.post("/api/v1/auth/token", json={"username": "sessionuser", "password": PW}).headers["set-cookie"].lower()
    assert c.get("/api/v1/auth/me").json()["user"]["name"] == "sessionuser"                  # cookie only, no bearer header
    assert c.delete("/api/v1/account/data").status_code == 403                                # write without CSRF header
    assert c.delete("/api/v1/account/data", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert c.delete("/api/v1/account/data", headers={"X-CSRF-Token": c.cookies.get("veritas_csrf")}).status_code == 200
    c.post("/api/v1/auth/logout", headers={"X-CSRF-Token": c.cookies.get("veritas_csrf")})
    c.cookies.clear()
    assert c.get("/api/v1/auth/me").status_code == 401


def test_bearer_token_needs_no_csrf():
    c, body = _login("bearerguy", "bearer@example.com")
    c.cookies.clear()
    assert c.delete("/api/v1/account/data", headers={"Authorization": f"Bearer {body['access_token']}"}).status_code == 200


def test_api_keys_authenticate_can_be_revoked_and_are_stored_hashed():
    c, body = _login("keyguy", "key@example.com")
    bearer = {"Authorization": f"Bearer {body['access_token']}"}
    made = c.post("/api/v1/account/api-keys", json={"name": "ci"}, headers=bearer).json()
    raw = made["key"]
    assert raw.startswith("vk_")
    key = {"Authorization": f"Bearer {raw}"}
    assert c.get("/api/v1/auth/me", headers=key).json()["user"]["name"] == "keyguy"
    assert c.post("/api/v1/account/api-keys", json={"name": "nested"}, headers=key).status_code == 403
    listed = c.get("/api/v1/account/api-keys", headers=bearer).json()["keys"]
    assert listed[0]["prefix"] == raw[:10] and "key" not in listed[0] and raw not in str(listed)
    assert c.delete(f"/api/v1/account/api-keys/{made['id']}", headers=bearer).status_code == 200
    assert c.get("/api/v1/auth/me", headers=key).status_code == 401


def test_delete_account_needs_password_and_erases_everything():
    c, body = _login("goneguy", "gone@example.com")
    bearer = {"Authorization": f"Bearer {body['access_token']}"}
    c.cookies.clear()
    raw = c.post("/api/v1/account/api-keys", json={"name": "ci"}, headers=bearer).json()["key"]
    assert c.post("/api/v1/account/delete", json={"password": "wrong-password-1"}, headers=bearer).status_code == 403
    assert c.get("/api/v1/auth/me", headers=bearer).status_code == 200                         # a wrong password deletes nothing
    done = c.post("/api/v1/account/delete", json={"password": PW}, headers=bearer)
    assert done.status_code == 200 and done.json()["deleted"]["users"] == 1 and done.json()["deleted"]["api_keys"] == 1
    assert c.get("/api/v1/auth/me", headers=bearer).status_code == 401
    assert c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {raw}"}).status_code == 401
    assert c.post("/api/v1/auth/token", json={"username": "goneguy", "password": PW}).status_code == 401


def test_websocket_cookie_session_rejects_foreign_origin():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    c, _ = _login("wsguy", "ws@example.com")
    cookie = {"cookie": f"veritas_session={c.cookies.get('veritas_session')}"}
    with c.websocket_connect("/ws/v1/detection-stream", headers={"origin": "http://localhost:5173", **cookie}) as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/v1/detection-stream", headers={"origin": "https://evil.example", **cookie}) as ws:
            ws.receive_json()


def test_redis_rate_limiter_is_shared_between_instances():
    import os
    import uuid
    url = os.environ.get("VERITAS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set VERITAS_TEST_REDIS_URL to run against a Redis server")
    from app.core.rate_limit import RedisRateLimiter
    name = "t" + uuid.uuid4().hex[:8]
    a, b = RedisRateLimiter(url, name, 3, 60), RedisRateLimiter(url, name, 3, 60)   # two "processes"
    assert [a.allow("k"), b.allow("k"), a.allow("k"), b.allow("k")] == [True, True, True, False]
    assert a.allow("other")
    a.clear()
    assert b.allow("k")


def test_redis_limiter_degrades_to_local_when_redis_is_down():
    from app.core.rate_limit import RedisRateLimiter
    lim = RedisRateLimiter("redis://127.0.0.1:1", "down", 2, 60)
    assert [lim.allow("k"), lim.allow("k"), lim.allow("k")] == [True, True, False]


def test_metrics_endpoint_is_hidden_without_a_token_and_exposes_route_templates(monkeypatch):
    c = _client()
    c.get("/health")
    assert c.get("/metrics").status_code == 404
    monkeypatch.setenv("VERITAS_METRICS_TOKEN", "scrape-secret-123")
    assert c.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 404
    body = c.get("/metrics", headers={"Authorization": "Bearer scrape-secret-123"}).text
    assert 'veritas_http_requests_total{method="GET",route="/health",status="200"}' in body
