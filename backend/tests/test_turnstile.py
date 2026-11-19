"""Cloudflare Turnstile on the public forms: off when no secret is set, required (with a clear message) when it is."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from app.api.routes import auth
from app.core.config import settings
from app.main import app

PW = "correct-horse-battery"
FORMS = [
    ("/api/v1/auth/signup", lambda n: {"username": f"ts{n}", "email": f"ts{n}@example.com", "password": PW}, 201),
    ("/api/v1/auth/resend-verification", lambda n: {"email": f"nobody{n}@example.com"}, 200),
    ("/api/v1/auth/password-reset/request", lambda n: {"email": f"nobody{n}@example.com"}, 200),
]


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.signup_rate_limiter, auth.mail_rate_limiter):
        limiter._events.clear()
    yield


class _FakeCloudflare:
    """Stands in for httpx.AsyncClient so no test calls Cloudflare. `answer` is what siteverify says."""
    answer = {"success": True}
    seen = []

    def __init__(self, *a, **kw): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return False

    async def post(self, url, data):
        _FakeCloudflare.seen.append((url, data))
        answer = _FakeCloudflare.answer

        class _R:
            def json(self): return answer
        return _R()


def _secret(monkeypatch, value):
    """Settings are frozen; the auth routes read them through their own module name, so swap in a copy there."""
    monkeypatch.setattr(auth, "settings", dataclasses.replace(settings, turnstile_secret=value))


def _client():
    return TestClient(app, base_url="https://testserver")


@pytest.mark.parametrize("path,body,ok", FORMS)
def test_without_a_secret_no_token_is_needed(monkeypatch, path, body, ok):
    _secret(monkeypatch, "")
    r = _client().post(path, json=body("off"))
    assert r.status_code == ok, r.text


@pytest.mark.parametrize("path,body,ok", FORMS)
def test_with_a_secret_a_missing_token_is_refused_clearly(monkeypatch, path, body, ok):
    _secret(monkeypatch, "test-secret")
    monkeypatch.setattr(auth.httpx, "AsyncClient", _FakeCloudflare)
    r = _client().post(path, json=body("missing"))
    assert r.status_code == 400
    assert r.json()["detail"] == "Please complete the human check."


@pytest.mark.parametrize("path,body,ok", FORMS)
def test_with_a_secret_the_token_is_verified(monkeypatch, path, body, ok):
    _secret(monkeypatch, "test-secret")
    monkeypatch.setattr(auth.httpx, "AsyncClient", _FakeCloudflare)
    _FakeCloudflare.seen.clear()

    _FakeCloudflare.answer = {"success": False}
    r = _client().post(path, json={**body("bad"), "turnstile_token": "rejected-token"})
    assert r.status_code == 400 and r.json()["detail"] == "Human check failed. Please try again."

    _FakeCloudflare.answer = {"success": True}
    r = _client().post(path, json={**body("good"), "turnstile_token": "accepted-token"})
    assert r.status_code == ok, r.text
    url, data = _FakeCloudflare.seen[-1]
    assert url.endswith("/turnstile/v0/siteverify")
    assert data["secret"] == "test-secret" and data["response"] == "accepted-token"
