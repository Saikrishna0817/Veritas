"""A mail server that is down must not break sign-up or password reset with a 500."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.routes import auth
from app.core import mailer
from app.main import app

PW = "correct-horse-battery"


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.signup_rate_limiter, auth.mail_rate_limiter, auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()
    yield


@pytest.fixture
def mail_down(monkeypatch):
    def refuse(*a, **kw):
        raise ConnectionRefusedError("mail server unreachable")
    monkeypatch.setattr(mailer, "send", refuse)


def _client():
    return TestClient(app, base_url="https://testserver")


def test_signup_says_the_mail_was_not_sent_and_reset_recovers_the_account(mail_down, monkeypatch, caplog):
    c = _client()
    name = "md_" + uuid.uuid4().hex[:8]
    r = c.post("/api/v1/auth/signup", json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert r.status_code == 201
    assert r.json()["mail_sent"] is False and "could not be sent" in r.json()["message"]
    assert "Sending mail failed" in caplog.text

    # mail works again: a password reset link also confirms the address, so the account is not stuck
    monkeypatch.undo()
    r = c.post("/api/v1/auth/password-reset/request", json={"email": f"{name}@example.com"})
    assert r.status_code == 200
    import re
    token = re.search(r"token=([\w-]+)", mailer.outbox[-1]["body"]).group(1)
    assert c.post("/api/v1/auth/password-reset/confirm", json={"token": token, "new_password": "a-new-password-22"}).status_code == 200
    assert c.post("/api/v1/auth/token", json={"username": name, "password": "a-new-password-22"}).status_code == 200


def test_resend_and_reset_answer_the_same_when_mail_fails(mail_down):
    c = _client()
    name = "mr_" + uuid.uuid4().hex[:8]
    c.post("/api/v1/auth/signup", json={"username": name, "email": f"{name}@example.com", "password": PW})
    for path in ("/api/v1/auth/resend-verification", "/api/v1/auth/password-reset/request"):
        known = c.post(path, json={"email": f"{name}@example.com"})
        unknown = c.post(path, json={"email": f"nobody-{name}@example.com"})
        assert known.status_code == unknown.status_code == 200
        assert known.json() == unknown.json()        # never reveals which addresses have accounts
