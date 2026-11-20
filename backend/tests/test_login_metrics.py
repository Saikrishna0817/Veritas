"""Refused sign-ins are counted for alerting (P9b), without usernames or IPs in the series."""
import uuid

import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from app.api.routes import auth
from app.core.observability import LOGIN_FAILURES
from app.core.security import hash_password
from app.main import app
from app.models import database as db


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()
    yield
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter):
        limiter._events.clear()


def _n(outcome):
    return LOGIN_FAILURES.labels(outcome)._value.get()


def test_refused_sign_ins_are_counted_by_outcome():
    c = TestClient(app, base_url="https://testserver")
    name = "metric_" + uuid.uuid4().hex[:8]
    db.create_user(name, hash_password("metrics-test-password"), role="user")
    before = {o: _n(o) for o in ("wrong_password", "unknown_account", "rate_limited")}

    assert c.post("/api/v1/auth/token", json={"username": name, "password": "nope-nope-nope"}).status_code == 401
    assert c.post("/api/v1/auth/token", json={"username": "ghost_" + name, "password": "nope-nope-nope"}).status_code == 401
    assert _n("wrong_password") == before["wrong_password"] + 1
    assert _n("unknown_account") == before["unknown_account"] + 1

    codes = [c.post("/api/v1/auth/token", json={"username": name, "password": "nope-nope-nope"}).status_code for _ in range(12)]
    assert 429 in codes
    assert _n("rate_limited") - before["rate_limited"] == codes.count(429)

    exposed = generate_latest().decode()
    assert "veritas_auth_login_failures_total" in exposed and name not in exposed
