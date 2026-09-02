import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.routes.auth import LoginRequest, login_rate_limiter, token
from app.core.security import require_user


def make_request(ip: str = "198.51.100.1") -> Request:
    return Request({"type": "http", "method": "POST", "path": "/auth/token", "headers": [], "client": (ip, 50000)})


@pytest.fixture(autouse=True)
def clear_login_rate_limiter():
    login_rate_limiter.clear()
    yield
    login_rate_limiter.clear()


def test_login_rate_limit_blocks_after_threshold():
    credentials = LoginRequest(username="test-admin", password="wrong")
    for _ in range(5):
        with pytest.raises(HTTPException) as error:
            asyncio.run(token(credentials, make_request()))
        assert error.value.status_code == 401
    with pytest.raises(HTTPException) as error:
        asyncio.run(token(credentials, make_request()))
    assert error.value.status_code == 429


def test_login_rate_limit_resets_after_window():
    for index in range(5):
        assert login_rate_limiter.allow("198.51.100.2", now=float(index))
    assert not login_rate_limiter.allow("198.51.100.2", now=59.0)
    assert login_rate_limiter.allow("198.51.100.2", now=60.0)


def test_other_routes_are_not_rate_limited():
    for _ in range(20):
        with pytest.raises(HTTPException) as error:
            require_user(None)
        assert error.value.status_code == 401
    assert login_rate_limiter.allow("198.51.100.3", now=0.0)
