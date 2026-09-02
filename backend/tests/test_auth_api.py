import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from app.api.router import api_router

from app.core.security import (
    authenticate_user,
    create_access_token,
    decode_access_token,
    require_user,
)


def test_missing_bearer_token_is_rejected():
    with pytest.raises(HTTPException) as error:
        require_user(None)
    assert error.value.status_code == 401


def test_token_issues_and_verifies():
    user = authenticate_user("test-admin", "test-password")
    assert user is not None
    token = create_access_token(user)
    assert decode_access_token(token) == user


def test_invalid_credentials_are_rejected():
    assert authenticate_user("test-admin", "wrong") is None


def test_invalid_token_is_rejected():
    with pytest.raises(HTTPException) as error:
        decode_access_token("not-a-token")
    assert error.value.status_code == 401


def test_every_rest_endpoint_except_token_issuance_requires_authentication():
    unprotected = []
    for route in api_router.routes:
        if not isinstance(route, APIRoute) or route.path == "/auth/token":
            continue
        calls = {dependency.call for dependency in route.dependant.dependencies}
        if require_user not in calls:
            unprotected.append(route.path)
    assert unprotected == []
