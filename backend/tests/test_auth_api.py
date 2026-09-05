import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.main import app
from app.api.router import api_router

from app.core.security import (
    authenticate_user,
    create_access_token,
    decode_access_token,
    require_admin,
    require_user,
)

client = TestClient(app)


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
        if require_user not in calls and require_admin not in calls:
            unprotected.append(route.path)
    assert unprotected == []


def test_regular_user_and_admin_authentication_and_role_detection():
    # Admin login
    admin_token = create_access_token({"id": "admin-1", "name": "admin", "role": "admin"})
    decoded_admin = decode_access_token(admin_token)
    assert decoded_admin["role"] == "admin"

    # Regular user login
    user_token = create_access_token({"id": "user-1", "name": "regular_user", "role": "user"})
    decoded_user = decode_access_token(user_token)
    assert decoded_user["role"] == "user"


def test_regular_user_cannot_access_admin_apis():
    user_token = create_access_token({"id": "user-1", "name": "regular_user", "role": "user"})
    headers = {"Authorization": f"Bearer {user_token}"}

    # Regular user calling /users (Admin-only)
    res_users = client.get("/api/v1/users", headers=headers)
    assert res_users.status_code == 403
    assert "Admin role required" in res_users.json()["detail"]

    # Regular user calling /defense/quarantine (Admin-only)
    res_quarantine = client.post("/api/v1/defense/quarantine", headers=headers)
    assert res_quarantine.status_code == 403
    assert "Admin role required" in res_quarantine.json()["detail"]


def test_admin_can_access_admin_apis():
    admin_token = create_access_token({"id": "admin", "name": "admin", "role": "admin"})
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Admin calling /users
    res_users = client.get("/api/v1/users", headers=headers)
    assert res_users.status_code == 200
    assert isinstance(res_users.json(), list)


def test_regular_user_can_access_user_apis():
    user_token = create_access_token({"id": "user-regular-001", "name": "user", "role": "user"})
    headers = {"Authorization": f"Bearer {user_token}"}

    # Regular user calling /auth/me
    res_me = client.get("/api/v1/auth/me", headers=headers)
    assert res_me.status_code == 200
    assert res_me.json()["user"]["role"] == "user"

    # Regular user calling /datasets/demo
    res_demo = client.get("/api/v1/datasets/demo", headers=headers)
    assert res_demo.status_code == 200


def test_portal_required_role_enforcement():
    from app.models.database import init_db
    init_db()

    # Regular user attempting admin portal login
    res_user_in_admin_portal = client.post("/api/v1/auth/token", json={"username": "user", "password": "user123", "required_role": "admin"})
    assert res_user_in_admin_portal.status_code == 403
    assert "Access denied" in res_user_in_admin_portal.json()["detail"]

    # Admin attempting user portal login
    res_admin_in_user_portal = client.post("/api/v1/auth/token", json={"username": "test-admin", "password": "test-password", "required_role": "user"})
    assert res_admin_in_user_portal.status_code == 400
    assert "Administrator account detected" in res_admin_in_user_portal.json()["detail"]

    # Valid user portal login
    res_valid_user = client.post("/api/v1/auth/token", json={"username": "user", "password": "user123", "required_role": "user"})
    assert res_valid_user.status_code == 200

    # Valid admin portal login
    res_valid_admin = client.post("/api/v1/auth/token", json={"username": "test-admin", "password": "test-password", "required_role": "admin"})
    assert res_valid_admin.status_code == 200


def test_delete_historical_result_permissions():
    user_token = create_access_token({"id": "user-1", "name": "user", "role": "user"})
    admin_token = create_access_token({"id": "admin", "name": "admin", "role": "admin"})

    # Regular user attempting DELETE /history/non-existent
    res_user = client.delete("/api/v1/history/fake-id", headers={"Authorization": f"Bearer {user_token}"})
    assert res_user.status_code == 403
    assert "Admin role required" in res_user.json()["detail"]

    # Admin attempting DELETE /history/non-existent
    res_admin = client.delete("/api/v1/history/fake-id", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin.status_code == 404




