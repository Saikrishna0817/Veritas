"""A deleted result leaves the caches (P4); duplicate names are 409 in every letter case (P7, P8)."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies as deps
from app.api.routes import auth
from app.core.security import hash_password
from app.main import app
from app.models import database as db
from auth_helpers import token_for

PW = "username-test-password-1"


@pytest.fixture(autouse=True)
def _fresh_limiters():
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter, auth.signup_rate_limiter):
        limiter._events.clear()
    yield
    for limiter in (auth.login_rate_limiter, auth.login_user_rate_limiter, auth.signup_rate_limiter):
        limiter._events.clear()


def _as(user):
    c = TestClient(app, base_url="https://testserver")
    c.headers["Authorization"] = f"Bearer {token_for(user)}"
    return c


def test_a_deleted_result_is_no_longer_served_from_the_cache():
    owner = {"id": "cache-" + uuid.uuid4().hex[:8], "name": "cache-owner", "role": "user"}
    rid = "ds-" + uuid.uuid4().hex[:8]
    result = {"dataset_id": rid, "verdict": "SUSPICIOUS", "overall_suspicion_score": 0.7, "n_samples": 10,
              "attack_classification": {"attack_type": "label_flip"}, "dataset_info": {"dataset_id": rid}}
    db.save_result(result, "upload", "x.csv", owner_id=owner["id"])
    deps.upload_result_cache[rid] = result
    deps.upload_result_cache[deps.latest_key(owner["id"])] = result
    other = {"dataset_id": "keep-" + rid}
    deps.upload_result_cache["keep-" + rid] = other

    admin = _as({"id": os.environ["VERITAS_ADMIN_USERNAME"], "name": os.environ["VERITAS_ADMIN_USERNAME"], "role": "admin"})
    assert admin.delete(f"/api/v1/history/{rid}").status_code == 200
    assert rid not in deps.upload_result_cache
    assert deps.latest_key(owner["id"]) not in deps.upload_result_cache
    assert deps.upload_result_cache.get("keep-" + rid) is other                # other results stay
    assert _as(owner).get("/api/v1/forensics/latest?source=upload").json().get("verdict") != "SUSPICIOUS"


def test_admin_duplicate_is_409_in_any_letter_case():
    admin = _as({"id": os.environ["VERITAS_ADMIN_USERNAME"], "name": os.environ["VERITAS_ADMIN_USERNAME"], "role": "admin"})
    name = "Case" + uuid.uuid4().hex[:8]
    assert admin.post("/api/v1/users", json={"username": name, "password": PW}).status_code == 201
    for again in (name, name.lower(), name.upper()):
        r = admin.post("/api/v1/users", json={"username": again, "password": PW})
        assert r.status_code == 409 and "already exists" in r.json()["detail"], again


def test_sign_up_refuses_a_name_that_differs_only_in_case():
    name = "Mixed" + uuid.uuid4().hex[:8]
    db.create_user(name, hash_password(PW), role="user")
    r = TestClient(app, base_url="https://testserver").post(
        "/api/v1/auth/signup", json={"username": name.lower(), "email": f"{name.lower()}@example.com", "password": PW})
    assert r.status_code == 409
    assert db.get_user_by_username(name.lower()) is None
