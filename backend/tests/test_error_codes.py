from fastapi.testclient import TestClient

from app.core.errors import ApiError
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


@app.get("/__boom")
async def _boom():
    raise RuntimeError("secret internals")


@app.get("/__quota")
async def _quota():
    raise ApiError(429, "Daily quota reached", code="quota_exceeded")


def test_unauthorized_and_not_found_codes():
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401 and r.json()["code"] == "unauthorized" and isinstance(r.json()["detail"], str)
    r = client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_validation_error_keeps_detail_list():
    r = client.post("/api/v1/auth/signup", json={})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    assert isinstance(r.json()["detail"], list) and r.json()["detail"][0]["loc"]


def test_specific_code_and_internal_error_hides_internals():
    r = client.get("/__quota")
    assert r.status_code == 429 and r.json() == {"detail": "Daily quota reached", "code": "quota_exceeded"}
    r = client.get("/__boom")
    assert r.status_code == 500 and r.json() == {"detail": "Internal Server Error", "code": "internal_error"}
