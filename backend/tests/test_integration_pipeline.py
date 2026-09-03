"""Full API & Pipeline Integration Tests.

Validates:
- Health check & DB readiness (L4)
- Security headers & CSP (H1)
- X-Request-ID propagation (H3)
- Auth enforcement across API endpoints
- Analysis execution & SQLite persistence (M5)
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token = create_access_token({"id": "test-admin", "name": "test-admin", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint_returns_db_status(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("ok", "degraded")
    assert data["platform"] == "AI Trust Forensics"
    assert "total_analyses" in data


def test_security_headers_and_request_id_present(client):
    res = client.get("/health")
    assert res.status_code == 200
    headers = res.headers
    assert "content-security-policy" in headers
    assert "x-content-type-options" in headers
    assert "x-frame-options" in headers
    assert "x-request-id" in headers
    assert headers["x-frame-options"] == "DENY"


def test_custom_request_id_header_honoured(client):
    res = client.get("/health", headers={"X-Request-ID": "custom-rid-12345"})
    assert res.status_code == 200
    assert res.headers["x-request-id"] == "custom-rid-12345"


def test_unauthenticated_request_rejected(client):
    res = client.post("/api/v1/defense/quarantine")
    assert res.status_code == 401


def test_demo_run_and_defense_persistence(client, auth_headers):
    res = client.post("/api/v1/demo/run", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "overall_suspicion_score" in data
    assert "verdict" in data
    assert "layer_scores" in data

    # Test defense status endpoint
    res_def = client.get("/api/v1/defense/status", headers=auth_headers)
    assert res_def.status_code == 200
    def_data = res_def.json()
    assert "mode" in def_data


def test_csv_upload_integration(client, auth_headers):
    csv_content = "feature1,feature2,feature3,label\n" + "\n".join(
        [f"{i*0.1},{i*0.2},{i*0.3},{i%2}" for i in range(50)]
    )
    files = {"file": ("test_dataset.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    res = client.post("/api/v1/analyze/upload", headers=auth_headers, files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_info"]["filename"] == "test_dataset.csv"
    assert data["dataset_info"]["n_rows"] == 50
    assert "verdict" in data
