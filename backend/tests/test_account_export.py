"""F2 (launch gap plan): a person downloads a copy of the data Veritas holds about them (GET /account/export).
The privacy policy promised a copy on request by email; this makes it self-service."""
import json
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.main import app
from app.models import database as db
from app.services import auth as service

PW = "export-password-1"


@pytest.fixture(autouse=True)
def _limiters():
    for lim in (service.login_rate_limiter, service.login_user_rate_limiter):
        lim.clear()


def _signed_in():
    name = "f2_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW))
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert r.status_code == 200
    return c, name, uid, {"X-CSRF-Token": r.json()["csrf_token"]}


def _csv(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 3)
    return pd.DataFrame(X, columns=list("abc")).assign(label=(X[:, 0] > 0).astype(int)).to_csv(index=False).encode()


def test_export_has_my_data_and_nothing_secret_or_anyone_elses():
    me, name, uid, h = _signed_in()
    mine = me.post("/api/v1/analyze/upload", headers=h, files={"file": ("mine.csv", _csv(1), "text/csv")}).json()["dataset_info"]["dataset_id"]
    key = me.post("/api/v1/account/api-keys", headers=h, json={"name": "ci"}).json()
    other, _, _, oh = _signed_in()
    theirs = other.post("/api/v1/analyze/upload", headers=oh, files={"file": ("theirs.csv", _csv(2), "text/csv")}).json()["dataset_info"]["dataset_id"]

    r = me.get("/api/v1/account/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "attachment" in r.headers.get("content-disposition", "") and ".json" in r.headers["content-disposition"]
    body = r.json()
    assert body["account"]["id"] == uid and body["account"]["username"] == name
    assert [x["id"] for x in body["analyses"]] == [mine]
    assert body["analyses"][0]["result"]["dataset_info"]["dataset_id"] == mine          # the full stored result
    assert [k["name"] for k in body["api_keys"]] == ["ci"]
    assert any(e["action"] for e in body["audit_events"])
    assert body["exported_at"] and "retention_days" in body

    text = json.dumps(body)
    assert theirs not in text
    for secret in ("password_hash", "totp_secret", "key_hash", key.get("key", "vk_never")):
        assert secret not in text


def test_export_needs_a_session():
    assert TestClient(app).get("/api/v1/account/export").status_code == 401
