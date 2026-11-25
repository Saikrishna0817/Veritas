"""S1 (launch gap plan): when admins must use 2FA, an admin who has not enrolled reads only their own data.

Before the fix only `require_admin` routes checked enrollment, so such an admin still read every person's results,
scans and the whole audit log through the routes that widen an admin's view (`owner_scope`, `_owned`, `/audit/events`).
"""
import dataclasses
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import security, totp
from app.core.config import settings
from app.core.security import hash_password
from app.main import app
from app.models import database as db
from app.services import auth as service

PW = "scope-password-1"


@pytest.fixture(autouse=True)
def _limiters():
    for lim in (service.login_rate_limiter, service.login_user_rate_limiter, service.twofa_ip_limiter, service.twofa_user_limiter):
        lim.clear()


def _signed_in(role):
    name = "s1_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role=role)
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/api/v1/auth/token", json={"username": name, "password": PW})
    assert r.status_code == 200
    return c, uid, {"X-CSRF-Token": r.json()["csrf_token"]}


def _csv():
    rng = np.random.RandomState(3)
    X = rng.randn(400, 4)
    y = (X[:, 0] > 0).astype(int)
    return pd.DataFrame(X, columns=list("abcd")).assign(label=y).to_csv(index=False).encode()


@pytest.fixture()
def someone_elses_result():
    c, uid, h = _signed_in("user")
    r = c.post("/api/v1/analyze/upload", headers=h, files={"file": ("private.csv", _csv(), "text/csv")})
    assert r.status_code == 200, r.text
    return uid, r.json()["dataset_info"]["dataset_id"]


def test_unenrolled_admin_sees_only_own_data_when_2fa_is_required(monkeypatch, someone_elses_result):
    owner_id, rid = someone_elses_result
    admin, admin_id, h = _signed_in("admin")
    assert admin.get(f"/api/v1/history/{rid}").status_code == 200                     # requirement off: admins see all

    monkeypatch.setattr(security, "settings", dataclasses.replace(settings, require_admin_2fa=True))
    assert admin.get(f"/api/v1/history/{rid}").status_code == 404
    assert admin.get(f"/api/v1/analyze/upload/{rid}").status_code == 404
    listed = admin.get("/api/v1/history", params={"limit": 200}).json()
    assert rid not in str(listed)
    events = admin.get("/api/v1/audit/events", params={"limit": 200}).json()["events"]
    assert all(e.get("actor_id") == admin_id for e in events)

    # enrolling restores the admin view
    setup = admin.post("/api/v1/auth/2fa/setup", headers=h).json()
    assert admin.post("/api/v1/auth/2fa/enable", json={"code": totp.code_at(setup["secret"])}, headers=h).status_code == 200
    assert admin.get(f"/api/v1/history/{rid}").status_code == 200
    assert any(e.get("actor_id") == owner_id for e in admin.get("/api/v1/audit/events", params={"limit": 200}).json()["events"])
