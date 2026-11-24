"""The auth service works without a Request: plain values in, the HTTP error or the sign-in body out."""
import uuid

import pytest
from fastapi import HTTPException, Response

from app.core.security import decode_access_token, hash_password
from app.models import database as db
from app.services import auth as service

PW = "service-test-password-1"
db.init_db()


def _clear():
    service.login_rate_limiter.clear()
    service.login_user_rate_limiter.clear()


def test_login_issue_and_sign_out_everywhere_without_a_request():
    name = "svc_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password(PW), role="user")
    _clear()
    user = service.check_password_login(name, PW, "198.51.100.200", "user")
    body = service.issue_session(Response(), user)
    assert decode_access_token(body["access_token"])["id"] == uid
    service.sign_out_everywhere(Response(), uid)
    with pytest.raises(HTTPException) as e:
        decode_access_token(body["access_token"])
    assert e.value.status_code == 401
    with pytest.raises(HTTPException) as e:
        service.check_password_login(name, "wrong", "198.51.100.200", None)
    assert e.value.status_code == 401
    _clear()
