"""Authentication routes: login (bearer + cookie session), sign-up with email verification, password reset."""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.config import settings
from app.core.security import require_user
from app.services import auth as service
from app.services.auth import login_rate_limiter, login_user_rate_limiter, mail_rate_limiter, signup_rate_limiter  # noqa: F401  (tests clear these)

logger = logging.getLogger(__name__)
router = APIRouter()


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _password(value: str) -> str:
    if len(value.encode("utf-8")) > 72:     # bcrypt ignores bytes past 72: reject instead of silently truncating
        raise ValueError("password must be at most 72 bytes")
    return value


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)
    required_role: Optional[str] = Field(default=None, pattern="^(user|analyst|admin)$")


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    turnstile_token: Optional[str] = None
    _pw = field_validator("password")(lambda cls, v: _password(v))


class EmailRequest(BaseModel):
    email: EmailStr
    turnstile_token: Optional[str] = None


class TokenRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)


class ResetConfirm(TokenRequest):
    new_password: str = Field(min_length=10, max_length=128)
    _pw = field_validator("new_password")(lambda cls, v: _password(v))


async def _check_turnstile(token: Optional[str], request: Request) -> None:
    """Cloudflare Turnstile on public forms. Skipped when no secret is configured."""
    if not settings.turnstile_secret:
        return
    if not token:
        raise HTTPException(status_code=400, detail="Please complete the human check.")
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post("https://challenges.cloudflare.com/turnstile/v0/siteverify",
                                  data={"secret": settings.turnstile_secret, "response": token, "remoteip": _ip(request)})
        ok = r.json().get("success") is True
    except Exception:
        logger.exception("Turnstile verification failed to run")
        ok = False
    if not ok:
        raise HTTPException(status_code=400, detail="Human check failed. Please try again.")


@router.post("/auth/token")
async def token(credentials: LoginRequest, request: Request, response: Response):
    user = service.check_password_login(credentials.username, credentials.password, _ip(request), credentials.required_role)
    return service.issue_session(response, user)


@router.post("/auth/logout")
async def logout(response: Response):
    """Clears this browser's cookies. The token itself stays valid until it expires; /auth/logout-all ends it."""
    service.sign_out(response)
    return {"message": "Signed out."}


@router.post("/auth/logout-all")
async def logout_everywhere(response: Response, user: dict = Depends(require_user)):
    """End every session of the caller's account on every device, this one included. API keys keep working."""
    service.sign_out_everywhere(response, user["id"])
    return {"message": "Signed out on every device."}


@router.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, request: Request):
    service.check_signup_allowed(_ip(request))
    await _check_turnstile(body.turnstile_token, request)
    return await service.register(body.username, body.email, body.password)


@router.post("/auth/verify-email")
async def verify_email(body: TokenRequest):
    service.verify_email(body.token)
    return {"message": "Email confirmed. You can sign in."}


@router.post("/auth/resend-verification")
async def resend_verification(body: EmailRequest, request: Request):
    """Always answers the same way so it cannot be used to discover which emails have accounts."""
    service.check_mail_allowed(_ip(request))
    await _check_turnstile(body.turnstile_token, request)
    await service.resend_verification(body.email)
    return {"message": "If that address has an unconfirmed account, a new link is on its way."}


@router.post("/auth/password-reset/request")
async def password_reset_request(body: EmailRequest, request: Request):
    service.check_mail_allowed(_ip(request))
    await _check_turnstile(body.turnstile_token, request)
    await service.request_password_reset(body.email)
    return {"message": "If that address has an account, a reset link is on its way."}


@router.post("/auth/password-reset/confirm")
async def password_reset_confirm(body: ResetConfirm):
    service.confirm_password_reset(body.token, body.new_password)
    return {"message": "Password changed. You can sign in."}


@router.get("/auth/me")
async def me(user: dict = Depends(require_user)):
    return {"user": user}
