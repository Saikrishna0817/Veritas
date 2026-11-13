"""Authentication routes: login (bearer + cookie session), sign-up with email verification, password reset."""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core import mailer
from app.core.config import settings
from app.core.rate_limit import make_limiter
from app.core.security import (authenticate_user, clear_session_cookies, create_access_token, hash_password, require_user,
                               set_session_cookies)
from app.models import database as db

logger = logging.getLogger(__name__)
router = APIRouter()
login_rate_limiter = make_limiter("login", 5, 60)
login_user_rate_limiter = make_limiter("login-user", 10, 900)   # per username, so guessing spread over many IPs is slowed too
signup_rate_limiter = make_limiter("signup", 5, 3600)
mail_rate_limiter = make_limiter("mail", 5, 3600)

VERIFY_TTL_MIN, RESET_TTL_MIN = 24 * 60, 60


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


async def _send_verification(user_id: str, email: str) -> None:
    token = db.create_auth_token(user_id, "verify_email", VERIFY_TTL_MIN)
    await run_in_threadpool(mailer.send, email, "Confirm your Veritas email",
                f"Welcome to Veritas.\n\nConfirm your email address (valid 24 hours):\n{settings.public_url}/verify-email?token={token}\n\n"
                "If you did not sign up, ignore this message.")


@router.post("/auth/token")
async def token(credentials: LoginRequest, request: Request, response: Response):
    if not login_rate_limiter.allow(_ip(request)):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts. Try again in one minute.")
    if not login_user_rate_limiter.allow(credentials.username.lower()):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts for this account. Try again in 15 minutes.")
    user = authenticate_user(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})

    # Flow-specific role enforcement
    if credentials.required_role == "admin" and user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Account does not have administrator privileges.")
    if credentials.required_role in ("user", "analyst") and user.get("role") == "admin":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Administrator account detected. Please use the Administrator Login portal at /admin/login.")

    access = create_access_token(user)
    csrf = set_session_cookies(response, access)       # browsers use the cookies; API clients can keep using the bearer token
    return {"access_token": access, "token_type": "bearer", "user": user, "csrf_token": csrf}


@router.post("/auth/logout")
async def logout(response: Response):
    clear_session_cookies(response)
    return {"message": "Signed out."}


@router.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, request: Request):
    if not settings.signup_enabled:
        raise HTTPException(status_code=403, detail="Sign-up is closed.")
    if not signup_rate_limiter.allow(_ip(request)):
        raise HTTPException(status_code=429, detail="Too many sign-ups from this address. Try again later.")
    await _check_turnstile(body.turnstile_token, request)
    email = body.email.lower()
    if db.get_user_by_username(body.username) or db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="That username or email is already registered.")
    verified = not settings.require_email_verification
    user_id = db.create_user(body.username, hash_password(body.password), role="user", email=email, email_verified=verified)
    if not verified:
        await _send_verification(user_id, email)
    db.log_audit_event(user_id, "account.signup", "account", user_id, {})
    return {"message": "Account created." + ("" if verified else " Check your email to confirm your address."), "verification_required": not verified}


@router.post("/auth/verify-email")
async def verify_email(body: TokenRequest):
    user_id = db.consume_auth_token(body.token, "verify_email")
    if user_id is None:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired. Request a new one.")
    db.set_email_verified(user_id)
    return {"message": "Email confirmed. You can sign in."}


@router.post("/auth/resend-verification")
async def resend_verification(body: EmailRequest, request: Request):
    """Always answers the same way so it cannot be used to discover which emails have accounts."""
    if not mail_rate_limiter.allow(_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
    await _check_turnstile(body.turnstile_token, request)
    user = db.get_user_by_email(body.email.lower())
    if user and not user.get("email_verified", 1):
        await _send_verification(user["id"], user["email"])
    return {"message": "If that address has an unconfirmed account, a new link is on its way."}


@router.post("/auth/password-reset/request")
async def password_reset_request(body: EmailRequest, request: Request):
    if not mail_rate_limiter.allow(_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
    await _check_turnstile(body.turnstile_token, request)
    user = db.get_user_by_email(body.email.lower())
    if user:
        token = db.create_auth_token(user["id"], "reset_password", RESET_TTL_MIN)
        await run_in_threadpool(mailer.send, user["email"], "Reset your Veritas password",  # SMTP blocks: keep it off the event loop
                    f"Reset your password (valid 1 hour):\n{settings.public_url}/reset-password?token={token}\n\nIf you did not ask for this, ignore it.")
    return {"message": "If that address has an account, a reset link is on its way."}


@router.post("/auth/password-reset/confirm")
async def password_reset_confirm(body: ResetConfirm):
    user_id = db.consume_auth_token(body.token, "reset_password")
    if user_id is None:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired. Request a new one.")
    db.set_password_hash(user_id, hash_password(body.new_password))
    db.set_email_verified(user_id)      # receiving the reset mail proves the address
    db.log_audit_event(user_id, "account.password_reset", "account", user_id, {})
    return {"message": "Password changed. You can sign in."}


@router.get("/auth/me")
async def me(user: dict = Depends(require_user)):
    return {"user": user}
