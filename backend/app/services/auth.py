"""Sign-in, session issue and end, sign-up and the mail flows (decision 2: auth first).

Routes in app/api/routes/auth.py call these; none of them touches a Request object, only plain values (the caller's
IP is passed in) and the Response whose cookies are set.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException, Response, status
from starlette.concurrency import run_in_threadpool

from app.core import mailer
from app.core.config import settings
from app.core.observability import LOGIN_FAILURES
from app.core.rate_limit import make_limiter
from app.core.security import (authenticate_user, clear_session_cookies, create_access_token, hash_password,
                               set_session_cookies)
from app.models import database as db

logger = logging.getLogger(__name__)
login_rate_limiter = make_limiter("login", 5, 60)
login_user_rate_limiter = make_limiter("login-user", 10, 900)   # per username, so guessing spread over many IPs is slowed too
signup_rate_limiter = make_limiter("signup", 5, 3600)
mail_rate_limiter = make_limiter("mail", 5, 3600)

VERIFY_TTL_MIN, RESET_TTL_MIN = 24 * 60, 60


# ── Sign-in and sessions ─────────────────────────────────────────────────────

def audit_failed_login(username: str, ip: str) -> None:
    """Failed sign-ins go to the audit log. For an existing account the event is the account's own, so its owner
    sees it under their activity; for an unknown name only the IP is kept (people type passwords into the name field)."""
    known = db.get_user_by_username(username)
    LOGIN_FAILURES.labels("wrong_password" if known else "unknown_account").inc()
    if known:
        db.log_audit_event(known["id"], "auth.login_failed", "account", known["id"], {"ip": ip})
        logger.warning("Failed sign-in for account %s from %s", known["id"], ip)
    else:
        db.log_audit_event("anonymous", "auth.login_failed", "account", None, {"ip": ip, "known_account": False})
        logger.warning("Failed sign-in for an unknown username from %s", ip)


def check_password_login(username: str, password: str, ip: str, required_role: Optional[str]) -> dict:
    """Rate limits, password check and the portal's role rule. Returns the user, or raises the HTTP error."""
    if not login_rate_limiter.allow(ip):
        LOGIN_FAILURES.labels("rate_limited").inc()
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts. Try again in one minute.")
    if not login_user_rate_limiter.allow(username.lower()):
        LOGIN_FAILURES.labels("rate_limited").inc()
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts for this account. Try again in 15 minutes.")
    user = authenticate_user(username, password)
    if user is None:
        audit_failed_login(username, ip)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    if required_role == "admin" and user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Account does not have administrator privileges.")
    if required_role in ("user", "analyst") and user.get("role") == "admin":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Administrator account detected. Please use the Administrator Login portal at /admin/login.")
    return user


def issue_session(response: Response, user: dict) -> dict:
    """Sign the access token, set the browser cookies, and return the sign-in body. API clients can keep using the bearer token."""
    access = create_access_token(user)
    csrf = set_session_cookies(response, access)
    return {"access_token": access, "token_type": "bearer", "user": user, "csrf_token": csrf}


def sign_out(response: Response) -> None:
    """This browser only: the token itself stays valid until it expires; sign_out_everywhere ends it."""
    clear_session_cookies(response)


def sign_out_everywhere(response: Response, user_id: str) -> None:
    db.end_all_sessions(user_id)
    clear_session_cookies(response)
    db.log_audit_event(user_id, "auth.signed_out_everywhere", "account", user_id, {})


# ── Mail ─────────────────────────────────────────────────────────────────────

async def send_mail(to: str, subject: str, body: str) -> bool:
    """Send one mail off the event loop (SMTP blocks). A mail server that is down or refuses must not turn the request
    into a 500: the failure is logged and the caller decides what to tell the person."""
    try:
        await run_in_threadpool(mailer.send, to, subject, body)
        return True
    except Exception:
        logger.exception("Sending mail failed (subject=%r): check VERITAS_SMTP_* and the mail server", subject)
        return False


async def send_verification(user_id: str, email: str) -> bool:
    token = db.create_auth_token(user_id, "verify_email", VERIFY_TTL_MIN)
    return await send_mail(email, "Confirm your Veritas email",
                           f"Welcome to Veritas.\n\nConfirm your email address (valid 24 hours):\n{settings.public_url}/verify-email?token={token}\n\n"
                           "If you did not sign up, ignore this message.")


# ── Sign-up, verification, reset ─────────────────────────────────────────────

def check_signup_allowed(ip: str) -> None:
    if not settings.signup_enabled:
        raise HTTPException(status_code=403, detail="Sign-up is closed.")
    if not signup_rate_limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Too many sign-ups from this address. Try again later.")


async def register(username: str, email: str, password: str) -> dict:
    email = email.lower()
    if db.username_taken(username) or db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="That username or email is already registered.")
    verified = not settings.require_email_verification
    user_id = db.create_user(username, hash_password(password), role="user", email=email, email_verified=verified)
    sent = verified or await send_verification(user_id, email)
    db.log_audit_event(user_id, "account.signup", "account", user_id, {} if verified else {"mail_sent": sent})
    if not sent:      # the account exists; a new link can be requested once mail works again
        return {"message": "Account created, but the confirmation email could not be sent right now. "
                           "In a few minutes, use “Forgot your password?” on the sign-in page: that link also confirms your address.",
                "verification_required": True, "mail_sent": False}
    return {"message": "Account created." + ("" if verified else " Check your email to confirm your address."), "verification_required": not verified}


def verify_email(token: str) -> None:
    user_id = db.consume_auth_token(token, "verify_email")
    if user_id is None:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired. Request a new one.")
    db.set_email_verified(user_id)


def check_mail_allowed(ip: str) -> None:
    if not mail_rate_limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")


async def resend_verification(email: str) -> None:
    """Silent when there is nothing to send, so it cannot be used to discover which emails have accounts."""
    user = db.get_user_by_email(email.lower())
    if user and not user.get("email_verified", 1):
        await send_verification(user["id"], user["email"])


async def request_password_reset(email: str) -> None:
    user = db.get_user_by_email(email.lower())
    if user:
        token = db.create_auth_token(user["id"], "reset_password", RESET_TTL_MIN)
        # A failed send is logged and the answer stays the same, so the reply never reveals whether the address has an account.
        await send_mail(user["email"], "Reset your Veritas password",
                        f"Reset your password (valid 1 hour):\n{settings.public_url}/reset-password?token={token}\n\nIf you did not ask for this, ignore it.")


def confirm_password_reset(token: str, new_password: str) -> None:
    user_id = db.consume_auth_token(token, "reset_password")
    if user_id is None:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired. Request a new one.")
    db.set_password_hash(user_id, hash_password(new_password))
    db.set_email_verified(user_id)      # receiving the reset mail proves the address
    db.log_audit_event(user_id, "account.password_reset", "account", user_id, {})
