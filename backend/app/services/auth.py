"""Sign-in, session issue and end, sign-up and the mail flows (decision 2: auth first).

Routes in app/api/routes/auth.py call these; none of them touches a Request object, only plain values (the caller's
IP is passed in) and the Response whose cookies are set.
"""
from __future__ import annotations

import logging
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from fastapi import HTTPException, Response, status
from starlette.concurrency import run_in_threadpool

from app.core import mailer, totp
from app.core.config import settings
from app.core.observability import LOGIN_FAILURES
from app.core.rate_limit import make_limiter
from app.core.security import (authenticate_user, clear_session_cookies, create_access_token, hash_password,
                               set_refresh_cookie, set_session_cookies)
from app.models import database as db

logger = logging.getLogger(__name__)
login_rate_limiter = make_limiter("login", 5, 60)
login_user_rate_limiter = make_limiter("login-user", 10, 900)   # per username, so guessing spread over many IPs is slowed too
signup_rate_limiter = make_limiter("signup", 5, 3600)
mail_rate_limiter = make_limiter("mail", 5, 3600)
twofa_ip_limiter = make_limiter("2fa-ip", 10, 300)
twofa_user_limiter = make_limiter("2fa-user", 5, 300)     # wrong codes per account: 6 digits must not be guessable
CHALLENGE_TTL_MIN = 5
RECOVERY_CODES = 8

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


def issue_session(response: Response, user: dict, family_id: Optional[str] = None) -> dict:
    """Sign the access token, set the browser cookies (session, CSRF, refresh), and return the sign-in body.
    API clients can keep using the bearer token. A new sign-in starts a new refresh family; a refresh continues its own."""
    access = create_access_token(user)
    csrf = set_session_cookies(response, access)
    set_refresh_cookie(response, db.create_refresh_token(user["id"], family_id))
    return {"access_token": access, "token_type": "bearer", "user": user, "csrf_token": csrf}


def sign_in(response: Response, user: dict) -> dict:
    """Password was right. Accounts with 2FA get a short-lived challenge instead of a session."""
    stored = db.get_user_by_id(user["id"]) or {}
    if stored.get("totp_enabled_at"):
        return {"two_factor_required": True, "challenge": _make_challenge(user["id"])}
    return issue_session(response, user)


def refresh_session(response: Response, raw_refresh: Optional[str], ip: str) -> dict:
    """Spend the refresh token and issue a new access session plus the next refresh token (rotation).
    Presenting a token that was already spent ends its whole family and is audited."""
    def refuse():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session ended: sign in again")
    if not raw_refresh:
        refuse()
    outcome, row = db.use_refresh_token(raw_refresh)
    if outcome == "reuse":
        db.log_audit_event(row["user_id"], "auth.refresh_reuse_detected", "account", row["user_id"],
                           {"ip": ip, "family_id": row["family_id"]})
        logger.warning("Refresh token reuse for account %s from %s: family %s revoked", row["user_id"], ip, row["family_id"])
    if outcome != "ok":
        refuse()
    stored = db.get_user_by_id(row["user_id"])
    if stored is None or stored.get("disabled_at"):
        refuse()
    user = {"id": stored["id"], "name": stored["username"], "role": stored["role"]}
    return issue_session(response, user, family_id=row["family_id"])


def sign_out(response: Response, raw_refresh: Optional[str] = None) -> None:
    """This browser only: the access token stays valid until it expires, but its refresh family is ended;
    sign_out_everywhere ends every token."""
    if raw_refresh:
        db.revoke_refresh_family(raw_refresh)
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


# ── Two-factor (TOTP) ────────────────────────────────────────────────────────

def _make_challenge(user_id: str) -> str:
    """Signed, five minutes, and carries neither name nor role, so decode_access_token refuses it as a session."""
    exp = datetime.now(timezone.utc) + timedelta(minutes=CHALLENGE_TTL_MIN)
    return jwt.encode({"sub": user_id, "typ": "2fa", "exp": exp}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _read_challenge(challenge: str) -> str:
    try:
        claims = jwt.decode(challenge, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="This sign-in step expired. Enter your password again.") from exc
    if claims.get("typ") != "2fa" or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="This sign-in step expired. Enter your password again.")
    return claims["sub"]


def _hash_recovery(code: str) -> str:
    return hashlib.sha256(code.strip().lower().replace(" ", "").encode()).hexdigest()


def _check_code(stored: dict, code: str) -> bool:
    """A current authenticator code (each 30 s step works once) or an unused recovery code."""
    step = totp.verify(totp.decrypt_secret(stored["totp_secret"]), code, stored.get("totp_last_step") or 0)
    if step is not None and db.claim_totp_step(stored["id"], step):
        return True
    return db.use_recovery_code(stored["id"], _hash_recovery(code))


def _attempt(stored: dict, code: str, ip: str, event: str) -> None:
    """Rate-limited code check for verify and disable; wrong codes are audited."""
    if not twofa_ip_limiter.allow(ip) or not twofa_user_limiter.allow(stored["id"]):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again in a few minutes.")
    if not _check_code(stored, code):
        db.log_audit_event(stored["id"], event, "account", stored["id"], {"ip": ip})
        raise HTTPException(status_code=401, detail="That code is not right.")


def complete_two_factor(response: Response, challenge: str, code: str, ip: str) -> dict:
    user_id = _read_challenge(challenge)
    stored = db.get_user_by_id(user_id)
    if stored is None or stored.get("disabled_at") or not stored.get("totp_enabled_at"):
        raise HTTPException(status_code=401, detail="This sign-in step expired. Enter your password again.")
    _attempt(stored, code, ip, "auth.2fa_failed")
    return issue_session(response, {"id": stored["id"], "name": stored["username"], "role": stored["role"]})


def two_factor_setup(user: dict) -> dict:
    stored = db.get_user_by_id(user["id"])
    if stored.get("totp_enabled_at"):
        raise HTTPException(status_code=409, detail="Two-factor sign-in is already on.")
    secret = totp.new_secret()
    db.set_totp_pending(user["id"], totp.encrypt_secret(secret))
    return {"secret": secret, "otpauth_uri": totp.otpauth_uri(secret, stored["username"])}   # the only time the secret leaves the server


def two_factor_enable(user: dict, code: str, ip: str) -> dict:
    stored = db.get_user_by_id(user["id"])
    if stored.get("totp_enabled_at") or not stored.get("totp_secret"):
        raise HTTPException(status_code=409, detail="Start setup first." if not stored.get("totp_secret") else "Two-factor sign-in is already on.")
    if not twofa_user_limiter.allow(user["id"]):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again in a few minutes.")
    step = totp.verify(totp.decrypt_secret(stored["totp_secret"]), code)
    if step is None:
        raise HTTPException(status_code=400, detail="That code is not right. Check the clock on your phone and try the next one.")
    codes = [f"{secrets.token_hex(3)}-{secrets.token_hex(3)}" for _ in range(RECOVERY_CODES)]
    db.enable_totp(user["id"], step, [_hash_recovery(c) for c in codes])
    db.log_audit_event(user["id"], "auth.2fa_enabled", "account", user["id"], {"ip": ip})
    return {"recovery_codes": codes}           # shown once; only their hashes are kept


def two_factor_disable(user: dict, code: str, ip: str) -> None:
    stored = db.get_user_by_id(user["id"])
    if not stored.get("totp_enabled_at"):
        raise HTTPException(status_code=409, detail="Two-factor sign-in is not on.")
    _attempt(stored, code, ip, "auth.2fa_failed")
    db.disable_totp(user["id"])
    db.log_audit_event(user["id"], "auth.2fa_disabled", "account", user["id"], {"ip": ip})
