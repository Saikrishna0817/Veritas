"""Authentication and input-security helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import hmac

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt import PyJWTError as JWTError
import bcrypt

from app.core.config import settings


def is_safe_filename(name: str) -> bool:
    # Basic guard: avoid path traversal
    return not any(x in name for x in ("..", "\\", "/"))


bearer_scheme = HTTPBearer(auto_error=False)


def _auth_configured() -> bool:
    return bool(settings.jwt_secret and settings.admin_username and settings.admin_password)


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = password_hash.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False


# Verified against when the username is unknown so response time doesn't reveal
# which usernames exist.
_DUMMY_HASH = bcrypt.hashpw(b"veritas-dummy", bcrypt.gensalt()).decode("utf-8")


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    """Authenticate against the SQLite users table (the env admin is seeded there at startup)."""
    if not _auth_configured():
        return None

    from app.models import database as db

    stored = db.get_user_by_username(username)
    if verify_password(password, stored["password_hash"] if stored else _DUMMY_HASH) and stored:
        if not stored.get("email_verified", 1):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Verify your email address first. Check your inbox, or request a new link.")
        return {"id": stored["id"], "name": stored["username"], "role": stored["role"]}
    return None


def create_access_token(user: dict[str, str]) -> str:
    if not _auth_configured():
        raise RuntimeError("Authentication is not configured")
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    return jwt.encode({"sub": user["id"], "name": user["name"], "role": user["role"], "exp": expires_at}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token") from exc
    if not all(claims.get(key) for key in ("sub", "name", "role")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token claims")
    return {"id": claims["sub"], "name": claims["name"], "role": claims["role"]}


def token_expiry(token: str) -> float:
    """Unix expiry time of an already-verified token."""
    return float(jwt.decode(token, options={"verify_signature": False})["exp"])


SESSION_COOKIE = "veritas_session"      # httpOnly: the JWT, never readable by page scripts
CSRF_COOKIE = "veritas_csrf"            # readable by the page, echoed back in X-CSRF-Token
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def set_session_cookies(response, token: str) -> str:
    """Set the httpOnly session cookie and the CSRF cookie (double-submit). Returns the CSRF token."""
    import secrets
    csrf = secrets.token_urlsafe(24)
    max_age = settings.jwt_expiry_minutes * 60
    common = {"max_age": max_age, "secure": settings.cookie_secure, "samesite": "strict", "path": "/"}
    response.set_cookie(SESSION_COOKIE, token, httponly=True, **common)
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, **common)
    return csrf


def clear_session_cookies(response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def require_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict[str, Any]:
    """Authenticate by API key (``Bearer vk_...``), bearer JWT, or the session cookie.

    Cookie sessions are ambient credentials, so every state-changing request must also carry an X-CSRF-Token
    header equal to the CSRF cookie. Bearer tokens and API keys are not sent automatically by browsers.
    """
    if credentials is not None and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
        if token.startswith("vk_"):
            from app.models import database as db
            row = db.get_user_by_api_key(token)
            if row is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key", headers={"WWW-Authenticate": "Bearer"})
            return {"id": row["id"], "name": row["username"], "role": row["role"], "via": "api_key"}
        return decode_access_token(token)
    cookie = request.cookies.get(SESSION_COOKIE)
    if cookie:
        if request.method not in SAFE_METHODS:
            sent, expected = request.headers.get("x-csrf-token", ""), request.cookies.get(CSRF_COOKIE, "")
            if not expected or not hmac.compare_digest(sent, expected):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing or invalid CSRF token")
        return decode_access_token(cookie)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})


def require_admin(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
    """Dependency that requires role='admin' in addition to a valid token.

    Use on endpoints that perform destructive or privileged actions (quarantine,
    user management, system configuration).
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation.",
        )
    return user
