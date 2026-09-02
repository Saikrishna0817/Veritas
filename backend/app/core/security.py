"""Authentication and input-security helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hmac import compare_digest
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings


def is_safe_filename(name: str) -> bool:
    # Basic guard: avoid path traversal
    return not any(x in name for x in ("..", "\\", "/"))


bearer_scheme = HTTPBearer(auto_error=False)


def _auth_configured() -> bool:
    return bool(settings.jwt_secret and settings.admin_username and settings.admin_password)


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    """Authenticate the configured bootstrap administrator without logging secrets."""
    if not _auth_configured():
        return None
    if not (compare_digest(username, settings.admin_username) and compare_digest(password, settings.admin_password)):
        return None
    return {"id": username, "name": username, "role": "admin"}


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


def require_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})
    return decode_access_token(credentials.credentials)
