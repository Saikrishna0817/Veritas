"""JWT authentication routes."""

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.security import authenticate_user, create_access_token, require_user
from app.core.rate_limit import SlidingWindowRateLimiter

router = APIRouter()
login_rate_limiter = SlidingWindowRateLimiter(limit=5, window_seconds=60)


from typing import Optional

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)
    required_role: Optional[str] = Field(default=None, pattern="^(user|analyst|admin)$")


@router.post("/auth/token")
async def token(credentials: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not login_rate_limiter.allow(client_ip):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts. Try again in one minute.")
    user = authenticate_user(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})

    # Flow-specific role enforcement
    if credentials.required_role == "admin" and user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Account does not have administrator privileges.",
        )
    if credentials.required_role in ("user", "analyst") and user.get("role") == "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Administrator account detected. Please use the Administrator Login portal at /admin/login.",
        )

    return {"access_token": create_access_token(user), "token_type": "bearer", "user": user}



@router.get("/auth/me")
async def me(user: dict = Depends(require_user)):
    return {"user": user}
