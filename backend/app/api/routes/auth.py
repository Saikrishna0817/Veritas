"""JWT authentication routes."""

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import authenticate_user, create_access_token, require_user

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


@router.post("/auth/token")
async def token(credentials: LoginRequest):
    user = authenticate_user(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": create_access_token(user), "token_type": "bearer", "user": user}


@router.get("/auth/me")
async def me(user: dict = Depends(require_user)):
    return {"user": user}
