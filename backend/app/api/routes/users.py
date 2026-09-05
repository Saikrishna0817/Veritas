"""User management endpoints for Administrators."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import require_admin, hash_password
from app.models import database as db

router = APIRouter()


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=4, max_length=128)
    role: str = Field(default="user", pattern="^(user|analyst|admin)$")



@router.get("/users", response_model=List[Dict[str, Any]])
async def get_users(admin: dict = Depends(require_admin)):
    """List all registered users in the platform."""
    return db.list_users(limit=200)


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user_account(payload: CreateUserRequest, admin: dict = Depends(require_admin)):
    """Create a new analyst or admin user."""
    existing = db.get_user_by_username(payload.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with username '{payload.username}' already exists.",
        )
    pwd_hash = hash_password(payload.password)
    user_id = db.create_user(payload.username, pwd_hash, role=payload.role)
    return {
        "message": f"User '{payload.username}' created successfully.",
        "user": {
            "id": user_id,
            "username": payload.username,
            "role": payload.role,
        },
    }


@router.delete("/users/{user_id}")
async def revoke_user_account(user_id: str, admin: dict = Depends(require_admin)):
    """Revoke and delete a user account."""
    if user_id == admin.get("id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own active administrator account.",
        )
    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return {"message": "User account successfully revoked."}
