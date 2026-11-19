"""User management endpoints for Administrators."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import require_admin, hash_password
from app.models import database as db

router = APIRouter()


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")   # same rule as self sign-up
    password: str = Field(min_length=10, max_length=128)
    role: str = Field(default="user", pattern="^(user|analyst|admin)$")

    @field_validator("password")
    @classmethod
    def _fits_bcrypt(cls, value: str) -> str:
        # bcrypt silently ignores bytes past 72; reject instead of truncating.
        if len(value.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        return value



class RoleRequest(BaseModel):
    role: str = Field(pattern="^(user|analyst|admin)$")


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
    db.log_audit_event(admin["id"], "user.created", "user", user_id, {"username": payload.username, "role": payload.role})
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
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    # Same erasure as self-service deletion: the id is recorded as deleted, so a session token the person still
    # holds is refused at once instead of working until it expires, and none of their rows are left behind.
    db.delete_account(user_id)
    from app.api.routes.account import _forget_cached
    _forget_cached(user_id)
    db.log_audit_event(admin["id"], "user.deleted", "user", user_id, {"username": (target or {}).get("username")})
    return {"message": "User account successfully revoked."}


@router.patch("/users/{user_id}")
async def change_user_role(user_id: str, payload: RoleRequest, admin: dict = Depends(require_admin)):
    """Change another account's role. An admin cannot change their own, so the last admin cannot lock everyone out."""
    if user_id == admin.get("id"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot change the role of the account you are signed in with.")
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if target["role"] != payload.role:
        db.set_user_role(user_id, payload.role)
        db.log_audit_event(admin["id"], "user.role_changed", "user", user_id,
                           {"username": target["username"], "from": target["role"], "to": payload.role})
    return {"message": f"'{target['username']}' is now {payload.role}.",
            "user": {"id": user_id, "username": target["username"], "role": payload.role}}
