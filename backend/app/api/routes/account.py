"""Account-level data controls for the signed-in user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app.api import dependencies as deps
from app.core.security import clear_session_cookies, require_user, verify_password

router = APIRouter()


@router.get("/account/usage")
async def my_usage(user: dict = Depends(require_user)):
    """Today's usage against the daily quotas."""
    from app.core.config import settings
    return {"used": deps.db.get_usage(user["id"]),
            "limits": {"analyses": settings.quota_analyses_per_day, "rows": settings.quota_rows_per_day,
                       "scan_mb": settings.quota_scan_mb_per_day},
            "retention_days": settings.retention_days}


def _forget_cached(user_id: str) -> None:
    for cache in (deps.upload_result_cache, deps.demo_result_cache):
        with cache._lock:
            for key in [k for k in cache._data if k == deps.latest_key(user_id) or k.endswith(f":{user_id}")]:
                del cache._data[key]
            for key in [k for k, v in cache._data.items() if isinstance(v, dict) and v.get("owner_id") == user_id]:
                del cache._data[key]


@router.delete("/account/data")
async def delete_my_data(user: dict = Depends(require_user)):
    """Erase every result, scan, defense action, review case and usage counter stored for the caller. The account stays."""
    counts = deps.db.delete_owner_data(user["id"])
    _forget_cached(user["id"])
    deps.db.log_audit_event(user["id"], "account.data_deleted", "account", user["id"], counts)
    return {"deleted": counts}


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


@router.post("/account/delete")
async def delete_my_account(body: DeleteAccountRequest, response: Response, user: dict = Depends(require_user)):
    """Permanently delete the account and everything stored for it, including API keys and the audit log. Needs the password."""
    stored = deps.db.get_user_by_id(user["id"])
    if not stored or not verify_password(body.password, stored["password_hash"]):
        raise HTTPException(status_code=403, detail="Wrong password.")
    if stored["role"] == "admin":
        raise HTTPException(status_code=403, detail="Admin accounts cannot be deleted here.")
    counts = deps.db.delete_account(user["id"])
    _forget_cached(user["id"])
    clear_session_cookies(response)
    return {"deleted": counts}


class ApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@router.post("/account/api-keys", status_code=201)
async def create_api_key(body: ApiKeyRequest, user: dict = Depends(require_user)):
    """Create a key for CLI / CI use. The key is shown once; only its hash is stored."""
    if user.get("via") == "api_key":
        raise HTTPException(status_code=403, detail="API keys cannot create other API keys.")
    if len([k for k in deps.db.list_api_keys(user["id"]) if not k["revoked_at"]]) >= 10:
        raise HTTPException(status_code=409, detail="At most 10 active API keys. Revoke one first.")
    raw, row = deps.db.create_api_key(user["id"], body.name)
    deps.db.log_audit_event(user["id"], "apikey.created", "api_key", row["id"], {"name": body.name})
    return {"key": raw, **row, "note": "Store this key now; it cannot be shown again."}


@router.get("/account/api-keys")
async def list_api_keys(user: dict = Depends(require_user)):
    return {"keys": deps.db.list_api_keys(user["id"])}


@router.delete("/account/api-keys/{key_id}")
async def revoke_api_key(key_id: str, user: dict = Depends(require_user)):
    if not deps.db.revoke_api_key(user["id"], key_id):
        raise HTTPException(status_code=404, detail="Key not found.")
    deps.db.log_audit_event(user["id"], "apikey.revoked", "api_key", key_id, {})
    return {"message": "Key revoked."}
