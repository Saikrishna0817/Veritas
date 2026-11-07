"""Account-level data controls for the signed-in user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api import dependencies as deps
from app.core.security import require_user

router = APIRouter()


@router.get("/account/usage")
async def my_usage(user: dict = Depends(require_user)):
    """Today's usage against the daily quotas."""
    from app.core.config import settings
    return {"used": deps.db.get_usage(user["id"]),
            "limits": {"analyses": settings.quota_analyses_per_day, "rows": settings.quota_rows_per_day,
                       "scan_mb": settings.quota_scan_mb_per_day},
            "retention_days": settings.retention_days}


@router.delete("/account/data")
async def delete_my_data(user: dict = Depends(require_user)):
    """Erase every result, scan, defense action and review case stored for the caller, and their cached latest results."""
    counts = deps.db.delete_owner_data(user["id"])
    for cache in (deps.upload_result_cache, deps.demo_result_cache):
        with cache._lock:
            for key in [k for k in cache._data if k == deps.latest_key(user["id"]) or k.endswith(f":{user['id']}")]:
                del cache._data[key]
            for key in [k for k, v in cache._data.items() if isinstance(v, dict) and v.get("owner_id") == user["id"]]:
                del cache._data[key]
    deps.db.log_audit_event(user["id"], "account.data_deleted", "account", user["id"], counts)
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
