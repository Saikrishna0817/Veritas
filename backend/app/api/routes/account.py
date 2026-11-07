"""Account-level data controls for the signed-in user."""
from __future__ import annotations

from fastapi import APIRouter, Depends

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
