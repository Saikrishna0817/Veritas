"""Per-user daily usage quotas (UTC day). Admins are exempt; a limit of 0 disables it."""
from fastapi import HTTPException

from app.core.config import settings
from app.models import database as db


def charge(user: dict, kind: str, amount: int, limit: int) -> None:
    """Count ``amount`` against today's ``kind`` usage or raise 429 with what is left."""
    if user.get("role") == "admin" or not limit:
        return
    allowed, used = db.charge_usage(user["id"], kind, amount, limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Daily {kind.replace('_', ' ')} quota reached ({used} of {limit} used today, UTC). It resets at 00:00 UTC.",
        )


def charge_analysis(user: dict, rows: int = 0) -> None:
    charge(user, "analyses", 1, settings.quota_analyses_per_day)
    if rows:
        charge(user, "rows", rows, settings.quota_rows_per_day)


def charge_scan(user: dict, size_bytes: int) -> None:
    charge(user, "analyses", 1, settings.quota_analyses_per_day)
    charge(user, "scan_mb", max(1, size_bytes // (1024 * 1024)), settings.quota_scan_mb_per_day)
