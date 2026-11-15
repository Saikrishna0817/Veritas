"""Per-user daily usage quotas (UTC day). Admins are exempt; a limit of 0 disables it."""
from datetime import datetime, timezone

from fastapi import HTTPException

from app.core.config import settings
from app.models import database as db


def charge(user: dict, kind: str, amount: int, limit: int) -> dict:
    """Count ``amount`` against today's ``kind`` usage or raise 429 with what is left. Returns what was charged
    ({} for admins and unlimited kinds), to hand to ``refund`` if the work then fails."""
    if user.get("role") == "admin" or not limit:
        return {}
    allowed, used = db.charge_usage(user["id"], kind, amount, limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Daily {kind.replace('_', ' ')} quota reached ({used} of {limit} used today, UTC). It resets at 00:00 UTC.",
        )
    return {"day": datetime.now(timezone.utc).strftime("%Y-%m-%d"), kind: amount}


def refund(owner_id: str, charged: dict | None) -> None:
    """Undo a charge (a failed analysis should not use up the day's quota)."""
    for kind, amount in (charged or {}).items():
        if kind != "day":
            db.refund_usage(owner_id, charged["day"], kind, amount)


def _charge_both(user: dict, first: tuple, second: tuple) -> dict:
    charged = charge(user, *first)
    try:
        return {**charged, **charge(user, *second)}
    except HTTPException:
        refund(user["id"], charged)      # the second limit failed: the first must not stay charged
        raise


def charge_analysis(user: dict, rows: int = 0) -> dict:
    if not rows:
        return charge(user, "analyses", 1, settings.quota_analyses_per_day)
    return _charge_both(user, ("analyses", 1, settings.quota_analyses_per_day), ("rows", rows, settings.quota_rows_per_day))


def charge_scan(user: dict, size_bytes: int) -> dict:
    return _charge_both(user, ("analyses", 1, settings.quota_analyses_per_day),
                        ("scan_mb", max(1, size_bytes // (1024 * 1024)), settings.quota_scan_mb_per_day))
