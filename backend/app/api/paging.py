"""Cursor paging for list endpoints: ``?before=<created_at of the last row you have>`` returns the next older page.

A cursor beats offset paging here: rows arrive all the time, and an offset would skip or repeat them. The cursor is
normalised to the stored format (UTC ``isoformat()``), so a created_at copied from a response always works.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, Query


def before_cursor(before: Optional[str] = Query(None, max_length=40, description="created_at of the last row of the previous page")) -> Optional[str]:
    if before is None:
        return None
    # an unencoded "+00:00" reaches us as " 00:00" (a "+" in a query string means a space)
    before = re.sub(r" (\d{2}:\d{2})$", r"+\1", before.strip())
    try:
        when = datetime.fromisoformat(before)
    except ValueError:
        raise HTTPException(status_code=422, detail="'before' must be a created_at value from a previous page (ISO 8601).")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat()


def next_cursor(rows: list, limit: int) -> Optional[str]:
    """The cursor for the following page, or None when this page is the last."""
    if not rows or len(rows) < limit:
        return None
    last = str(rows[-1]["created_at"])
    return last[:-6] + "Z" if last.endswith("+00:00") else last     # "Z" needs no URL encoding; "+" does
