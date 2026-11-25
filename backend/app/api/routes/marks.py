"""Trial feedback: a person marks flagged items of their own results as a real problem or fine."""
from __future__ import annotations

import csv
import io
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api import dependencies as deps
from app.core.rate_limit import make_limiter
from app.core.security import require_admin, require_user

router = APIRouter()
marks_rate_limiter = make_limiter("marks", 120, 60)

COLUMNS = ["username", "result_id", "source", "filename", "result_created_at", "item", "rank", "verdict", "updated_at"]


class MarkRequest(BaseModel):
    item: str = Field(min_length=1, max_length=1000)
    rank: Optional[int] = Field(default=None, ge=0)
    verdict: Optional[Literal["bad", "fine"]] = None      # null clears the mark


def _own_result(result_id: str, user: dict) -> None:
    """404 unless the caller owns the result (admins included: marks are personal)."""
    if not deps.db.get_result(result_id, user["id"]):
        raise HTTPException(status_code=404, detail="Result not found.")


@router.put("/results/{result_id}/marks")
async def set_mark(result_id: str, body: MarkRequest, request: Request, user: dict = Depends(require_user)):
    _own_result(result_id, user)
    ip = request.client.host if request.client else "unknown"
    if not marks_rate_limiter.allow(f"marks:{user['id']}:{ip}"):
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    if body.verdict is None:
        deps.db.delete_mark(user["id"], result_id, body.item)
    else:
        deps.db.upsert_mark(user["id"], result_id, body.item, body.rank, body.verdict)
    return {"item": body.item, "verdict": body.verdict}


@router.get("/results/{result_id}/marks")
async def get_marks(result_id: str, user: dict = Depends(require_user)):
    _own_result(result_id, user)
    return {"marks": deps.db.list_marks(user["id"], result_id)}


@router.get("/marks/export")
async def export_marks(admin: dict = Depends(require_admin)):
    """All trial marks as CSV (admin only)."""
    rows = deps.db.export_marks()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(COLUMNS)
    for r in rows:
        # a leading = + - @ would run as a formula when the CSV is opened in a spreadsheet
        w.writerow(["'" + v if isinstance(v, str) and v[:1] in "=+-@" else v for v in (r[c] for c in COLUMNS)])
    deps.db.log_audit_event(admin["id"], "marks.exported", "marks", None, {"rows": len(rows)})
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="trial-marks.csv"'})
