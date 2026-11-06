"""Background analysis job status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_user
from app.jobs.queue import job_store

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, user: dict = Depends(require_user)):
    """Poll the status/result of an async analysis job (owner or admin only)."""
    job = job_store.get(job_id)
    # 404 (not 403) for other users' jobs so job ids can't be probed.
    if not job or (job["owner_id"] != user["id"] and user.get("role") != "admin"):
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
