"""Background analysis job status routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.jobs.queue import job_store

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status/result of an async analysis job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
