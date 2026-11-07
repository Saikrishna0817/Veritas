"""Direct-to-storage uploads: the browser or CLI PUTs the file straight to S3/R2 and the API never carries the bytes.

1. ``POST /uploads/presign``  -> a URL good for one PUT of exactly the declared size (checked here against the limits)
2. the client PUTs the file to that URL
3. ``POST /uploads/complete`` -> the API checks the stored object, charges the quota and queues the job (202 + job_id)

Needs ``VERITAS_STORAGE_URL=s3://...``; with local storage ``/uploads/presign`` answers 501 and clients use ``/analyze/upload?background=true``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api import dependencies as deps
from app.api.routes.models import MAX_SCAN_SIZE, MODEL_EXTENSIONS
from app.api.routes.upload import MAX_CSV_SIZE
from app.core import quota
from app.core.security import is_safe_filename, require_user
from app.core.storage import new_key, storage
from app.jobs import tasks  # noqa: F401
from app.jobs.queue import job_store

router = APIRouter()
PRESIGN_TTL = 900
MAX_OPEN_UPLOADS = 10           # unfinished uploads per user per hour; abandoned blobs also need a bucket lifecycle rule
BYTES_PER_ROW = 64              # ponytail: rows are charged from size (the API never reads the file); exact count if quotas get strict


class PresignRequest(BaseModel):
    role: Literal["csv", "csv_reference", "model"]
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)


class CompleteRequest(BaseModel):
    upload_id: str
    reference_upload_id: str | None = None
    split_column: str | None = None
    split_cutoff: str | None = None


@router.post("/uploads/presign")
async def presign_upload(body: PresignRequest, request: Request, user: dict = Depends(require_user)):
    store = storage()
    if getattr(store, "presign_put", None) is None:
        raise HTTPException(status_code=501, detail="Direct upload needs S3-compatible storage. Use /analyze/upload?background=true.")
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"presign:{user['id']}:{client_ip}"):
        raise HTTPException(status_code=429, detail="Too many analysis requests. Try again in one minute.")

    model = body.role == "model"
    extensions, limit = (MODEL_EXTENSIONS, MAX_SCAN_SIZE) if model else ((".csv",), MAX_CSV_SIZE)
    if not is_safe_filename(body.filename) or not body.filename.lower().endswith(extensions):
        raise HTTPException(status_code=400, detail=f"Supported files: {', '.join(extensions)}.")
    if body.size > limit:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {limit // 1024 // 1024} MB.")
    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    if deps.db.count_open_uploads(user["id"], since) >= MAX_OPEN_UPLOADS:
        raise HTTPException(status_code=429, detail="Too many unfinished uploads. Finish or wait an hour.")

    key, upload_id = new_key(user["id"], "model" if model else "upload"), uuid.uuid4().hex
    deps.db.create_pending_upload(upload_id, user["id"], body.role, body.filename, body.size, key)
    return {"upload_id": upload_id, "url": store.presign_put(key, body.size, PRESIGN_TTL), "method": "PUT",
            "headers": {"Content-Length": str(body.size)}, "expires_in": PRESIGN_TTL}


def _take(upload_id: str, user: dict, roles: tuple[str, ...]) -> dict:
    """The caller's unused upload of a suitable role, verified against what actually reached storage."""
    row = deps.db.claim_pending_upload(upload_id, user["id"])
    if not row or row["role"] not in roles:
        raise HTTPException(status_code=404, detail="Unknown or already used upload_id.")
    if storage().size(row["storage_key"]) != row["size"]:
        storage().delete(row["storage_key"])
        raise HTTPException(status_code=400, detail="The uploaded object is missing or its size differs from the declared size.")
    return row


@router.post("/uploads/complete", status_code=202)
async def complete_upload(body: CompleteRequest, user: dict = Depends(require_user)):
    main = _take(body.upload_id, user, ("csv", "model"))
    store, rows = storage(), [main]
    try:
        if main["role"] == "model":
            if body.reference_upload_id:
                raise HTTPException(status_code=400, detail="A model scan has no reference file.")
            quota.charge_scan(user, main["size"])
            job_id = job_store.submit("model_scan", {"key": main["storage_key"], "filename": main["filename"]}, owner_id=user["id"])
        else:
            payload = {"key": main["storage_key"], "filename": main["filename"]}
            if body.reference_upload_id:
                ref = _take(body.reference_upload_id, user, ("csv_reference",))
                rows.append(ref)
                payload["reference_key"] = ref["storage_key"]
            if body.split_column or body.split_cutoff:
                payload.update(split_column=body.split_column or None, split_cutoff=body.split_cutoff or None)
            quota.charge_analysis(user, rows=max(1, main["size"] // BYTES_PER_ROW))
            job_id = job_store.submit("csv_upload", payload, owner_id=user["id"])
    except Exception:
        for row in rows:                     # the job never started, so nothing else will delete these
            store.delete(row["storage_key"])
        raise
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})
