"""CSV upload analysis routes.

All pipeline orchestration is delegated to analysis_helpers.run_csv_analysis()
(H2 — no more duplicated code in this file).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps
from app.api.analysis_helpers import run_csv_analysis
from app.core.security import is_safe_filename, require_user

logger = logging.getLogger(__name__)

router = APIRouter()
MAX_CSV_SIZE = 200 * 1024 * 1024


async def read_upload_limited(file: UploadFile, limit: int) -> bytes:
    """Read an upload in bounded chunks, rejecting files that exceed the limit."""
    parts: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {limit // 1024 // 1024} MB.",
            )
        parts.append(chunk)
    return b"".join(parts)


@router.post("/analyze/upload")
async def analyze_uploaded_csv(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    """Upload a CSV file and run the full 5-layer poisoning detection pipeline.

    - Auto-detects label column, feature columns, data types
    - Runs supervised detection if label column found, unsupervised otherwise
    - Self-contained: uses internal 70/30 split (no external baseline needed)
    - Max 200,000 rows, 200 MB file size
    """
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"upload:{user['id']}:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many analysis requests. Try again in one minute.",
        )

    if (
        not file.filename
        or not is_safe_filename(file.filename)
        or not file.filename.lower().endswith(".csv")
    ):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await read_upload_limited(file, MAX_CSV_SIZE)
    filename = file.filename

    try:
        full_result, dataset_id = await run_csv_analysis(content, filename, source="upload")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        rid = getattr(request.state, "request_id", "-")
        logger.exception("CSV analysis failed [rid=%s]", rid)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed. Check server logs (request ID: {rid}).",
        )

    deps.upload_result_cache[dataset_id] = full_result
    deps.upload_result_cache["latest"] = full_result

    background_tasks.add_task(deps.db.save_result, full_result, "upload", filename)
    background_tasks.add_task(
        deps.db.log_audit_event,
        user["id"],
        "analysis.uploaded",
        "dataset",
        dataset_id,
        {
            "filename": filename,
            "n_samples": full_result.get("n_samples"),
            "verdict": full_result.get("verdict"),
        },
    )

    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)


@router.get("/analyze/upload/latest")
async def get_latest_upload_result():
    """Get the latest uploaded dataset analysis result (memory cache → DB fallback)."""
    cached = deps.upload_result_cache.get("latest")
    if cached:
        return cached
    result = deps.db.get_latest(source="upload")
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No upload analysis yet. POST to /analyze/upload first.",
        )
    return result


@router.get("/analyze/upload/{dataset_id}")
async def get_upload_result(dataset_id: str):
    """Get analysis result for a specific uploaded dataset (memory → DB fallback)."""
    cached = deps.upload_result_cache.get(dataset_id)
    if cached:
        return cached
    result = deps.db.get_result(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return result
