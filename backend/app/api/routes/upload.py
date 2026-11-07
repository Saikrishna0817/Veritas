"""CSV upload analysis routes.

All pipeline orchestration is delegated to analysis_helpers.run_csv_analysis()
(H2 — no more duplicated code in this file).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps
from app.api.analysis_helpers import _orchestrate_csv, run_csv_analysis
from app.core.security import is_safe_filename, require_user
from app.jobs.queue import job_store

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
    reference_file: UploadFile | None = File(None, description="Optional trusted reference CSV with the same columns; the main file is then all incoming"),
    split_column: str | None = Form(None, description="Optional column to split on (numbers or dates)"),
    split_cutoff: str | None = Form(None, description="Rows with split_column strictly below this are the trusted reference"),
    background: bool = Query(False, description="Return 202 + job_id immediately; poll GET /jobs/{job_id}"),
    user: dict = Depends(require_user),
):
    """Upload a CSV file and check it against a trusted reference set.

    - Auto-detects label column, feature columns, data types
    - The reference is a separate file (`reference_file`), the rows before a cutoff
      (`split_column` + `split_cutoff`), or by default the first 70% in file order
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
    split: dict = {}
    if reference_file is not None:
        if not reference_file.filename or not is_safe_filename(reference_file.filename) \
                or not reference_file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="The reference file must be a CSV.")
        split["reference_bytes"] = await read_upload_limited(reference_file, MAX_CSV_SIZE)
    if split_column or split_cutoff:
        split.update(split_column=split_column or None, split_cutoff=split_cutoff or None)

    if background:
        def _job() -> dict:
            result, dataset_id = _orchestrate_csv(content, filename, user["id"], split)
            result["source"] = "upload"
            result["owner_id"] = user["id"]
            result = deps.to_serializable(result)
            deps.upload_result_cache[dataset_id] = result
            deps.upload_result_cache[deps.latest_key(user["id"])] = result
            deps.db.save_result(result, "upload", filename, owner_id=user["id"])
            deps.db.log_audit_event(
                user["id"], "analysis.uploaded", "dataset", dataset_id,
                {"filename": filename, "n_samples": result.get("n_samples"), "verdict": result.get("verdict")},
            )
            return result

        job_id = job_store.submit("csv_upload", _job, owner_id=user["id"])
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

    try:
        full_result, dataset_id = await run_csv_analysis(content, filename, user["id"], source="upload", split=split)
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
    deps.upload_result_cache[deps.latest_key(user["id"])] = full_result

    background_tasks.add_task(deps.db.save_result, full_result, "upload", filename, user["id"])
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
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result, user["id"])

    return JSONResponse(content=full_result)


def _owned(result: dict | None, user: dict) -> dict | None:
    """A cached result is returned only to the user who made it (admins can read any)."""
    if result and (result.get("owner_id") == user["id"] or user.get("role") == "admin"):
        return result
    return None


@router.get("/analyze/upload/latest")
async def get_latest_upload_result(user: dict = Depends(require_user)):
    """The caller's latest uploaded dataset analysis (memory cache, then DB)."""
    result = deps.upload_result_cache.get(deps.latest_key(user["id"])) or deps.db.get_latest(user["id"], source="upload")
    if not result:
        raise HTTPException(status_code=404, detail="No upload analysis yet. POST to /analyze/upload first.")
    return result


@router.get("/analyze/upload/{dataset_id}")
async def get_upload_result(dataset_id: str, user: dict = Depends(require_user)):
    """One of the caller's uploaded dataset results. 404 for anyone else's, so ids cannot be probed."""
    result = _owned(deps.upload_result_cache.get(dataset_id), user) or deps.db.get_result(dataset_id, deps.owner_scope(user))
    if not result:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return result
