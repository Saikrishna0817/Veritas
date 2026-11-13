"""Model scanning, federated trust, and trust score routes.

All model-scan pipeline orchestration delegated to analysis_helpers.run_model_analysis()
(H2 — deduplication).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps
from app.api.analysis_helpers import run_model_analysis
from app.api.routes.upload import read_upload_limited
from app.core import quota
from app.core.storage import new_key, storage
from app.jobs import tasks  # noqa: F401
from app.jobs.queue import job_store
from app.core.security import is_safe_filename, require_user

logger = logging.getLogger(__name__)
router = APIRouter()


MODEL_EXTENSIONS = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx",
                    ".npy", ".npz", ".gguf", ".h5", ".keras", ".zip")
MAX_SCAN_SIZE = 200 * 1024 * 1024


@router.post("/analyze/model")
async def scan_model(
    background_tasks: BackgroundTasks,
    request: Request,
    model_file: UploadFile = File(...),
    background: bool = Query(False, description="Return 202 + job_id immediately; poll GET /jobs/{job_id}"),
    user: dict = Depends(require_user),
):
    """Statically scan a model file for anything that can run code when it is loaded. The file is never executed."""
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"model:{user['id']}:{client_ip}"):
        raise HTTPException(status_code=429, detail="Too many analysis requests. Try again in one minute.")

    if (
        not model_file.filename
        or not is_safe_filename(model_file.filename)
        or not model_file.filename.lower().endswith(MODEL_EXTENSIONS)
    ):
        raise HTTPException(status_code=400, detail=f"Supported model files: {', '.join(MODEL_EXTENSIONS)}.")

    model_bytes = await read_upload_limited(model_file, MAX_SCAN_SIZE)
    model_filename = model_file.filename
    charged = quota.charge_scan(user, len(model_bytes))

    if background:
        store = storage()
        key = new_key(user["id"], "model")
        store.put(key, model_bytes)
        job_id = job_store.submit("model_scan", {"key": key, "filename": model_filename, "charged": charged}, owner_id=user["id"])
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

    try:
        full_result = await run_model_analysis(model_bytes, model_filename)
    except Exception:
        quota.refund(user["id"], charged)
        rid = getattr(request.state, "request_id", "-")
        logger.exception("Model scan failed [rid=%s]", rid)
        raise HTTPException(status_code=500, detail=f"Model scan failed. Check server logs (request ID: {rid}).")

    background_tasks.add_task(deps.db.save_model_scan, full_result, user["id"])
    background_tasks.add_task(
        deps.db.log_audit_event, user["id"], "analysis.model_scanned", "model", full_result["scan_id"],
        {"filename": model_filename, "format": full_result["scan"]["format"], "verdict": full_result["verdict"]},
    )
    return JSONResponse(content=full_result)


@router.get("/analyze/model/history")
async def get_model_scan_history(limit: int = 20, user: dict = Depends(require_user)):
    return {"scans": deps.db.get_model_scan_history(deps.owner_scope(user), limit=limit)}


@router.get("/analyze/model/{scan_id}")
async def get_model_scan_result(scan_id: str, user: dict = Depends(require_user)):
    result = deps.db.get_model_scan(scan_id, deps.owner_scope(user))
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return result


@router.get("/trust/score")
async def get_trust_score(user: dict = Depends(require_user)):
    """Headline numbers of the latest analysis: only what the detector measured, nothing derived from it."""
    r = deps.resolve_latest_result(user["id"], prefer="auto")
    if not r:
        return {"has_analysis": False, "data_source": "none", "verdict": None, "poison_risk": None, "flagged_rows": None,
                "likely_attack": None, "evidence": {}, "updated_at": datetime.now(timezone.utc).isoformat(),
                "notice": "No analysis yet. Upload a CSV or analyse a real dataset."}
    layers = r.get("layer_results") or {}
    profile = r.get("attack_profile") or (r.get("details") or {}).get("attack_profile")
    return {
        "has_analysis": True,
        "data_source": r.get("source", "unknown"),
        "verdict": r.get("verdict"),
        "poison_risk": round((r.get("overall_suspicion_score") or 0.0) * 100, 1),   # the verdict's 0-1 suspicion, as a percentage
        "flagged_rows": len((layers.get("row_flags") or {}).get("flagged_indices_incoming") or []),
        "likely_attack": ({"type": profile["likely"], "label": profile["label"], "probability": profile["probabilities"][profile["likely"]]}
                          if profile else None),
        "evidence": {name: (layers.get(name) or {}).get("evidence") for name in ("label_check", "shift_check", "resid_check")},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
