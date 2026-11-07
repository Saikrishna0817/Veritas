"""Model scanning, federated trust, and trust score routes.

All model-scan pipeline orchestration delegated to analysis_helpers.run_model_analysis()
(H2 — deduplication).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps
from app.api.analysis_helpers import run_model_analysis
from app.api.routes.upload import read_upload_limited
from app.core import quota
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
    quota.charge_scan(user, len(model_bytes))

    try:
        full_result = await run_model_analysis(model_bytes, model_filename)
    except Exception:
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
    """Trust scores from latest real analysis, or honest empty state (no fake Grade A)."""
    r = deps.resolve_latest_result(user["id"], prefer="auto")
    if not r:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "has_analysis": False,
            "data_source": "none",
            "dataset_trust": {
                "data_quality": None,
                "poison_risk": None,
                "behavioral_trust": None,
                "overall": None,
            },
            "model_safety": {
                "backdoor_risk": None,
                "adversarial_robustness": None,
                "prediction_stability": None,
                "grade": None,
            },
            "updated_at": now,
            "notice": "No analysis yet. Upload a CSV or analyse a real dataset.",
        }

    suspicion = r.get("overall_suspicion_score", 0.0) or 0.0
    data_source = r.get("source", "unknown")

    poison_risk = round(suspicion * 100, 1)
    data_quality = round(max(0, 100 - poison_risk * 1.2), 1)
    behavioral_trust = round(max(0, 100 - poison_risk * 0.8), 1)
    overall = round((data_quality * 0.4 + (100 - poison_risk) * 0.35 + behavioral_trust * 0.25), 1)

    backdoor_risk = "HIGH" if suspicion > 0.7 else "MEDIUM" if suspicion > 0.4 else "LOW"
    adversarial_robustness = "LOW" if suspicion > 0.7 else "MEDIUM" if suspicion > 0.4 else "HIGH"
    prediction_stability = round(max(70, 100 - suspicion * 30), 1)
    grade = (
        "F" if overall < 40
        else "D" if overall < 55
        else "C" if overall < 70
        else "B" if overall < 85
        else "A"
    )

    return {
        "has_analysis": True,
        "dataset_trust": {
            "data_quality": data_quality,
            "poison_risk": poison_risk,
            "behavioral_trust": behavioral_trust,
            "overall": overall,
        },
        "model_safety": {
            "backdoor_risk": backdoor_risk,
            "adversarial_robustness": adversarial_robustness,
            "prediction_stability": prediction_stability,
            "grade": grade,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
    }
