"""Model scanning, federated trust, and trust score routes.

All model-scan pipeline orchestration delegated to analysis_helpers.run_model_analysis()
(H2 — deduplication).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps
from app.api.analysis_helpers import run_model_analysis
from app.api.routes.upload import read_upload_limited
from app.core.security import is_safe_filename, require_user
from app.ingestion.model_engine import MAX_MODEL_SIZE

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze/model")
async def scan_model(
    background_tasks: BackgroundTasks,
    request: Request,
    model_file: UploadFile = File(...),
    dataset_file: Optional[UploadFile] = File(None),
    user: dict = Depends(require_user),
):
    """Upload a trained sklearn .pkl model (+ optional CSV dataset) and scan its parameters."""
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"model:{user['id']}:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many analysis requests. Try again in one minute.",
        )

    if (
        not model_file.filename
        or not is_safe_filename(model_file.filename)
        or not model_file.filename.lower().endswith(".pkl")
    ):
        raise HTTPException(status_code=400, detail="Only .pkl (pickle) model files are accepted.")

    model_bytes = await read_upload_limited(model_file, MAX_MODEL_SIZE)
    model_filename = model_file.filename

    dataset_bytes: bytes | None = None
    dataset_filename: str | None = None
    if dataset_file and dataset_file.filename:
        if not is_safe_filename(dataset_file.filename) or not dataset_file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Dataset must be a .csv file.")
        dataset_bytes = await read_upload_limited(dataset_file, 200 * 1024 * 1024)
        dataset_filename = dataset_file.filename

    try:
        full_result = await run_model_analysis(
            model_bytes, model_filename, dataset_bytes, dataset_filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        rid = getattr(request.state, "request_id", "-")
        logger.exception("Model scan failed [rid=%s]", rid)
        raise HTTPException(
            status_code=500,
            detail=f"Model scan failed. Check server logs (request ID: {rid}).",
        )

    background_tasks.add_task(deps.db.save_model_scan, full_result)
    background_tasks.add_task(
        deps.db.log_audit_event,
        user["id"],
        "analysis.model_scanned",
        "model",
        full_result["scan_id"],
        {
            "filename": model_filename,
            "model_type": full_result.get("model_type"),
            "verdict": full_result.get("verdict"),
        },
    )

    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)


@router.get("/analyze/model/history")
async def get_model_scan_history(limit: int = 20):
    return {"scans": deps.db.get_model_scan_history(limit=limit)}


@router.get("/analyze/model/{scan_id}")
async def get_model_scan_result(scan_id: str):
    result = deps.db.get_model_scan(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return result


@router.get("/federated/clients")
async def get_federated_clients():
    """Get federated client trust scores."""
    from app.detection.layer5_federated import FederatedTrustAnalyzer, generate_demo_clients

    clients = generate_demo_clients()
    analyzer = FederatedTrustAnalyzer()
    return analyzer.analyze_clients(clients)


@router.get("/trust/score")
async def get_trust_score():
    """Get current dataset and model trust scores — uses latest upload or demo result."""
    r = deps.upload_result_cache.get("latest") or deps.demo_result_cache.get("latest")
    if r:
        suspicion = r.get("overall_suspicion_score", 0.0)
        causal = ((r.get("layer_results") or {}).get("layer4_causal") or {}).get("causal_effect", 0.0)
        data_source = r.get("source", "demo")
    else:
        suspicion = 0.0
        causal = 0.0
        data_source = "none"

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
        "updated_at": datetime.now(timezone.utc).isoformat(),  # L1: utcnow() deprecated
        "data_source": data_source,
        "debug": {"causal_effect": causal},
    }
