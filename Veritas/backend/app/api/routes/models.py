"""Model scanning, federated trust, and trust score routes."""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps

router = APIRouter()


@router.post("/analyze/model")
async def scan_model(
    background_tasks: BackgroundTasks,
    request: Request,
    model_file: UploadFile = File(...),
    dataset_file: Optional[UploadFile] = File(None),
):
    """Upload a trained sklearn .pkl model (+ optional CSV dataset) and scan its parameters."""
    if not model_file.filename.lower().endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl (pickle) model files are accepted.")

    model_bytes = await model_file.read()
    model_filename = model_file.filename

    dataset_bytes = None
    dataset_filename = None
    if dataset_file and dataset_file.filename:
        if not dataset_file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Dataset must be a .csv file.")
        dataset_bytes = await dataset_file.read()
        dataset_filename = dataset_file.filename

    def _run_model_scan(m_bytes, m_name, d_bytes, d_name):
        ingested = deps.model_engine.ingest(m_bytes, m_name, d_bytes, d_name)

        scan_pipeline = deps.DetectionPipeline()
        detection_result = scan_pipeline.run_on_upload(ingested)

        samples = ingested["samples"]
        incoming = samples[ingested["reference_split"] :]

        attack_class = deps.classifier.classify(detection_result["layer_results"], incoming)
        pattern = deps.reconstructor.reconstruct(incoming, attack_class, detection_result["layer_results"])
        sophistication = deps.sophistication.score(attack_class, pattern, detection_result)
        blast = deps.blast_mapper.map(incoming, detection_result["layer_results"])
        defense_action = deps.defense.decide_action(
            incoming, detection_result["overall_suspicion_score"], detection_result["verdict"]
        )

        full_result = {
            **detection_result,
            "scan_id": ingested["scan_id"],
            "model_filename": m_name,
            "dataset_filename": d_name,
            "model_type": ingested["model_type"],
            "model_metadata": ingested["model_metadata"],
            "extraction_info": ingested["extraction_info"],
            "attack_classification": attack_class,
            "injection_pattern": pattern,
            "sophistication": sophistication,
            "blast_radius": blast,
            "defense_action": defense_action,
            "source": "model_scan",
            "interpretation": (
                "Parameters extracted from the model's learned weights/trees were "
                "analyzed for statistical anomalies consistent with poisoning. "
                "A high suspicion score suggests the model was trained on poisoned data."
            ),
        }
        return full_result

    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            full_result = await loop.run_in_executor(
                pool, _run_model_scan, model_bytes, model_filename, dataset_bytes, dataset_filename
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model scan error: {str(e)}")

    full_result = deps.to_serializable(full_result)

    background_tasks.add_task(deps.db.save_model_scan, full_result)

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
    """Get current dataset and model trust scores."""
    if "latest" in deps.demo_result_cache:
        r = deps.demo_result_cache["latest"]
        suspicion = r.get("overall_suspicion_score", 0.0)
        causal = ((r.get("layer_results") or {}).get("layer4_causal") or {}).get("causal_effect", 0.0)
    else:
        suspicion = 0.0
        causal = 0.0

    poison_risk = round(suspicion * 100, 1)
    data_quality = round(max(0, 100 - poison_risk * 1.2), 1)
    behavioral_trust = round(max(0, 100 - poison_risk * 0.8), 1)
    overall = round((data_quality * 0.4 + (100 - poison_risk) * 0.35 + behavioral_trust * 0.25), 1)

    backdoor_risk = "HIGH" if suspicion > 0.7 else "MEDIUM" if suspicion > 0.4 else "LOW"
    adversarial_robustness = "LOW" if suspicion > 0.7 else "MEDIUM" if suspicion > 0.4 else "HIGH"
    prediction_stability = round(max(70, 100 - suspicion * 30), 1)
    grade = "F" if overall < 40 else "D" if overall < 55 else "C" if overall < 70 else "B" if overall < 85 else "A"

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
        "updated_at": datetime.utcnow().isoformat(),
        "debug": {"causal_effect": causal},
    }

