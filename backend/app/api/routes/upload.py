"""CSV upload analysis routes."""

from __future__ import annotations

import asyncio
import concurrent.futures
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps
from app.core.security import is_safe_filename
from app.core.security import require_user

router = APIRouter()
MAX_CSV_SIZE = 200 * 1024 * 1024


async def read_upload_limited(file: UploadFile, limit: int) -> bytes:
    """Read an upload in bounded chunks and reject it before retaining excess bytes."""
    parts: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {limit // 1024 // 1024} MB.")
        parts.append(chunk)
    return b"".join(parts)


@router.post("/analyze/upload")
async def analyze_uploaded_csv(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    """
    Upload a CSV file and run the full 5-layer poisoning detection pipeline.

    - Auto-detects label column, feature columns, data types
    - Runs supervised detection if label column found, unsupervised otherwise
    - Self-contained: uses internal 70/30 split (no external baseline needed)
    - Max 200,000 rows, 200 MB file size
    """
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"upload:{user['id']}:{client_ip}"):
        raise HTTPException(status_code=429, detail="Too many analysis requests. Try again in one minute.")

    if not file.filename or not is_safe_filename(file.filename) or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await read_upload_limited(file, MAX_CSV_SIZE)

    filename = file.filename

    def _run_analysis(csv_bytes: bytes, fname: str):
        engine = deps.CSVIngestionEngine()
        ingested = engine.ingest(csv_bytes, filename=fname)

        upload_pipeline = deps.DetectionPipeline()
        detection_result = upload_pipeline.run_on_upload(ingested)

        samples = ingested["samples"]
        incoming_samples = samples[ingested["reference_split"] :]

        attack_class = deps.classifier.classify(detection_result["layer_results"], incoming_samples)

        ensemble_scores = detection_result["layer_results"]["layer3_ensemble"].get("ensemble_scores", [])
        if ensemble_scores:
            for i, s in enumerate(incoming_samples):
                if i < len(ensemble_scores) and ensemble_scores[i] > 0.6:
                    s["poison_status"] = "suspected"

        pattern = deps.reconstructor.reconstruct(incoming_samples, attack_class, detection_result["layer_results"])
        sophistication = deps.sophistication.score(attack_class, pattern, detection_result)
        blast = deps.blast_mapper.map(incoming_samples, detection_result["layer_results"])
        counterfactual = deps.counterfactual.simulate(detection_result["layer_results"], blast)

        defense_action = deps.defense.decide_action(
            incoming_samples,
            detection_result["overall_suspicion_score"],
            detection_result["verdict"],
        )

        full_result = {
            **detection_result,
            "dataset_info": {
                "dataset_id": ingested["dataset_id"],
                "filename": ingested["filename"],
                "n_rows": ingested["n_rows"],
                "n_features": ingested["n_features"],
                "feature_names": ingested["feature_names"],
                "label_column": ingested["label_column"],
                "has_labels": ingested["has_labels"],
                "detection_mode": ingested["detection_mode"],
                "reference_split": ingested["reference_split"],
                "schema": ingested["schema"],
                "warnings": ingested["warnings"],
                "created_at": ingested["created_at"],
            },
            "attack_classification": attack_class,
            "injection_pattern": pattern,
            "sophistication": sophistication,
            "blast_radius": blast,
            "counterfactual": counterfactual,
            "defense_action": defense_action,
            "source": "upload",
        }

        return full_result, ingested["dataset_id"]

    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            full_result, dataset_id = await loop.run_in_executor(pool, _run_analysis, content, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed. Check server logs with the request ID.")

    full_result = deps.to_serializable(full_result)
    deps.upload_result_cache[dataset_id] = full_result
    deps.upload_result_cache["latest"] = full_result

    background_tasks.add_task(deps.db.save_result, full_result, "upload", filename)
    background_tasks.add_task(
        deps.db.log_audit_event, user["id"], "analysis.uploaded", "dataset", dataset_id,
        {"filename": filename, "n_samples": full_result.get("n_samples"), "verdict": full_result.get("verdict")},
    )

    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)


@router.get("/analyze/upload/latest")
async def get_latest_upload_result():
    """Get the latest uploaded dataset analysis result (memory cache → DB fallback)."""
    if "latest" in deps.upload_result_cache:
        return deps.upload_result_cache["latest"]
    result = deps.db.get_latest(source="upload")
    if not result:
        raise HTTPException(status_code=404, detail="No upload analysis yet. POST to /analyze/upload first.")
    return result


@router.get("/analyze/upload/{dataset_id}")
async def get_upload_result(dataset_id: str):
    """Get analysis result for a specific uploaded dataset (memory → DB fallback)."""
    if dataset_id in deps.upload_result_cache:
        return deps.upload_result_cache[dataset_id]
    result = deps.db.get_result(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return result
