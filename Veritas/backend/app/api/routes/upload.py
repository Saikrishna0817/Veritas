"""CSV upload analysis routes."""

from __future__ import annotations

import asyncio
import concurrent.futures
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api import dependencies as deps

router = APIRouter()


@router.post("/analyze/upload")
async def analyze_uploaded_csv(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
):
    """
    Upload a CSV file and run the full 5-layer poisoning detection pipeline.

    - Auto-detects label column, feature columns, data types
    - Runs supervised detection if label column found, unsupervised otherwise
    - Self-contained: uses internal 70/30 split (no external baseline needed)
    - Max 50,000 rows, 10MB file size
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    full_result = deps.to_serializable(full_result)
    deps.upload_result_cache[dataset_id] = full_result
    deps.upload_result_cache["latest"] = full_result

    background_tasks.add_task(deps.db.save_result, full_result, "upload", filename)

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

