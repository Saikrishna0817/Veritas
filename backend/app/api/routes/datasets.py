"""Demo + real datasets routes (including demo run)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional

from app.api import dependencies as deps

router = APIRouter()


@router.post("/demo/run")
async def run_demo(background_tasks: BackgroundTasks, request: Request):
    """Run the full demo pipeline with a fresh random scenario each time."""
    import app.demo.data_generator as dg
    data = dg.refresh_demo_data(scenario="random")
    samples = data["samples"]

    clean = [s for s in samples if s["poison_status"] == "clean"][:200]
    all_samples = samples[:300]

    # Fit baseline
    deps.pipeline.fit_baseline(clean[:150])

    # Run detection
    result = deps.pipeline.run(all_samples, run_causal=True)

    # Forensics
    attack_class = deps.classifier.classify(result["layer_results"], all_samples)
    pattern = deps.reconstructor.reconstruct(all_samples, attack_class, result["layer_results"])
    sophistication = deps.sophistication.score(attack_class, pattern, result)
    blast = deps.blast_mapper.map(all_samples, result["layer_results"])
    counterfactual = deps.counterfactual.simulate(result["layer_results"], blast)

    # Defense decision
    defense_action = deps.defense.decide_action(all_samples, result["overall_suspicion_score"], result["verdict"])

    # Human review if borderline
    hitl_case = None
    if result.get("requires_human_review"):
        hitl_case = deps.hitl.enqueue(all_samples, result["layer_results"], result["overall_suspicion_score"])

    full_result = {
        **result,
        "attack_classification": attack_class,
        "injection_pattern": pattern,
        "sophistication": sophistication,
        "blast_radius": blast,
        "counterfactual": counterfactual,
        "defense_action": defense_action,
        "hitl_case": hitl_case,
        "timeline": data["timeline"],
    }

    full_result = deps.to_serializable(full_result)
    deps.demo_result_cache["latest"] = full_result

    background_tasks.add_task(deps.db.save_result, full_result, "demo", "demo_dataset")

    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)


@router.get("/datasets/demo")
async def get_demo_dataset():
    data = deps.get_demo_data()
    return {
        "dataset_id": data["dataset_id"],
        "name": data["name"],
        "total_samples": data["total_samples"],
        "clean_samples": data["clean_samples"],
        "poisoned_samples": data["poisoned_samples"],
        "poison_rate": data["poison_rate"],
        "feature_names": data["feature_names"],
        "created_at": data["created_at"],
    }


@router.get("/datasets/demo/samples")
async def get_demo_samples(limit: int = 50, offset: int = 0, filter_status: Optional[str] = None):
    data = deps.get_demo_data()
    samples = data["samples"]

    if filter_status:
        samples = [s for s in samples if s.get("poison_status") == filter_status]

    return {"total": len(samples), "samples": samples[offset : offset + limit]}


@router.post("/detect/analyze")
async def analyze_dataset(request: Request):
    """Run full 5-layer detection pipeline."""
    body = await request.json()
    sample_ids = body.get("sample_ids", [])

    data = deps.get_demo_data()
    samples = data["samples"]

    if sample_ids:
        samples = [s for s in samples if s["id"] in sample_ids]

    clean = [s for s in samples if s["poison_status"] == "clean"][:100]
    deps.pipeline.fit_baseline(clean)

    result = deps.pipeline.run(samples[:200], run_causal=True)
    result = deps.to_serializable(result)
    deps.demo_result_cache["detection"] = result
    return JSONResponse(content=result)


@router.get("/detect/results/latest")
async def get_latest_results():
    if "latest" not in deps.demo_result_cache:
        raise HTTPException(status_code=404, detail="No results yet. Run /demo/run first.")
    return deps.demo_result_cache["latest"]


@router.get("/datasets/real")
async def list_real_datasets():
    return {"datasets": deps.DATASET_CATALOG}


@router.get("/datasets/real/{name}/download")
async def download_real_dataset(name: str):
    try:
        data = deps.get_real_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return StreamingResponse(
        io.BytesIO(data["csv_bytes"]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{data["filename"]}"'},
    )


@router.post("/datasets/real/{name}/analyze")
async def analyze_real_dataset(name: str, background_tasks: BackgroundTasks, request: Request):
    import concurrent.futures

    try:
        data = deps.get_real_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def _run(csv_bytes, fname):
        engine = deps.CSVIngestionEngine()
        ingested = engine.ingest(csv_bytes, filename=fname)
        pipeline = deps.DetectionPipeline()
        detection_result = pipeline.run_on_upload(ingested)

        samples = ingested["samples"]
        incoming = samples[ingested["reference_split"] :]

        attack_class = deps.classifier.classify(detection_result["layer_results"], incoming)
        pattern = deps.reconstructor.reconstruct(incoming, attack_class, detection_result["layer_results"])
        sophistication = deps.sophistication.score(attack_class, pattern, detection_result)
        blast = deps.blast_mapper.map(incoming, detection_result["layer_results"])
        defense_action = deps.defense.decide_action(
            incoming, detection_result["overall_suspicion_score"], detection_result["verdict"]
        )

        return {
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
            "real_dataset_name": name,
            "real_dataset_description": data["description"],
            "poison_note": data["poison_note"],
            "attack_classification": attack_class,
            "injection_pattern": pattern,
            "sophistication": sophistication,
            "blast_radius": blast,
            "defense_action": defense_action,
            "source": "real_dataset",
        }

    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            full_result = await loop.run_in_executor(pool, _run, data["csv_bytes"], data["filename"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    full_result = deps.to_serializable(full_result)

    # Make this result available to forensics, reports, and trust-score endpoints
    deps.upload_result_cache["latest"] = full_result
    deps.demo_result_cache["latest"] = full_result  # backward-compat for old code paths

    background_tasks.add_task(deps.db.save_result, full_result, "real_dataset", data["filename"])

    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)

