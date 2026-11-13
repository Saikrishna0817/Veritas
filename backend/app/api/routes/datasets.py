"""Demo + real datasets routes (including demo run).

Fixes applied:
  H2 — real-dataset analysis delegates to run_csv_analysis (no more 4th copy of the pipeline)
  M1 — /demo/run, /datasets/real/{name}/analyze are now rate-limited
  M7 — Pydantic request bodies for POST endpoints
  M10 — error responses do not leak internal details
  L1 — datetime.utcnow() replaced
  L2 — asyncio.get_running_loop()
"""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.api import dependencies as deps
from app.api.analysis_helpers import run_csv_analysis
from app.core import quota
from app.core.security import require_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic request models (M7) ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    sample_ids: list[str] = []


# ── Demo endpoints ─────────────────────────────────────────────────────────────

@router.post("/demo/run")
async def run_demo(
    background_tasks: BackgroundTasks,
    request: Request,
    user: dict = Depends(require_user),
):
    """Run the full demo pipeline with a fresh random scenario each time."""
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"demo:{user['id']}:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many analysis requests. Try again in one minute.",
        )

    import app.demo.data_generator as dg

    def _run():
        data = dg.refresh_demo_data(scenario="random")
        all_samples = data["samples"][:300]
        # pipeline.run() fits its own baseline on the first 70% of the batch.
        result = deps.new_detection_pipeline().run(all_samples)
        attack_class = deps.classifier.classify(result)
        pattern = deps.reconstructor.reconstruct(all_samples, attack_class, result["layer_results"])
        blast = deps.blast_mapper.map(all_samples, result["layer_results"])
        defense_action = deps.defense.decide_action(
            all_samples, result["overall_suspicion_score"], result["verdict"], scope="demo", owner_id=user["id"]
        )
        hitl_case = None
        if result.get("requires_human_review"):
            hitl_case = deps.hitl.enqueue(
                all_samples,
                {**result["layer_results"], "attack_classification": attack_class},
                result["overall_suspicion_score"],
                scope="demo",
                owner_id=user["id"],
            )
        full = {
            **result,
            "attack_classification": attack_class,
            "injection_pattern": pattern,
            "blast_radius": blast,
            "defense_action": defense_action,
            "hitl_case": hitl_case,
            "timeline": data["timeline"],
        }
        return deps.to_serializable(full)

    loop = asyncio.get_running_loop()
    full_result = await loop.run_in_executor(deps.get_thread_pool(), _run)

    full_result["owner_id"] = user["id"]
    deps.demo_result_cache[deps.latest_key(user["id"])] = full_result
    background_tasks.add_task(deps.db.save_result, full_result, "demo", "demo_dataset", user["id"])
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result, user["id"])
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
async def get_demo_samples(
    limit: int = 50,
    offset: int = 0,
    filter_status: Optional[str] = None,
):
    data = deps.get_demo_data()
    samples = data["samples"]
    if filter_status:
        samples = [s for s in samples if s.get("poison_status") == filter_status]
    return {"total": len(samples), "samples": samples[offset: offset + limit]}


@router.post("/detect/analyze")
async def analyze_dataset(
    body: AnalyzeRequest,  # M7: was request.json() raw body
    request: Request,
    user: dict = Depends(require_user),
):
    """Run full 5-layer detection pipeline on demo dataset samples."""
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"detect:{user['id']}:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many analysis requests. Try again in one minute.",
        )

    data = deps.get_demo_data()
    samples = data["samples"]
    if body.sample_ids:
        wanted = set(body.sample_ids)
        samples = [s for s in samples if s["id"] in wanted]
    batch = samples[:200]
    if len(batch) < 20:
        raise HTTPException(status_code=422, detail="At least 20 samples are required.")

    def _run():
        return deps.to_serializable(deps.new_detection_pipeline().run(batch))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(deps.get_thread_pool(), _run)
    deps.demo_result_cache[f"detection:{user['id']}"] = result
    return JSONResponse(content=result)


@router.get("/detect/results/latest")
async def get_latest_results(user: dict = Depends(require_user)):
    """The caller's latest analysis: real first, demo only as last resort."""
    r = deps.resolve_latest_result(user["id"], prefer="auto")
    if not r:
        raise HTTPException(
            status_code=404,
            detail=(
                "No analysis results found. Upload a CSV via /analyze/upload "
                "or analyse a real dataset via /datasets/real/{name}/analyze. "
                "Synthetic demo is available only via explicit POST /demo/run."
            ),
        )
    return r


# ── Real Dataset Library ───────────────────────────────────────────────────────

@router.get("/datasets/real")
async def list_real_datasets():
    return {"datasets": deps.DATASET_CATALOG}


@router.get("/datasets/real/{name}/download")
async def download_real_dataset(name: str, user: dict = Depends(require_user)):
    try:
        data = deps.get_real_dataset(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return StreamingResponse(
        io.BytesIO(data["csv_bytes"]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{data["filename"]}"'},
    )


@router.post("/datasets/real/{name}/analyze")
async def analyze_real_dataset(
    name: str,
    background_tasks: BackgroundTasks,
    request: Request,
    user: dict = Depends(require_user),
):
    """Analyze a real scikit-learn dataset using the full detection pipeline."""
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"real:{user['id']}:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many analysis requests. Try again in one minute.",
        )

    try:
        data = deps.get_real_dataset(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    quota.charge_analysis(user)

    try:
        full_result, dataset_id = await run_csv_analysis(
            data["csv_bytes"],
            data["filename"],
            user["id"],
            source="real_dataset",
            extra={
                "real_dataset_name": name,
                "real_dataset_description": data["description"],
                "poison_note": data["poison_note"],
            },
        )
    except Exception:
        rid = getattr(request.state, "request_id", "-")
        logger.exception("Real-dataset analysis failed [dataset=%s, rid=%s]", name, rid)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error. Check server logs (request ID: {rid}).",
        )

    # Make result available to forensics, reports, and trust-score endpoints
    deps.upload_result_cache[deps.latest_key(user["id"])] = full_result

    background_tasks.add_task(deps.db.save_result, full_result, "real_dataset", data["filename"], user["id"])
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result, user["id"])
    return JSONResponse(content=full_result)
