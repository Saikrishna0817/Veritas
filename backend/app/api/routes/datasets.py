"""Demo + real datasets routes (including demo run).

Fixes applied:
  H2 — real-dataset analysis delegates to run_csv_analysis (no more 4th copy of the pipeline)
  M1 — /demo/run, /redteam/simulate, /datasets/real/{name}/analyze are now rate-limited
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
from app.core.security import require_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic request models (M7) ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    sample_ids: list[str] = []


class RedTeamRequest(BaseModel):
    attack_type: Literal[
        "label_flip", "backdoor", "boiling_frog", "clean_label", "gradient_poisoning"
    ] = "label_flip"


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
        samples = data["samples"]
        clean = [s for s in samples if s["poison_status"] == "clean"][:200]
        all_samples = samples[:300]
        deps.pipeline.fit_baseline(clean[:150])
        result = deps.pipeline.run(all_samples, run_causal=True)
        attack_class = deps.classifier.classify(result["layer_results"], all_samples)
        pattern = deps.reconstructor.reconstruct(all_samples, attack_class, result["layer_results"])
        sophistication = deps.sophistication.score(attack_class, pattern, result)
        blast = deps.blast_mapper.map(all_samples, result["layer_results"])
        counterfactual = deps.counterfactual.simulate(result["layer_results"], blast)
        defense_action = deps.defense.decide_action(
            all_samples, result["overall_suspicion_score"], result["verdict"]
        )
        hitl_case = None
        if result.get("requires_human_review"):
            hitl_case = deps.hitl.enqueue(all_samples, result["layer_results"], result["overall_suspicion_score"])
        full = {
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
        return deps.to_serializable(full)

    loop = asyncio.get_running_loop()
    full_result = await loop.run_in_executor(deps.get_thread_pool(), _run)

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
    user: dict = Depends(require_user),
):
    """Run full 5-layer detection pipeline on demo dataset samples."""
    data = deps.get_demo_data()
    samples = data["samples"]
    if body.sample_ids:
        samples = [s for s in samples if s["id"] in body.sample_ids]
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

    try:
        full_result, dataset_id = await run_csv_analysis(
            data["csv_bytes"],
            data["filename"],
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
    deps.upload_result_cache["latest"] = full_result
    deps.demo_result_cache["latest"] = full_result  # backward-compat

    background_tasks.add_task(deps.db.save_result, full_result, "real_dataset", data["filename"])
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(deps.broadcast_demo_events, ws_manager, full_result)
    return JSONResponse(content=full_result)


# ── Red Team ───────────────────────────────────────────────────────────────────

@router.post("/redteam/simulate")
async def run_red_team(
    body: RedTeamRequest,  # M7: Pydantic model instead of raw request.json()
    request: Request,
    user: dict = Depends(require_user),
):
    """Inject a synthetic attack and measure detection resilience."""
    client_ip = request.client.host if request.client else "unknown"
    if not deps.analysis_rate_limiter.allow(f"redteam:{user['id']}:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many red-team requests. Try again in one minute.",
        )

    data = deps.get_demo_data()
    deps.red_team.pipeline = deps.pipeline
    result = deps.red_team.run_simulation(body.attack_type, data["samples"][:200])
    return result


@router.get("/redteam/history")
async def get_red_team_history(user: dict = Depends(require_user)):
    return {"simulations": deps.red_team.simulation_results}
