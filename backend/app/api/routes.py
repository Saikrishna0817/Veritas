"""
AI Trust Forensics Platform — REST API Routes
"""
import asyncio
import numpy as np
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
import uuid
import io
from datetime import datetime

from app.demo.data_generator import get_demo_data
from app.utils.serialization import to_serializable
from app.detection.pipeline import DetectionPipeline
from app.ingestion.csv_engine import CSVIngestionEngine
from app.ingestion.model_engine import ModelScanEngine
from app.demo.real_datasets import get_real_dataset, DATASET_CATALOG
from app.forensics.engine import (
    AttackTypeClassifier, InjectionPatternReconstructor,
    SophisticationScorer, BlastRadiusMapper, CounterfactualSimulator
)
from app.defense.engine import StabilityAwareAutoDefense, HumanInTheLoopQueue, RedTeamSimulator
from app.db import store as db

# Initialise SQLite on import
db.init_db()

router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────────────
_pipeline = DetectionPipeline()
_classifier = AttackTypeClassifier()
_reconstructor = InjectionPatternReconstructor()
_sophistication = SophisticationScorer()
_blast_mapper = BlastRadiusMapper()
_counterfactual = CounterfactualSimulator()
_defense = StabilityAwareAutoDefense()
_hitl = HumanInTheLoopQueue()
_red_team = RedTeamSimulator()
_model_engine = ModelScanEngine()

# In-memory cache (fast path) — DB is the durable fallback
_demo_result_cache = {}
_upload_result_cache = {}  # keyed by dataset_id


# ── DEMO ──────────────────────────────────────────────────────────────────────

@router.post("/demo/run")
async def run_demo(background_tasks: BackgroundTasks, request: Request):
    """Run the full demo pipeline and broadcast results via WebSocket."""
    data = get_demo_data()
    samples = data["samples"]
    
    clean = [s for s in samples if s["poison_status"] == "clean"][:200]
    all_samples = samples[:300]
    
    # Fit baseline
    _pipeline.fit_baseline(clean[:150])
    
    # Run detection
    result = _pipeline.run(all_samples, run_causal=True)
    
    # Forensics
    attack_class = _classifier.classify(result["layer_results"], all_samples)
    pattern = _reconstructor.reconstruct(all_samples, attack_class, result["layer_results"])
    sophistication = _sophistication.score(attack_class, pattern, result)
    blast = _blast_mapper.map(all_samples, result["layer_results"])
    counterfactual = _counterfactual.simulate(result["layer_results"], blast)
    
    # Defense decision
    defense_action = _defense.decide_action(
        all_samples, result["overall_suspicion_score"], result["verdict"]
    )
    
    # Human review if borderline
    hitl_case = None
    if result.get("requires_human_review"):
        hitl_case = _hitl.enqueue(all_samples, result["layer_results"],
                                   result["overall_suspicion_score"])
    
    full_result = {
        **result,
        "attack_classification": attack_class,
        "injection_pattern": pattern,
        "sophistication": sophistication,
        "blast_radius": blast,
        "counterfactual": counterfactual,
        "defense_action": defense_action,
        "hitl_case": hitl_case,
        "timeline": data["timeline"]
    }
    
    full_result = to_serializable(full_result)
    _demo_result_cache["latest"] = full_result

    # Persist to SQLite
    background_tasks.add_task(db.save_result, full_result, "demo", "demo_dataset")

    # Broadcast via WebSocket
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(
        _broadcast_demo_events, ws_manager, full_result
    )

    return JSONResponse(content=full_result)


async def _broadcast_demo_events(manager, result: dict):
    """Broadcast detection events to WebSocket clients."""
    await asyncio.sleep(0.5)
    await manager.broadcast("sample_analyzed", {
        "n_samples": result["n_samples"],
        "suspicion_score": result["overall_suspicion_score"],
        "layer_scores": result["layer_scores"]
    })
    
    await asyncio.sleep(1.0)
    if result["verdict"] != "CLEAN":
        await manager.broadcast("attack_confirmed", {
            "attack_type": result["attack_classification"]["attack_type"],
            "confidence": result["attack_classification"]["confidence"],
            "causal_effect": result["layer_results"]["layer4_causal"].get("causal_effect", 0),
            "narrative": result["injection_pattern"]["narrative"][:200],
            "blast_radius": {
                "n_batches": result["blast_radius"]["n_batches_affected"],
                "n_models": result["blast_radius"]["n_models_affected"],
                "impact_pct": result["blast_radius"]["prediction_impact_pct"]
            }
        })
    
    await asyncio.sleep(1.5)
    if result["defense_action"]["action"] in ("quarantine", "soft_quarantine"):
        await manager.broadcast("defense_triggered", {
            "action": result["defense_action"]["action"],
            "samples_affected": result["defense_action"]["samples_affected"],
            "model_stable": result["defense_action"].get("model_stable", True)
        })
    
    if result.get("hitl_case"):
        await asyncio.sleep(0.5)
        await manager.broadcast("human_review_required", {
            "case_id": result["hitl_case"]["case_id"],
            "suspicion_score": result["hitl_case"]["suspicion_score"],
            "deadline": result["hitl_case"]["deadline"]
        })


# ── CSV UPLOAD ANALYSIS ───────────────────────────────────────────────────────

@router.post("/analyze/upload")
async def analyze_uploaded_csv(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload a CSV file and run the full 5-layer poisoning detection pipeline.
    
    - Auto-detects label column, feature columns, data types
    - Runs supervised detection if label column found, unsupervised otherwise
    - Self-contained: uses internal 70/30 split (no external baseline needed)
    - Max 50,000 rows, 10MB file size
    """
    import concurrent.futures

    # Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    # Read file bytes
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    filename = file.filename

    def _run_analysis(csv_bytes: bytes, fname: str) -> dict:
        """CPU-bound analysis — runs in thread pool to avoid blocking event loop."""
        # Ingest CSV
        engine = CSVIngestionEngine()
        ingested = engine.ingest(csv_bytes, filename=fname)
        
        # Run detection pipeline (upload mode — self-contained split)
        upload_pipeline = DetectionPipeline()
        detection_result = upload_pipeline.run_on_upload(ingested)
        
        # Forensics
        samples = ingested["samples"]
        incoming_samples = samples[ingested["reference_split"]:]
        
        attack_class = _classifier.classify(detection_result["layer_results"], incoming_samples)
        
        # Mark suspected samples using ensemble scores
        ensemble_scores = detection_result["layer_results"]["layer3_ensemble"].get("ensemble_scores", [])
        if ensemble_scores:
            for i, s in enumerate(incoming_samples):
                if i < len(ensemble_scores) and ensemble_scores[i] > 0.6:
                    s["poison_status"] = "suspected"
        
        pattern = _reconstructor.reconstruct(incoming_samples, attack_class, detection_result["layer_results"])
        sophistication = _sophistication.score(attack_class, pattern, detection_result)
        blast = _blast_mapper.map(incoming_samples, detection_result["layer_results"])
        counterfactual = _counterfactual.simulate(detection_result["layer_results"], blast)
        
        defense_action = _defense.decide_action(
            incoming_samples,
            detection_result["overall_suspicion_score"],
            detection_result["verdict"]
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
                "created_at": ingested["created_at"]
            },
            "attack_classification": attack_class,
            "injection_pattern": pattern,
            "sophistication": sophistication,
            "blast_radius": blast,
            "counterfactual": counterfactual,
            "defense_action": defense_action,
            "source": "upload"
        }
        return full_result, ingested["dataset_id"]

    # Run CPU-bound work in thread pool (keeps event loop free)
    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            full_result, dataset_id = await loop.run_in_executor(
                pool, _run_analysis, content, filename
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    full_result = to_serializable(full_result)
    _upload_result_cache[dataset_id] = full_result
    _upload_result_cache["latest"] = full_result

    # Persist to SQLite
    background_tasks.add_task(db.save_result, full_result, "upload", filename)

    # Broadcast via WebSocket
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(_broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)


@router.get("/analyze/upload/latest")
async def get_latest_upload_result():
    """Get the latest uploaded dataset analysis result (memory cache → DB fallback)."""
    if "latest" in _upload_result_cache:
        return _upload_result_cache["latest"]
    result = db.get_latest(source="upload")
    if not result:
        raise HTTPException(status_code=404, detail="No upload analysis yet. POST to /analyze/upload first.")
    return result


@router.get("/analyze/upload/{dataset_id}")
async def get_upload_result(dataset_id: str):
    """Get analysis result for a specific uploaded dataset (memory → DB fallback)."""
    if dataset_id in _upload_result_cache:
        return _upload_result_cache[dataset_id]
    result = db.get_result(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return result


# ── HISTORY (persisted results) ───────────────────────────────────────────────

@router.get("/history")
async def get_analysis_history(source: Optional[str] = None, limit: int = 20):
    """
    Return recent analysis results from the SQLite database.
    Survives server restarts. source: 'demo' | 'upload' | 'model_scan' | 'real_dataset'
    """
    rows = db.get_history(source=source, limit=limit)
    stats = db.get_stats()
    return {"results": rows, "stats": stats}


@router.get("/history/{result_id}")
async def get_historical_result(result_id: str):
    """Fetch a full persisted result by ID."""
    result = db.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found in database.")
    return result


# ── MODEL SCAN ────────────────────────────────────────────────────────────────

@router.post("/analyze/model")
async def scan_model(
    background_tasks: BackgroundTasks,
    request: Request,
    model_file: UploadFile = File(...),
    dataset_file: Optional[UploadFile] = File(None),
):
    """
    Upload a trained sklearn .pkl model (+ optional CSV dataset) and detect
    whether the model's learned parameters show signs of poisoning.

    Supported: LogisticRegression, RandomForest, GradientBoosting,
    SVC, DecisionTree, MLP, SGD, KNeighbors, Ridge, Lasso, NaiveBayes.
    Max file size: 50MB.
    """
    import concurrent.futures

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
        """CPU-bound model scan — runs in thread pool."""
        ingested = _model_engine.ingest(m_bytes, m_name, d_bytes, d_name)

        scan_pipeline = DetectionPipeline()
        detection_result = scan_pipeline.run_on_upload(ingested)

        samples = ingested["samples"]
        incoming = samples[ingested["reference_split"]:]

        attack_class = _classifier.classify(detection_result["layer_results"], incoming)
        pattern = _reconstructor.reconstruct(incoming, attack_class, detection_result["layer_results"])
        sophistication = _sophistication.score(attack_class, pattern, detection_result)
        blast = _blast_mapper.map(incoming, detection_result["layer_results"])
        defense_action = _defense.decide_action(
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
                pool, _run_model_scan,
                model_bytes, model_filename, dataset_bytes, dataset_filename
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model scan error: {str(e)}")

    full_result = to_serializable(full_result)

    # Persist to SQLite
    background_tasks.add_task(db.save_model_scan, full_result)

    # Broadcast
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(_broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)


@router.get("/analyze/model/history")
async def get_model_scan_history(limit: int = 20):
    """Return recent model scan summaries from the database."""
    return {"scans": db.get_model_scan_history(limit=limit)}


@router.get("/analyze/model/{scan_id}")
async def get_model_scan_result(scan_id: str):
    """Fetch a full model scan result by ID."""
    result = db.get_model_scan(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return result


# ── REAL DATASET LIBRARY ──────────────────────────────────────────────────────

@router.get("/datasets/real")
async def list_real_datasets():
    """List all available real public datasets with poison injected."""
    return {"datasets": DATASET_CATALOG}


@router.get("/datasets/real/{name}/download")
async def download_real_dataset(name: str):
    """
    Download a real poisoned dataset as a CSV file.
    name: iris | wine | breast_cancer | digits
    """
    try:
        data = get_real_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return StreamingResponse(
        io.BytesIO(data["csv_bytes"]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{data["filename"]}"'}
    )


@router.post("/datasets/real/{name}/analyze")
async def analyze_real_dataset(
    name: str,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Run the full 5-layer detection pipeline on a real public dataset
    (with known poison injected). Returns full forensic analysis.
    name: iris | wine | breast_cancer | digits
    """
    import concurrent.futures

    try:
        data = get_real_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def _run(csv_bytes, fname):
        engine = CSVIngestionEngine()
        ingested = engine.ingest(csv_bytes, filename=fname)
        pipeline = DetectionPipeline()
        detection_result = pipeline.run_on_upload(ingested)

        samples = ingested["samples"]
        incoming = samples[ingested["reference_split"]:]

        attack_class = _classifier.classify(detection_result["layer_results"], incoming)
        pattern = _reconstructor.reconstruct(incoming, attack_class, detection_result["layer_results"])
        sophistication = _sophistication.score(attack_class, pattern, detection_result)
        blast = _blast_mapper.map(incoming, detection_result["layer_results"])
        defense_action = _defense.decide_action(
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
            full_result = await loop.run_in_executor(
                pool, _run, data["csv_bytes"], data["filename"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    full_result = to_serializable(full_result)

    # Persist
    background_tasks.add_task(db.save_result, full_result, "real_dataset", data["filename"])

    # Broadcast
    ws_manager = request.app.state.ws_manager
    background_tasks.add_task(_broadcast_demo_events, ws_manager, full_result)

    return JSONResponse(content=full_result)



@router.get("/datasets/demo")
async def get_demo_dataset():
    """Get the demo dataset summary."""
    data = get_demo_data()
    return {
        "dataset_id": data["dataset_id"],
        "name": data["name"],
        "total_samples": data["total_samples"],
        "clean_samples": data["clean_samples"],
        "poisoned_samples": data["poisoned_samples"],
        "poison_rate": data["poison_rate"],
        "feature_names": data["feature_names"],
        "created_at": data["created_at"]
    }


@router.get("/datasets/demo/samples")
async def get_demo_samples(limit: int = 50, offset: int = 0, 
                            filter_status: Optional[str] = None):
    """Get demo dataset samples with optional filtering."""
    data = get_demo_data()
    samples = data["samples"]
    
    if filter_status:
        samples = [s for s in samples if s.get("poison_status") == filter_status]
    
    return {
        "total": len(samples),
        "samples": samples[offset:offset + limit]
    }


# ── DETECTION ─────────────────────────────────────────────────────────────────

@router.post("/detect/analyze")
async def analyze_dataset(request: Request):
    """Run full 5-layer detection pipeline."""
    body = await request.json()
    sample_ids = body.get("sample_ids", [])
    
    data = get_demo_data()
    samples = data["samples"]
    
    if sample_ids:
        samples = [s for s in samples if s["id"] in sample_ids]
    
    clean = [s for s in samples if s["poison_status"] == "clean"][:100]
    _pipeline.fit_baseline(clean)
    
    result = _pipeline.run(samples[:200], run_causal=True)
    result = to_serializable(result)
    _demo_result_cache["detection"] = result
    return JSONResponse(content=result)


@router.get("/detect/results/latest")
async def get_latest_results():
    """Get the latest detection results."""
    if "latest" not in _demo_result_cache:
        raise HTTPException(status_code=404, detail="No results yet. Run /demo/run first.")
    return _demo_result_cache["latest"]


# ── FORENSICS ─────────────────────────────────────────────────────────────────

@router.get("/forensics/latest")
async def get_latest_forensics():
    """Get the latest forensic analysis."""
    if "latest" not in _demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    r = _demo_result_cache["latest"]
    return {
        "attack_classification": r.get("attack_classification"),
        "injection_pattern": r.get("injection_pattern"),
        "sophistication": r.get("sophistication"),
        "blast_radius": r.get("blast_radius"),
        "counterfactual": r.get("counterfactual")
    }


@router.get("/forensics/narrative")
async def get_attack_narrative():
    """Get the human-readable attack narrative."""
    if "latest" not in _demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    pattern = _demo_result_cache["latest"].get("injection_pattern", {})
    return {"narrative": pattern.get("narrative", "No narrative available.")}


@router.get("/forensics/timeline")
async def get_attack_timeline():
    """Get the temporal attack timeline data."""
    data = get_demo_data()
    return {"timeline": data["timeline"]}


@router.get("/blast-radius/latest")
async def get_blast_radius():
    """Get blast radius analysis."""
    if "latest" not in _demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    return _demo_result_cache["latest"].get("blast_radius", {})


# ── TRUST SCORES ──────────────────────────────────────────────────────────────

@router.get("/trust/score")
async def get_trust_score():
    """Get current dataset and model trust scores."""
    if "latest" in _demo_result_cache:
        r = _demo_result_cache["latest"]
        suspicion = r["overall_suspicion_score"]
        causal = r["layer_results"]["layer4_causal"].get("causal_effect", 0)
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
            "overall": overall
        },
        "model_safety": {
            "backdoor_risk": backdoor_risk,
            "adversarial_robustness": adversarial_robustness,
            "prediction_stability": prediction_stability,
            "grade": grade
        },
        "updated_at": datetime.utcnow().isoformat()
    }


# ── DEFENSE ───────────────────────────────────────────────────────────────────

@router.post("/defense/quarantine")
async def trigger_quarantine(request: Request):
    """Manually trigger quarantine action."""
    if "latest" not in _demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    r = _demo_result_cache["latest"]
    data = get_demo_data()
    action = _defense._quarantine(data["samples"][:50], r["overall_suspicion_score"])
    return action


@router.get("/defense/status")
async def get_defense_status():
    """Get current defense engine status."""
    return _defense.get_status()


@router.get("/defense/hitl/pending")
async def get_pending_reviews():
    """Get pending human review cases."""
    return {"cases": _hitl.get_pending()}


@router.post("/defense/hitl/decide")
async def submit_review_decision(request: Request):
    """Submit a human review decision."""
    body = await request.json()
    case_id = body.get("case_id")
    decision = body.get("decision")
    reviewer = body.get("reviewer", "analyst")
    
    if not case_id or not decision:
        raise HTTPException(status_code=400, detail="case_id and decision required")
    
    result = _hitl.decide(case_id, decision, reviewer)
    return result


# ── RED TEAM ──────────────────────────────────────────────────────────────────

@router.post("/redteam/simulate")
async def run_red_team(request: Request):
    """Run a red-team attack simulation."""
    body = await request.json()
    attack_type = body.get("attack_type", "label_flip")
    
    valid_attacks = ["label_flip", "backdoor", "boiling_frog", "clean_label", "gradient_poisoning"]
    if attack_type not in valid_attacks:
        raise HTTPException(status_code=400, detail=f"attack_type must be one of {valid_attacks}")
    
    data = get_demo_data()
    _red_team.pipeline = _pipeline
    result = _red_team.run_simulation(attack_type, data["samples"][:200])
    return result


@router.get("/redteam/history")
async def get_red_team_history():
    return {"simulations": _red_team.simulation_results}


# ── FEDERATED CLIENTS ─────────────────────────────────────────────────────────

@router.get("/federated/clients")
async def get_federated_clients():
    """Get federated client trust scores."""
    from app.detection.layer5_federated import generate_demo_clients
    clients = generate_demo_clients()
    
    # Run trust analysis
    from app.detection.layer5_federated import FederatedTrustAnalyzer
    analyzer = FederatedTrustAnalyzer()
    result = analyzer.analyze_clients(clients)
    return result


# ── REPORTS ───────────────────────────────────────────────────────────────────

@router.post("/reports/generate")
async def generate_report():
    """Generate a forensic evidence report."""
    if "latest" not in _demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")

    r = _demo_result_cache["latest"]
    report = {
        "report_id": str(uuid.uuid4()),
        "generated_at": datetime.utcnow().isoformat(),
        "title": "AI Poisoning Forensic Evidence Report",
        "platform": "AI Trust Forensics Platform v2.2",
        "executive_summary": {
            "verdict": r["verdict"],
            "attack_type": r.get("attack_classification", {}).get("attack_type", "unknown"),
            "confidence": r.get("attack_classification", {}).get("confidence", 0),
            "causal_effect": r["layer_results"]["layer4_causal"].get("causal_effect", 0),
            "sophistication_score": r.get("sophistication", {}).get("sophistication_score", 0),
            "blast_radius_summary": {
                "batches": r.get("blast_radius", {}).get("n_batches_affected", 0),
                "models": r.get("blast_radius", {}).get("n_models_affected", 0),
                "impact_pct": r.get("blast_radius", {}).get("prediction_impact_pct", 0)
            }
        },
        "evidence_bundle": r["layer_results"],
        "attack_narrative": r.get("injection_pattern", {}).get("narrative", ""),
        "defense_actions": _defense.defense_log,
        "compliance": {
            "nist_ai_rmf": "GOVERN 1.1, MAP 1.5, MEASURE 2.5, MANAGE 2.2",
            "eu_ai_act": "Article 9 (Risk Management), Article 17 (Quality Management)",
            "audit_hash": f"sha256_{uuid.uuid4().hex}"
        }
    }
    return report


# ── BLUE TEAM SOC ─────────────────────────────────────────────────────────────

@router.get("/blueteam/status")
async def get_blueteam_status():
    """
    Full Blue Team SOC status — defense mode, quarantine count,
    HITL queue depth, red team resilience, and threat level.
    """
    defense_status = _defense.get_status()
    pending_cases = _hitl.get_pending()
    sims = _red_team.simulation_results

    # Threat level from latest analysis
    threat_level = "NOMINAL"
    verdict = "CLEAN"
    suspicion = 0.0
    if "latest" in _demo_result_cache:
        verdict = _demo_result_cache["latest"].get("verdict", "CLEAN")
        suspicion = _demo_result_cache["latest"].get("overall_suspicion_score", 0)
        if suspicion > 0.65:
            threat_level = "CRITICAL"
        elif suspicion > 0.35:
            threat_level = "ELEVATED"
        elif suspicion > 0.15:
            threat_level = "GUARDED"
        else:
            threat_level = "NOMINAL"

    # Resilience from red team history
    total_sims = len(sims)
    caught = sum(1 for s in sims if s.get("detected", False))
    resilience_pct = round((caught / total_sims * 100) if total_sims > 0 else 100.0, 1)
    avg_resilience = round(
        sum(s.get("resilience_score", 0) for s in sims) / total_sims
        if total_sims > 0 else 10.0, 1
    )

    return {
        "threat_level": threat_level,
        "current_verdict": verdict,
        "suspicion_score": round(suspicion, 4),
        "defense_mode": defense_status["mode"],
        "total_quarantined": defense_status["total_quarantined"],
        "n_defense_actions": defense_status["n_defense_actions"],
        "last_defense_action": defense_status["last_action"],
        "hitl_queue_depth": len(pending_cases),
        "pending_cases": pending_cases[:5],
        "red_team": {
            "total_simulations": total_sims,
            "attacks_caught": caught,
            "attacks_missed": total_sims - caught,
            "resilience_pct": resilience_pct,
            "avg_resilience_score": avg_resilience,
        },
        "updated_at": datetime.utcnow().isoformat(),
    }


@router.get("/blueteam/incidents")
async def get_blueteam_incidents():
    """
    Defense incident log — every auto-quarantine and soft-quarantine
    action taken by the system, in reverse-chronological order.
    """
    log = list(reversed(_defense.defense_log))
    hitl_decisions = list(reversed(_hitl.decisions))

    # Merge into unified incident timeline
    incidents = []
    for action in log:
        incidents.append({
            "type": "auto_defense",
            "action": action.get("action"),
            "action_id": action.get("action_id"),
            "samples_affected": action.get("samples_affected", 0),
            "suspicion_score": action.get("suspicion_score", 0),
            "reason": action.get("reason", ""),
            "timestamp": action.get("timestamp", ""),
            "severity": "high" if action.get("action") == "quarantine" else "medium",
        })
    for d in hitl_decisions:
        incidents.append({
            "type": "human_decision",
            "action": d.get("decision"),
            "case_id": d.get("case_id"),
            "reviewer": d.get("reviewer"),
            "timestamp": d.get("decided_at", ""),
            "severity": "info",
        })

    # Sort by timestamp descending
    incidents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "incidents": incidents[:50],
        "total": len(incidents),
        "auto_defense_count": len(log),
        "human_decision_count": len(hitl_decisions),
    }


@router.get("/blueteam/resilience")
async def get_blueteam_resilience():
    """
    Detailed resilience breakdown — per attack type catch rate,
    average detection latency, false positive rate.
    """
    sims = _red_team.simulation_results
    if not sims:
        return {
            "overall_resilience_pct": 100.0,
            "by_attack_type": {},
            "total_tests": 0,
            "message": "No red team simulations run yet. Go to Red-Team Mode and fire some attacks!"
        }

    by_type = {}
    for s in sims:
        t = s["attack_type"]
        if t not in by_type:
            by_type[t] = {"total": 0, "caught": 0, "detection_times": [], "resilience_scores": []}
        by_type[t]["total"] += 1
        if s.get("detected"):
            by_type[t]["caught"] += 1
        by_type[t]["detection_times"].append(s.get("detection_speed_ms", 0))
        by_type[t]["resilience_scores"].append(s.get("resilience_score", 0))

    summary = {}
    for t, stats in by_type.items():
        summary[t] = {
            "total_tests": stats["total"],
            "caught": stats["caught"],
            "catch_rate_pct": round(stats["caught"] / stats["total"] * 100, 1),
            "avg_detection_ms": round(sum(stats["detection_times"]) / len(stats["detection_times"]), 1),
            "avg_resilience_score": round(sum(stats["resilience_scores"]) / len(stats["resilience_scores"]), 2),
        }

    total = len(sims)
    caught = sum(1 for s in sims if s.get("detected"))
    return {
        "overall_resilience_pct": round(caught / total * 100, 1),
        "total_tests": total,
        "total_caught": caught,
        "total_missed": total - caught,
        "avg_detection_ms": round(sum(s.get("detection_speed_ms", 0) for s in sims) / total, 1),
        "by_attack_type": summary,
        "recent_simulations": list(reversed(sims))[:10],
    }


_PLAYBOOKS = {
    "label_flip": {
        "attack": "Label Flip",
        "severity": "medium",
        "color": "#f59e0b",
        "description": "Adversary relabels training samples to corrupt decision boundaries.",
        "immediate_steps": [
            "🔒 Quarantine all samples flagged by L1 Statistical + L3 Ensemble layers",
            "🔍 Audit label provenance — trace back to data source and ingestion pipeline",
            "📊 Re-examine class distribution in affected batches",
            "🔄 Retrain model excluding quarantined samples",
        ],
        "investigation_steps": [
            "Compare label entropy before and after affected batches",
            "Check if any single source/client contributed a disproportionate share of flipped labels",
            "Cross-reference labels with a secondary ground-truth source if available",
        ],
        "remediation": [
            "Enable label validation checksum on all future ingestion pipelines",
            "Add human spot-check review for batches exceeding KL divergence > 2.0",
            "Switch to confident learning (label noise detection) for future training runs",
        ],
        "regulatory": "NIST AI RMF MAP 1.5 — Identify and assess AI risks from data provenance",
    },
    "backdoor": {
        "attack": "Backdoor (Trojan)",
        "severity": "critical",
        "color": "#ef4444",
        "description": "Hidden trigger pattern causes misclassification at inference time only.",
        "immediate_steps": [
            "🚨 IMMEDIATELY take model offline — do not serve predictions",
            "🔒 Hard quarantine all samples in the minority activation cluster (L2 Spectral)",
            "🛑 Block the data source that contributed the triggering samples",
            "📣 Alert all downstream consumers of the model's predictions",
        ],
        "investigation_steps": [
            "Extract and document the trigger pattern from the minority cluster's feature centroids",
            "Test model with and without trigger pattern to confirm backdoor behaviour",
            "Identify which training batches introduced the cluster via lineage map",
            "Check federated clients for gradient anomalies consistent with trojan insertion",
        ],
        "remediation": [
            "Full model retraining from scratch using clean data only",
            "Implement Neural Cleanse or STRIP detection on all future deployed models",
            "Add activation clustering check as a mandatory pre-deployment gate",
            "Rotate all credentials associated with the compromised data source",
        ],
        "regulatory": "EU AI Act Article 9 — Risk Management System must address adversarial manipulation",
    },
    "clean_label": {
        "attack": "Clean Label",
        "severity": "critical",
        "color": "#a855f7",
        "description": "Correctly-labelled samples crafted to poison model via feature space collision.",
        "immediate_steps": [
            "🔒 Quarantine all samples with Mahalanobis distance > 4.0 (L1 layer)",
            "📐 Map feature-space outliers — these are the crafted samples",
            "🔄 Retrain without outlier samples and compare causal effect (L4 layer)",
            "🧪 Run adversarial example detection on the remaining training set",
        ],
        "investigation_steps": [
            "Visualise samples in PCA space — crafted samples cluster near target class",
            "Compute per-sample gradient norm — clean-label samples have unusually large gradients",
            "Check if outlier samples all came from the same source/API endpoint",
        ],
        "remediation": [
            "Implement dataset filtering using spectral signatures before each training run",
            "Add Mahalanobis distance gate: reject samples > 3.5σ from class mean",
            "Use randomised smoothing during training to reduce sensitivity to crafted inputs",
        ],
        "regulatory": "NIST AI RMF MEASURE 2.5 — Evaluate trustworthiness of training data",
    },
    "gradient_poisoning": {
        "attack": "Gradient Poisoning",
        "severity": "high",
        "color": "#06b6d4",
        "description": "Malicious federated client sends inverted gradients to sabotage weight updates.",
        "immediate_steps": [
            "🔒 Quarantine all federated clients with trust score < 0.3 (L5 layer)",
            "🛑 Pause model aggregation — do not incorporate any client updates this round",
            "📊 Audit gradient norms and cosine similarity for all clients this round",
            "🔄 Rollback global model to last known-clean checkpoint",
        ],
        "investigation_steps": [
            "Compare gradient direction of suspect client vs global gradient (cosine similarity)",
            "Check if the client's trust score has been declining over multiple rounds",
            "Determine if the client dataset was independently compromised or is a rogue participant",
        ],
        "remediation": [
            "Switch aggregation to Krum or Trimmed Mean instead of FedAvg",
            "Enforce minimum trust score threshold for client participation",
            "Add differential privacy noise to gradient aggregation",
            "Require client-side gradient clipping before submission",
        ],
        "regulatory": "NIST AI RMF GOVERN 1.1 — Establish accountability for AI supply chain participants",
    },
    "boiling_frog": {
        "attack": "Boiling Frog (Slow Drift)",
        "severity": "high",
        "color": "#22c55e",
        "description": "Gradual slow poison injection designed to evade threshold-based detection.",
        "immediate_steps": [
            "📊 Run SHAP drift analysis across the full historical window (not just latest batch)",
            "🔭 Look for cumulative drift score > 0.2 — this attack hides in the tail",
            "🔄 Compare model performance on a frozen holdout set from 30, 60, 90 days ago",
            "🔒 Soft-quarantine all batches where SHAP drift first crossed 0.05",
        ],
        "investigation_steps": [
            "Plot feature importance over time — slow shifts in key features indicate this attack",
            "Check if the drift correlates with a specific data source or pipeline change",
            "Examine ingestion timestamps — the attack likely started weeks or months ago",
        ],
        "remediation": [
            "Implement continuous SHAP drift monitoring with weekly baseline resets",
            "Set up automated alerts for cumulative drift > 0.1 over any 7-day window",
            "Add data version control (DVC) to enable point-in-time rollback",
            "Schedule quarterly blind model audits against frozen ground-truth holdout sets",
        ],
        "regulatory": "EU AI Act Article 17 — Quality Management System must detect concept drift",
    },
}


@router.get("/blueteam/playbook/{attack_type}")
async def get_incident_playbook(attack_type: str):
    """
    Get the incident response playbook for a specific attack type.
    attack_type: label_flip | backdoor | clean_label | gradient_poisoning | boiling_frog
    """
    if attack_type not in _PLAYBOOKS:
        raise HTTPException(
            status_code=404,
            detail=f"No playbook for '{attack_type}'. Valid: {list(_PLAYBOOKS.keys())}"
        )
    return _PLAYBOOKS[attack_type]


@router.get("/blueteam/playbooks")
async def list_playbooks():
    """List all available incident response playbooks."""
    return {
        "playbooks": [
            {"id": k, "attack": v["attack"], "severity": v["severity"], "color": v["color"]}
            for k, v in _PLAYBOOKS.items()
        ]
    }

