"""Forensics, defense, blue-team, history, and report routes.

Changes:
  M2 — RBAC: quarantine and HITL decide now require authentication
  M7 — Pydantic request body for /defense/hitl/decide
  L1 — datetime.utcnow() replaced with timezone-aware datetime
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.api import dependencies as deps
from app.core.security import require_user

router = APIRouter()


# ── Pydantic request models (M7) ──────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    case_id: str
    decision: Literal["approve_quarantine", "mark_safe"]
    reviewer: str = "analyst"


# ── Helper: resolve the best available result ─────────────────────────────────

def _get_best_result(prefer: str = "auto") -> dict:
    """
    Return the most recent analysis result.
    prefer: 'demo'   → only demo cache
            'upload' → only upload cache
            'auto'   → whichever is freshest (upload wins tie)
    Raises HTTPException 404 if nothing available.
    """
    demo = deps.demo_result_cache.get("latest")
    upload = deps.upload_result_cache.get("latest")

    if prefer == "demo":
        if demo:
            return demo
        raise HTTPException(status_code=404, detail="Run /demo/run first.")

    if prefer == "upload":
        if upload:
            return upload
        raise HTTPException(status_code=404, detail="Upload a CSV via /analyze/upload first.")

    # auto: prefer upload if it's available (it's the user's own data)
    if upload:
        return upload
    if demo:
        return demo

    # DB fallback
    db_result = deps.db.get_latest()
    if db_result:
        return db_result

    raise HTTPException(
        status_code=404,
        detail="No analysis results found. Run /demo/run or upload a CSV first.",
    )


# ── HISTORY (persisted results) ───────────────────────────────────────────────


@router.get("/history")
async def get_analysis_history(
    source: Optional[str] = Query(None, pattern="^(demo|upload|real_dataset|model_scan)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Return combined history from analysis_results + model_scans tables."""
    rows = deps.db.get_history(source=source, limit=limit)
    model_rows = []
    if source in (None, "model_scan"):
        model_rows = deps.db.get_model_scan_history(limit=limit)
        for r in model_rows:
            r["source"] = "model_scan"
            r["filename"] = r.get("model_filename")
            r["n_samples"] = r.get("n_samples")

    combined = rows + model_rows
    combined.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    combined = combined[:limit]

    stats = deps.db.get_stats()
    return {"results": combined, "stats": stats}


@router.get("/audit/events")
async def get_my_audit_events(
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(require_user),
):
    """Return the authenticated analyst's own recent audit activity."""
    return {"events": deps.db.get_audit_events(actor_id=user["id"], limit=limit)}


@router.get("/history/{result_id}")
async def get_historical_result(result_id: str):
    result = deps.db.get_result(result_id)
    if not result:
        result = deps.db.get_model_scan(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found in database.")
    return result


# ── FORENSICS ─────────────────────────────────────────────────────────────────


@router.get("/forensics/latest")
async def get_latest_forensics(source: str = Query("auto", description="auto|demo|upload")):
    r = _get_best_result(source)
    return {
        "attack_classification": r.get("attack_classification"),
        "injection_pattern": r.get("injection_pattern"),
        "sophistication": r.get("sophistication"),
        "blast_radius": r.get("blast_radius"),
        "counterfactual": r.get("counterfactual"),
        "source": r.get("source", "demo"),
        "verdict": r.get("verdict"),
        "overall_suspicion_score": r.get("overall_suspicion_score"),
        "dataset_info": r.get("dataset_info"),
    }


@router.get("/forensics/narrative")
async def get_attack_narrative(source: str = Query("auto", description="auto|demo|upload")):
    r = _get_best_result(source)
    pattern = r.get("injection_pattern") or {}
    return {"narrative": pattern.get("narrative", "No narrative available."), "source": r.get("source", "demo")}


@router.get("/forensics/timeline")
async def get_attack_timeline():
    data = deps.get_demo_data()
    return {"timeline": data["timeline"]}


@router.get("/blast-radius/latest")
async def get_blast_radius(source: str = Query("auto", description="auto|demo|upload")):
    r = _get_best_result(source)
    return r.get("blast_radius") or {}


# ── DEFENSE ───────────────────────────────────────────────────────────────────


@router.post("/defense/quarantine")
async def trigger_quarantine(user: dict = Depends(require_user)):  # M2: auth required
    r = _get_best_result("auto")
    data = deps.get_demo_data()
    action = deps.defense._quarantine(data["samples"][:50], r["overall_suspicion_score"])
    return action


@router.get("/defense/status")
async def get_defense_status():
    return deps.defense.get_status()


@router.get("/defense/hitl/pending")
async def get_pending_reviews():
    return {"cases": deps.hitl.get_pending()}


@router.post("/defense/hitl/decide")
async def submit_review_decision(
    body: HITLDecisionRequest,  # M7: Pydantic instead of raw request.json()
    user: dict = Depends(require_user),  # M2: auth required
):
    result = deps.hitl.decide(body.case_id, body.decision, body.reviewer)
    return result


# ── REPORTS ───────────────────────────────────────────────────────────────────


@router.post("/reports/generate")
async def generate_report(source: str = Query("auto", description="auto|demo|upload")):
    """
    Generate a forensic evidence report.
    source=auto  → use the most recent result (upload wins over demo)
    source=demo  → use the latest demo run result
    source=upload → use the latest uploaded CSV result
    """
    r = _get_best_result(source)

    report = {
        "report_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),  # L1
        "title": "AI Poisoning Analyst Evidence Summary",
        "platform": "AI Trust Forensics Platform v2.2",
        "data_source": r.get("source", "demo"),
        "dataset_info": r.get("dataset_info"),
        "executive_summary": {
            "verdict": r.get("verdict"),
            "attack_type": (r.get("attack_classification") or {}).get("attack_type", "unknown"),
            "confidence": (r.get("attack_classification") or {}).get("confidence", 0),
            "causal_effect": ((r.get("layer_results") or {}).get("layer4_causal") or {}).get("causal_effect", 0),
            "sophistication_score": (r.get("sophistication") or {}).get("sophistication_score", 0),
            "blast_radius_summary": {
                "batches": (r.get("blast_radius") or {}).get("n_batches_affected", 0),
                "models": (r.get("blast_radius") or {}).get("n_models_affected", 0),
                "impact_pct": (r.get("blast_radius") or {}).get("prediction_impact_pct", 0),
            },
        },
        "evidence_bundle": r.get("layer_results"),
        "layer_scores": r.get("layer_scores"),
        "attack_narrative": (r.get("injection_pattern") or {}).get("narrative", ""),
        "defense_actions": deps.defense.defense_log,
        "limitations": {
            "status": "experimental_analyst_support",
            "notice": (
                "This summary contains heuristic risk signals and proxy-model analysis. "
                "It is not a certification, legal evidence, attack attribution, or proof "
                "that a dataset or model was poisoned."
            ),
            "framework_references": (
                "NIST AI RMF and EU AI Act references are implementation prompts only; "
                "they do not establish compliance."
            ),
            "report_reference": f"summary_{uuid.uuid4().hex}",
        },
    }
    return report


# ── BLUE TEAM SOC ─────────────────────────────────────────────────────────────


@router.get("/blueteam/status")
async def get_blueteam_status():
    defense_status = deps.defense.get_status()
    pending_cases = deps.hitl.get_pending()
    sims = deps.red_team.simulation_results

    threat_level = "NOMINAL"
    verdict = "CLEAN"
    suspicion = 0.0
    result = deps.demo_result_cache.get("latest") or deps.upload_result_cache.get("latest")
    if result:
        verdict = result.get("verdict", "CLEAN")
        suspicion = result.get("overall_suspicion_score", 0)
        if suspicion > 0.65:
            threat_level = "CRITICAL"
        elif suspicion > 0.35:
            threat_level = "ELEVATED"
        elif suspicion > 0.15:
            threat_level = "GUARDED"

    total_sims = len(sims)
    caught = sum(1 for s in sims if s.get("detected", False))
    resilience_pct = round((caught / total_sims * 100) if total_sims > 0 else 100.0, 1)
    avg_resilience = round(
        sum(s.get("resilience_score", 0) for s in sims) / total_sims if total_sims > 0 else 10.0, 1
    )

    return {
        "threat_level": threat_level,
        "current_verdict": verdict,
        "suspicion_score": round(float(suspicion), 4),
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
        "updated_at": datetime.now(timezone.utc).isoformat(),  # L1
    }


@router.get("/blueteam/incidents")
async def get_blueteam_incidents():
    log = list(reversed(deps.defense.defense_log))
    hitl_decisions = list(reversed(deps.hitl.decisions))

    incidents = []
    for action in log:
        incidents.append(
            {
                "type": "auto_defense",
                "action": action.get("action"),
                "action_id": action.get("action_id"),
                "samples_affected": action.get("samples_affected", 0),
                "suspicion_score": action.get("suspicion_score", 0),
                "reason": action.get("reason", ""),
                "timestamp": action.get("timestamp", ""),
                "severity": "high" if action.get("action") == "quarantine" else "medium",
            }
        )
    for d in hitl_decisions:
        incidents.append(
            {
                "type": "human_decision",
                "action": d.get("decision"),
                "case_id": d.get("case_id"),
                "reviewer": d.get("reviewer"),
                "timestamp": d.get("decided_at", ""),
                "severity": "info",
            }
        )

    incidents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "incidents": incidents[:50],
        "total": len(incidents),
        "auto_defense_count": len(log),
        "human_decision_count": len(hitl_decisions),
    }


@router.get("/blueteam/resilience")
async def get_blueteam_resilience():
    sims = deps.red_team.simulation_results
    if not sims:
        return {
            "overall_resilience_pct": 100.0,
            "by_attack_type": {},
            "total_tests": 0,
            "message": "No red team simulations run yet. Go to Red-Team Mode and fire some attacks!",
        }

    by_type: dict = {}
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
    if attack_type not in _PLAYBOOKS:
        raise HTTPException(
            status_code=404,
            detail=f"No playbook for '{attack_type}'. Valid: {list(_PLAYBOOKS.keys())}",
        )
    return _PLAYBOOKS[attack_type]


@router.get("/blueteam/playbooks")
async def list_playbooks():
    return {
        "playbooks": [
            {"id": k, "attack": v["attack"], "severity": v["severity"], "color": v["color"]}
            for k, v in _PLAYBOOKS.items()
        ]
    }
