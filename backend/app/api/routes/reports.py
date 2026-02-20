"""Forensics, defense, blue-team, red-team, history, and report routes."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from fastapi import APIRouter, HTTPException, Request

from app.api import dependencies as deps

router = APIRouter()


# ── HISTORY (persisted results) ───────────────────────────────────────────────


@router.get("/history")
async def get_analysis_history(source: Optional[str] = None, limit: int = 20):
    rows = deps.db.get_history(source=source, limit=limit)
    stats = deps.db.get_stats()
    return {"results": rows, "stats": stats}


@router.get("/history/{result_id}")
async def get_historical_result(result_id: str):
    result = deps.db.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found in database.")
    return result


# ── FORENSICS ─────────────────────────────────────────────────────────────────


@router.get("/forensics/latest")
async def get_latest_forensics():
    if "latest" not in deps.demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    r = deps.demo_result_cache["latest"]
    return {
        "attack_classification": r.get("attack_classification"),
        "injection_pattern": r.get("injection_pattern"),
        "sophistication": r.get("sophistication"),
        "blast_radius": r.get("blast_radius"),
        "counterfactual": r.get("counterfactual"),
    }


@router.get("/forensics/narrative")
async def get_attack_narrative():
    if "latest" not in deps.demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    pattern = deps.demo_result_cache["latest"].get("injection_pattern", {})
    return {"narrative": pattern.get("narrative", "No narrative available.")}


@router.get("/forensics/timeline")
async def get_attack_timeline():
    data = deps.get_demo_data()
    return {"timeline": data["timeline"]}


@router.get("/blast-radius/latest")
async def get_blast_radius():
    if "latest" not in deps.demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    return deps.demo_result_cache["latest"].get("blast_radius", {})


# ── DEFENSE ───────────────────────────────────────────────────────────────────


@router.post("/defense/quarantine")
async def trigger_quarantine():
    if "latest" not in deps.demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")
    r = deps.demo_result_cache["latest"]
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
async def submit_review_decision(request: Request):
    body = await request.json()
    case_id = body.get("case_id")
    decision = body.get("decision")
    reviewer = body.get("reviewer", "analyst")

    if not case_id or not decision:
        raise HTTPException(status_code=400, detail="case_id and decision required")

    result = deps.hitl.decide(case_id, decision, reviewer)
    return result


# ── RED TEAM ──────────────────────────────────────────────────────────────────


@router.post("/redteam/simulate")
async def run_red_team(request: Request):
    body = await request.json()
    attack_type = body.get("attack_type", "label_flip")

    valid_attacks = ["label_flip", "backdoor", "boiling_frog", "clean_label", "gradient_poisoning"]
    if attack_type not in valid_attacks:
        raise HTTPException(status_code=400, detail=f"attack_type must be one of {valid_attacks}")

    data = deps.get_demo_data()
    deps.red_team.pipeline = deps.pipeline
    result = deps.red_team.run_simulation(attack_type, data["samples"][:200])
    return result


@router.get("/redteam/history")
async def get_red_team_history():
    return {"simulations": deps.red_team.simulation_results}


# ── REPORTS ───────────────────────────────────────────────────────────────────


@router.post("/reports/generate")
async def generate_report():
    if "latest" not in deps.demo_result_cache:
        raise HTTPException(status_code=404, detail="Run /demo/run first.")

    r = deps.demo_result_cache["latest"]
    report = {
        "report_id": str(uuid.uuid4()),
        "generated_at": datetime.utcnow().isoformat(),
        "title": "AI Poisoning Forensic Evidence Report",
        "platform": "AI Trust Forensics Platform v2.2",
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
        "attack_narrative": (r.get("injection_pattern") or {}).get("narrative", ""),
        "defense_actions": deps.defense.defense_log,
        "compliance": {
            "nist_ai_rmf": "GOVERN 1.1, MAP 1.5, MEASURE 2.5, MANAGE 2.2",
            "eu_ai_act": "Article 9 (Risk Management), Article 17 (Quality Management)",
            "audit_hash": f"sha256_{uuid.uuid4().hex}",
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
    if "latest" in deps.demo_result_cache:
        verdict = deps.demo_result_cache["latest"].get("verdict", "CLEAN")
        suspicion = deps.demo_result_cache["latest"].get("overall_suspicion_score", 0)
        if suspicion > 0.65:
            threat_level = "CRITICAL"
        elif suspicion > 0.35:
            threat_level = "ELEVATED"
        elif suspicion > 0.15:
            threat_level = "GUARDED"
        else:
            threat_level = "NOMINAL"

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
        "updated_at": datetime.utcnow().isoformat(),
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
    if attack_type not in _PLAYBOOKS:
        raise HTTPException(
            status_code=404, detail=f"No playbook for '{attack_type}'. Valid: {list(_PLAYBOOKS.keys())}"
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

