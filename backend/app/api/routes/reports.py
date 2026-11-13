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
from app.core.security import require_admin, require_user

router = APIRouter()


# ── Pydantic request models (M7) ──────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    case_id: str
    decision: Literal["approve_quarantine", "mark_safe"]


# ── Helper: resolve the best available result ─────────────────────────────────

def _get_best_result(user: dict, prefer: str = "auto") -> dict:
    """Resolve analysis without inventing synthetic data.

    prefer: auto | real | upload | real_dataset | demo
    Raises HTTPException 404 with a clear empty-state message if nothing exists.
    """
    result = deps.resolve_latest_result(user["id"], prefer=prefer)
    if result is not None:
        return result

    messages = {
        "demo": "No demo analysis found. POST /api/v1/demo/run to generate a synthetic scenario.",
        "upload": "No upload analysis found. POST /api/v1/analyze/upload with a CSV first.",
        "real_dataset": "No real-dataset analysis found. POST /api/v1/datasets/real/{name}/analyze first.",
        "auto": (
            "No analysis results found. Upload a CSV via /analyze/upload "
            "or analyse a real dataset via /datasets/real/{name}/analyze. "
            "Synthetic demo data is available only via explicit POST /demo/run."
        ),
        "real": (
            "No real analysis found. Upload a CSV or analyse a real dataset. "
            "Demo is available only via the Demo source or POST /demo/run."
        ),
    }
    raise HTTPException(status_code=404, detail=messages.get(prefer, messages["auto"]))


# ── HISTORY (persisted results) ───────────────────────────────────────────────


@router.get("/history")
async def get_analysis_history(
    source: Optional[str] = Query(None, pattern="^(demo|upload|real_dataset|model_scan)$"),
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(require_user),
):
    """The caller's history (every user's for an admin): analysis_results + model_scans."""
    owner = deps.owner_scope(user)
    rows = deps.db.get_history(owner, source=source, limit=limit)
    model_rows = []
    if source in (None, "model_scan"):
        model_rows = deps.db.get_model_scan_history(owner, limit=limit)
        for r in model_rows:
            r["source"] = "model_scan"
            r["filename"] = r.get("model_filename")
            r["n_samples"] = r.get("n_samples")

    combined = rows + model_rows
    combined.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    combined = combined[:limit]

    stats = deps.db.get_stats(owner)
    return {"results": combined, "stats": stats}


@router.get("/audit/events")
async def get_my_audit_events(
    limit: int = Query(50, ge=1, le=200),
    user: dict = Depends(require_user),
):
    """Return recent audit activity (all system activity for admins, user's own activity for regular users)."""
    actor_id = None if user.get("role") == "admin" else user["id"]
    return {"events": deps.db.get_audit_events(actor_id=actor_id, limit=limit)}



@router.get("/history/{result_id}")
async def get_historical_result(result_id: str, user: dict = Depends(require_user)):
    owner = deps.owner_scope(user)
    result = deps.db.get_result(result_id, owner)
    if not result:
        result = deps.db.get_model_scan(result_id, owner)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found in database.")
    return result


@router.delete("/history/{result_id}")
async def delete_historical_result(result_id: str, admin: dict = Depends(require_admin)):
    """Delete a historical dataset or model scan analysis record (Admin only)."""
    success = deps.db.delete_result(result_id, deps.db.ALL_OWNERS)
    if not success:
        raise HTTPException(status_code=404, detail="Result record not found.")
    deps.db.log_audit_event(
        admin["id"],
        "analysis.deleted",
        "result",
        result_id,
        {"deleted_by": admin["name"]}
    )
    return {"message": "Historical result record purged successfully."}



# ── FORENSICS ─────────────────────────────────────────────────────────────────


@router.get("/forensics/latest")
async def get_latest_forensics(source: str = Query("auto", description="auto|demo|upload"), user: dict = Depends(require_user)):
    r = _get_best_result(user, source)
    return {
        "attack_classification": r.get("attack_classification"),
        "injection_pattern": r.get("injection_pattern"),
        "blast_radius": r.get("blast_radius"),
        "source": r.get("source", "demo"),
        "verdict": r.get("verdict"),
        "overall_suspicion_score": r.get("overall_suspicion_score"),
        "dataset_info": r.get("dataset_info"),
    }


@router.get("/forensics/narrative")
async def get_attack_narrative(source: str = Query("auto", description="auto|demo|upload"), user: dict = Depends(require_user)):
    r = _get_best_result(user, source)
    pattern = r.get("injection_pattern") or {}
    return {"narrative": pattern.get("narrative", "No narrative available."), "source": r.get("source", "demo")}


@router.get("/forensics/timeline")
async def get_attack_timeline():
    data = deps.get_demo_data()
    return {"timeline": data["timeline"]}


@router.get("/blast-radius/latest")
async def get_blast_radius(source: str = Query("auto", description="auto|demo|upload"), user: dict = Depends(require_user)):
    r = _get_best_result(user, source)
    return r.get("blast_radius") or {}


# ── DEFENSE ───────────────────────────────────────────────────────────────────


@router.post("/defense/quarantine")
async def trigger_quarantine(user: dict = Depends(require_admin)):  # Admin-only quarantine trigger
    """Quarantine rows the latest dataset analysis flagged (capped at 5% of the batch)."""
    r = _get_best_result(user, "auto")
    info = r.get("dataset_info") or {}
    layer3 = (r.get("layer_results") or {}).get("row_flags") or {}
    split = info.get("reference_split")
    flagged = layer3.get("flagged_indices_incoming")
    if split is None or flagged is None:
        raise HTTPException(
            status_code=409,
            detail="Manual quarantine needs a CSV or real-dataset analysis; the latest result has no row-level flags.",
        )
    flagged_set = set(flagged)
    scope = info.get("dataset_id", "")
    rows = [
        {"id": f"row_{split + i}", "poison_status": "suspected" if i in flagged_set else "unknown"}
        for i in range(r.get("n_samples", 0))
    ]
    action = deps.defense._quarantine(rows, r["overall_suspicion_score"], scope=scope, owner_id=user["id"])
    deps.db.log_audit_event(
        user["id"], "defense.quarantine", "dataset", scope,
        {"samples_affected": action["samples_affected"], "action_id": action["action_id"]},
    )
    return action


@router.get("/defense/status")
async def get_defense_status(user: dict = Depends(require_user)):
    return deps.defense.get_status(user["id"])


@router.get("/defense/hitl/pending")
async def get_pending_reviews(user: dict = Depends(require_user)):
    return {"cases": deps.hitl.get_pending(user["id"])}


@router.post("/defense/hitl/decide")
async def submit_review_decision(
    body: HITLDecisionRequest,  # M7: Pydantic instead of raw request.json()
    user: dict = Depends(require_user),  # M2: auth required
):
    # The reviewer is the authenticated user, never a client-supplied name.
    record = deps.hitl.decide(body.case_id, body.decision, reviewer=user["name"], owner_id=user["id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Case not found or already resolved.")
    case = record.pop("case")
    if body.decision == "approve_quarantine":
        record["defense_action"] = deps.defense.quarantine_ids(
            case.get("sample_ids", []),
            case.get("scope", ""),
            reason=f"HITL case {body.case_id} approved by {user['name']}",
            owner_id=user["id"],
        )
    deps.db.log_audit_event(
        user["id"], "hitl.decided", "hitl_case", body.case_id, {"decision": body.decision}
    )
    return record


# ── REPORTS ───────────────────────────────────────────────────────────────────


@router.post("/reports/generate")
async def generate_report(source: str = Query("auto", description="auto|demo|upload|real_dataset"), user: dict = Depends(require_user)):
    """
    Generate a forensic evidence report.
    source=auto  → use the most recent result (upload wins over demo)
    source=demo  → use the latest demo run result
    source=upload → use the latest uploaded CSV result
    """
    prefer = "real" if source == "auto" else source
    r = _get_best_result(user, prefer)

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
            "blast_radius_summary": {   # flagged rows only: no model lineage is supplied, so no downstream impact
                "flagged_rows": (r.get("blast_radius") or {}).get("n_poisoned_samples", 0),
            },
            "attack_profile": (r.get("layer_results") or {}).get("attack_profile") or r.get("attack_profile"),
        },
        "evidence_bundle": r.get("layer_results"),
        "layer_scores": r.get("layer_scores"),
        "attack_narrative": (r.get("injection_pattern") or {}).get("narrative", ""),
        "defense_actions": deps.defense.defense_log(user["id"]),
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
async def get_blueteam_status(user: dict = Depends(require_user)):
    defense_status = deps.defense.get_status(user["id"])
    pending_cases = deps.hitl.get_pending(user["id"])
    threat_level = "NOMINAL"
    verdict = "CLEAN"
    suspicion = 0.0
    result = deps.resolve_latest_result(user["id"], prefer="auto")
    if result:
        verdict = result.get("verdict", "CLEAN")
        suspicion = result.get("overall_suspicion_score", 0)
        threat_level = {"CONFIRMED_POISONED": "CRITICAL", "SUSPICIOUS": "ELEVATED", "LOW_RISK": "GUARDED"}.get(verdict, "NOMINAL")

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
        "updated_at": datetime.now(timezone.utc).isoformat(),  # L1
    }


@router.get("/blueteam/incidents")
async def get_blueteam_incidents(user: dict = Depends(require_user)):
    log = list(reversed(deps.defense.defense_log(user["id"])))
    hitl_decisions = list(reversed(deps.hitl.decisions(user["id"])))

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


# One playbook per attack family the detector names (forensics.engine.FAMILIES). Every step uses what the product
# actually produces: the batch verdict, the label / shift / residual checks, flagged rows, cell scores, the attack
# profile, quarantine and human review. Framework references are pointers for your own process, not compliance claims.
_PLAYBOOKS = {
    "label_flip": {
        "attack": "Label Flipping",
        "severity": "medium",
        "color": "#f59e0b",
        "description": "Some rows carry the wrong label, so a model trained on them learns a distorted decision boundary.",
        "immediate_steps": [
            "Do not train on this batch until it is reviewed.",
            "Open the flagged rows: their label cell should carry the highest cell score.",
            "Approve quarantine of the flagged rows in the review queue, or mark them safe if the labels are right.",
        ],
        "investigation_steps": [
            "If the attack profile says targeted, check which claimed class the flagged rows share and who supplied them.",
            "Compare the flagged labels against a second source (original records, a second annotator).",
            "Look for a common source, annotator or upload time among the flagged rows.",
        ],
        "remediation": [
            "Correct or drop the flipped labels, then re-run the analysis against the same trusted reference.",
            "Require review or double annotation for the sources that produced the flagged rows.",
            "Keep a trusted reference file and check every new batch against it before training.",
        ],
        "regulatory": "Reference: NIST AI RMF MAP 1.5 (data provenance risks)",
    },
    "backdoor": {
        "attack": "Backdoor Trigger",
        "severity": "critical",
        "color": "#ef4444",
        "description": "Rows carry a trigger pattern in a few columns plus a target label, so a model learns to misclassify whenever the trigger appears.",
        "immediate_steps": [
            "Do not train on this batch; if a model was already trained on it, stop using it until it is checked.",
            "Open the flagged rows: the trigger columns carry the highest cell scores, usually the same few columns in every row.",
            "Quarantine the flagged rows and block the source that supplied them.",
        ],
        "investigation_steps": [
            "Note the trigger columns and values; out-of-range values mean the trigger was not even disguised.",
            "Check whether the flagged rows share one claimed label (the backdoor's target class).",
            "If a model was trained on the data, test it on clean rows with the trigger values pasted in: a backdoored model flips its prediction.",
        ],
        "remediation": [
            "Retrain from a clean reference without the flagged rows.",
            "Add range checks on the trigger columns at ingestion.",
            "Re-run the analysis on every new batch from that source before it is used.",
        ],
        "regulatory": "Reference: EU AI Act Article 15 (robustness against attempts to alter use or performance)",
    },
    "clean_label": {
        "attack": "Feature Collision (Clean-Label)",
        "severity": "critical",
        "color": "#a855f7",
        "description": "Rows with correct labels were moved close together in feature space to poison the model without any wrong label.",
        "immediate_steps": [
            "Do not train on this batch until it is reviewed.",
            "Look at the flagged rows as whole rows: their values are pulled towards one point, so no single cell stands out.",
            "Quarantine the flagged rows.",
        ],
        "investigation_steps": [
            "Check how similar the flagged rows are to each other (the attack profile's tightness feature is low).",
            "Find the row they cluster around: it is usually the victim the attacker wants misclassified.",
            "Check whether the flagged rows came from one source.",
        ],
        "remediation": [
            "Retrain without the flagged rows and re-check the victim row's prediction.",
            "Cap how many near-duplicate rows one source can contribute.",
            "Re-run the analysis on every new batch before training.",
        ],
        "regulatory": "Reference: NIST AI RMF MEASURE 2.5 (validity and reliability of training data)",
    },
    "covariate_shift": {
        "attack": "Covariate Shift / Corrupted Source",
        "severity": "medium",
        "color": "#06b6d4",
        "description": "Some rows come from a shifted or corrupted source: several columns move together, so the rows no longer fit the reference.",
        "immediate_steps": [
            "Hold the batch; check whether the shift or residual check raised the alarm.",
            "Open the flagged rows and note which columns carry unusual values.",
            "Soft-quarantine the flagged rows until the source is understood.",
        ],
        "investigation_steps": [
            "Ask whether something legitimate changed (a new sensor, unit or region) before treating it as an attack.",
            "Compare the flagged rows' columns with the same columns in the reference.",
            "Trace the flagged rows to their source system or pipeline step.",
        ],
        "remediation": [
            "Fix or drop the corrupted source, or add the new legitimate data to the trusted reference after review.",
            "Add schema and range checks for the affected columns at ingestion.",
            "Re-run the analysis against the updated reference.",
        ],
        "regulatory": "Reference: EU AI Act Article 10 (data governance and quality)",
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
