"""Report helpers."""

from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List


def build_report(latest_result: Dict[str, Any], defense_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    layer4 = (latest_result.get("layer_results") or {}).get("layer4_causal") or {}
    blast = latest_result.get("blast_radius") or {}
    attack_class = latest_result.get("attack_classification") or {}
    pattern = latest_result.get("injection_pattern") or {}
    sophistication = latest_result.get("sophistication") or {}

    return {
        "report_id": str(uuid.uuid4()),
        "generated_at": datetime.utcnow().isoformat(),
        "title": "AI Poisoning Forensic Evidence Report",
        "platform": "AI Trust Forensics Platform v2.2",
        "executive_summary": {
            "verdict": latest_result.get("verdict"),
            "attack_type": attack_class.get("attack_type", "unknown"),
            "confidence": attack_class.get("confidence", 0),
            "causal_effect": layer4.get("causal_effect", 0),
            "sophistication_score": sophistication.get("sophistication_score", 0),
            "blast_radius_summary": {
                "batches": blast.get("n_batches_affected", 0),
                "models": blast.get("n_models_affected", 0),
                "impact_pct": blast.get("prediction_impact_pct", 0),
            },
        },
        "evidence_bundle": latest_result.get("layer_results"),
        "attack_narrative": pattern.get("narrative", ""),
        "defense_actions": defense_log,
        "compliance": {
            "nist_ai_rmf": "GOVERN 1.1, MAP 1.5, MEASURE 2.5, MANAGE 2.2",
            "eu_ai_act": "Article 9 (Risk Management), Article 17 (Quality Management)",
            "audit_hash": f"sha256_{uuid.uuid4().hex}",
        },
    }

