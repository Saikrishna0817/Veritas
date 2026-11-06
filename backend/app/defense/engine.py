"""Defense Engine: Auto-Defense + Human-in-the-Loop + Red-Team

All state lives in SQLite via app.models.database, so defense actions, the HITL
queue, decisions and red-team history survive restarts and the singletons in
app.api.dependencies hold no per-request state.

Sample ids such as ``row_5`` repeat across datasets, so quarantine bookkeeping is
keyed by ``scope`` (the dataset id) as well as sample id.
"""
import copy
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.models import database as db

QUARANTINE_ACTIONS = ("quarantine", "soft_quarantine")


class StabilityAwareAutoDefense:
    """
    Stability-aware auto-defense that prevents self-damage.
    Max 5% quarantine per epoch, rate-limited actions (per dataset scope).
    """

    MAX_QUARANTINE_RATIO = 0.05
    QUARANTINE_COOLDOWN_SECS = 30

    def __init__(self):
        self.mode = "active"  # active | observe_only

    @property
    def defense_log(self) -> List[Dict[str, Any]]:
        """Fetch defense log from SQLite DB."""
        return db.get_defense_actions(limit=100)

    @property
    def quarantined_ids(self) -> set:
        """Hard-quarantined ``scope:sample_id`` keys, recovered from persisted actions."""
        return {
            f"{a.get('scope', '')}:{sid}"
            for a in db.get_defense_actions(limit=200)
            if a.get("action") == "quarantine"
            for sid in a.get("sample_ids", [])
        }

    def decide_action(self, samples: List[Dict], suspicion_score: float,
                      verdict: str, scope: str = "") -> Dict[str, Any]:
        """Decide defense action based on suspicion score."""
        if self.mode == "observe_only":
            return {"action": "observe", "reason": "observe_only_mode", "samples_affected": 0}

        # Rate limiting: only repeated actions on the same dataset are throttled.
        now = datetime.now(timezone.utc)
        last = next((a for a in self.defense_log
                     if a.get("scope", "") == scope and a.get("action") in QUARANTINE_ACTIONS), None)
        if last:
            elapsed = (now - datetime.fromisoformat(last["timestamp"])).total_seconds()
            if elapsed < self.QUARANTINE_COOLDOWN_SECS:
                return {"action": "cooldown", "reason": "rate_limited", "samples_affected": 0,
                        "cooldown_remaining_secs": round(self.QUARANTINE_COOLDOWN_SECS - elapsed, 1)}

        if verdict == "CONFIRMED_POISONED" and suspicion_score > 0.7:
            return self._quarantine(samples, suspicion_score, scope)
        elif verdict == "SUSPICIOUS" and suspicion_score > 0.5:
            return self._soft_quarantine(samples, suspicion_score, scope)
        else:
            return {"action": "monitor", "reason": "below_threshold", "samples_affected": 0}

    def _quarantine(self, samples: List[Dict], score: float, scope: str = "") -> Dict[str, Any]:
        """Hard quarantine — remove from training."""
        already = self.quarantined_ids
        candidates = [s for s in samples
                      if s.get("poison_status") in ("confirmed", "suspected")
                      and f"{scope}:{s['id']}" not in already]

        max_quarantine = max(1, int(len(samples) * self.MAX_QUARANTINE_RATIO))
        to_quarantine = candidates[:max_quarantine]
        return self._record_quarantine(
            [s["id"] for s in to_quarantine], score, scope,
            reason=f"Suspicion score {score:.2f} > 0.70 threshold",
        )

    def quarantine_ids(self, sample_ids: List[str], scope: str, reason: str) -> Dict[str, Any]:
        """Quarantine explicit ids (used when an analyst approves a HITL case)."""
        already = self.quarantined_ids
        fresh = [i for i in sample_ids if f"{scope}:{i}" not in already]
        return self._record_quarantine(fresh, None, scope, reason=reason)

    def _record_quarantine(self, ids: List[str], score, scope: str, reason: str) -> Dict[str, Any]:
        action = {
            "action": "quarantine",
            "action_id": str(uuid.uuid4()),
            "scope": scope,
            "samples_affected": len(ids),
            "sample_ids": ids,
            "suspicion_score": score,
            "model_stable": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        db.save_defense_action(action)
        return action

    def _soft_quarantine(self, samples: List[Dict], score: float, scope: str = "") -> Dict[str, Any]:
        """Soft quarantine — down-weight in training."""
        candidates = [s for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        n = min(len(candidates), max(1, int(len(samples) * self.MAX_QUARANTINE_RATIO * 0.5)))

        action = {
            "action": "soft_quarantine",
            "action_id": str(uuid.uuid4()),
            "scope": scope,
            "samples_affected": n,
            "weight_factor": 0.1,
            "suspicion_score": score,
            "model_stable": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": f"Suspicion score {score:.2f} in borderline range",
        }
        db.save_defense_action(action)
        return action

    def get_status(self) -> Dict[str, Any]:
        log = self.defense_log
        return {
            "mode": self.mode,
            "total_quarantined": len(self.quarantined_ids),
            "n_defense_actions": len(log),
            "last_action": log[0] if log else None
        }


class HumanInTheLoopQueue:
    """Manages borderline cases requiring human review with SQLite persistence."""

    MAX_CASE_IDS = 500

    def enqueue(self, samples: List[Dict], evidence: Dict,
                suspicion_score: float, scope: str = "") -> Dict[str, Any]:
        """Add a case to the human review queue."""
        now = datetime.now(timezone.utc)
        flagged = [s["id"] for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        case = {
            "case_id": str(uuid.uuid4()),
            "scope": scope,
            "suspicion_score": suspicion_score,
            "n_samples": len(samples),
            # Rows an approval would quarantine; fall back to a sample if nothing is flagged.
            "sample_ids": (flagged or [s["id"] for s in samples[:5]])[: self.MAX_CASE_IDS],
            "evidence_summary": {
                "kl_divergence": evidence.get("layer1_statistical", {}).get("kl_divergence", 0),
                "causal_effect": evidence.get("layer4_causal", {}).get("causal_effect", 0),
                "attack_type": evidence.get("attack_classification", {}).get("attack_type", "unknown")
            },
            "status": "pending",
            "created_at": now.isoformat(),
            "deadline": now.replace(hour=23, minute=59).isoformat(),
        }
        db.save_hitl_case(case)
        return case

    def decide(self, case_id: str, decision: str, reviewer: str = "analyst") -> Dict[str, Any] | None:
        """Record a human decision. Returns None if the case is unknown or already resolved."""
        case = next((c for c in db.get_pending_hitl_cases() if c["case_id"] == case_id), None)
        if not case:
            return None

        decision_record = {
            "case_id": case_id,
            "decision": decision,  # "approve_quarantine" | "mark_safe"
            "reviewer": reviewer,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "case": case,
        }
        db.save_hitl_decision(decision_record)
        return decision_record

    def get_pending(self) -> List[Dict]:
        return db.get_pending_hitl_cases()

    @property
    def decisions(self) -> List[Dict]:
        return db.get_hitl_decisions()


class RedTeamSimulator:
    """Injects synthetic attacks and measures detection resilience."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    @property
    def simulation_results(self) -> List[Dict[str, Any]]:
        return db.get_redteam_simulations()

    def run_simulation(self, attack_type: str, samples: List[Dict]) -> Dict[str, Any]:
        """Inject a synthetic attack and measure detection performance."""
        from app.demo.data_generator import (
            inject_label_flip_attack, inject_backdoor_attack, inject_boiling_frog_attack,
            inject_clean_label_attack, inject_gradient_poisoning_attack
        )
        from app.detection.pipeline import DetectionPipeline

        control_samples = copy.deepcopy(samples)
        test_samples = copy.deepcopy(samples)

        if attack_type == "label_flip":
            test_samples = inject_label_flip_attack(test_samples, n_poison=20)
        elif attack_type == "backdoor":
            test_samples = inject_backdoor_attack(test_samples, n_poison=15)
        elif attack_type == "boiling_frog":
            test_samples = inject_boiling_frog_attack(test_samples, n_poison=25)
        elif attack_type == "clean_label":
            test_samples = inject_clean_label_attack(test_samples, n_poison=18)
        elif attack_type == "gradient_poisoning":
            test_samples = inject_gradient_poisoning_attack(test_samples, n_poison=12)

        n_injected = sum(1 for s in test_samples if s.get("poison_status") == "confirmed")

        started = time.perf_counter()
        result = (self.pipeline or DetectionPipeline()).run(test_samples)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        # Control run on the same data without the injection: a flagged control
        # is a false positive for this batch.
        control = DetectionPipeline().run(control_samples)
        control_flagged = control["verdict"] not in ("CLEAN", "LOW_RISK")

        detected = result["verdict"] not in ("CLEAN", "LOW_RISK")
        suspicion = result["overall_suspicion_score"]
        verdict = result.get("verdict", "CLEAN")

        detection_component = 4.0 if detected else 0.0
        speed_component = min(3.0, 3.0 * (500.0 / max(elapsed_ms, 50.0)))
        confidence_component = min(3.0, suspicion * 3.0)
        resilience = round(detection_component + speed_component + confidence_component, 1)

        sim_result = {
            "simulation_id": str(uuid.uuid4()),
            "attack_type": attack_type,
            "n_injected": n_injected,
            "detected": detected,
            "verdict": verdict,
            "suspicion_score": round(suspicion, 4),
            "resilience_score": resilience,
            "detection_speed_ms": elapsed_ms,
            "clean_control_verdict": control["verdict"],
            "clean_control_flagged": control_flagged,
            "layer_scores": result.get("layer_scores", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        db.save_redteam_simulation(sim_result)
        return sim_result
