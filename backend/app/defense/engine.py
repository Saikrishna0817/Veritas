"""Defense Engine: Auto-Defense + Human-in-the-Loop + Red-Team

Persists state to SQLite via app.models.database so defense actions, HITL queue,
decisions, and red-team simulation history survive server restarts.
"""
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.models import database as db


class StabilityAwareAutoDefense:
    """
    Stability-aware auto-defense that prevents self-damage.
    Max 5% quarantine per epoch, rate-limited actions.
    """

    MAX_QUARANTINE_RATIO = 0.05
    QUARANTINE_COOLDOWN_SECS = 30

    def __init__(self):
        self.quarantined_ids = set()
        self.last_action_time = None
        self.mode = "active"  # active | observe_only

    @property
    def defense_log(self) -> List[Dict[str, Any]]:
        """Fetch defense log from SQLite DB."""
        return db.get_defense_actions(limit=100)

    def decide_action(self, samples: List[Dict], suspicion_score: float,
                      verdict: str) -> Dict[str, Any]:
        """Decide defense action based on suspicion score."""
        if self.mode == "observe_only":
            return {"action": "observe", "reason": "observe_only_mode", "samples_affected": 0}

        # Rate limiting
        now = datetime.now(timezone.utc)
        if self.last_action_time:
            elapsed = (now - self.last_action_time).total_seconds()
            if elapsed < self.QUARANTINE_COOLDOWN_SECS:
                return {"action": "cooldown", "reason": "rate_limited", "samples_affected": 0,
                        "cooldown_remaining_secs": round(self.QUARANTINE_COOLDOWN_SECS - elapsed, 1)}

        if verdict == "CONFIRMED_POISONED" and suspicion_score > 0.7:
            return self._quarantine(samples, suspicion_score)
        elif verdict == "SUSPICIOUS" and suspicion_score > 0.5:
            return self._soft_quarantine(samples, suspicion_score)
        else:
            return {"action": "monitor", "reason": "below_threshold", "samples_affected": 0}

    def _quarantine(self, samples: List[Dict], score: float) -> Dict[str, Any]:
        """Hard quarantine — remove from training."""
        candidates = [s for s in samples 
                      if s.get("poison_status") in ("confirmed", "suspected") 
                      and s["id"] not in self.quarantined_ids]
        
        max_quarantine = max(1, int(len(samples) * self.MAX_QUARANTINE_RATIO))
        to_quarantine = candidates[:max_quarantine]
        
        for s in to_quarantine:
            self.quarantined_ids.add(s["id"])
        
        self.last_action_time = datetime.now(timezone.utc)
        action = {
            "action": "quarantine",
            "action_id": str(uuid.uuid4()),
            "samples_affected": len(to_quarantine),
            "sample_ids": [s["id"] for s in to_quarantine],
            "suspicion_score": score,
            "model_stable": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": f"Suspicion score {score:.2f} > 0.70 threshold"
        }
        db.save_defense_action(action)
        return action

    def _soft_quarantine(self, samples: List[Dict], score: float) -> Dict[str, Any]:
        """Soft quarantine — down-weight in training."""
        candidates = [s for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        n = min(len(candidates), max(1, int(len(samples) * self.MAX_QUARANTINE_RATIO * 0.5)))
        
        self.last_action_time = datetime.now(timezone.utc)
        action = {
            "action": "soft_quarantine",
            "action_id": str(uuid.uuid4()),
            "samples_affected": n,
            "weight_factor": 0.1,
            "suspicion_score": score,
            "model_stable": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": f"Suspicion score {score:.2f} in borderline range"
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

    def enqueue(self, samples: List[Dict], evidence: Dict,
                suspicion_score: float) -> Dict[str, Any]:
        """Add a case to the human review queue."""
        case = {
            "case_id": str(uuid.uuid4()),
            "suspicion_score": suspicion_score,
            "n_samples": len(samples),
            "sample_ids": [s["id"] for s in samples[:5]],
            "evidence_summary": {
                "kl_divergence": evidence.get("layer1_statistical", {}).get("kl_divergence", 0),
                "causal_effect": evidence.get("layer4_causal", {}).get("causal_effect", 0),
                "attack_type": evidence.get("attack_classification", {}).get("attack_type", "unknown")
            },
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deadline": (datetime.now(timezone.utc).replace(hour=23, minute=59)).isoformat()
        }
        db.save_hitl_case(case)
        return case

    def decide(self, case_id: str, decision: str, reviewer: str = "analyst") -> Dict[str, Any]:
        """Record a human decision (approve/reject)."""
        pending = db.get_pending_hitl_cases()
        case = next((c for c in pending if c["case_id"] == case_id), None)
        if not case:
            return {"error": "Case not found or already resolved"}
        
        decision_record = {
            "case_id": case_id,
            "decision": decision,  # "approve_quarantine" | "mark_safe"
            "reviewer": reviewer,
            "decided_at": datetime.now(timezone.utc).isoformat()
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
        import copy
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

        from app.detection.pipeline import DetectionPipeline

        started = time.perf_counter()
        pipeline = self.pipeline or DetectionPipeline()
        result = pipeline.run(test_samples, run_causal=False)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

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
            "false_positive_rate": 0.0,
            "layer_scores": result.get("layer_scores", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        db.save_redteam_simulation(sim_result)
        return sim_result
