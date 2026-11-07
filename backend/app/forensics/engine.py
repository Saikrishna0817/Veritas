"""Forensics: Attack Type Classifier + Injection Pattern Reconstructor"""
import numpy as np
from typing import Dict, Any, List


ATTACK_TYPES = {
    "label_flip": {
        "subtypes": ["random_flip", "targeted_flip"],
        "indicators": ["label_entropy_spike", "class_imbalance_shift"],
        "severity": "medium",
        "description": "Adversary flips labels of training samples to corrupt model boundaries"
    },
    "backdoor": {
        "subtypes": ["patch_trigger", "blend_trigger", "wanet"],
        "indicators": ["activation_clustering", "spectral_signature"],
        "severity": "critical",
        "description": "Hidden trigger pattern causes misclassification at inference time"
    },
    "clean_label": {
        "subtypes": ["feature_collision", "witches_brew"],
        "indicators": ["feature_space_outlier", "gradient_conflict"],
        "severity": "critical",
        "description": "Correctly-labeled samples crafted to poison model via feature space manipulation"
    }
}


class AttackTypeClassifier:
    """
    Classifies the type of poisoning attack from evidence bundle signals.
    Uses rule-based + statistical classification.
    """

    def classify(self, evidence: Dict[str, Any], samples: List[Dict]) -> Dict[str, Any]:
        """Classify attack type from evidence bundle."""
        scores = {attack: 0.0 for attack in ATTACK_TYPES}

        # Veritas v2 checks: a label alarm is direct evidence of wrong labels. A shift alarm says the
        # incoming rows are distinguishable from clean ones, which backdoors and collisions both cause
        # and the two checks cannot tell apart.
        if evidence.get("label_check", {}).get("alarm"):
            scores["label_flip"] += 0.6
        if evidence.get("shift_check", {}).get("alarm"):
            scores["backdoor"] += 0.25
            scores["clean_label"] += 0.25
        labels = [s["label"] for s in samples if s.get("label", -1) >= 0]
        if labels and _entropy(labels) < 0.5:
            scores["label_flip"] += 0.3   # one label dominates the batch

        # Do not force a precise attack label when the rules have no material
        # support. This is an analyst hypothesis, not an attribution engine.
        total = sum(scores.values())
        probabilities = ({k: round(v / total, 4) for k, v in scores.items()}
                         if total else {k: 0.0 for k in scores})
        best_attack = max(scores, key=scores.get)
        confidence = round(probabilities[best_attack], 4)
        if scores[best_attack] < 0.2:
            return {
                "attack_type": "unknown",
                "attack_subtype": "insufficient_evidence",
                "confidence": 0.0,
                "severity": "low",
                "description": "Available heuristic signals do not support a specific attack classification.",
                "probabilities": probabilities,
                "indicators_triggered": [],
                "classification_status": "insufficient_evidence",
            }
        attack_info = ATTACK_TYPES[best_attack]

        return {
            "attack_type": best_attack,
            "attack_subtype": "not_determined",
            "confidence": confidence,
            "severity": attack_info["severity"],
            "description": attack_info["description"],
            "probabilities": probabilities,
            "indicators_triggered": attack_info["indicators"],
            "classification_status": "heuristic",
        }


class InjectionPatternReconstructor:
    """Summarises which rows were flagged and how they differ from the rest. Facts only: when the file was
    parsed is not when rows were injected, so no timeline, schedule or source is inferred."""

    def reconstruct(self, samples: List[Dict], attack_classification: Dict[str, Any],
                    evidence: Dict[str, Any]) -> Dict[str, Any]:
        flagged = [s for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        if not flagged:
            return {
                "narrative": "No rows were flagged. The batch either looks like the reference or the evidence was too weak to single out rows.",
                "analysis_status": "insufficient_evidence",
            }
        rest = [s for s in samples if s.get("poison_status") not in ("confirmed", "suspected")]
        sigma_shift = 0.0
        if rest:
            f_vec, r_vec = np.array([s["feature_vector"] for s in flagged]), np.array([s["feature_vector"] for s in rest])
            sigma_shift = round(float(np.mean(np.abs(f_vec.mean(0) - r_vec.mean(0))) / (r_vec.std() + 1e-8)), 2)

        label, shift = evidence.get("label_check", {}), evidence.get("shift_check", {})
        lines = [f"{len(flagged)} of {len(samples)} incoming rows were flagged.",
                 f"Label check evidence {label.get('evidence', 0):.1f} (-log10 p){' - ALARM' if label.get('alarm') else ''}; "
                 f"shift check evidence {shift.get('evidence', 0):.1f}{' - ALARM' if shift.get('alarm') else ''}.",
                 f"Flagged rows differ from the others by {sigma_shift} standard deviations on average.",
                 "This says which rows look wrong, not who changed them or when."]
        return {"narrative": "\n".join(lines), "n_poisoned_samples": len(flagged), "sigma_shift": sigma_shift,
                "analysis_status": "observed"}


class BlastRadiusMapper:
    """Counts the flagged rows. There is no model registry or lineage input, so downstream impact is not estimated."""

    def map(self, samples: List[Dict], evidence: Dict) -> Dict[str, Any]:
        flagged = [s for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        return {
            "n_poisoned_samples": len(flagged),
            "scope_status": "observed_input_only",
            "limitation": "No production model lineage or prediction volume was supplied; downstream impact is not estimated.",
        }


def _entropy(labels: List) -> float:
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p + 1e-10) for p in probs)
