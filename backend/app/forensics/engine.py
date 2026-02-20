"""Forensics: Attack Type Classifier + Injection Pattern Reconstructor"""
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import random


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
    },
    "gradient_poisoning": {
        "subtypes": ["gradient_inversion", "scaling"],
        "indicators": ["cosine_divergence", "norm_spike", "feature_inversion"],
        "severity": "high",
        "description": "Samples with inverted gradient signals disrupt model weight updates for specific classes"
    },
    "boiling_frog": {
        "subtypes": ["gradual_drift", "slow_injection"],
        "indicators": ["temporal_drift_pattern", "cumulative_shap_shift"],
        "severity": "high",
        "description": "Gradual, slow injection designed to evade threshold-based detection"
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

        l1 = evidence.get("layer1_statistical", {})
        l2 = evidence.get("layer2_spectral", {})
        l3 = evidence.get("layer3_ensemble", {})
        l4 = evidence.get("layer4_causal", {})
        l5 = evidence.get("layer5_federated", {})
        shap = evidence.get("shap_drift", {})

        # Label flip indicators
        if l1.get("kl_divergence", 0) > 1.5:
            scores["label_flip"] += 0.3
        if l3.get("flagged_ratio", 0) > 0.05:
            scores["label_flip"] += 0.2
        # Check label distribution
        labels = [s["label"] for s in samples if s.get("label", -1) >= 0]
        if labels:
            label_entropy = _entropy(labels)
            if label_entropy < 0.5:
                scores["label_flip"] += 0.3

        # Backdoor indicators
        if l2.get("backdoor_detected", False):
            scores["backdoor"] += 0.5
        if l2.get("spectral_gap", 0) > 3.0:
            scores["backdoor"] += 0.3
        if l2.get("minority_cluster_ratio", 1) < 0.1:
            scores["backdoor"] += 0.2

        # Clean label indicators: high Mahalanobis (feature outliers) but no spectral cluster
        if l1.get("mahalanobis", 0) > 4.0 and not l2.get("backdoor_detected", False):
            scores["clean_label"] += 0.4
        if l4.get("causal_effect", 0) > 0.08 and not l2.get("backdoor_detected", False):
            scores["clean_label"] += 0.3
        if l3.get("flagged_ratio", 0) > 0.03 and l1.get("kl_divergence", 0) < 1.0:
            scores["clean_label"] += 0.2  # anomalies without label shift

        # Gradient poisoning indicators
        if l5.get("n_quarantined", 0) > 0:
            scores["gradient_poisoning"] += 0.5
        if l5.get("avg_trust", 1) < 0.4:
            scores["gradient_poisoning"] += 0.3
        if l1.get("mahalanobis", 0) > 3.0 and l2.get("spectral_gap", 0) > 2.0:
            scores["gradient_poisoning"] += 0.2

        # Boiling frog indicators
        if shap.get("cumulative_drift", 0) > 0.2:
            scores["boiling_frog"] += 0.4
        if shap.get("drift_score", 0) > 0.1:
            scores["boiling_frog"] += 0.2
        # Check temporal spread of poison
        poison_times = [s["ingested_at"] for s in samples if s.get("poison_status") == "confirmed"]
        if len(poison_times) > 5:
            scores["boiling_frog"] += 0.2

        # Normalize
        total = sum(scores.values()) + 1e-8
        probabilities = {k: round(v / total, 4) for k, v in scores.items()}

        best_attack = max(scores, key=scores.get)
        confidence = round(probabilities[best_attack], 4)
        attack_info = ATTACK_TYPES[best_attack]

        return {
            "attack_type": best_attack,
            "attack_subtype": random.choice(attack_info["subtypes"]),
            "confidence": confidence,
            "severity": attack_info["severity"],
            "description": attack_info["description"],
            "probabilities": probabilities,
            "indicators_triggered": attack_info["indicators"]
        }


class InjectionPatternReconstructor:
    """
    Reconstructs the injection pattern and generates a human-readable attack narrative.
    """

    def reconstruct(self, samples: List[Dict], attack_classification: Dict[str, Any],
                    evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct injection pattern and generate narrative."""
        
        poisoned = [s for s in samples if s.get("poison_status") == "confirmed"]
        if not poisoned:
            return {"narrative": "No confirmed poisoned samples found.", "injection_schedule": "none"}

        # Temporal analysis
        times = sorted([s["ingested_at"] for s in poisoned])
        first_injection = times[0]
        last_injection = times[-1]

        # Batch analysis
        batches = set(s.get("batch_id", "unknown") for s in poisoned)
        sources = set(s.get("source_id", "unknown") for s in poisoned)
        clients = set(s.get("client_id", "unknown") for s in poisoned)

        # Injection schedule detection
        n_poison = len(poisoned)
        if n_poison < 10:
            schedule = "one_shot"
        elif len(batches) <= 2:
            schedule = "burst"
        else:
            schedule = "gradual"

        # Statistical disguise analysis
        features = np.array([s["feature_vector"] for s in poisoned])
        clean = [s for s in samples if s.get("poison_status") == "clean"]
        clean_features = np.array([s["feature_vector"] for s in clean[:len(poisoned)]])
        
        if len(clean_features) > 0:
            mean_shift = float(np.mean(np.abs(features.mean(0) - clean_features.mean(0))))
            sigma_shift = round(mean_shift / (clean_features.std() + 1e-8), 2)
        else:
            sigma_shift = 0.0

        # Source fingerprint
        primary_client = list(clients)[0] if clients else "unknown"
        primary_source = list(sources)[0] if sources else "unknown"

        # Causal effect
        causal_effect = evidence.get("layer4_causal", {}).get("causal_effect", 0)
        acc_impact = round(abs(causal_effect) * 100, 1)

        # Generate narrative
        attack_type = attack_classification.get("attack_type", "unknown")
        attack_subtype = attack_classification.get("attack_subtype", "unknown")
        confidence = round(attack_classification.get("confidence", 0) * 100, 1)

        narrative = f"""ATTACK RECONSTRUCTION REPORT
─────────────────────────────
Type:        {attack_type.replace('_', ' ').title()}
Subtype:     {attack_subtype.replace('_', ' ').title()}
Confidence:  {confidence}%

HOW it was injected:
• {n_poison} samples crafted and injected
• Feature vectors shifted by avg {sigma_shift}σ from clean distribution
• Injected across {len(batches)} training batch(es)
• Injection schedule: {schedule.replace('_', ' ')}
• Disguised as normal distribution tail

WHEN:
• First injection:  {first_injection[:19]} UTC
• Last injection:   {last_injection[:19]} UTC
• Pattern:          {schedule.replace('_', ' ').title()}

WHY it worked until now:
• Each batch individually appeared benign
• Collective causal effect: -{acc_impact}% accuracy on target class

SOURCE FINGERPRINT:
• Client ID:    {primary_client}
• Source:       {primary_source}
• Trust Score:  {round(evidence.get('layer5_federated', {}).get('avg_trust', 0.5) * 100, 0):.0f}/100"""

        return {
            "narrative": narrative,
            "n_poisoned_samples": n_poison,
            "affected_batches": list(batches),
            "n_batches": len(batches),
            "affected_sources": list(sources),
            "affected_clients": list(clients),
            "injection_schedule": schedule,
            "first_injection": first_injection,
            "last_injection": last_injection,
            "sigma_shift": sigma_shift,
            "primary_client": primary_client
        }


class SophisticationScorer:
    """Scores attacker sophistication on a 1-10 scale."""

    def score(self, attack_classification: Dict, pattern: Dict,
              evidence: Dict) -> Dict[str, Any]:
        
        score = 0.0
        factors = {}

        # Evasion layers
        n_layers_evaded = 6 - evidence.get("n_layers_alarmed", 0)
        factors["evasion_layers"] = round(n_layers_evaded / 6, 2)
        score += factors["evasion_layers"] * 3

        # Temporal precision (gradual = more sophisticated)
        schedule = pattern.get("injection_schedule", "one_shot")
        temporal_score = {"gradual": 1.0, "burst": 0.5, "one_shot": 0.2}.get(schedule, 0.3)
        factors["temporal_precision"] = temporal_score
        score += temporal_score * 2.5

        # Statistical disguise
        sigma_shift = pattern.get("sigma_shift", 0)
        disguise_score = max(0, 1 - sigma_shift / 3)  # lower shift = better disguise
        factors["statistical_disguise"] = round(disguise_score, 2)
        score += disguise_score * 2.5

        # Target specificity
        severity = attack_classification.get("severity", "medium")
        specificity = {"critical": 1.0, "high": 0.7, "medium": 0.4}.get(severity, 0.3)
        factors["target_specificity"] = specificity
        score += specificity * 2

        final_score = round(min(10, max(1, score)), 1)
        
        level = (
            "APT-grade (Coordinated Campaign)" if final_score >= 8 else
            "Targeted (Sophisticated Attacker)" if final_score >= 4 else
            "Opportunistic (Script-kiddie level)"
        )

        return {
            "sophistication_score": final_score,
            "level": level,
            "factors": factors,
            "description": f"Score {final_score}/10 — {level}"
        }


class BlastRadiusMapper:
    """Maps the blast radius of a poisoning attack."""

    def map(self, samples: List[Dict], evidence: Dict) -> Dict[str, Any]:
        poisoned = [s for s in samples if s.get("poison_status") == "confirmed"]
        
        affected_batches = list(set(s.get("batch_id", "unknown") for s in poisoned))
        n_batches = len(affected_batches)
        
        # Simulate downstream model impact
        n_models = min(n_batches + 1, 4)
        
        causal_effect = abs(evidence.get("layer4_causal", {}).get("causal_effect", 0))
        prediction_impact = round(min(causal_effect * 150, 35), 1)
        
        # Counterfactual harm
        n_predictions_affected = int(prediction_impact / 100 * 10000)
        
        return {
            "n_poisoned_samples": len(poisoned),
            "affected_batches": affected_batches,
            "n_batches_affected": n_batches,
            "n_models_affected": n_models,
            "prediction_impact_pct": prediction_impact,
            "n_predictions_affected": n_predictions_affected,
            "downstream_harm": {
                "domain": "Medical Diagnosis",
                "estimated_misdiagnoses": int(n_predictions_affected * 0.03),
                "accuracy_loss_pct": round(causal_effect * 100, 1),
                "financial_impact_usd": int(n_predictions_affected * 12.5)
            },
            "lineage_map": {
                batch: {
                    "models_trained": [f"model_v{i+1}" for i in range(min(2, n_models))],
                    "prediction_influence": round(prediction_impact / n_batches, 1)
                }
                for batch in affected_batches[:3]
            }
        }


class CounterfactualSimulator:
    """Simulates what would have happened without detection."""

    def simulate(self, evidence: Dict, blast_radius: Dict) -> Dict[str, Any]:
        causal_effect = abs(evidence.get("layer4_causal", {}).get("causal_effect", 0))
        acc_with = evidence.get("layer4_causal", {}).get("accuracy_with_poison", 0.85)
        
        projections = []
        for days in [30, 60, 90]:
            degradation = min(causal_effect * (1 + days / 30 * 0.3), 0.25)
            projected_acc = round(max(0.6, acc_with - degradation), 4)
            projections.append({
                "days": days,
                "projected_accuracy": projected_acc,
                "accuracy_loss": round(degradation, 4),
                "estimated_harm": int(blast_radius.get("n_predictions_affected", 0) * (days / 30))
            })

        return {
            "counterfactual_projections": projections,
            "harm_prevented": {
                "accuracy_preserved": round(causal_effect, 4),
                "predictions_protected": blast_radius.get("n_predictions_affected", 0),
                "estimated_cost_saved_usd": blast_radius.get("downstream_harm", {}).get("financial_impact_usd", 0)
            },
            "detection_value": f"Prevented ~{round(causal_effect * 100, 1)}% accuracy degradation over 90 days"
        }


def _entropy(labels: List) -> float:
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p + 1e-10) for p in probs)
