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
        poison_times = [s["ingested_at"] for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        if len(poison_times) > 5:
            scores["boiling_frog"] += 0.2

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
    """
    Reconstructs the injection pattern and generates a human-readable attack narrative.
    """

    def reconstruct(self, samples: List[Dict], attack_classification: Dict[str, Any],
                    evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct injection pattern and generate narrative."""
        
        poisoned = [s for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        if not poisoned:
            return {
                "narrative": "No samples crossed the configured suspicion threshold. No injection pattern can be reconstructed from this analysis.",
                "injection_schedule": "not_determined",
                "analysis_status": "insufficient_evidence",
            }

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
        clean = [s for s in samples if s.get("poison_status") not in ("confirmed", "suspected")]
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

        narrative = f"""ANALYST HYPOTHESIS — NOT ATTRIBUTION
─────────────────────────────
Candidate type: {attack_type.replace('_', ' ').title()}
Subtype:     {attack_subtype.replace('_', ' ').title()}
Heuristic confidence: {confidence}%

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

INTERPRETATION LIMITS:
• This reconstruction is based on rows flagged by heuristic detectors.
• It does not establish attacker identity, source attribution, or real-world harm.
• Proxy-model effect observed in this analysis: {acc_impact}% accuracy difference.

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
            "primary_client": primary_client,
            "analysis_status": "heuristic_hypothesis",
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
            "description": f"Heuristic score {final_score}/10 — {level}; not an attacker-capability assessment.",
            "analysis_status": "heuristic_estimate",
        }


class BlastRadiusMapper:
    """Maps the blast radius of a poisoning attack."""

    def map(self, samples: List[Dict], evidence: Dict) -> Dict[str, Any]:
        poisoned = [s for s in samples if s.get("poison_status") in ("confirmed", "suspected")]
        
        affected_batches = list(set(s.get("batch_id", "unknown") for s in poisoned))
        n_batches = len(affected_batches)
        
        # The application has no model registry or production lineage input.
        # Report the observed input scope only; do not fabricate harm.
        causal_effect = abs(evidence.get("layer4_causal", {}).get("causal_effect", 0))
        return {
            "n_poisoned_samples": len(poisoned),
            "affected_batches": affected_batches,
            "n_batches_affected": n_batches,
            "proxy_accuracy_effect_pct": round(causal_effect * 100, 1),
            "scope_status": "observed_input_only",
            "limitation": "No production model lineage, prediction volume, domain impact, or financial impact was supplied; downstream impact is not estimated.",
        }


class CounterfactualSimulator:
    """Describes the available proxy comparison without inventing deployment harm."""

    def simulate(self, evidence: Dict, blast_radius: Dict) -> Dict[str, Any]:
        causal_effect = abs(evidence.get("layer4_causal", {}).get("causal_effect", 0))
        acc_with = evidence.get("layer4_causal", {}).get("acc_with_poison", evidence.get("layer4_causal", {}).get("accuracy_with_poison", 0.85))
        
        return {
            "proxy_accuracy_effect": round(causal_effect, 4),
            "accuracy_with_flagged_rows": acc_with,
            "analysis_status": "not_a_deployment_counterfactual",
            "limitation": "No deployed-model telemetry or intervention outcome is available. Harm prevented, future degradation, and financial impact are not estimated."
        }


def _entropy(labels: List) -> float:
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p + 1e-10) for p in probs)
