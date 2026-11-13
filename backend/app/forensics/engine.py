"""Forensics: attack classification (from the detector's attack profile), injection pattern summary, blast radius."""
import numpy as np
from typing import Dict, Any, List


# The detector's six attack profiles grouped into the families the UI colours.
FAMILIES = {
    "label_flip": {"profiles": ("label_flip_random", "label_flip_targeted"), "severity": "medium",
                   "description": "Labels of some rows were changed to corrupt the model's decision boundaries"},
    "backdoor": {"profiles": ("backdoor_ood", "backdoor_inrange"), "severity": "critical",
                 "description": "Rows carry a trigger pattern and a target label so a model learns to misclassify on the trigger"},
    "clean_label": {"profiles": ("feature_collision",), "severity": "critical",
                    "description": "Correctly labelled rows moved in feature space to poison the model without wrong labels"},
    "covariate_shift": {"profiles": ("covariate_shift",), "severity": "medium",
                        "description": "Some rows come from a shifted or corrupted source (several columns moved together)"},
}
_FAMILY_OF = {p: fam for fam, info in FAMILIES.items() for p in info["profiles"]}


class AttackTypeClassifier:
    """The report's attack classification, taken from ``veritas2.attack_profile`` (trained on dev batches, held-out
    accuracy in docs/benchmark-v2-findings.md). No profile (a CLEAN or unlabeled batch) means no classification."""

    def classify(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        profile = detection_result.get("attack_profile")
        if not profile:
            return {"attack_type": "unknown", "attack_subtype": "insufficient_evidence", "confidence": 0.0, "severity": "low",
                    "description": "The batch is not suspicious enough (or has no labels) to name an attack.",
                    "probabilities": {}, "classification_status": "insufficient_evidence"}
        family = _FAMILY_OF[profile["likely"]]
        return {"attack_type": family, "attack_subtype": profile["likely"],
                "confidence": profile["probabilities"][profile["likely"]], "severity": FAMILIES[family]["severity"],
                "description": f"Most like {profile['label']}. {FAMILIES[family]['description']}.",
                "probabilities": profile["probabilities"], "classification_status": "model",
                "evidence": profile["features"]}


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

        checks = "; ".join(f"{name} check evidence {c.get('evidence', 0):.1f}{' - ALARM' if c.get('alarm') else ''}"
                           for name, c in (("label", evidence.get("label_check", {})), ("shift", evidence.get("shift_check", {})),
                                           ("residual", evidence.get("resid_check", {}))))
        lines = [f"{len(flagged)} of {len(samples)} incoming rows were flagged.",
                 f"{checks} (-log10 p).",
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
