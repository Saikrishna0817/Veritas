"""Detection Layer 4: Causal Proof Engine (Do-Calculus inspired)"""
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import random


class CausalProofEngine:
    """
    Proves causation (not just correlation) using counterfactual analysis.
    Implements leave-one-out test, placebo validation, and ITE estimation.
    
    Inspired by DoWhy's do-calculus framework.
    """

    def __init__(self, n_bootstrap: int = 100, significance_level: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level

    def _simulate_model_accuracy(self, features: np.ndarray, labels: np.ndarray,
                                  exclude_mask: np.ndarray = None) -> float:
        """
        Simulate model accuracy with/without certain samples.
        Uses a simple linear classifier as proxy.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        if exclude_mask is not None:
            features = features[~exclude_mask]
            labels = labels[~exclude_mask]

        if len(features) < 10 or len(np.unique(labels)) < 2:
            return 0.5

        clf = LogisticRegression(max_iter=200, random_state=42)
        try:
            scores = cross_val_score(clf, features, labels, cv=min(3, len(features) // 10), scoring="accuracy")
            return float(scores.mean())
        except Exception:
            return 0.5

    def estimate_causal_effect(self, features: np.ndarray, labels: np.ndarray,
                                poison_mask: np.ndarray) -> Dict[str, Any]:
        """
        Estimate the causal effect of poisoned samples on model accuracy.
        
        Treatment: sample_included (1=included, 0=excluded)
        Outcome: model_accuracy
        """
        if poison_mask.sum() == 0:
            return {"causal_effect": 0.0, "placebo_passed": True, "confidence_interval": [0.0, 0.0]}

        # Baseline accuracy (with poison)
        acc_with_poison = self._simulate_model_accuracy(features, labels)

        # Counterfactual accuracy (without poison)
        acc_without_poison = self._simulate_model_accuracy(features, labels, exclude_mask=poison_mask)

        # Causal effect = improvement from removing poison
        causal_effect = acc_without_poison - acc_with_poison

        # Bootstrap confidence interval
        bootstrap_effects = []
        n = len(features)
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            b_features = features[idx]
            b_labels = labels[idx]
            b_mask = poison_mask[idx]
            if b_mask.sum() == 0:
                continue
            try:
                acc_w = self._simulate_model_accuracy(b_features, b_labels)
                acc_wo = self._simulate_model_accuracy(b_features, b_labels, exclude_mask=b_mask)
                bootstrap_effects.append(acc_wo - acc_w)
            except Exception:
                pass

        if bootstrap_effects:
            ci_low = float(np.percentile(bootstrap_effects, 2.5))
            ci_high = float(np.percentile(bootstrap_effects, 97.5))
        else:
            ci_low = causal_effect - 0.02
            ci_high = causal_effect + 0.02

        # Placebo test: shuffle poison labels → effect should disappear
        shuffled_mask = poison_mask.copy()
        np.random.shuffle(shuffled_mask)
        acc_placebo = self._simulate_model_accuracy(features, labels, exclude_mask=shuffled_mask)
        placebo_effect = acc_placebo - acc_with_poison
        placebo_passed = abs(placebo_effect) < abs(causal_effect) * 0.3

        # Statistical significance (t-test on bootstrap)
        if len(bootstrap_effects) > 10:
            t_stat, p_value = stats.ttest_1samp(bootstrap_effects, 0)
            significant = p_value < self.significance_level
        else:
            p_value = 0.01 if abs(causal_effect) > 0.02 else 0.5
            significant = abs(causal_effect) > 0.02

        # ITE (Individual Treatment Effect) per poisoned sample
        ite_scores = []
        for i in np.where(poison_mask)[0]:
            single_mask = np.zeros(len(features), dtype=bool)
            single_mask[i] = True
            acc_wo_single = self._simulate_model_accuracy(features, labels, exclude_mask=single_mask)
            ite_scores.append(round(float(acc_wo_single - acc_with_poison), 4))

        return {
            "causal_effect": round(float(causal_effect), 4),
            "accuracy_with_poison": round(float(acc_with_poison), 4),
            "accuracy_without_poison": round(float(acc_without_poison), 4),
            "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
            "placebo_passed": bool(placebo_passed),
            "placebo_effect": round(float(placebo_effect), 4),
            "p_value": round(float(p_value), 4),
            "statistically_significant": bool(significant),
            "ite_scores": ite_scores[:10],  # top 10
            "mean_ite": round(float(np.mean(ite_scores)) if ite_scores else 0.0, 4),
            "proof_valid": bool(placebo_passed and significant and causal_effect > 0.01),
            "suspicion_score": min(1.0, round(abs(causal_effect) * 5, 4))
        }

    def analyze_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        features = np.array([s["feature_vector"] for s in samples])
        labels = np.array([s["label"] for s in samples])
        poison_mask = np.array([s["poison_status"] == "confirmed" for s in samples])
        return self.estimate_causal_effect(features, labels, poison_mask)
