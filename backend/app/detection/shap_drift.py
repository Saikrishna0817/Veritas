"""SHAP Explainability Drift Monitor"""
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, Any, List
from datetime import datetime


class SHAPDriftMonitor:
    """
    Monitors SHAP value distributions over time.
    Detects attacks BEFORE accuracy degrades by tracking explanation drift.
    """

    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.shap_history: List[Dict] = []
        self.feature_names: List[str] = []

    def _compute_shap_values(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute approximate SHAP values using feature permutation importance.
        In production, use the shap library with the actual model.
        """
        from sklearn.linear_model import LogisticRegression
        
        if len(np.unique(labels)) < 2:
            return np.zeros_like(features)
        
        clf = LogisticRegression(max_iter=200, random_state=42)
        try:
            clf.fit(features, labels)
            # Approximate SHAP as |coefficient * feature_value|
            shap_approx = np.abs(features * clf.coef_[0])
            # Normalize per sample
            row_sums = shap_approx.sum(axis=1, keepdims=True) + 1e-8
            return shap_approx / row_sums
        except Exception:
            return np.abs(features) / (np.abs(features).sum(axis=1, keepdims=True) + 1e-8)

    def record_snapshot(self, features: np.ndarray, labels: np.ndarray,
                        timestamp: str = None, batch_id: str = None) -> Dict[str, Any]:
        """Record a SHAP snapshot for the current batch."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        shap_values = self._compute_shap_values(features, labels)
        mean_shap = shap_values.mean(axis=0)
        std_shap = shap_values.std(axis=0)

        snapshot = {
            "timestamp": timestamp,
            "batch_id": batch_id,
            "mean_shap": mean_shap.tolist(),
            "std_shap": std_shap.tolist(),
            "n_samples": len(features)
        }
        self.shap_history.append(snapshot)
        return snapshot

    def compute_drift(self) -> Dict[str, Any]:
        """Compute drift between consecutive SHAP snapshots."""
        if len(self.shap_history) < 2:
            return {"drift_score": 0.0, "alarm": False, "drift_per_feature": []}

        current = np.array(self.shap_history[-1]["mean_shap"])
        previous = np.array(self.shap_history[-2]["mean_shap"])

        # Wasserstein distance per feature
        drift_per_feature = []
        for i in range(len(current)):
            # Use distributions from history
            curr_vals = np.array([s["mean_shap"][i] for s in self.shap_history[-5:]])
            prev_vals = np.array([s["mean_shap"][i] for s in self.shap_history[:-1][-5:]])
            if len(curr_vals) > 1 and len(prev_vals) > 1:
                w = wasserstein_distance(curr_vals, prev_vals)
            else:
                w = abs(float(current[i]) - float(previous[i]))
            drift_per_feature.append(round(float(w), 4))

        overall_drift = float(np.mean(drift_per_feature))
        max_drift_feature = int(np.argmax(drift_per_feature))

        # Cumulative drift over all history
        cumulative_drift = 0.0
        if len(self.shap_history) > 2:
            first = np.array(self.shap_history[0]["mean_shap"])
            cumulative_drift = float(np.mean(np.abs(current - first)))

        return {
            "drift_score": round(overall_drift, 4),
            "cumulative_drift": round(cumulative_drift, 4),
            "drift_per_feature": drift_per_feature,
            "max_drift_feature_idx": max_drift_feature,
            "n_snapshots": len(self.shap_history),
            "alarm": overall_drift > self.drift_threshold,
            "suspicion_score": min(1.0, round(overall_drift / self.drift_threshold, 4))
        }

    def get_drift_timeline(self) -> List[Dict]:
        """Get drift scores over time for visualization."""
        if len(self.shap_history) < 2:
            return []
        
        timeline = []
        for i in range(1, len(self.shap_history)):
            curr = np.array(self.shap_history[i]["mean_shap"])
            prev = np.array(self.shap_history[i-1]["mean_shap"])
            drift = float(np.mean(np.abs(curr - prev)))
            timeline.append({
                "timestamp": self.shap_history[i]["timestamp"],
                "drift_score": round(drift, 4),
                "batch_id": self.shap_history[i].get("batch_id")
            })
        return timeline
