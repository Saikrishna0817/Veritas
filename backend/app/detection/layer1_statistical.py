"""Detection Layer 1: Statistical Shift Detection"""
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import mahalanobis
from typing import Dict, Any, List


class StatisticalShiftDetector:
    """
    Detects distributional shifts between incoming data and clean baseline.
    Uses KL-Divergence, Wasserstein Distance, and Mahalanobis Distance.
    """

    def __init__(self, threshold_kl: float = 2.0, threshold_mhd: float = 3.5):
        self.threshold_kl = threshold_kl
        self.threshold_mhd = threshold_mhd
        self.baseline_mean: np.ndarray = None
        self.baseline_cov_inv: np.ndarray = None
        self.baseline_distribution: np.ndarray = None

    def fit_baseline(self, clean_features: np.ndarray):
        """Fit the clean baseline distribution."""
        self.baseline_mean = clean_features.mean(axis=0)
        cov = np.cov(clean_features.T)
        # Add small regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self.baseline_cov_inv = np.linalg.inv(cov)
        self.baseline_distribution = clean_features

    def compute_divergences(self, incoming: np.ndarray) -> Dict[str, Any]:
        """Compute all divergence metrics between incoming and baseline."""
        if self.baseline_distribution is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        results = {}

        # KL Divergence (per-feature histograms)
        kl_scores = []
        for i in range(incoming.shape[1]):
            hist_in, bins = np.histogram(incoming[:, i], bins=20, density=True)
            hist_base, _ = np.histogram(self.baseline_distribution[:, i], bins=bins, density=True)
            # Smooth to avoid zeros
            hist_in = hist_in + 1e-10
            hist_base = hist_base + 1e-10
            hist_in /= hist_in.sum()
            hist_base /= hist_base.sum()
            kl_scores.append(float(entropy(hist_in, hist_base)))
        
        results["kl_divergence"] = round(float(np.mean(kl_scores)), 4)
        results["kl_per_feature"] = [round(k, 4) for k in kl_scores]

        # Wasserstein Distance (per-feature)
        w_scores = []
        for i in range(incoming.shape[1]):
            w = wasserstein_distance(incoming[:, i], self.baseline_distribution[:, i])
            w_scores.append(float(w))
        results["wasserstein"] = round(float(np.mean(w_scores)), 4)

        # Mahalanobis Distance (mean of incoming vs baseline mean)
        incoming_mean = incoming.mean(axis=0)
        try:
            mhd = mahalanobis(incoming_mean, self.baseline_mean, self.baseline_cov_inv)
        except Exception:
            mhd = float(np.linalg.norm(incoming_mean - self.baseline_mean))
        results["mahalanobis"] = round(float(mhd), 4)

        # Sigma deviation
        std = self.baseline_distribution.std(axis=0)
        diff = np.abs(incoming.mean(axis=0) - self.baseline_mean)
        sigma_dev = float(np.mean(diff / (std + 1e-8)))
        results["sigma_deviation"] = round(sigma_dev, 4)

        # Alarm
        results["alarm"] = (
            results["kl_divergence"] > self.threshold_kl or
            results["mahalanobis"] > self.threshold_mhd
        )
        results["suspicion_score"] = min(1.0, round(
            (results["kl_divergence"] / (self.threshold_kl * 2)) * 0.4 +
            (results["mahalanobis"] / (self.threshold_mhd * 2)) * 0.4 +
            (results["wasserstein"] / 5.0) * 0.2,
            4
        ))

        return results

    def analyze_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze a list of sample dicts."""
        features = np.array([s["feature_vector"] for s in samples])
        return self.compute_divergences(features)
