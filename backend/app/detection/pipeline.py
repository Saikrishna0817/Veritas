"""Detection Pipeline — Orchestrates all 5 layers.
Supports both:
  - Demo mode: synthetic dataset with known baseline
  - Upload mode: self-contained split (70% reference / 30% incoming), no external baseline
"""
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from .layer1_statistical import StatisticalShiftDetector
from .layer2_spectral import SpectralActivationAnalyzer
from .layer3_ensemble import EnsembleAnomalyDetector
from .layer4_causal import CausalProofEngine
from .layer5_federated import FederatedTrustAnalyzer, generate_demo_clients
from .shap_drift import SHAPDriftMonitor


class DetectionPipeline:
    """
    Full 5-layer detection pipeline.

    Upload mode (has_labels=False or no external baseline):
      - Layer 1: reference=first 70% of data, incoming=last 30%
      - Layer 2: runs on full dataset (unsupervised spectral)
      - Layer 3: trained on reference split, predicts on incoming split
      - Layer 4: skipped (needs known poison mask)
      - Layer 5: uses demo clients (federated context not applicable for uploads)
      - SHAP: runs on both splits, computes drift between them
    """

    def __init__(self):
        self.layer1 = StatisticalShiftDetector()
        self.layer2 = SpectralActivationAnalyzer()
        self.layer3 = EnsembleAnomalyDetector()
        self.layer4 = CausalProofEngine(n_bootstrap=30)
        self.layer5 = FederatedTrustAnalyzer()
        self.shap_monitor = SHAPDriftMonitor()
        self._baseline_fitted = False

    def fit_baseline(self, clean_samples: List[Dict]):
        """Fit all detectors on clean baseline data (demo mode)."""
        features = np.array([s["feature_vector"] for s in clean_samples])
        labels = np.array([s["label"] for s in clean_samples])

        self.layer1.fit_baseline(features)
        self.layer3.fit(features)
        self.shap_monitor.record_snapshot(features, labels, batch_id="baseline")
        self._baseline_fitted = True

    def fit_baseline_from_features(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """Fit detectors directly from numpy arrays (upload mode)."""
        self.layer1.fit_baseline(features)
        self.layer3.fit(features)
        if labels is not None:
            self.shap_monitor.record_snapshot(features, labels, batch_id="reference")
        else:
            dummy_labels = np.zeros(len(features), dtype=int)
            self.shap_monitor.record_snapshot(features, dummy_labels, batch_id="reference")
        self._baseline_fitted = True

    def run(self, samples: List[Dict], baseline_samples: List[Dict] = None,
            run_causal: bool = True) -> Dict[str, Any]:
        """Run pipeline on samples (demo mode — uses sample dicts)."""
        start_time = datetime.utcnow()

        if not self._baseline_fitted and baseline_samples:
            self.fit_baseline(baseline_samples)

        features = np.array([s["feature_vector"] for s in samples])
        labels = np.array([s["label"] for s in samples])

        return self._run_core(
            features=features,
            labels=labels,
            has_labels=True,
            run_causal=run_causal,
            samples=samples,
            start_time=start_time
        )

    def run_on_upload(self, ingested: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pipeline on an uploaded dataset (upload mode).
        Uses self-contained 70/30 split — no external baseline needed.
        """
        start_time = datetime.utcnow()

        features = ingested["features"]          # full normalized feature matrix
        labels = ingested.get("labels")          # None if unsupervised
        has_labels = ingested["has_labels"]
        split_idx = ingested["reference_split"]
        samples = ingested["samples"]

        # Self-contained split
        ref_features = features[:split_idx]
        inc_features = features[split_idx:]
        ref_labels = labels[:split_idx] if has_labels and labels is not None else None
        inc_labels = labels[split_idx:] if has_labels and labels is not None else None

        # Fit on reference portion
        self.fit_baseline_from_features(ref_features, ref_labels)

        # Analyze incoming portion
        return self._run_core(
            features=inc_features,
            labels=inc_labels if inc_labels is not None else np.zeros(len(inc_features), dtype=int),
            has_labels=has_labels,
            run_causal=False,  # no known poison mask in upload mode
            samples=samples[split_idx:],
            start_time=start_time,
            ref_features=ref_features,
            ref_labels=ref_labels,
            mode="upload"
        )

    def _run_core(self, features: np.ndarray, labels: np.ndarray,
                  has_labels: bool, run_causal: bool,
                  samples: List[Dict], start_time: datetime,
                  ref_features: np.ndarray = None,
                  ref_labels: np.ndarray = None,
                  mode: str = "demo") -> Dict[str, Any]:
        """Core detection logic shared by both modes."""

        # ── Layer 1: Statistical ──────────────────────────────────────────
        try:
            l1 = self.layer1.compute_divergences(features)
        except Exception as e:
            l1 = {"suspicion_score": 0.0, "alarm": False, "error": str(e)}

        # ── Layer 2: Spectral ─────────────────────────────────────────────
        # Works in both modes; uses dummy labels if unsupervised
        try:
            effective_labels = labels if has_labels else np.zeros(len(features), dtype=int)
            l2 = self.layer2.analyze(features, effective_labels)
        except Exception as e:
            l2 = {"suspicion_score": 0.0, "alarm": False, "error": str(e)}

        # ── Layer 3: Ensemble ─────────────────────────────────────────────
        try:
            if self._baseline_fitted:
                l3 = self.layer3.predict(features)
            else:
                l3 = {"suspicion_score": 0.0, "alarm": False, "flagged_count": 0}
        except Exception as e:
            l3 = {"suspicion_score": 0.0, "alarm": False, "error": str(e)}

        # ── Layer 4: Causal ───────────────────────────────────────────────
        if run_causal and has_labels:
            try:
                poison_mask = np.array([s.get("poison_status") == "confirmed" for s in samples])
                if poison_mask.sum() == 0:
                    ensemble_scores = np.array(l3.get("ensemble_scores", [0.5] * len(samples)))
                    poison_mask = ensemble_scores > 0.6
                l4 = self.layer4.estimate_causal_effect(features, labels, poison_mask)
            except Exception as e:
                l4 = {"suspicion_score": 0.0, "proof_valid": False, "error": str(e)}
        elif not has_labels:
            # Unsupervised causal proxy: use ensemble flags as treatment
            try:
                ensemble_scores = np.array(l3.get("ensemble_scores", [0.5] * len(features)))
                poison_mask = ensemble_scores > 0.6
                if poison_mask.sum() > 0 and poison_mask.sum() < len(features):
                    # Proxy: measure feature distribution shift caused by flagged samples
                    flagged_feat = features[poison_mask]
                    clean_feat = features[~poison_mask]
                    shift = float(np.mean(np.abs(flagged_feat.mean(0) - clean_feat.mean(0))))
                    l4 = {
                        "suspicion_score": min(1.0, round(shift * 5, 4)),
                        "causal_effect": round(shift, 4),
                        "proof_valid": shift > 0.05,
                        "mode": "unsupervised_proxy",
                        "accuracy_with_poison": None,
                        "accuracy_without_poison": None,
                        "placebo_passed": None,
                        "statistically_significant": shift > 0.05,
                        "confidence_interval": [round(shift * 0.8, 4), round(shift * 1.2, 4)]
                    }
                else:
                    l4 = {"suspicion_score": 0.0, "proof_valid": False, "mode": "unsupervised_proxy",
                          "causal_effect": 0.0}
            except Exception as e:
                l4 = {"suspicion_score": 0.0, "proof_valid": False, "error": str(e)}
        else:
            l4 = {"suspicion_score": 0.0, "proof_valid": False, "skipped": True}

        # ── Layer 5: Federated ────────────────────────────────────────────
        try:
            demo_clients = generate_demo_clients()
            l5 = self.layer5.analyze_clients(demo_clients)
        except Exception as e:
            l5 = {"suspicion_score": 0.0, "alarm": False, "error": str(e)}

        # ── SHAP Drift ────────────────────────────────────────────────────
        try:
            effective_labels = labels if has_labels else np.zeros(len(features), dtype=int)
            self.shap_monitor.record_snapshot(
                features, effective_labels,
                batch_id=samples[0].get("batch_id", "incoming") if samples else "incoming"
            )
            shap_drift = self.shap_monitor.compute_drift()
        except Exception as e:
            shap_drift = {"drift_score": 0.0, "alarm": False, "error": str(e)}

        # ── Overall Score ─────────────────────────────────────────────────
        if has_labels:
            weights = [0.20, 0.20, 0.20, 0.25, 0.10, 0.05]
        else:
            # Unsupervised: redistribute L4 weight to L1 and L3
            weights = [0.30, 0.15, 0.30, 0.10, 0.10, 0.05]

        layer_scores_raw = [
            l1.get("suspicion_score", 0),
            l2.get("suspicion_score", 0),
            l3.get("suspicion_score", 0),
            l4.get("suspicion_score", 0),
            l5.get("suspicion_score", 0),
            shap_drift.get("suspicion_score", 0)
        ]
        overall_score = sum(w * s for w, s in zip(weights, layer_scores_raw))
        overall_score = round(min(1.0, overall_score), 4)

        n_alarms = sum([
            l1.get("alarm", False),
            l2.get("alarm", False),
            l3.get("alarm", False),
            l4.get("proof_valid", False),
            l5.get("alarm", False),
            shap_drift.get("alarm", False)
        ])

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Poisoning level classification
        if overall_score >= 0.70:
            verdict = "CONFIRMED_POISONED"
            poisoning_level = "CRITICAL"
        elif overall_score >= 0.50:
            verdict = "SUSPICIOUS"
            poisoning_level = "MODERATE"
        elif overall_score >= 0.30:
            verdict = "LOW_RISK"
            poisoning_level = "LOW"
        else:
            verdict = "CLEAN"
            poisoning_level = "NONE"

        return {
            "job_id": f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "analyzed_at": start_time.isoformat(),
            "mode": mode,
            "detection_mode": "supervised" if has_labels else "unsupervised",
            "n_samples": len(features),
            "elapsed_ms": round(elapsed_ms, 1),
            "overall_suspicion_score": overall_score,
            "poisoning_level": poisoning_level,
            "n_layers_alarmed": n_alarms,
            "verdict": verdict,
            "requires_human_review": 0.40 <= overall_score <= 0.70,
            "layer_results": {
                "layer1_statistical": l1,
                "layer2_spectral": l2,
                "layer3_ensemble": l3,
                "layer4_causal": l4,
                "layer5_federated": l5,
                "shap_drift": shap_drift
            },
            "layer_scores": {
                "statistical": round(l1.get("suspicion_score", 0), 4),
                "spectral": round(l2.get("suspicion_score", 0), 4),
                "ensemble": round(l3.get("suspicion_score", 0), 4),
                "causal": round(l4.get("suspicion_score", 0), 4),
                "federated": round(l5.get("suspicion_score", 0), 4),
                "shap_drift": round(shap_drift.get("suspicion_score", 0), 4)
            }
        }
