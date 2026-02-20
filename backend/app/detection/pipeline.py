"""
Detection Pipeline — Orchestrator
====================================
FIX SUMMARY:
- OLD: The overall suspicion score was a flat weighted average of all 5 layers.
  This meant a high L3 score (which on clean data was ~0.3 due to contamination)
  contributed 25% to the final score, pushing clean data to ~0.15–0.25 suspicion,
  which then crossed the SUSPICIOUS threshold.
- FIX 1: Layer 3's contribution is gated — it only contributes its full weight
  if the flagged_ratio significantly exceeds the expected clean baseline (5%).
- FIX 2: Layer 4 (causal) now GATES the final verdict. If causal proof is not
  valid, the verdict cannot be CONFIRMED_POISONED regardless of other layers.
  This prevents false positives where statistical noise in L1/L3 triggers a
  false confirmation.
- FIX 3: The verdict thresholds are adjusted to account for the corrected
  score distributions.
- FIX 4: The pipeline now passes the L3 flagged_indices to L4 so the causal
  experiment runs on the right samples (was previously running on empty list).
"""

import numpy as np
from .layer1_statistical import StatisticalShiftDetector
from .layer2_spectral import SpectralActivationAnalyzer
from .layer3_ensemble import EnsembleAnomalyDetector
from .layer4_causal import CausalProofEngine
from .layer5_federated import FederatedTrustAnalyzer


# ── Layer weights for overall suspicion score ──
LAYER_WEIGHTS = {
    "l1_statistical" : 0.30,   # was 0.25 — primary signal, most reliable
    "l2_spectral"    : 0.20,
    "l3_ensemble"    : 0.25,   # was 0.25 — but now gated (see below)
    "l4_causal"      : 0.20,   # causal proof is gating verdict, not just score
    "l5_federated"   : 0.05,   # was 0.10 — reduced: only relevant in FL scenarios
}

# ── Verdict thresholds ──
THRESHOLD_CONFIRMED  = 0.65
THRESHOLD_SUSPICIOUS = 0.35
THRESHOLD_LOW_RISK   = 0.15


class DetectionPipeline:
    """
    Orchestrates all 5 detection layers and produces a combined verdict.

    Usage:
        pipeline = DetectionPipeline()
        pipeline.fit_baseline(X_reference, y_reference)
        result = pipeline.analyze(X_incoming, y_incoming)
    """

    def __init__(self, random_state: int = 42):
        self.l1 = StatisticalShiftDetector()
        self.l2 = SpectralActivationAnalyzer(random_state=random_state)
        self.l3 = EnsembleAnomalyDetector(random_state=random_state)
        self.l4 = CausalProofEngine(random_state=random_state)
        self.l5 = FederatedTrustAnalyzer()
        self._baseline_fitted = False
        self.X_reference = None
        self.y_reference = None

    def fit_baseline(self, X_reference: np.ndarray, y_reference: np.ndarray) -> None:
        """
        Fit all baseline models on the known-clean reference partition (70% split).
        Must be called before analyze().
        """
        X = np.array(X_reference, dtype=float)
        y = np.array(y_reference)
        self.X_reference = X
        self.y_reference = y
        self.l1.fit_baseline(X)
        self.l3.fit(X)   # L3 learns what "normal" looks like
        self._baseline_fitted = True

    def analyze(
        self,
        X_incoming: np.ndarray,
        y_incoming: np.ndarray,
        federated_data: dict = None,
    ) -> dict:
        """
        Run the full 5-layer analysis on incoming data.

        Parameters
        ----------
        X_incoming      : incoming data features (30% split)
        y_incoming      : incoming data labels
        federated_data  : optional dict {client_id: gradient} for L5

        Returns
        -------
        Full analysis result dict with all layer outputs + combined verdict.
        """
        X = np.array(X_incoming, dtype=float)
        y = np.array(y_incoming)

        if not self._baseline_fitted:
            # Fit on incoming if no baseline (unsupervised mode)
            mid = len(X) // 2
            self.fit_baseline(X[:mid], y[:mid])

        # ── Run all layers ──────────────────────────────────────────────────
        r1 = self.l1.analyze(X)
        r2 = self.l2.analyze(X, y)
        r3 = self.l3.analyze(X)

        # Pass L3 flagged indices to L4 (THE BUG FIX — was passing empty list)
        flagged_idx = r3.get("flagged_indices", [])
        # Also combine with reference data for causal proof (need full picture)
        if self.X_reference is not None and len(self.X_reference) > 0:
            X_full = np.vstack([self.X_reference, X])
            y_full = np.concatenate([self.y_reference, y])
            # Shift flagged indices to account for prepended reference
            n_ref  = len(self.X_reference)
            flagged_full = [i + n_ref for i in flagged_idx]
        else:
            X_full, y_full, flagged_full = X, y, flagged_idx

        r4 = self.l4.run(X_full, y_full, flagged_full)

        # L5: use provided federated data or simulate
        if federated_data and "client_gradients" in federated_data:
            r5 = self.l5.analyze(
                federated_data["client_gradients"],
                federated_data.get("global_gradient", np.zeros(10)),
            )
        else:
            r5 = self.l5.simulate_round()

        # ── Combine suspicion scores ─────────────────────────────────────────
        # FIX: L3's contribution is capped if flagged_ratio ≈ expected baseline
        l3_score_raw = r3["suspicion_score"]
        l3_excess    = r3.get("flagged_ratio", 0.0) - r3.get("expected_clean_flag_rate", 0.05)
        if l3_excess <= 0.02:  # barely above expected clean baseline → near-zero contribution
            l3_score_gated = l3_score_raw * 0.2
        else:
            l3_score_gated = l3_score_raw

        raw_score = (
            LAYER_WEIGHTS["l1_statistical"] * r1["suspicion_score"]
            + LAYER_WEIGHTS["l2_spectral"]    * r2["suspicion_score"]
            + LAYER_WEIGHTS["l3_ensemble"]    * l3_score_gated
            + LAYER_WEIGHTS["l4_causal"]      * r4["suspicion_score"]
            + LAYER_WEIGHTS["l5_federated"]   * r5["suspicion_score"]
        )
        overall_suspicion = float(np.clip(raw_score, 0.0, 1.0))

        # ── Verdict ──────────────────────────────────────────────────────────
        # FIX: CONFIRMED_POISONED requires causal proof to be valid.
        # Without causal proof, max verdict is SUSPICIOUS even if score > 0.65.
        if overall_suspicion >= THRESHOLD_CONFIRMED and r4["proof_valid"]:
            verdict = "CONFIRMED_POISONED"
        elif overall_suspicion >= THRESHOLD_CONFIRMED and not r4["proof_valid"]:
            verdict = "SUSPICIOUS"   # high score but no causal proof → downgrade
        elif overall_suspicion >= THRESHOLD_SUSPICIOUS:
            verdict = "SUSPICIOUS"
        elif overall_suspicion >= THRESHOLD_LOW_RISK:
            verdict = "LOW_RISK"
        else:
            verdict = "CLEAN"

        # ── Degradation score (from L4 — fixed) ──────────────────────────────
        degradation_score = r4.get("degradation_score", 0.0)

        return {
            "verdict"            : verdict,
            "overall_suspicion"  : overall_suspicion,
            "degradation_score"  : degradation_score,
            "causal_proof_valid" : r4["proof_valid"],
            "layer_scores"       : {
                "l1_statistical" : r1["suspicion_score"],
                "l2_spectral"    : r2["suspicion_score"],
                "l3_ensemble"    : l3_score_raw,
                "l3_ensemble_gated": l3_score_gated,
                "l4_causal"      : r4["suspicion_score"],
                "l5_federated"   : r5["suspicion_score"],
            },
            "layer_weights"      : LAYER_WEIGHTS,
            "thresholds"         : {
                "confirmed"      : THRESHOLD_CONFIRMED,
                "suspicious"     : THRESHOLD_SUSPICIOUS,
                "low_risk"       : THRESHOLD_LOW_RISK,
            },
            "details"            : {
                "layer1"         : r1,
                "layer2"         : r2,
                "layer3"         : r3,
                "layer4"         : r4,
                "layer5"         : r5,
            },
        }

    @classmethod
    def from_split(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        reference_fraction: float = 0.70,
        random_state: int = 42,
    ) -> tuple:
        """
        Convenience: split data into reference/incoming, fit, and analyse.
        Returns (pipeline, result).
        """
        n = len(X)
        split = int(n * reference_fraction)
        idx = np.random.RandomState(random_state).permutation(n)
        ref_idx = idx[:split]
        inc_idx = idx[split:]

        pipeline = cls(random_state=random_state)
        pipeline.fit_baseline(X[ref_idx], y[ref_idx])
        result = pipeline.analyze(X[inc_idx], y[inc_idx])
        return pipeline, result
