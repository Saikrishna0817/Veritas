"""
Detection Pipeline — Orchestrator  (RECTIFIED v2)
===================================================

RECTIFICATIONS:

  RECT 1 ── fit_baseline does NOT pass y_reference to L1
    OLD: self.l1.fit_baseline(X)
         StatisticalShiftDetector.fit_baseline() accepts y_reference but
         receives nothing here. This means baseline_label_ratio = None
         and label-flip is NEVER detected by Layer 1 in the pipeline.
    FIX: self.l1.fit_baseline(X, y_reference=y)

  RECT 2 ── L2 receives X_incoming (30% split) but should see the FULL dataset
    OLD: r2 = self.l2.analyze(X, y)
         Spectral analysis on a 30% slice means only ~1500 samples are analysed.
         The backdoor trigger set (e.g. 0.8% of 5000 = 40 samples) shrinks to
         ~12 samples in the 30% slice, making the minority cluster invisible.
    FIX: If X_reference is available, concatenate it with X_incoming for L2.
         L2 is read-only (no model is fitted), so this is safe.

  RECT 3 ── L1 also receives X_incoming (30% split) not the full dataset
    OLD: r1 = self.l1.analyze(X)  — same slicing problem as L2.
         For the backdoor attack (39 trigger samples in 5000), only ~12
         appear in the 30% split — below the binomial test threshold.
    FIX: Pass the full dataset (X_full, y_full) to L1 as well.
         L1's analyze() is also read-only relative to the fitted baseline.

  RECT 4 ── L3 flagged_indices are relative to X_incoming but are shifted
            by n_ref before passing to L4 — but L3 ran on X_incoming only
    OLD: flagged_full = [i + n_ref for i in flagged_idx]
         Then L4 receives X_full (ref + incoming) and flagged_full.
         But the flagged indices from L3 were produced by running L3 on
         X_incoming only (0-indexed from 0 to len(X_incoming)-1).
         After shift: flagged_full correctly maps into X_full. ✓
         HOWEVER: L3 was not fitted to correctly detect patterns in X_full.
         L3 should also analyze X_full (not just the incoming slice) so
         it sees the full anomaly picture.
    FIX: Pass X_full to L3.analyze() and do NOT shift indices (they are
         already 0-indexed into X_full since L3 now processes X_full).
         Remove the n_ref shift since L3 operates on X_full directly.

  RECT 5 ── from_split() uses np.random.RandomState directly — not reproducible
            if caller sets a global numpy seed beforehand
    FIX: Use the instance's random_state parameter consistently.

  RECT 6 ── L5 simulate_round() is called when federated_data is None, but
            simulated gradients are synthetic noise unrelated to the actual data.
            This means L5 always contributes some non-zero suspicion in
            non-federated scenarios, polluting the overall score.
    FIX: When federated_data is None and this is not a federated scenario,
         return a zero-suspicion L5 result instead of simulating.
         Add a `federated_mode` flag to the pipeline.

  Previously fixed bugs (L3 gating, L4 causal gate on verdict,
  L3 flagged_indices passed to L4) are preserved.
"""

import numpy as np
from .layer1_statistical import StatisticalShiftDetector
from .layer2_spectral     import SpectralActivationAnalyzer
from .layer3_ensemble     import EnsembleAnomalyDetector
from .layer4_causal       import CausalProofEngine
from .layer5_federated    import FederatedTrustAnalyzer


# ── Layer weights ──────────────────────────────────────────────────────────────
LAYER_WEIGHTS = {
    "l1_statistical": 0.30,
    "l2_spectral"   : 0.20,
    "l3_ensemble"   : 0.25,
    "l4_causal"     : 0.20,
    "l5_federated"  : 0.05,
}

# ── Verdict thresholds ─────────────────────────────────────────────────────────
THRESHOLD_CONFIRMED  = 0.65
THRESHOLD_SUSPICIOUS = 0.35
THRESHOLD_LOW_RISK   = 0.15


class DetectionPipeline:
    """
    Orchestrates all 5 detection layers and produces a combined verdict.

    Key architectural guarantees:
      - CONFIRMED_POISONED requires L4 causal proof to be valid.
      - L1, L2, L3 all operate on the FULL dataset (ref + incoming) so
        low-density attacks (e.g. 0.8% backdoor triggers) are visible.
      - L3 flagged indices map directly into X_full (no index shift needed).
      - L5 only contributes when federated_mode=True.

    Usage:
        pipeline = DetectionPipeline(federated_mode=False)
        pipeline.fit_baseline(X_reference, y_reference)
        result = pipeline.analyze(X_incoming, y_incoming)
    """

    def __init__(self, random_state: int = 42, federated_mode: bool = False):
        self.l1              = StatisticalShiftDetector()
        self.l2              = SpectralActivationAnalyzer(random_state=random_state)
        self.l3              = EnsembleAnomalyDetector(random_state=random_state)
        self.l4              = CausalProofEngine(random_state=random_state)
        self.l5              = FederatedTrustAnalyzer()
        self.random_state    = random_state
        self.federated_mode  = federated_mode   # RECT 6
        self._baseline_fitted = False
        self.X_reference     = None
        self.y_reference     = None

    def fit_baseline(self, X_reference: np.ndarray, y_reference: np.ndarray) -> None:
        """
        Fit all baseline models on the known-clean reference partition.
        RECT 1: y_reference now passed to L1.
        """
        X = np.array(X_reference, dtype=float)
        y = np.array(y_reference)
        self.X_reference = X
        self.y_reference = y

        # RECT 1: pass y so label-flip baseline is stored
        self.l1.fit_baseline(X, y_reference=y)
        self.l3.fit(X)   # L3 learns normal distribution on clean reference
        self._baseline_fitted = True

    def analyze(
        self,
        X_incoming: np.ndarray,
        y_incoming: np.ndarray,
        federated_data: dict = None,
    ) -> dict:
        X = np.array(X_incoming, dtype=float)
        y = np.array(y_incoming)

        if not self._baseline_fitted:
            mid = len(X) // 2
            self.fit_baseline(X[:mid], y[:mid])

        # ── Build full dataset for analysis (RECT 2, 3, 4) ──────────────────
        if self.X_reference is not None and len(self.X_reference) > 0:
            X_full = np.vstack([self.X_reference, X])
            y_full = np.concatenate([self.y_reference, y])
        else:
            X_full, y_full = X, y

        # ── L1: full dataset (RECT 3) ────────────────────────────────────────
        r1 = self.l1.analyze(X_full, y_incoming=y_full)

        # ── L2: full dataset (RECT 2) ────────────────────────────────────────
        r2 = self.l2.analyze(X_full, y_full)

        # ── L3: full dataset (RECT 4) — indices map directly into X_full ────
        r3           = self.l3.analyze(X_full)
        flagged_full = r3.get("flagged_indices", [])

        # ── L4: receives X_full and directly-mapped flagged indices ──────────
        r4 = self.l4.run(X_full, y_full, flagged_full)

        # ── L5: only meaningful in federated mode (RECT 6) ──────────────────
        if self.federated_mode:
            if federated_data and "client_gradients" in federated_data:
                r5 = self.l5.analyze(
                    federated_data["client_gradients"],
                    federated_data.get("global_gradient", np.zeros(10)),
                )
            else:
                r5 = self.l5.simulate_round()
        else:
            # Non-federated: L5 contributes zero so it doesn't pollute score
            r5 = self._zero_l5_result()

        # ── Combine suspicion scores ─────────────────────────────────────────
        l3_score_raw = r3["suspicion_score"]
        l3_excess    = r3.get("flagged_ratio", 0.0) - r3.get("expected_clean_flag_rate", 0.05)
        l3_score_gated = l3_score_raw if l3_excess > 0.02 else l3_score_raw * 0.2

        raw_score = (
            LAYER_WEIGHTS["l1_statistical"] * r1["suspicion_score"]
            + LAYER_WEIGHTS["l2_spectral"]  * r2["suspicion_score"]
            + LAYER_WEIGHTS["l3_ensemble"]  * l3_score_gated
            + LAYER_WEIGHTS["l4_causal"]    * r4["suspicion_score"]
            + LAYER_WEIGHTS["l5_federated"] * r5["suspicion_score"]
        )
        overall_suspicion = float(np.clip(raw_score, 0.0, 1.0))

        # ── Verdict: CONFIRMED requires causal proof ─────────────────────────
        if overall_suspicion >= THRESHOLD_CONFIRMED and r4["proof_valid"]:
            verdict = "CONFIRMED_POISONED"
        elif overall_suspicion >= THRESHOLD_CONFIRMED:
            verdict = "SUSPICIOUS"
        elif overall_suspicion >= THRESHOLD_SUSPICIOUS:
            verdict = "SUSPICIOUS"
        elif overall_suspicion >= THRESHOLD_LOW_RISK:
            verdict = "LOW_RISK"
        else:
            verdict = "CLEAN"

        return {
            "verdict"           : verdict,
            "overall_suspicion" : overall_suspicion,
            "degradation_score" : r4.get("degradation_score", 0.0),
            "causal_proof_valid": r4["proof_valid"],
            "layer_scores": {
                "l1_statistical"  : r1["suspicion_score"],
                "l2_spectral"     : r2["suspicion_score"],
                "l3_ensemble"     : l3_score_raw,
                "l3_ensemble_gated": l3_score_gated,
                "l4_causal"       : r4["suspicion_score"],
                "l5_federated"    : r5["suspicion_score"],
            },
            "layer_weights": LAYER_WEIGHTS,
            "thresholds": {
                "confirmed"  : THRESHOLD_CONFIRMED,
                "suspicious" : THRESHOLD_SUSPICIOUS,
                "low_risk"   : THRESHOLD_LOW_RISK,
            },
            "details": {
                "layer1": r1,
                "layer2": r2,
                "layer3": r3,
                "layer4": r4,
                "layer5": r5,
            },
        }

    @staticmethod
    def _zero_l5_result() -> dict:
        """Return a neutral L5 result for non-federated scenarios."""
        return {
            "suspicion_score"    : 0.0,
            "n_clients"          : 0,
            "n_quarantined"      : 0,
            "n_suspicious"       : 0,
            "quarantined_clients": [],
            "suspicious_clients" : [],
            "per_client"         : {},
            "avg_trust_score"    : 1.0,
            "skip_reason"        : "non_federated_mode",
        }

    @classmethod
    def from_split(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        reference_fraction: float = 0.70,
        random_state: int         = 42,
        federated_mode: bool      = False,
    ) -> tuple:
        """
        Convenience: split, fit, and analyse in one call.
        RECT 5: uses random_state consistently.
        Returns (pipeline, result).
        """
        n   = len(X)
        rng = np.random.RandomState(random_state)   # RECT 5: use instance seed
        idx     = rng.permutation(n)
        split   = int(n * reference_fraction)
        ref_idx = idx[:split]
        inc_idx = idx[split:]

        pipeline = cls(random_state=random_state, federated_mode=federated_mode)
        pipeline.fit_baseline(X[ref_idx], y[ref_idx])
        result = pipeline.analyze(X[inc_idx], y[inc_idx])
        return pipeline, result
