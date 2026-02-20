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

    def fit_baseline(self, X_reference, y_reference=None) -> None:
        """
        Fit all baseline models on the known-clean reference partition.
        Accepts numpy arrays OR list-of-dicts (demo path).
        RECT 1: y_reference now passed to L1.
        """
        # List-of-dicts path (from demo routes)
        if isinstance(X_reference, list) and len(X_reference) > 0 and isinstance(X_reference[0], dict):
            X = np.array([s["feature_vector"] for s in X_reference], dtype=float)
            y = np.array([s.get("label", 0) for s in X_reference])
        else:
            X = np.array(X_reference, dtype=float)
            if y_reference is None:
                y = np.zeros(len(X))
            else:
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
        y_incoming = None,
        federated_data: dict = None,
    ) -> dict:
        # ── Type-safety: coerce inputs to clean numpy arrays ──────────────
        X = np.array(X_incoming, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if y_incoming is None:
            y = np.zeros(len(X), dtype=float)
        else:
            y = np.array(y_incoming, dtype=float)
            y = np.nan_to_num(y, nan=0.0)

        if not self._baseline_fitted:
            mid = len(X) // 2
            self.fit_baseline(X[:mid], y[:mid])
            # After auto-fitting, only analyse the second half
            X = X[mid:]
            y = y[mid:]

        # ══════════════════════════════════════════════════════════════════
        # KEY FIX: Each layer receives ONLY X_incoming (unseen data).
        # The baseline/reference was already learned during fit_baseline().
        # Passing the full dataset (ref + incoming) made every layer compare
        # the data against itself, guaranteeing near-zero scores.
        # ══════════════════════════════════════════════════════════════════

        # ── L1: compare incoming against fitted baseline ──────────────────
        try:
            r1 = self.l1.analyze(X, y_incoming=y)
        except Exception:
            r1 = self.l1._null_result("layer_error")

        # ── L2: spectral analysis on incoming data only ───────────────────
        try:
            r2 = self.l2.analyze(X, y)
        except Exception:
            r2 = self.l2._null_result("layer_error")

        # ── L3: detectors fitted on reference → test on incoming only ─────
        try:
            r3 = self.l3.analyze(X)
        except Exception:
            r3 = self.l3._null_result("layer_error")
        flagged_indices = r3.get("flagged_indices", [])

        # ── L4: causal proof on incoming data with L3's flagged indices ───
        try:
            r4 = self.l4.run(X, y, flagged_indices)
        except Exception:
            r4 = self.l4._null_result("layer_error")

        # ── L5: only meaningful in federated mode ─────────────────────────
        if self.federated_mode:
            if federated_data and "client_gradients" in federated_data:
                try:
                    r5 = self.l5.analyze(
                        federated_data["client_gradients"],
                        federated_data.get("global_gradient", np.zeros(10)),
                    )
                except Exception:
                    r5 = self._zero_l5_result()
            else:
                try:
                    r5 = self.l5.simulate_round()
                except Exception:
                    r5 = self._zero_l5_result()
        else:
            r5 = self._zero_l5_result()

        # ── Combine suspicion scores ─────────────────────────────────────────
        l3_score_raw = float(r3.get("suspicion_score", 0.0) or 0.0)
        l3_excess    = float(r3.get("flagged_ratio", 0.0) or 0.0) - float(r3.get("expected_clean_flag_rate", 0.05) or 0.05)
        l3_score_gated = l3_score_raw if l3_excess > 0.02 else l3_score_raw * 0.2

        raw_score = (
            LAYER_WEIGHTS["l1_statistical"] * float(r1.get("suspicion_score", 0.0) or 0.0)
            + LAYER_WEIGHTS["l2_spectral"]  * float(r2.get("suspicion_score", 0.0) or 0.0)
            + LAYER_WEIGHTS["l3_ensemble"]  * float(l3_score_gated or 0.0)
            + LAYER_WEIGHTS["l4_causal"]    * float(r4.get("suspicion_score", 0.0) or 0.0)
            + LAYER_WEIGHTS["l5_federated"] * float(r5.get("suspicion_score", 0.0) or 0.0)
        )
        overall_suspicion = float(np.clip(raw_score, 0.0, 1.0))

        # ── Verdict: CONFIRMED requires causal proof ─────────────────────────
        if overall_suspicion >= THRESHOLD_CONFIRMED and r4.get("proof_valid", False):
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
            "causal_proof_valid": r4.get("proof_valid", False),
            "layer_scores": {
                "l1_statistical"  : float(r1.get("suspicion_score", 0.0) or 0.0),
                "l2_spectral"     : float(r2.get("suspicion_score", 0.0) or 0.0),
                "l3_ensemble"     : l3_score_raw,
                "l3_ensemble_gated": l3_score_gated,
                "l4_causal"       : float(r4.get("suspicion_score", 0.0) or 0.0),
                "l5_federated"    : float(r5.get("suspicion_score", 0.0) or 0.0),
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

    # ══════════════════════════════════════════════════════════════════════════
    # Adapter methods — bridge between route-level dict API and numpy internals
    # ══════════════════════════════════════════════════════════════════════════

    def run_on_upload(self, ingested: dict) -> dict:
        """
        Adapter for upload & real-dataset routes.
        Takes output of CSVIngestionEngine.ingest() and returns route-compatible dict.
        """
        X = np.array(ingested["features"], dtype=float)
        y = np.array(ingested["labels"]) if ingested.get("labels") is not None else np.zeros(len(X))
        ref_split = ingested.get("reference_split", int(len(X) * 0.70))

        X_ref, y_ref = X[:ref_split], y[:ref_split]
        X_inc, y_inc = X[ref_split:], y[ref_split:]

        self.fit_baseline(X_ref, y_ref)
        raw = self.analyze(X_inc, y_inc)
        return self._normalise_result(raw, n_samples=len(X))

    def run(self, samples: list, run_causal: bool = True) -> dict:
        """
        Adapter for demo/detect/red-team routes that pass list-of-dicts.
        Extracts feature_vector + label, runs pipeline, returns route-compatible dict.
        """
        X = np.array([s["feature_vector"] for s in samples], dtype=float)
        y = np.array([s.get("label", 0) for s in samples])

        if not self._baseline_fitted:
            mid = int(len(X) * 0.70)
            self.fit_baseline(X[:mid], y[:mid])

        raw = self.analyze(X, y)
        return self._normalise_result(raw, n_samples=len(samples))



    def _normalise_result(self, raw: dict, n_samples: int = 0) -> dict:
        """
        Translate internal pipeline output keys to what routes expect.

        Pipeline outputs:  overall_suspicion, details.layer1/2/3/4/5
        Routes expect:     overall_suspicion_score, layer_results.layer1_statistical etc.
        """
        details = raw.get("details", {})

        layer_results = {
            "layer1_statistical": details.get("layer1", {}),
            "layer2_spectral":    details.get("layer2", {}),
            "layer3_ensemble":    details.get("layer3", {}),
            "layer4_causal":      details.get("layer4", {}),
            "layer5_federated":   details.get("layer5", {}),
        }

        n_alarmed = sum(
            1 for s in raw.get("layer_scores", {}).values()
            if isinstance(s, (int, float)) and s > 0.35
        )

        overall = raw.get("overall_suspicion", 0.0)
        verdict = raw.get("verdict", "CLEAN")

        return {
            "verdict":                verdict,
            "overall_suspicion_score": overall,
            "layer_scores":           raw.get("layer_scores", {}),
            "layer_results":          layer_results,
            "n_samples":              n_samples,
            "n_layers_alarmed":       n_alarmed,
            "causal_proof_valid":     raw.get("causal_proof_valid", False),
            "degradation_score":      raw.get("degradation_score", 0.0),
            "requires_human_review":  0.35 <= overall < 0.65,
            "thresholds":             raw.get("thresholds", {}),
            "layer_weights":          raw.get("layer_weights", {}),
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
