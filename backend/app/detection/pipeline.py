"""
Detection Pipeline — Orchestrator  (RECTIFIED v3)
===================================================

NEW IN v3 — BLEND TRIGGER SCORE PROPAGATION:

  RECT 7 ── L1 blend_trigger alarm not surfaced in combined score
    OLD: The orchestrator reads r1["suspicion_score"] which in v4 did not
         include blend_contrib. Now that L1 v5 adds blend_contrib to its
         suspicion_score, this flows through correctly — BUT the alarm flag
         "alarm_blend_trigger" is not reflected in n_layers_alarmed counting
         or in the Attack Classification confidence routing.
    FIX: Add alarm_blend_trigger to the layer alarm tally. Also update
         _normalise_result to expose blend_score in the layer1 summary.

  RECT 8 ── L2 partial_backdoor (3/4 signals) gives 0 suspicion in v2
    OLD: L2 v2 only contributes suspicion when all 4 signals fire.
         3/4 signals → fires=3 → suspicion = 3 * 0.10 = 0.30, but the
         _normalise_result caps contributions at LAYER_WEIGHTS["l2"] = 0.20.
         In practice, 3/4 signals on a real backdoor gave L2 ≈ 0.03 overall.
    FIX: L2 v3 now returns partial_backdoor suspicion in [0.15, 0.50] range.
         The orchestrator uses the raw L2 suspicion_score which now reflects this.
         No orchestrator code change needed — L2 v3 handles it internally.

  RECT 9 ── LAYER_WEIGHTS["l1_statistical"] = 0.30 but blend trigger adds 0.35
             to L1's internal score. When L1 suspicion_score = 0.35 (blend only),
             the contribution to overall = 0.30 × 0.35 = 0.105 — below SUSPICIOUS.
    FIX: Increase l1_statistical weight to 0.35 and reduce l3_ensemble to 0.20
         so total remains 1.0. L1 is the most versatile detector; giving it
         more weight improves sensitivity for both spike and blend attacks.

  RECT 10 ── Verdict logic: if overall >= CONFIRMED but proof_valid is False,
             we emit SUSPICIOUS. But a blend backdoor rarely gets causal proof
             because L4 runs on flagged_indices from L3 (which may be 0).
             This creates a dead-end where a clear blend attack stays SUSPICIOUS.
    FIX: If alarm_blend_trigger=True AND alarm_l2_spectral=True, allow
         CONFIRMED_POISONED even without L4 causal proof (spectral + correlation
         evidence is sufficient for high-confidence blend backdoor calls).
         Otherwise preserve existing causal gate.

All previously documented rectifications (RECT 1–6 from v2) are preserved.
"""

import numpy as np
try:
    from .layer1_statistical import StatisticalShiftDetector
    from .layer2_spectral     import SpectralActivationAnalyzer
    from .layer3_ensemble     import EnsembleAnomalyDetector
    from .layer4_causal       import CausalProofEngine
    from .layer5_federated    import FederatedTrustAnalyzer
except ImportError:
    from layer1_statistical import StatisticalShiftDetector
    from layer2_spectral     import SpectralActivationAnalyzer
    from layer3_ensemble     import EnsembleAnomalyDetector
    from layer4_causal       import CausalProofEngine
    from layer5_federated    import FederatedTrustAnalyzer


# ── Layer weights (RECT 9: rebalanced) ────────────────────────────────────────
LAYER_WEIGHTS = {
    "l1_statistical": 0.35,   # RECT 9: was 0.30
    "l2_spectral"   : 0.20,
    "l3_ensemble"   : 0.20,   # RECT 9: was 0.25
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

    v3 key improvements:
      - L1 blend trigger alarm counted in layer alarm tally (RECT 7)
      - L2 partial_backdoor signal flows through (RECT 8, handled in L2 v3)
      - L1 weight increased to 0.35 for better blend/spike sensitivity (RECT 9)
      - Blend+spectral co-fire allows CONFIRMED without causal proof (RECT 10)
    """

    def __init__(self, random_state: int = 42, federated_mode: bool = False):
        self.l1              = StatisticalShiftDetector()
        self.l2              = SpectralActivationAnalyzer(random_state=random_state)
        self.l3              = EnsembleAnomalyDetector(random_state=random_state)
        self.l4              = CausalProofEngine(random_state=random_state)
        self.l5              = FederatedTrustAnalyzer()
        self.random_state    = random_state
        self.federated_mode  = federated_mode
        self._baseline_fitted = False
        self.X_reference     = None
        self.y_reference     = None

    def fit_baseline(self, X_reference, y_reference=None) -> None:
        if isinstance(X_reference, list):
            samples = X_reference
            X = np.array([s["feature_vector"] for s in samples], dtype=float)
            y = np.array([s.get("label", 0) for s in samples], dtype=int)
            col_min = X.min(axis=0); col_rng = X.max(axis=0) - col_min
            col_rng[col_rng == 0] = 1; X = (X - col_min) / col_rng
        else:
            X = np.array(X_reference, dtype=float)
            y = np.array(y_reference if y_reference is not None else
                         np.zeros(len(X_reference), dtype=int))

        self.X_reference = X
        self.y_reference = y
        self.l1.fit_baseline(X, y_reference=y)
        self.l3.fit(X)
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
            raise ValueError(
                "DetectionPipeline.analyze() called before fit_baseline(). "
                "Call pipeline.fit_baseline(X_clean, y_clean) first."
            )

        # Build full dataset
        if self.X_reference is not None and len(self.X_reference) > 0:
            X_full = np.vstack([self.X_reference, X])
            y_full = np.concatenate([self.y_reference, y])
        else:
            X_full, y_full = X, y

        # ── L1: full dataset ────────────────────────────────────────────────
        r1 = self.l1.analyze(X_full, y_incoming=y_full)

        # ── L2: full dataset ────────────────────────────────────────────────
        r2 = self.l2.analyze(X_full, y_full)

        # ── L3: full dataset ────────────────────────────────────────────────
        r3           = self.l3.analyze(X_full)
        flagged_full = r3.get("flagged_indices", [])

        # ── L4: receives X_full ─────────────────────────────────────────────
        r4 = self.l4.run(X_full, y_full, flagged_full)

        # ── L5: only in federated mode ──────────────────────────────────────
        if self.federated_mode:
            if federated_data and "client_gradients" in federated_data:
                r5 = self.l5.analyze(
                    federated_data["client_gradients"],
                    federated_data.get("global_gradient", np.zeros(10)),
                )
            else:
                r5 = self.l5.simulate_round()
        else:
            r5 = self._zero_l5_result()

        # ── Combine suspicion scores ─────────────────────────────────────────
        l3_score_raw   = r3["suspicion_score"]
        l3_excess      = r3.get("flagged_ratio", 0.0) - r3.get("expected_clean_flag_rate", 0.05)
        l3_score_gated = l3_score_raw if l3_excess > 0.02 else l3_score_raw * 0.2

        raw_score = (
            LAYER_WEIGHTS["l1_statistical"] * r1["suspicion_score"]
            + LAYER_WEIGHTS["l2_spectral"]  * r2["suspicion_score"]
            + LAYER_WEIGHTS["l3_ensemble"]  * l3_score_gated
            + LAYER_WEIGHTS["l4_causal"]    * r4["suspicion_score"]
            + LAYER_WEIGHTS["l5_federated"] * r5["suspicion_score"]
        )
        overall_suspicion = float(np.clip(raw_score, 0.0, 1.0))

        # RECT 10: Blend + spectral co-fire allows CONFIRMED without L4 proof
        blend_alarm    = r1.get("alarm_blend_trigger", False)
        spectral_alarm = r2.get("backdoor_detected", False) or r2.get("partial_backdoor", False)
        blend_spectral_confirmed = blend_alarm and spectral_alarm

        # ── Verdict ──────────────────────────────────────────────────────────
        if overall_suspicion >= THRESHOLD_CONFIRMED and (r4["proof_valid"] or blend_spectral_confirmed):
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
            "verdict"                 : verdict,
            "overall_suspicion"       : overall_suspicion,
            "degradation_score"       : r4.get("degradation_score", 0.0),
            "causal_proof_valid"      : r4["proof_valid"],
            "blend_spectral_confirmed": blend_spectral_confirmed,
            "layer_scores": {
                "l1_statistical"   : r1["suspicion_score"],
                "l2_spectral"      : r2["suspicion_score"],
                "l3_ensemble"      : l3_score_raw,
                "l3_ensemble_gated": l3_score_gated,
                "l4_causal"        : r4["suspicion_score"],
                "l5_federated"     : r5["suspicion_score"],
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
        n   = len(X)
        rng = np.random.RandomState(random_state)
        idx     = rng.permutation(n)
        split   = int(n * reference_fraction)
        ref_idx = idx[:split]
        inc_idx = idx[split:]

        pipeline = cls(random_state=random_state, federated_mode=federated_mode)
        pipeline.fit_baseline(X[ref_idx], y[ref_idx])
        result = pipeline.analyze(X[inc_idx], y[inc_idx])
        return pipeline, result

    # ═══════════════════════════════════════════════════════════════════════════
    # ADAPTER METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def run_on_upload(self, ingested: dict) -> dict:
        X        = ingested["features"]
        labels   = ingested.get("labels")
        split    = ingested["reference_split"]
        samples  = ingested["samples"]

        if labels is None:
            labels = np.zeros(len(X), dtype=int)

        X_ref    = X[:split]
        y_ref    = labels[:split]
        X_inc    = X[split:]
        y_inc    = labels[split:]

        self.fit_baseline(X_ref, y_ref)
        result   = self.analyze(X_inc, y_inc)

        return self._normalise_result(result, samples, X_inc)

    def run(self, samples: list, run_causal: bool = True) -> dict:
        X = np.array([s["feature_vector"] for s in samples], dtype=float)
        y = np.array([s.get("label", 0) for s in samples], dtype=int)

        col_min   = X.min(axis=0)
        col_range = X.max(axis=0) - col_min
        col_range[col_range == 0] = 1
        X = (X - col_min) / col_range

        n   = len(X)
        split = int(n * 0.70)
        self.fit_baseline(X[:split], y[:split])
        result = self.analyze(X[split:], y[split:])

        return self._normalise_result(result, samples, X[split:])

    @staticmethod
    def _normalise_result(result: dict, samples: list, X_inc: np.ndarray) -> dict:
        details = result.get("details", {})

        layer_results = {
            "layer1_statistical" : details.get("layer1", {}),
            "layer2_spectral"    : details.get("layer2", {}),
            "layer3_ensemble"    : details.get("layer3", {}),
            "layer4_causal"      : details.get("layer4", {}),
            "layer5_federated"   : details.get("layer5", {}),
        }

        l4 = layer_results["layer4_causal"]
        if "acc_with_poison" in l4 and "accuracy_with_poison" not in l4:
            l4["accuracy_with_poison"] = l4["acc_with_poison"]

        l5 = layer_results["layer5_federated"]
        if "avg_trust_score" in l5 and "avg_trust" not in l5:
            l5["avg_trust"] = l5["avg_trust_score"]

        # RECT 7: include blend_trigger alarm in layer count
        l1 = layer_results["layer1_statistical"]
        alarms = [
            l1.get("alarm_kl", False) or l1.get("alarm_mahal", False) or
            l1.get("alarm_blend_trigger", False),          # RECT 7
            layer_results["layer2_spectral"].get("backdoor_detected", False) or
            layer_results["layer2_spectral"].get("partial_backdoor", False),  # RECT 8
            layer_results["layer3_ensemble"].get("flagged_ratio", 0) > 0.08,
            layer_results["layer4_causal"].get("proof_valid", False),
            layer_results["layer5_federated"].get("n_quarantined", 0) > 0,
        ]
        n_layers_alarmed = sum(1 for a in alarms if a)

        overall = result.get("overall_suspicion", 0.0)
        verdict = result.get("verdict", "CLEAN")

        flagged_idx  = set(layer_results["layer3_ensemble"].get("flagged_indices", []))
        n_ref_stored = len(samples) - len(X_inc)
        ensemble_scores = [
            1.0 if (i + n_ref_stored) in flagged_idx else 0.1
            for i in range(len(X_inc))
        ]
        layer_results["layer3_ensemble"]["ensemble_scores"] = ensemble_scores

        return {
            "overall_suspicion_score" : overall,
            "verdict"                 : verdict,
            "layer_results"           : layer_results,
            "layer_scores"            : result.get("layer_scores", {}),
            "n_layers_alarmed"        : n_layers_alarmed,
            "n_samples"               : len(X_inc),
            "requires_human_review"   : 0.35 <= overall < 0.65,
            "degradation_score"       : result.get("degradation_score", 0.0),
            "causal_proof_valid"      : result.get("causal_proof_valid", False),
            "blend_spectral_confirmed": result.get("blend_spectral_confirmed", False),
            "thresholds"              : result.get("thresholds", {}),
            "overall_suspicion"       : overall,
            "details"                 : details,
        }
