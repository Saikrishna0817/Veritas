"""Detection pipeline: orchestrates the five layers plus SHAP drift.

Layer suspicion scores are combined with ``LAYER_WEIGHTS``. Layer 3 is fitted on the
reference rows and scores only the incoming batch; a low-confidence Layer 3 score is
suppressed (see the calibration notes below). ``CONFIRMED_POISONED`` requires a valid
Layer-4 causal proof, unless a blend-trigger alarm and a spectral alarm co-fire.
Design history is in git; measured performance is in ``backend/research/results``.
"""

import numpy as np
try:
    from .layer1_statistical import StatisticalShiftDetector
    from .layer2_spectral     import SpectralActivationAnalyzer
    from .layer3_ensemble     import EnsembleAnomalyDetector
    from .layer4_causal       import CausalProofEngine
    from .layer5_federated    import FederatedTrustAnalyzer
    from .shap_drift          import SHAPDriftMonitor
    from .veritas2            import Veritas2, SUSPICIOUS_AT, verdict_for
except ImportError:
    from layer1_statistical import StatisticalShiftDetector
    from layer2_spectral     import SpectralActivationAnalyzer
    from layer3_ensemble     import EnsembleAnomalyDetector
    from layer4_causal       import CausalProofEngine
    from layer5_federated    import FederatedTrustAnalyzer
    from shap_drift          import SHAPDriftMonitor
    from veritas2            import Veritas2, SUSPICIOUS_AT, verdict_for


# ── Layer weights (RECT 9: rebalanced) ────────────────────────────────────────
# Step 3 retained these values: `backend/research/results/2026-09-02-benchmark-v1-step2-l3-gate.json`
# reports 0.00 incident F1 for every individual layer and
# for the combination. The only non-zero Layer-3 evidence is row-level, so
# changing a batch-level blend weight would not be evidence-based calibration.
LAYER_WEIGHTS = {
    "l1_statistical": 0.35,   # RECT 9: was 0.30
    "l2_spectral"   : 0.20,
    "l3_ensemble"   : 0.20,   # RECT 9: was 0.25
    "l4_causal"     : 0.20,
    "l5_federated"  : 0.05,
}

# Calibration evidence: `backend/research/results/2026-09-02-benchmark-v1-
# step1-incoming-ratio.json` measured a 0.00 clean-batch Layer-3 flag rate and
# a 0.06 boiling-frog incoming flag rate. The former 0.02 extra margin
# suppressed that measured attack signal by 80%; retaining any rate above the
# detector's own expected clean rate is the least permissive correction tested
# by that benchmark.
L3_GATE_EXCESS_MARGIN = 0.00
L3_LOW_CONFIDENCE_SUPPRESSION = 0.20

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

    def __init__(self, random_state: int = 42, federated_mode: bool = False, engine: str = "v2"):
        # "v2" (default): label check + gated shift check, ~2-5 s per batch, measured in docs/benchmark-v2-findings.md.
        # "legacy": the original five layers; kept for benchmark comparison, at chance on label noise and ~40-130 s per batch.
        if engine not in ("v2", "legacy"):
            raise ValueError(f"unknown engine {engine!r}")
        self.engine          = engine
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
        if self.engine == "legacy":
            self.l1.fit_baseline(X, y_reference=y)
            self.l3.fit(X)
        else:
            self.v2 = Veritas2(self.random_state).fit(X, y)
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

        if self.engine == "v2":
            return self._analyze_v2(X, y)

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

        # ── L3: reference-fit, incoming-score ───────────────────────────────
        # Layer 3 is fitted in `fit_baseline()` on clean reference data. Score
        # only the incoming batch here so its flagged ratio describes the batch
        # under judgment, rather than being diluted by reference rows. Layer 4
        # still receives full-dataset indices for its existing API.
        r3 = self._analyze_l3_incoming(X)
        flagged_full = r3["flagged_indices"]

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

        # ── SHAP drift: reference vs incoming batches ───────────────────────
        r_shap = self._analyze_shap_drift(X, y)

        # ── Combine suspicion scores ─────────────────────────────────────────
        l3_score_raw = r3["suspicion_score"]
        l3_score_gated = self._gate_l3_score(l3_score_raw, r3)

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
                "shap_drift": r_shap,
            },
        }

    def _analyze_v2(self, X: np.ndarray, y: np.ndarray) -> dict:
        r = self.v2.analyze(X, y)
        flagged = r["flagged_indices"]
        label_ev, shift_ev = r["evidence"].get("label", 0.0), r["evidence"]["shift"]
        # Same keys the forensics/defense/report code reads; old layers that no longer run are empty.
        details = {
            "engine": "v2",
            "label_check": {"evidence": label_ev, "alarm": label_ev >= SUSPICIOUS_AT, "ran": "label" in r["evidence"]},
            "shift_check": {"evidence": shift_ev, "alarm": shift_ev >= SUSPICIOUS_AT, "in_row_ranking": r["shift_in_ranking"]},
            "layer3": {"flagged_indices": [len(self.X_reference) + i for i in flagged], "flagged_indices_incoming": flagged,
                       "flagged_ratio": len(flagged) / max(len(X), 1), "row_scores": [round(float(v), 3) for v in r["row_scores"]],
                       "suspicion_score": r["suspicion"]},
            "warnings": r["warnings"],
        }
        return {
            "verdict": r["verdict"], "overall_suspicion": r["suspicion"], "degradation_score": 0.0,
            "causal_proof_valid": False, "blend_spectral_confirmed": False,
            "layer_scores": {"label_check": verdict_for(label_ev)[1], "shift_check": verdict_for(shift_ev)[1]},  # 0-1 for the UI
            "layer_weights": {}, "details": details,
            "thresholds": {"confirmed": THRESHOLD_CONFIRMED, "suspicious": THRESHOLD_SUSPICIOUS, "low_risk": THRESHOLD_LOW_RISK},
        }

    def _analyze_shap_drift(self, X_incoming: np.ndarray, y_incoming: np.ndarray) -> dict:
        """Compare explanation drift between the reference baseline and incoming batch."""
        if self.X_reference is None or len(self.X_reference) < 30 or len(X_incoming) < 10:
            return {
                "suspicion_score": 0.0,
                "drift_score": 0.0,
                "cumulative_drift": 0.0,
                "alarm": False,
                "skip_reason": "insufficient_samples",
            }

        monitor = SHAPDriftMonitor(feature_names=[f"feature_{i}" for i in range(X_incoming.shape[1])])
        monitor.record_snapshot(self.X_reference, self.y_reference, batch_id="reference")
        monitor.record_snapshot(X_incoming, y_incoming, batch_id="incoming")
        drift = monitor.compute_drift()
        drift["suspicion_score"] = drift.get("suspicion_score", 0.0)
        drift["drift_timeline"] = monitor.get_drift_timeline()
        return drift

    def _analyze_l3_incoming(self, X_incoming: np.ndarray) -> dict:
        """Score only incoming rows and map local flags to full-dataset indices."""
        result = self.l3.analyze(X_incoming)
        flagged_incoming = result.get("flagged_indices", [])
        reference_count = len(self.X_reference) if self.X_reference is not None else 0
        result["flagged_indices_incoming"] = flagged_incoming
        result["flagged_indices"] = [reference_count + index for index in flagged_incoming]
        return result

    @staticmethod
    def _gate_l3_score(l3_score_raw: float, l3_result: dict) -> float:
        """Suppress only Layer-3 rates at or below its expected clean baseline."""
        l3_excess = (l3_result.get("flagged_ratio", 0.0)
                     - l3_result.get("expected_clean_flag_rate", 0.05))
        if l3_excess > L3_GATE_EXCESS_MARGIN:
            return l3_score_raw
        return l3_score_raw * L3_LOW_CONFIDENCE_SUPPRESSION

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

    def run(self, samples: list) -> dict:
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
    def _normalise_v2(result: dict, details: dict, X_inc: np.ndarray) -> dict:
        label, shift = details["label_check"], details["shift_check"]
        overall = result["overall_suspicion"]
        return {
            "overall_suspicion_score": overall, "overall_suspicion": overall, "verdict": result["verdict"],
            "layer_results": {"label_check": label, "shift_check": shift, "layer3_ensemble": details["layer3"]},
            "layer_scores": result["layer_scores"], "n_layers_alarmed": int(label["alarm"]) + int(shift["alarm"]),
            "n_samples": len(X_inc), "requires_human_review": THRESHOLD_SUSPICIOUS <= overall < THRESHOLD_CONFIRMED,
            "degradation_score": 0.0, "causal_proof_valid": False, "blend_spectral_confirmed": False,
            "thresholds": result["thresholds"], "engine": "v2", "warnings": details["warnings"],
        }

    @staticmethod
    def _normalise_result(result: dict, samples: list, X_inc: np.ndarray) -> dict:
        details = result.get("details", {})
        if details.get("engine") == "v2":
            return DetectionPipeline._normalise_v2(result, details, X_inc)

        layer_results = {
            "layer1_statistical" : details.get("layer1", {}),
            "layer2_spectral"    : details.get("layer2", {}),
            "layer3_ensemble"    : details.get("layer3", {}),
            "layer4_causal"      : details.get("layer4", {}),
            "layer5_federated"   : details.get("layer5", {}),
            "shap_drift"         : details.get("shap_drift", {}),
        }

        l4 = layer_results["layer4_causal"]
        if "acc_with_poison" in l4 and "accuracy_with_poison" not in l4:
            l4["accuracy_with_poison"] = l4["acc_with_poison"]
        if "acc_without_poison" in l4 and "accuracy_without_poison" not in l4:
            l4["accuracy_without_poison"] = l4["acc_without_poison"]
        if "significant" in l4 and "statistically_significant" not in l4:
            l4["statistically_significant"] = l4["significant"]
        if "placebo_valid" in l4 and "placebo_passed" not in l4:
            l4["placebo_passed"] = l4["placebo_valid"]

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
            layer_results["shap_drift"].get("alarm", False),
        ]
        n_layers_alarmed = sum(1 for a in alarms if a)

        overall = result.get("overall_suspicion", 0.0)
        verdict = result.get("verdict", "CLEAN")

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
        }
