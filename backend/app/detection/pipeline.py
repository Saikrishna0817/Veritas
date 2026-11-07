"""Detection pipeline: a thin adapter around the Veritas v2 detector (``veritas2.py``).

The original five-layer pipeline was removed after benchmark v2 showed it at chance on label noise and unable
to reach a verdict on single-layer attacks (docs/benchmark-v2-findings.md). It is recoverable from the git tag
``legacy-five-layer-pipeline``.
"""

import numpy as np

from .veritas2 import Veritas2, SHIFT_PENALTY, SUSPICIOUS_AT, verdict_for

# Suspicion thresholds used by the defense and review-queue code; ``veritas2.verdict_for`` maps onto them.
THRESHOLD_CONFIRMED  = 0.65
THRESHOLD_SUSPICIOUS = 0.35
THRESHOLD_LOW_RISK   = 0.15


class DetectionPipeline:
    """fit_baseline() on a trusted reference set, then analyze() an incoming batch."""

    def __init__(self, random_state: int = 42):
        self.random_state     = random_state
        self._baseline_fitted = False
        self.X_reference      = None
        self.y_reference      = None

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
        self.v2 = Veritas2(self.random_state).fit(X, y)
        self._baseline_fitted = True

    def analyze(self, X_incoming: np.ndarray, y_incoming: np.ndarray) -> dict:
        if not self._baseline_fitted:
            raise ValueError(
                "DetectionPipeline.analyze() called before fit_baseline(). "
                "Call pipeline.fit_baseline(X_clean, y_clean) first."
            )
        return self._analyze_v2(np.array(X_incoming, dtype=float), np.array(y_incoming))

    def _analyze_v2(self, X: np.ndarray, y: np.ndarray) -> dict:
        r = self.v2.analyze(X, y)
        flagged = r["flagged_indices"]
        label_ev = r["evidence"].get("label", 0.0)
        shift_ev = max(r["evidence"]["shift"] - SHIFT_PENALTY, 0.0)  # same weighting as the batch statistic
        # "layer3" keeps the row-flag keys the report and quarantine code read.
        details = {
            "label_check": {"evidence": label_ev, "alarm": label_ev >= SUSPICIOUS_AT, "ran": "label" in r["evidence"]},
            "shift_check": {"evidence": shift_ev, "alarm": shift_ev >= SUSPICIOUS_AT, "in_row_ranking": r["shift_in_ranking"]},
            "row_flags": {"flagged_indices": [len(self.X_reference) + i for i in flagged], "flagged_indices_incoming": flagged,
                       "flagged_ratio": len(flagged) / max(len(X), 1), "row_scores": [round(float(v), 3) for v in r["row_scores"]],
                       "suspicion_score": r["suspicion"]},
            "warnings": r["warnings"],
        }
        return {
            "verdict": r["verdict"], "overall_suspicion": r["suspicion"], "layer_scores": {"label_check": verdict_for(label_ev)[1], "shift_check": verdict_for(shift_ev)[1]},  # 0-1 for the UI
            "details": details,
            "thresholds": {"confirmed": THRESHOLD_CONFIRMED, "suspicious": THRESHOLD_SUSPICIOUS, "low_risk": THRESHOLD_LOW_RISK},
        }

    @classmethod
    def from_split(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        reference_fraction: float = 0.70,
        random_state: int         = 42,
    ) -> tuple:
        n   = len(X)
        rng = np.random.RandomState(random_state)
        idx     = rng.permutation(n)
        split   = int(n * reference_fraction)
        ref_idx = idx[:split]
        inc_idx = idx[split:]

        pipeline = cls(random_state=random_state)
        pipeline.fit_baseline(X[ref_idx], y[ref_idx])
        result = pipeline.analyze(X[inc_idx], y[inc_idx])
        return pipeline, result

    # Adapters used by the upload, dataset and defense code.

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

        return self._normalise(result, X_inc)

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

        return self._normalise(result, X[split:])

    @staticmethod
    def _normalise(result: dict, X_inc: np.ndarray) -> dict:
        details = result["details"]
        label, shift = details["label_check"], details["shift_check"]
        overall = result["overall_suspicion"]
        return {
            "overall_suspicion_score": overall, "overall_suspicion": overall, "verdict": result["verdict"],
            "layer_results": {"label_check": label, "shift_check": shift, "row_flags": details["row_flags"]},
            "layer_scores": result["layer_scores"], "n_layers_alarmed": int(label["alarm"]) + int(shift["alarm"]),
            "n_samples": len(X_inc), "requires_human_review": THRESHOLD_SUSPICIOUS <= overall < THRESHOLD_CONFIRMED,
            "thresholds": result["thresholds"], "warnings": details["warnings"],
        }
