"""Detection pipeline: a thin adapter around the Veritas v2 detector (``veritas2.py``).

The original five-layer pipeline was removed after benchmark v2 showed it at chance on label noise and unable
to reach a verdict on single-layer attacks (docs/benchmark-v2-findings.md). It is recoverable from the git tag
``legacy-five-layer-pipeline``.
"""

import numpy as np

from .veritas2 import Veritas2, LABEL_PENALTY, RESID_PENALTY, SHIFT_PENALTY, SUSPICIOUS_AT, verdict_for

# Suspicion thresholds used by the defense and review-queue code; ``veritas2.verdict_for`` maps onto them.
THRESHOLD_CONFIRMED  = 0.65
THRESHOLD_SUSPICIOUS = 0.35
THRESHOLD_LOW_RISK   = 0.15
TOP_CELLS = 3


class DetectionPipeline:
    """fit_baseline() on a trusted reference set, then analyze() an incoming batch."""

    def __init__(self, random_state: int = 42):
        self.random_state     = random_state
        self._baseline_fitted = False
        self.X_reference      = None
        self.y_reference      = None

    def fit_baseline(self, X_reference, y_reference=None) -> None:
        X = np.array(X_reference, dtype=float)
        y = np.array(y_reference if y_reference is not None else np.zeros(len(X), dtype=int))

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
        label_ev = max(r["evidence"].get("label", 0.0) - LABEL_PENALTY, 0.0)
        shift_ev = max(r["evidence"]["shift"] - SHIFT_PENALTY, 0.0)  # same weighting as the batch statistic
        resid_ev = max(r["evidence"].get("resid", 0.0) - RESID_PENALTY, 0.0)
        cells = r["cell_scores"] if "label" in r["evidence"] else r["cell_scores"][:, :-1]   # unlabeled: no label cell to blame
        # "layer3" keeps the row-flag keys the report and quarantine code read.
        details = {
            "label_check": {"evidence": label_ev, "alarm": label_ev >= SUSPICIOUS_AT, "ran": "label" in r["evidence"]},
            "shift_check": {"evidence": shift_ev, "alarm": shift_ev >= SUSPICIOUS_AT, "in_row_ranking": r["shift_in_ranking"]},
            "resid_check": {"evidence": resid_ev, "alarm": resid_ev >= SUSPICIOUS_AT, "in_row_ranking": r["resid_in_ranking"]},
            "row_flags": {"flagged_indices": [len(self.X_reference) + i for i in flagged], "flagged_indices_incoming": flagged,
                       "flagged_ratio": len(flagged) / max(len(X), 1), "row_scores": [round(float(v), 3) for v in r["row_scores"]],
                       "suspicion_score": r["suspicion"],
                       # the 3 most suspicious cells of each flagged row: [column index, score]; index n_features = the label
                       "top_cells": {i: [[int(j), round(float(cells[i, j]), 3)]
                                         for j in np.argsort(-cells[i], kind="stable")[:TOP_CELLS]] for i in flagged}},
            "warnings": r["warnings"],
            "attack_profile": r["attack_profile"],   # None when the batch is CLEAN or unlabeled
        }
        return {
            "verdict": r["verdict"], "overall_suspicion": r["suspicion"], "layer_scores": {"label_check": verdict_for(label_ev)[1], "shift_check": verdict_for(shift_ev)[1], "resid_check": verdict_for(resid_ev)[1]},  # 0-1 for the UI
            "details": details,
            "thresholds": {"confirmed": THRESHOLD_CONFIRMED, "suspicious": THRESHOLD_SUSPICIOUS, "low_risk": THRESHOLD_LOW_RISK},
        }

    # Adapters used by the upload, dataset and defense code.

    def run_on_upload(self, ingested: dict) -> dict:
        X        = ingested["features"]
        labels   = ingested.get("labels")
        split    = ingested["reference_split"]

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
        X = np.array([s["feature_vector"] for s in samples], dtype=float)   # the detector standardises columns itself
        y = np.array([s.get("label", 0) for s in samples], dtype=int)
        split = int(len(X) * 0.70)
        self.fit_baseline(X[:split], y[:split])
        result = self.analyze(X[split:], y[split:])

        return self._normalise(result, X[split:])

    @staticmethod
    def _normalise(result: dict, X_inc: np.ndarray) -> dict:
        details = result["details"]
        label, shift, resid = details["label_check"], details["shift_check"], details["resid_check"]
        overall = result["overall_suspicion"]
        return {
            "overall_suspicion_score": overall, "overall_suspicion": overall, "verdict": result["verdict"],
            "layer_results": {"label_check": label, "shift_check": shift, "resid_check": resid, "row_flags": details["row_flags"]},
            "layer_scores": result["layer_scores"], "n_layers_alarmed": int(label["alarm"]) + int(shift["alarm"]) + int(resid["alarm"]),
            "n_samples": len(X_inc), "requires_human_review": THRESHOLD_SUSPICIOUS <= overall < THRESHOLD_CONFIRMED,
            "thresholds": result["thresholds"], "warnings": details["warnings"], "attack_profile": details["attack_profile"],
        }
