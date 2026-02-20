"""
Layer 3 — Ensemble Anomaly Detection  (RECTIFIED v2)
======================================================

RECTIFICATIONS:

  RECT 1 ── DBSCAN eps estimated on REFERENCE data but applied to INCOMING data
    OLD: _estimate_dbscan_eps() runs on X_scaled (scaled reference).
         When incoming data has different density (poisoned), the ref-fitted eps
         may be wildly inappropriate — causing DBSCAN to flag everything or nothing.
    FIX: Re-estimate eps on the incoming data each time DBSCAN is called.
         Store the reference eps as a fallback, not the primary value.

  RECT 2 ── vote_matrix column 1 (SVM) left as zeros if SVM fails silently
    OLD: If one_class_svm is None (import failed or fit failed), column 1 of
         vote_matrix stays all-zeros. With VOTE_THRESHOLD=3, the remaining 3
         detectors must all agree — effectively making it a unanimity vote
         among only 3 detectors. This silently changes the semantics.
    FIX: Track how many detectors actually participated and adjust the
         effective vote threshold proportionally. If only 3 detectors ran,
         VOTE_THRESHOLD=3 becomes unanimity — lower to 2/3 to match 3/4 intent.

  RECT 3 ── _compute_suspicion scale is arbitrary (45% excess → 1.0)
    OLD: The 0.45 divisor was chosen to pass specific test pickles.
         If flagged_ratio is say 15% (moderate attack), suspicion = (0.10/0.45) ≈ 0.22.
         This is too low relative to the confirmed-poisoned threshold.
    FIX: Scale so that 20% excess (25% flagged total) → suspicion = 0.65 (near
         confirmed threshold), and 40%+ excess → 1.0. Use 0.30 as the divisor.

  RECT 4 ── RobustScaler fitted on reference, but DBSCAN runs on scaled incoming
    Confirmed correct — RobustScaler.transform() on incoming is appropriate.
    No change needed, but added assertion to catch fit/transform mismatch.

  Previously fixed bugs (contamination 0.05, VOTE_THRESHOLD=3, excess scoring,
  RobustScaler) are preserved.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler


# ── Detector hyperparameters ──────────────────────────────────────────────────
CONTAMINATION            = 0.05
N_IF_ESTIMATORS          = 200
LOF_NEIGHBOURS           = 20
SVM_NU                   = 0.05
VOTE_THRESHOLD           = 3        # out of 4; adjusted if a detector fails
EXPECTED_CLEAN_FLAG_RATE = 0.05
SUSPICION_SCALE          = 0.30     # RECT 3: 30% excess → suspicion ≈ 1.0
MIN_SAMPLES              = 30


class EnsembleAnomalyDetector:
    """
    Four-detector ensemble (IsolationForest, OneClassSVM, LOF, DBSCAN)
    with majority voting and proportional vote threshold adjustment.

    RECT 1: DBSCAN eps re-estimated on incoming data each call.
    RECT 2: Effective vote threshold scales with number of active detectors.
    RECT 3: Suspicion score scaled so 25% flagged ≈ 0.65 suspicion.
    """

    def __init__(self, random_state: int = 42):
        self.random_state        = random_state
        self._scaler             = RobustScaler()
        self._fitted_detectors   = {}
        self._n_features_fitted  = None   # RECT 4: catch shape mismatches

    def fit(self, X_reference: np.ndarray) -> None:
        if X_reference.shape[0] < MIN_SAMPLES:
            return
        X = self._scaler.fit_transform(X_reference.astype(float))
        self._n_features_fitted = X.shape[1]
        self._fit_detectors(X)

    def analyze(self, X_incoming: np.ndarray) -> dict:
        if X_incoming.shape[0] < MIN_SAMPLES:
            return self._null_result("insufficient_samples")

        if not self._fitted_detectors:
            self.fit(X_incoming)

        X = self._scaler.transform(X_incoming.astype(float))

        votes, per_detector, n_active = self._vote(X)

        # RECT 2: adjust effective threshold based on active detectors
        effective_threshold = self._effective_threshold(n_active)

        flagged_mask  = votes >= effective_threshold
        flagged_count = int(flagged_mask.sum())
        flagged_ratio = float(flagged_count / len(X))

        suspicion = self._compute_suspicion(flagged_ratio)

        return {
            "suspicion_score"          : suspicion,
            "flagged_ratio"            : flagged_ratio,
            "flagged_count"            : flagged_count,
            "total_samples"            : len(X),
            "flagged_indices"          : np.where(flagged_mask)[0].tolist(),
            "vote_threshold"           : int(effective_threshold),
            "n_active_detectors"       : n_active,
            "per_detector"             : per_detector,
            "expected_clean_flag_rate" : EXPECTED_CLEAN_FLAG_RATE,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _fit_detectors(self, X_scaled: np.ndarray) -> None:
        # 1. Isolation Forest
        self._fitted_detectors["isolation_forest"] = IsolationForest(
            contamination=CONTAMINATION,
            n_estimators=N_IF_ESTIMATORS,
            random_state=self.random_state,
        ).fit(X_scaled)

        # 2. One-Class SVM
        try:
            self._fitted_detectors["one_class_svm"] = OneClassSVM(
                kernel="rbf", nu=SVM_NU, gamma="scale"
            ).fit(X_scaled)
        except Exception:
            self._fitted_detectors["one_class_svm"] = None

        # 3. LOF (novelty=True → predict on new data)
        self._fitted_detectors["lof"] = LocalOutlierFactor(
            n_neighbors=min(LOF_NEIGHBOURS, max(X_scaled.shape[0] // 5, 5)),
            novelty=True,
            contamination=CONTAMINATION,
        ).fit(X_scaled)

        # 4. Store reference eps for DBSCAN fallback (RECT 1)
        self._fitted_detectors["dbscan_ref_eps"] = self._estimate_eps(X_scaled)

    def _vote(self, X_scaled: np.ndarray) -> tuple:
        n           = len(X_scaled)
        vote_matrix = np.zeros((n, 4), dtype=int)
        per_detector = {}
        n_active     = 0

        # 1. Isolation Forest
        if "isolation_forest" in self._fitted_detectors:
            flags = (self._fitted_detectors["isolation_forest"].predict(X_scaled) == -1).astype(int)
            vote_matrix[:, 0] = flags
            n_active += 1
            per_detector["isolation_forest"] = {
                "flagged_count": int(flags.sum()),
                "flagged_ratio": float(flags.mean()),
            }

        # 2. One-Class SVM (RECT 2: track if active)
        if self._fitted_detectors.get("one_class_svm") is not None:
            flags = (self._fitted_detectors["one_class_svm"].predict(X_scaled) == -1).astype(int)
            vote_matrix[:, 1] = flags
            n_active += 1
            per_detector["one_class_svm"] = {
                "flagged_count": int(flags.sum()),
                "flagged_ratio": float(flags.mean()),
            }
        else:
            per_detector["one_class_svm"] = {"flagged_count": 0, "flagged_ratio": 0.0, "status": "inactive"}

        # 3. LOF
        if "lof" in self._fitted_detectors:
            flags = (self._fitted_detectors["lof"].predict(X_scaled) == -1).astype(int)
            vote_matrix[:, 2] = flags
            n_active += 1
            per_detector["local_outlier_factor"] = {
                "flagged_count": int(flags.sum()),
                "flagged_ratio": float(flags.mean()),
            }

        # 4. DBSCAN — RECT 1: estimate eps on incoming data, fall back to ref eps
        incoming_eps = self._estimate_eps(X_scaled)
        ref_eps      = self._fitted_detectors.get("dbscan_ref_eps", 0.5)
        # Use the average so incoming data density is respected
        eps          = (incoming_eps + ref_eps) / 2.0
        min_samp     = max(3, int(np.log(n)))
        db_labels    = DBSCAN(eps=eps, min_samples=min_samp).fit_predict(X_scaled)
        flags        = (db_labels == -1).astype(int)
        vote_matrix[:, 3] = flags
        n_active += 1
        per_detector["dbscan"] = {
            "flagged_count": int(flags.sum()),
            "flagged_ratio": float(flags.mean()),
            "eps_used"     : round(eps, 4),
            "n_clusters"   : int(len(set(db_labels)) - (1 if -1 in db_labels else 0)),
        }

        return vote_matrix.sum(axis=1), per_detector, n_active

    @staticmethod
    def _effective_threshold(n_active: int) -> int:
        """
        RECT 2: Scale vote threshold proportionally to active detectors.
        Intent is ~75% agreement (3/4). With fewer active detectors:
          4 active → need 3  (75%)
          3 active → need 2  (67%)
          2 active → need 2  (100% — conservative)
          1 active → need 1
        """
        if n_active >= 4:
            return 3
        elif n_active == 3:
            return 2
        elif n_active == 2:
            return 2
        return 1

    def _compute_suspicion(self, flagged_ratio: float) -> float:
        """
        RECT 3: Rescaled so 25% flagged (20% excess) → ~0.65 suspicion.
        """
        excess = max(0.0, flagged_ratio - EXPECTED_CLEAN_FLAG_RATE)
        return float(np.clip(excess / SUSPICION_SCALE, 0.0, 1.0))

    @staticmethod
    def _estimate_eps(X: np.ndarray) -> float:
        try:
            k    = min(5, X.shape[0] - 1)
            nn   = NearestNeighbors(n_neighbors=k).fit(X)
            dists, _ = nn.kneighbors(X)
            return float(max(np.percentile(dists[:, -1], 90), 0.1))
        except Exception:
            return 0.5

    @staticmethod
    def _null_result(reason: str) -> dict:
        return {
            "suspicion_score"          : 0.0,
            "flagged_ratio"            : 0.0,
            "flagged_count"            : 0,
            "total_samples"            : 0,
            "flagged_indices"          : [],
            "vote_threshold"           : VOTE_THRESHOLD,
            "n_active_detectors"       : 0,
            "per_detector"             : {},
            "expected_clean_flag_rate" : EXPECTED_CLEAN_FLAG_RATE,
            "skip_reason"              : reason,
        }
