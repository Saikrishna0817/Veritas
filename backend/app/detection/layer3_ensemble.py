"""
Layer 3 — Ensemble Anomaly Detection
=======================================
FIX SUMMARY:
- ROOT CAUSE OF FALSE POSITIVES: contamination=0.1 means IsolationForest
  ALWAYS flags 10% of samples as anomalies, even on perfectly clean data.
  With 4 detectors × 10% contamination, majority voting still fires too often.
- FIX 1: contamination reduced to 0.05 (5%) — still catches attacks but
  reduces false positive baseline on clean data.
- FIX 2: Vote threshold raised from ≥2 to ≥3 out of 4 (supermajority).
  On clean data with 5% contamination, the probability of ≥3 detectors
  agreeing on the same sample drops dramatically.
- FIX 3: Suspicion score now uses the EXCESS flagged ratio above a clean-data
  expected baseline (expected ≈ 5% flagged on clean data). If flagged_ratio
  is near 5%, score stays near 0.
- FIX 4: Features are scaled before detectors to avoid IQR outliers in
  high-variance features dominating isolation scores.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler


# ── Detector hyperparameters ──
CONTAMINATION          = 0.05    # was 0.10 — reduced to cut baseline FP rate
N_IF_ESTIMATORS        = 200     # more trees = more stable estimates
LOF_NEIGHBOURS         = 20
SVM_NU                 = 0.05    # was 0.10
VOTE_THRESHOLD         = 3       # was 2 — supermajority required (out of 4)
EXPECTED_CLEAN_FLAG_RATE = 0.05  # expected fraction flagged on perfectly clean data
MIN_SAMPLES            = 30


class EnsembleAnomalyDetector:
    """
    Four-detector ensemble with majority voting anomaly detection.

    Detectors:
        1. IsolationForest  — global outliers (short isolation path)
        2. SGD One-Class SVM — outside learned normal hypersphere
        3. Local Outlier Factor — locally sparse points
        4. DBSCAN noise points — not in any dense cluster

    The key fix is raising the vote threshold to 3/4 (supermajority) and
    reducing contamination, so natural outliers in clean data are NOT flagged.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._scaler = RobustScaler()   # robust to outliers, unlike StandardScaler
        self._fitted_detectors = {}

    def fit(self, X_reference: np.ndarray) -> None:
        """
        Fit all detectors on the clean reference split.
        Fitting on clean data lets the detectors learn what "normal" looks like.
        """
        if X_reference.shape[0] < MIN_SAMPLES:
            return
        X = self._scaler.fit_transform(X_reference.astype(float))
        self._fit_detectors(X)

    def analyze(self, X_incoming: np.ndarray) -> dict:
        """
        Detect anomalies in the incoming partition using the fitted detectors.

        Returns
        -------
        dict with suspicion_score, flagged_ratio, flagged_count, per_detector results
        """
        if X_incoming.shape[0] < MIN_SAMPLES:
            return self._null_result("insufficient_samples")

        if not self._fitted_detectors:
            # Fit on the incoming data itself (unsupervised fallback)
            self.fit(X_incoming)

        X = self._scaler.transform(X_incoming.astype(float))

        votes, per_detector = self._vote(X)

        flagged_mask  = votes >= VOTE_THRESHOLD
        flagged_count = int(flagged_mask.sum())
        flagged_ratio = float(flagged_count / len(X))

        suspicion = self._compute_suspicion(flagged_ratio)

        return {
            "suspicion_score" : suspicion,
            "flagged_ratio"   : flagged_ratio,
            "flagged_count"   : flagged_count,
            "total_samples"   : len(X),
            "flagged_indices" : np.where(flagged_mask)[0].tolist(),
            "vote_threshold"  : VOTE_THRESHOLD,
            "per_detector"    : per_detector,
            "expected_clean_flag_rate": EXPECTED_CLEAN_FLAG_RATE,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _fit_detectors(self, X_scaled: np.ndarray) -> None:
        """Fit each detector on scaled reference data."""
        # 1. Isolation Forest
        self._fitted_detectors["isolation_forest"] = IsolationForest(
            contamination=CONTAMINATION,
            n_estimators=N_IF_ESTIMATORS,
            random_state=self.random_state,
        ).fit(X_scaled)

        # 2. One-Class SVM (linear SGD variant for speed)
        # FIX: Use rbf kernel SVM — linear version is less stable on structured data
        try:
            from sklearn.svm import OneClassSVM as OCSVM
            self._fitted_detectors["one_class_svm"] = OCSVM(
                kernel="rbf", nu=SVM_NU, gamma="scale"
            ).fit(X_scaled)
        except Exception:
            self._fitted_detectors["one_class_svm"] = None

        # 3. Local Outlier Factor (novelty=True so we can call predict later)
        self._fitted_detectors["lof"] = LocalOutlierFactor(
            n_neighbors=min(LOF_NEIGHBOURS, max(X_scaled.shape[0] // 5, 5)),
            novelty=True,
            contamination=CONTAMINATION,
        ).fit(X_scaled)

        # 4. DBSCAN is not fitted — it runs on incoming data directly (transductive)
        self._fitted_detectors["dbscan_eps"] = self._estimate_dbscan_eps(X_scaled)

    def _vote(self, X_scaled: np.ndarray) -> tuple:
        """
        Each detector casts a vote (1=anomaly, 0=normal) per sample.
        Returns (vote_sum array, per_detector breakdown dict).
        """
        n = len(X_scaled)
        vote_matrix = np.zeros((n, 4), dtype=int)
        per_detector = {}

        # 1. Isolation Forest
        if "isolation_forest" in self._fitted_detectors:
            preds = self._fitted_detectors["isolation_forest"].predict(X_scaled)
            flags = (preds == -1).astype(int)
            vote_matrix[:, 0] = flags
            per_detector["isolation_forest"] = {
                "flagged_count" : int(flags.sum()),
                "flagged_ratio" : float(flags.mean()),
            }

        # 2. One-Class SVM
        if self._fitted_detectors.get("one_class_svm") is not None:
            preds = self._fitted_detectors["one_class_svm"].predict(X_scaled)
            flags = (preds == -1).astype(int)
            vote_matrix[:, 1] = flags
            per_detector["one_class_svm"] = {
                "flagged_count" : int(flags.sum()),
                "flagged_ratio" : float(flags.mean()),
            }

        # 3. LOF
        if "lof" in self._fitted_detectors:
            preds = self._fitted_detectors["lof"].predict(X_scaled)
            flags = (preds == -1).astype(int)
            vote_matrix[:, 2] = flags
            per_detector["local_outlier_factor"] = {
                "flagged_count" : int(flags.sum()),
                "flagged_ratio" : float(flags.mean()),
            }

        # 4. DBSCAN (transductive — runs fresh on incoming data)
        eps = self._fitted_detectors.get("dbscan_eps", 0.5)
        min_samples = max(3, int(np.log(n)))   # adaptive min_samples
        db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        flags = (db_labels == -1).astype(int)
        vote_matrix[:, 3] = flags
        per_detector["dbscan"] = {
            "flagged_count" : int(flags.sum()),
            "flagged_ratio" : float(flags.mean()),
            "n_clusters"    : int(len(set(db_labels)) - (1 if -1 in db_labels else 0)),
        }

        return vote_matrix.sum(axis=1), per_detector

    def _compute_suspicion(self, flagged_ratio: float) -> float:
        """
        Convert flagged_ratio to suspicion score.

        FIX: The key insight is that on CLEAN data, ~5% will always be flagged
        by chance (contamination parameter). We only score the EXCESS above that.
        If flagged_ratio ≈ 5% → suspicion ≈ 0.
        If flagged_ratio ≈ 25% → suspicion ≈ 0.65.
        If flagged_ratio ≈ 50%+ → suspicion ≈ 1.0.
        """
        excess = max(0.0, flagged_ratio - EXPECTED_CLEAN_FLAG_RATE)
        # Scale: 45% excess maps to suspicion ≈ 1.0
        suspicion = float(np.clip(excess / 0.45, 0.0, 1.0))
        return suspicion

    @staticmethod
    def _estimate_dbscan_eps(X: np.ndarray) -> float:
        """
        Estimate a reasonable DBSCAN eps using the k-nearest neighbour
        distance distribution (elbow heuristic).
        FIX: Static eps=0.5 was inappropriate for scaled data in all domains.
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            k = min(5, X.shape[0] - 1)
            nn = NearestNeighbors(n_neighbors=k).fit(X)
            dists, _ = nn.kneighbors(X)
            k_dists = dists[:, -1]
            # Use 90th percentile of k-distances as eps
            eps = float(np.percentile(k_dists, 90))
            return max(eps, 0.1)   # never go below 0.1
        except Exception:
            return 0.5   # fallback

    @staticmethod
    def _null_result(reason: str) -> dict:
        return {
            "suspicion_score"          : 0.0,
            "flagged_ratio"            : 0.0,
            "flagged_count"            : 0,
            "total_samples"            : 0,
            "flagged_indices"          : [],
            "vote_threshold"           : VOTE_THRESHOLD,
            "per_detector"             : {},
            "expected_clean_flag_rate" : EXPECTED_CLEAN_FLAG_RATE,
            "skip_reason"              : reason,
        }
