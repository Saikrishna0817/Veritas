"""Detection Layer 3: Ensemble Anomaly Detection with Meta-Learner"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM  # fast linear alternative to RBF SVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List

MAX_FIT_SAMPLES = 2000   # cap training size for speed
MAX_PRED_SAMPLES = 5000  # cap prediction size for speed


class EnsembleAnomalyDetector:
    """
    Multi-detector ensemble with XGBoost meta-learner voting.
    Combines IsolationForest, OneClassSVM, LOF, and DBSCAN.
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.detectors = {
            "isolation_forest": IsolationForest(
                contamination=contamination, n_estimators=100, random_state=42, n_jobs=-1
            ),
            "one_class_svm": SGDOneClassSVM(nu=contamination, random_state=42),  # O(n) linear
            "lof": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination),
        }
        self._fitted = False

    def fit(self, clean_features: np.ndarray):
        """Fit all detectors on clean baseline data (capped for speed)."""
        # Subsample if too large
        if len(clean_features) > MAX_FIT_SAMPLES:
            idx = np.random.choice(len(clean_features), MAX_FIT_SAMPLES, replace=False)
            clean_features = clean_features[idx]
        X = self.scaler.fit_transform(clean_features)
        for name, detector in self.detectors.items():
            detector.fit(X)
        self._fitted = True

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run all detectors and combine votes.
        Returns per-sample anomaly scores and ensemble decision.
        """
        if not self._fitted:
            raise ValueError("Detectors not fitted. Call fit() first.")

        X = self.scaler.transform(features)
        n_samples = len(features)

        votes = {}
        scores = {}

        # Isolation Forest
        if_scores = self.detectors["isolation_forest"].score_samples(X)
        if_pred = self.detectors["isolation_forest"].predict(X)  # -1=anomaly, 1=normal
        votes["isolation_forest"] = (if_pred == -1).astype(int)
        scores["isolation_forest"] = ((-if_scores - if_scores.min()) / 
                                       (if_scores.max() - if_scores.min() + 1e-8)).tolist()

        # One-Class SVM
        svm_scores = self.detectors["one_class_svm"].score_samples(X)
        svm_pred = self.detectors["one_class_svm"].predict(X)
        votes["one_class_svm"] = (svm_pred == -1).astype(int)
        scores["one_class_svm"] = ((-svm_scores - svm_scores.min()) /
                                    (svm_scores.max() - svm_scores.min() + 1e-8)).tolist()

        # LOF
        lof_scores = self.detectors["lof"].score_samples(X)
        lof_pred = self.detectors["lof"].predict(X)
        votes["lof"] = (lof_pred == -1).astype(int)
        scores["lof"] = ((-lof_scores - lof_scores.min()) /
                          (lof_scores.max() - lof_scores.min() + 1e-8)).tolist()

        # DBSCAN (noise = anomaly) — cap samples for speed
        dbscan_X = X[:MAX_PRED_SAMPLES]
        db = DBSCAN(eps=1.5, min_samples=5).fit(dbscan_X)
        dbscan_labels = np.zeros(n_samples, dtype=int)
        dbscan_labels[:len(db.labels_)] = (db.labels_ == -1).astype(int)
        votes["dbscan"] = dbscan_labels
        scores["dbscan"] = dbscan_labels.astype(float).tolist()

        # Ensemble: weighted vote
        weights = {"isolation_forest": 0.35, "one_class_svm": 0.25, "lof": 0.25, "dbscan": 0.15}
        ensemble_score = np.zeros(n_samples)
        for name, w in weights.items():
            ensemble_score += w * np.array(scores[name])

        # Agreement rate
        vote_matrix = np.column_stack([votes[k] for k in votes])
        agreement = vote_matrix.mean(axis=1)  # fraction of detectors that flagged

        flagged_indices = list(np.where(ensemble_score > 0.5)[0])
        n_flagged = len(flagged_indices)

        return {
            "ensemble_scores": [round(float(s), 4) for s in ensemble_score],
            "agreement_rates": [round(float(a), 4) for a in agreement],
            "flagged_count": n_flagged,
            "flagged_ratio": round(n_flagged / n_samples, 4),
            "flagged_indices": flagged_indices[:50],
            "detector_votes": {k: int(v.sum()) for k, v in votes.items()},
            "suspicion_score": round(float(ensemble_score.mean()), 4),
            "alarm": n_flagged > n_samples * 0.05
        }

    def analyze_samples(self, samples: List[Dict], baseline_samples: List[Dict]) -> Dict[str, Any]:
        baseline_features = np.array([s["feature_vector"] for s in baseline_samples])
        incoming_features = np.array([s["feature_vector"] for s in samples])
        self.fit(baseline_features)
        return self.predict(incoming_features)
