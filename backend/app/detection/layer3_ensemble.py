"""
Layer 3 — Ensemble Anomaly Detection  (EXPANDED v4 — with Model Inversion Detector)
======================================================================================

NEW IN v4:
  - ModelInversionDetector injected as a standalone sub-detector that runs
    alongside the 8-detector ensemble. It does NOT participate in the vote
    pool (to avoid distorting the existing calibration) but appends its own
    result block to the analyze() output dict.

  MODEL INVERSION DETECTOR
  ─────────────────────────
  Model inversion attacks (Fredrikson et al. 2015, subsequent gradient-based
  variants) query the model repeatedly to reconstruct training data. The
  queries leave a characteristic signature in the *incoming data distribution*
  that the ensemble sees:

    1. LOW INTER-SAMPLE DIVERSITY
       An attacker grid-searching the feature space to triangulate a class
       boundary sends queries that are systematically spaced — far more uniform
       than a natural i.i.d. sample. We measure this as the coefficient of
       variation (CV) of pairwise distances: low CV = uniform spacing = grid.

    2. BOUNDARY CONCENTRATION
       Inversion queries cluster near the decision boundary of each class,
       trying to locate the transition region. Proxy: incoming samples
       concentrate within the 25th-percentile Mahalanobis distance band
       of the reference distribution's centroid more than expected.

    3. NEAR-DUPLICATE QUERY BURSTS
       Gradient-based inversion sends near-identical queries in a burst,
       perturbing slightly each time to read off the gradient direction.
       Proxy: fraction of samples with a pairwise L2 distance < ε to at
       least one other incoming sample.

    4. SYSTEMATIC FEATURE SPACE COVERAGE
       The attacker wants to cover each class's support uniformly. Random
       legitimate queries cluster around the mode; adversarial queries spread
       toward the tails. Proxy: incoming tail-fraction (samples outside the
       80th-percentile reference ellipse) is higher than the expected 20%.

  Suspicion score: weighted mean of the four signal scores, clipped [0, 1].
  Alarm fires when suspicion ≥ MODEL_INVERSION_ALARM_THRESH (0.45).

EXISTING MODELS (v3, unchanged):
  1. IsolationForest, 2. OneClassSVM, 3. LOF, 4. DBSCAN,
  5. EllipticEnvelope, 6. XGBoost one-class, 7. RF one-class, 8. COPOD.
Vote pool: 8 detectors, dynamic threshold ≈ 75% agreement.
All v2 rectifications preserved.
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler


# ── Model Inversion hyperparameters ───────────────────────────────────────────
MODEL_INVERSION_ALARM_THRESH = 0.45   # suspicion score at which alarm fires
MI_NEAR_DUP_EPS              = 0.05   # L2 distance threshold for near-duplicates
MI_NEAR_DUP_ALARM_FRAC       = 0.15   # ≥15% near-dup rate triggers this signal
MI_BOUNDARY_BAND_PERCENTILE  = 25     # inner band = centroid distances < p25 of ref
MI_BOUNDARY_ALARM_FRAC       = 0.15   # ≥15% in boundary band triggers this signal
MI_CV_ALARM                  = 0.50   # inter-sample dist CV < 0.50 = suspiciously uniform
MI_TAIL_ALARM_FRAC           = 0.30   # ≥30% of samples in reference tail = systematic coverage
MI_MAX_SUBSAMPLE             = 400    # cap for O(n²) pairwise distance computations


class ModelInversionDetector:
    """
    Injected into Layer 3.  Detects model-inversion query patterns in the
    incoming data distribution.  Fitted on the reference distribution;
    called from EnsembleAnomalyDetector.analyze().

    Does NOT modify the vote pool — results are appended as a separate key
    ``model_inversion`` in the analyze() output dict.
    """

    def __init__(self, random_state: int = 42):
        self.random_state    = random_state
        self._rng            = np.random.RandomState(random_state)
        self._fitted         = False
        # Reference statistics cached at fit time
        self._ref_centroid   = None   # median of reference (IQR-normalised)
        self._ref_dist_p25   = None   # 25th-pctile dist to centroid in ref
        self._ref_dist_p80   = None   # 80th-pctile dist to centroid (tail boundary)
        self._ref_diversity  = None   # expected mean pairwise distance in ref
        self._ref_div_std    = None   # std of expected pairwise distance

    # ── Fitting ──────────────────────────────────────────────────────────────
    def fit(self, X_ref: np.ndarray) -> None:
        """Cache reference distribution statistics needed for inversion detection."""
        if len(X_ref) < 30:
            return
        X = X_ref.astype(float)
        # IQR normalise
        iqrs = np.array([max(float(np.percentile(X[:, j], 75)) -
                             float(np.percentile(X[:, j], 25)), 1e-6)
                         for j in range(X.shape[1])])
        Xn = X / (iqrs + 1e-9)

        self._ref_centroid = np.median(Xn, axis=0)
        dists              = np.linalg.norm(Xn - self._ref_centroid, axis=1)
        self._ref_dist_p25 = float(np.percentile(dists, MI_BOUNDARY_BAND_PERCENTILE))
        self._ref_dist_p80 = float(np.percentile(dists, 80))

        # Bootstrap expected pairwise diversity
        diversities = []
        for _ in range(20):
            n_sub = min(100, len(X))
            s     = Xn[self._rng.choice(len(Xn), n_sub, replace=False)]
            d     = np.sqrt(np.sum((s[:, None] - s[None, :]) ** 2, axis=2))
            np.fill_diagonal(d, np.nan)
            diversities.append(float(np.nanmean(d)))
        self._ref_diversity = float(np.mean(diversities))
        self._ref_div_std   = float(np.std(diversities) + 1e-9)
        self._iqrs          = iqrs
        self._fitted        = True

    # ── Analysis ─────────────────────────────────────────────────────────────
    def analyze(self, X_incoming: np.ndarray) -> dict:
        """
        Returns a dict with keys:
          suspicion_score  — float in [0, 1]
          alarm            — bool
          signals          — dict of per-signal scores and raw values
          interpretation   — human-readable summary
        """
        if not self._fitted:
            return {"suspicion_score": 0.0, "alarm": False,
                    "signals": {}, "interpretation": "not_fitted"}

        X  = X_incoming.astype(float)
        Xn = X / (self._iqrs + 1e-9)
        n  = len(Xn)
        signals = {}
        scores  = []

        # ── Signal 1: Near-duplicate rate ─────────────────────────────────────
        sub_idx  = self._rng.choice(n, min(n, MI_MAX_SUBSAMPLE), replace=False)
        Xs       = Xn[sub_idx]
        diffs    = Xs[:, None] - Xs[None, :]
        pdists   = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(pdists, np.inf)
        near_dup_rate = float((pdists < MI_NEAR_DUP_EPS).any(axis=1).mean())
        signals["near_duplicate_rate"] = round(near_dup_rate, 4)
        # Score: linearly scales 0→0 at 0%, 1.0 at MI_NEAR_DUP_ALARM_FRAC
        s1 = min(near_dup_rate / MI_NEAR_DUP_ALARM_FRAC, 1.0)
        signals["near_dup_score"] = round(s1, 4)
        scores.append(s1)

        # ── Signal 2: Boundary concentration ─────────────────────────────────
        dists_in   = np.linalg.norm(Xn - self._ref_centroid, axis=1)
        inner_frac = float((dists_in < self._ref_dist_p25).mean())
        signals["boundary_concentration_frac"] = round(inner_frac, 4)
        s2 = min(inner_frac / MI_BOUNDARY_ALARM_FRAC, 1.0)
        signals["boundary_score"] = round(s2, 4)
        scores.append(s2)

        # ── Signal 3: Inter-sample distance CV (uniformity) ───────────────────
        finite   = pdists[np.isfinite(pdists)]
        dist_cv  = float(np.std(finite) / (np.mean(finite) + 1e-9))
        signals["pairwise_dist_cv"] = round(dist_cv, 4)
        # Low CV = uniform = adversarial; score inversely proportional
        s3 = max(0.0, 1.0 - dist_cv / MI_CV_ALARM)
        signals["uniformity_score"] = round(s3, 4)
        scores.append(s3)

        # ── Signal 4: Tail coverage (systematic exploration) ─────────────────
        tail_frac = float((dists_in > self._ref_dist_p80).mean())
        signals["tail_fraction"] = round(tail_frac, 4)
        # Legitimate data: ~20% in tail by definition; adversarial: much more
        s4 = min(max(0.0, tail_frac - 0.20) / (MI_TAIL_ALARM_FRAC - 0.20), 1.0)
        signals["tail_coverage_score"] = round(s4, 4)
        scores.append(s4)

        # ── Aggregate ─────────────────────────────────────────────────────────
        # Weights: near-dup and uniformity are strongest signals
        weights = [0.30, 0.20, 0.30, 0.20]
        suspicion = float(np.clip(
            sum(w * s for w, s in zip(weights, scores)), 0.0, 1.0
        ))
        alarm = suspicion >= MODEL_INVERSION_ALARM_THRESH

        if alarm:
            interp = (
                f"Model inversion pattern detected (suspicion={suspicion:.3f}). "
                f"Near-dup rate={near_dup_rate:.1%}, "
                f"boundary concentration={inner_frac:.1%}, "
                f"pairwise CV={dist_cv:.3f}."
            )
        else:
            interp = (
                f"No model inversion signature (suspicion={suspicion:.3f}). "
                f"Query distribution appears consistent with legitimate i.i.d. sampling."
            )

        return {
            "suspicion_score" : suspicion,
            "alarm"           : alarm,
            "signals"         : signals,
            "interpretation"  : interp,
        }


# ── Hyperparameters ────────────────────────────────────────────────────────────
CONTAMINATION            = 0.05
N_IF_ESTIMATORS          = 200
LOF_NEIGHBOURS           = 20
SVM_NU                   = 0.05
# Target ~75% agreement; adjusted dynamically for active detectors
VOTE_FRACTION_TARGET     = 0.75
EXPECTED_CLEAN_FLAG_RATE = 0.05
SUSPICION_SCALE          = 0.30
MIN_SAMPLES              = 30


class EnsembleAnomalyDetector:
    """
    Eight-detector ensemble (up from 4) with dynamic vote threshold.

    Detectors:
      1. IsolationForest       — global outliers
      2. OneClassSVM (RBF)     — hypersphere boundary
      3. LocalOutlierFactor    — local density
      4. DBSCAN                — noise points
      5. EllipticEnvelope      — robust Gaussian ellipse  [NEW]
      6. XGBoost one-class     — PU learning, tree ensemble  [NEW]
      7. RandomForest one-class— PU learning, RF ensemble  [NEW]
      8. COPOD                 — copula-based tail prob.    [NEW]

    Vote threshold = ceil(n_active × VOTE_FRACTION_TARGET).
    """

    def __init__(self, random_state: int = 42):
        self.random_state          = random_state
        self._scaler               = RobustScaler()
        self._fitted_detectors     = {}
        self._n_features_fitted    = None
        # ── Injected sub-detector ─────────────────────────────────────────────
        self._model_inversion_det  = ModelInversionDetector(random_state=random_state)

    def fit(self, X_reference: np.ndarray) -> None:
        if X_reference.shape[0] < MIN_SAMPLES:
            return
        X = self._scaler.fit_transform(X_reference.astype(float))
        self._n_features_fitted = X.shape[1]
        self._fit_detectors(X)
        # Fit model inversion detector on the raw (unscaled) reference so it
        # can use IQR normalisation independently.
        self._model_inversion_det.fit(X_reference.astype(float))

    def analyze(self, X_incoming: np.ndarray) -> dict:
        if X_incoming.shape[0] < MIN_SAMPLES:
            return self._null_result("insufficient_samples")
        if not self._fitted_detectors:
            self.fit(X_incoming)

        X = self._scaler.transform(X_incoming.astype(float))
        votes, per_detector, n_active = self._vote(X)

        effective_threshold = int(np.ceil(n_active * VOTE_FRACTION_TARGET))
        effective_threshold = max(effective_threshold, 2)

        flagged_mask  = votes >= effective_threshold
        flagged_count = int(flagged_mask.sum())
        flagged_ratio = float(flagged_count / len(X))
        suspicion     = self._compute_suspicion(flagged_ratio)

        # ── Run model inversion sub-detector (on raw incoming, not scaled) ────
        # X_incoming may be scaled already if called from pipeline; we pass the
        # RobustScaler-inverse so ModelInversionDetector uses its own IQR scale.
        try:
            X_raw = self._scaler.inverse_transform(X)
        except Exception:
            X_raw = X_incoming.astype(float)
        mi_result = self._model_inversion_det.analyze(X_raw)

        return {
            "suspicion_score"         : suspicion,
            "flagged_ratio"           : flagged_ratio,
            "flagged_count"           : flagged_count,
            "total_samples"           : len(X),
            "flagged_indices"         : np.where(flagged_mask)[0].tolist(),
            "vote_threshold"          : effective_threshold,
            "n_active_detectors"      : n_active,
            "per_detector"            : per_detector,
            "expected_clean_flag_rate": EXPECTED_CLEAN_FLAG_RATE,
            # ── Injected sub-detector output ───────────────────────────────────
            "model_inversion"         : mi_result,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _fit_detectors(self, X: np.ndarray) -> None:
        n = X.shape[0]

        # 1. Isolation Forest
        self._fitted_detectors["isolation_forest"] = IsolationForest(
            contamination=CONTAMINATION, n_estimators=N_IF_ESTIMATORS,
            random_state=self.random_state).fit(X)

        # 2. One-Class SVM
        try:
            self._fitted_detectors["one_class_svm"] = OneClassSVM(
                kernel="rbf", nu=SVM_NU, gamma="scale").fit(X)
        except Exception:
            self._fitted_detectors["one_class_svm"] = None

        # 3. LOF
        self._fitted_detectors["lof"] = LocalOutlierFactor(
            n_neighbors=min(LOF_NEIGHBOURS, max(n // 5, 5)),
            novelty=True, contamination=CONTAMINATION).fit(X)

        # 4. DBSCAN eps for reference
        self._fitted_detectors["dbscan_ref_eps"] = self._estimate_eps(X)

        # 5. EllipticEnvelope (NEW)
        try:
            self._fitted_detectors["elliptic"] = EllipticEnvelope(
                contamination=CONTAMINATION, random_state=self.random_state,
                support_fraction=0.85).fit(X)
        except Exception:
            self._fitted_detectors["elliptic"] = None

        # 6. XGBoost PU one-class (NEW)
        self._fitted_detectors["xgb_oc"] = self._fit_pu_model(X, model_type="xgb")

        # 7. RandomForest PU one-class (NEW)
        self._fitted_detectors["rf_oc"]  = self._fit_pu_model(X, model_type="rf")

        # 8. COPOD baseline CDF (NEW) — store marginal quantile functions
        self._fitted_detectors["copod_cdfs"] = self._fit_copod(X)

    def _fit_pu_model(self, X_ref: np.ndarray, model_type: str):
        """
        Positive-Unlabelled one-class detector.
        Positive = reference samples.
        Unlabelled = synthetic uniform noise in the feature bounding box.
        """
        try:
            rng    = np.random.RandomState(self.random_state)
            n, p   = X_ref.shape
            lo     = X_ref.min(axis=0) - 0.5
            hi     = X_ref.max(axis=0) + 0.5
            noise  = rng.uniform(lo, hi, size=(n, p))
            X_pu   = np.vstack([X_ref, noise])
            y_pu   = np.array([1]*n + [0]*n)

            if model_type == "xgb":
                try:
                    from xgboost import XGBClassifier  # type: ignore
                    clf = XGBClassifier(n_estimators=100, max_depth=3,
                                        use_label_encoder=False,
                                        eval_metric="logloss",
                                        random_state=self.random_state,
                                        verbosity=0)
                except ImportError:
                    # Fallback to GradientBoosting if xgboost not installed
                    from sklearn.ensemble import GradientBoostingClassifier
                    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                     random_state=self.random_state)
            else:
                clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                             random_state=self.random_state,
                                             n_jobs=-1)
            clf.fit(X_pu, y_pu)
            # Threshold: flag samples with P(positive) < contamination
            ref_probs = clf.predict_proba(X_ref)[:, 1]
            threshold = float(np.percentile(ref_probs, CONTAMINATION * 100))
            return {"clf": clf, "threshold": threshold}
        except Exception:
            return None

    def _fit_copod(self, X: np.ndarray) -> list:
        """
        Store per-feature sorted arrays for empirical CDF lookup.
        COPOD score = -log(min(F(x), 1-F(x))) for each feature, summed.
        """
        cdfs = []
        for j in range(X.shape[1]):
            col = np.sort(X[:, j])
            cdfs.append(col)
        return cdfs

    def _vote(self, X: np.ndarray) -> tuple:
        n            = len(X)
        vote_matrix  = np.zeros((n, 8), dtype=int)
        per_detector = {}
        n_active     = 0

        def register(name, flags, col_idx):
            nonlocal n_active
            vote_matrix[:, col_idx] = flags
            n_active += 1
            per_detector[name] = {"flagged_count": int(flags.sum()),
                                  "flagged_ratio": float(flags.mean())}

        # 1. IsolationForest
        if "isolation_forest" in self._fitted_detectors:
            register("isolation_forest",
                     (self._fitted_detectors["isolation_forest"].predict(X) == -1).astype(int), 0)

        # 2. OneClassSVM
        if self._fitted_detectors.get("one_class_svm") is not None:
            register("one_class_svm",
                     (self._fitted_detectors["one_class_svm"].predict(X) == -1).astype(int), 1)

        # 3. LOF
        if "lof" in self._fitted_detectors:
            register("local_outlier_factor",
                     (self._fitted_detectors["lof"].predict(X) == -1).astype(int), 2)

        # 4. DBSCAN (re-estimate eps on incoming + ref avg)
        ref_eps = self._fitted_detectors.get("dbscan_ref_eps", 0.5)
        inc_eps = self._estimate_eps(X)
        eps     = (ref_eps + inc_eps) / 2.0
        labels  = DBSCAN(eps=eps, min_samples=max(3, int(np.log(n)))).fit_predict(X)
        flags   = (labels == -1).astype(int)
        vote_matrix[:, 3] = flags; n_active += 1
        per_detector["dbscan"] = {"flagged_count": int(flags.sum()),
                                  "flagged_ratio": float(flags.mean()),
                                  "eps_used": round(eps, 4),
                                  "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}

        # 5. EllipticEnvelope (NEW)
        if self._fitted_detectors.get("elliptic") is not None:
            try:
                register("elliptic_envelope",
                         (self._fitted_detectors["elliptic"].predict(X) == -1).astype(int), 4)
            except Exception:
                pass

        # 6. XGBoost PU (NEW)
        xgb_det = self._fitted_detectors.get("xgb_oc")
        if xgb_det is not None:
            try:
                probs = xgb_det["clf"].predict_proba(X)[:, 1]
                register("xgb_one_class", (probs < xgb_det["threshold"]).astype(int), 5)
            except Exception:
                pass

        # 7. RandomForest PU (NEW)
        rf_det = self._fitted_detectors.get("rf_oc")
        if rf_det is not None:
            try:
                probs = rf_det["clf"].predict_proba(X)[:, 1]
                register("rf_one_class", (probs < rf_det["threshold"]).astype(int), 6)
            except Exception:
                pass

        # 8. COPOD (NEW)
        copod_cdfs = self._fitted_detectors.get("copod_cdfs")
        if copod_cdfs is not None:
            try:
                scores = self._copod_scores(X, copod_cdfs)
                threshold = float(np.percentile(scores, (1 - CONTAMINATION) * 100))
                register("copod", (scores > threshold).astype(int), 7)
            except Exception:
                pass

        return vote_matrix.sum(axis=1), per_detector, n_active

    def _copod_scores(self, X: np.ndarray, cdfs: list) -> np.ndarray:
        """
        Per-sample COPOD score = sum of -log(min(F(x_j), 1-F(x_j))) over features.
        Higher score = more extreme in at least one tail.
        """
        n, p    = X.shape
        scores  = np.zeros(n)
        for j, ref_sorted in enumerate(cdfs):
            n_ref = len(ref_sorted)
            # Empirical CDF via searchsorted
            ranks = np.searchsorted(ref_sorted, X[:, j], side="right") / n_ref
            ranks = np.clip(ranks, 1e-9, 1 - 1e-9)
            tail  = np.minimum(ranks, 1 - ranks)
            scores += -np.log(tail)
        return scores

    def _compute_suspicion(self, flagged_ratio: float) -> float:
        excess = max(0.0, flagged_ratio - EXPECTED_CLEAN_FLAG_RATE)
        return float(np.clip(excess / SUSPICION_SCALE, 0.0, 1.0))

    @staticmethod
    def _estimate_eps(X: np.ndarray) -> float:
        try:
            k    = min(5, X.shape[0] - 1)
            nn   = NearestNeighbors(n_neighbors=k).fit(X)
            d, _ = nn.kneighbors(X)
            return float(max(np.percentile(d[:, -1], 90), 0.1))
        except Exception:
            return 0.5

    @staticmethod
    def _null_result(reason):
        return {"suspicion_score":0.0,"flagged_ratio":0.0,"flagged_count":0,
                "total_samples":0,"flagged_indices":[],"vote_threshold":6,
                "n_active_detectors":0,"per_detector":{},
                "expected_clean_flag_rate":EXPECTED_CLEAN_FLAG_RATE,
                "model_inversion":{"suspicion_score":0.0,"alarm":False,
                                   "signals":{},"interpretation":"skipped"},
                "skip_reason":reason}
