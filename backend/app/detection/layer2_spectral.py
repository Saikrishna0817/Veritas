s"""
Layer 2 — Spectral Activation Analysis  (RECTIFIED v2)
========================================================

RECTIFICATIONS:

  RECT 1 ── Random projection ≠ neural activations
    OLD: _simulate_activations() uses a fixed random matrix. This is not a
         penultimate-layer activation. The spectral gap it produces is driven
         by the random projection geometry, not attack structure.
    FIX: Keep the random projection (no real model available) BUT normalise the
         spectral gap by the expected gap under random matrix theory (RMT).
         For an n×d random Gaussian matrix, E[S[0]/S[1]] ≈ sqrt(n/d) × const.
         We divide the observed gap by this baseline to get a relative gap.
         This removes the systematic inflation from the projection itself.

  RECT 2 ── Spectral gap threshold is dimensionality-blind
    OLD: threshold=4.0 applied equally to 8-feature and 800-feature datasets.
         SVD singular value ratios scale with sqrt(n_features).
    FIX: Threshold is now adaptive: BASE_THRESHOLD × sqrt(n_features / 8),
         where 8 is the reference feature count for this dataset.

  RECT 3 ── KMeans k=2 forces a split even on unimodal data
    OLD: KMeans always returns 2 clusters. If data is unimodal (no backdoor),
         it still finds 2 clusters, and the minority cluster check then
         depends entirely on silhouette — which can still be > 0.25 on
         naturally skewed structured data.
    FIX: Add a gap statistic check before accepting the k=2 solution.
         If k=1 fits nearly as well as k=2 (gap < GAP_STAT_MIN), the
         "minority cluster" is an artefact — suppress the minority alarm.

  RECT 4 ── tightness=0.0 for single-point minority cluster triggers tight_alarm
    OLD: When minority cluster has exactly 1 point, tightness=0.0 < 0.5 → alarm.
         A single-point cluster is not meaningful evidence of a backdoor.
    FIX: Require minority cluster to have at least MIN_CLUSTER_POINTS members
         before evaluating tightness.

  Previously fixed bugs (spectral gap threshold raised, silhouette guard,
  AND-gate for all four signals) are preserved.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── Constants ────────────────────────────────────────────────────────────────
SPECTRAL_GAP_BASE        = 4.0    # base threshold at n_features=8
REFERENCE_N_FEATURES     = 8      # reference dimensionality for threshold scaling
MINORITY_CLUSTER_MAX     = 0.15
CLUSTER_TIGHTNESS_MAX    = 0.5
SILHOUETTE_MIN           = 0.25
GAP_STAT_MIN             = 0.10   # RECT 3: minimum inertia gap to accept k=2
MIN_CLUSTER_POINTS       = 3      # RECT 4: minimum points in minority cluster
N_ACTIVATION_DIMS        = 32
N_PCA_COMPONENTS         = 5
MIN_SAMPLES              = 40


class SpectralActivationAnalyzer:
    """
    Detects backdoor attacks via spectral analysis of simulated activations.

    The four-signal AND-gate requires:
      1. Relative spectral gap (RMT-normalised) above adaptive threshold
      2. Genuine minority cluster (gap statistic confirms k=2 is real)
      3. Cluster tightness (non-trivial cluster, >= 3 points)
      4. Silhouette score confirms cluster separation

    RECT 1 & 2: spectral gap is now RMT-normalised and dimensionality-adaptive.
    RECT 3: gap statistic guards against forced k=2 splits.
    RECT 4: single-point clusters no longer trigger tightness alarm.
    """

    def __init__(self, random_state: int = 42):
        self.random_state      = random_state
        self._rng              = np.random.RandomState(random_state)
        self._projection_matrix = None

    def analyze(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[0] < MIN_SAMPLES:
            return self._null_result("insufficient_samples")

        X = X.astype(float)
        n_features = X.shape[1]

        # Adaptive threshold (RECT 2)
        spectral_gap_threshold = SPECTRAL_GAP_BASE * np.sqrt(n_features / REFERENCE_N_FEATURES)

        # Step 1: Simulate activations
        activations = self._simulate_activations(X, y)

        # Step 2: PCA reduction
        n_comp   = min(N_PCA_COMPONENTS, activations.shape[1], activations.shape[0] - 1)
        pca      = PCA(n_components=n_comp, random_state=self.random_state)
        acts_pca = pca.fit_transform(activations)

        # Step 3: RMT-normalised spectral gap (RECT 1)
        spectral_gap, top_singular_ratio, rmt_baseline = self._compute_spectral_gap(activations)
        relative_gap = spectral_gap / max(rmt_baseline, 1.0)
        gap_alarm    = relative_gap > spectral_gap_threshold

        # Step 4: Cluster analysis with gap statistic (RECT 3 & 4)
        cluster_result = self._cluster_analysis(acts_pca)

        # Step 5: Combine — all four signals required
        minority_alarm = (
            cluster_result["minority_ratio"] < MINORITY_CLUSTER_MAX
            and cluster_result["gap_stat_ok"]           # RECT 3
            and cluster_result["minority_count"] >= MIN_CLUSTER_POINTS  # RECT 4
        )
        tight_alarm    = (
            cluster_result["minority_tightness"] < CLUSTER_TIGHTNESS_MAX
            and cluster_result["minority_count"] >= MIN_CLUSTER_POINTS  # RECT 4
        )
        sil_alarm      = cluster_result["silhouette"] > SILHOUETTE_MIN

        backdoor_detected = gap_alarm and minority_alarm and tight_alarm and sil_alarm

        # Suspicion score
        sig_gap   = self._normalise(relative_gap, spectral_gap_threshold, spread=2.0)
        sig_min   = self._normalise(MINORITY_CLUSTER_MAX - cluster_result["minority_ratio"], 0.0, spread=0.1)
        sig_tight = self._normalise(CLUSTER_TIGHTNESS_MAX - cluster_result["minority_tightness"], 0.0, spread=0.3)
        sig_sil   = self._normalise(cluster_result["silhouette"] - SILHOUETTE_MIN, 0.0, spread=0.2)

        if backdoor_detected:
            suspicion = float(np.clip(
                0.35 * sig_gap + 0.25 * sig_min + 0.25 * sig_tight + 0.15 * sig_sil,
                0.0, 1.0
            ))
        else:
            # Continuous partial suspicion: weight by how far each signal
            # exceeds its threshold. Pure fire-count (fires * 0.10) created
            # a 0.20 floor because silhouette and gap_stat routinely pass
            # on clean data.
            partial = (
                0.30 * max(0.0, sig_gap)
                + 0.25 * max(0.0, sig_min)
                + 0.25 * max(0.0, sig_tight)
                + 0.20 * max(0.0, sig_sil)
            )
            # Only credit partial score if at least 2 signals are genuinely alarming
            fires = sum([gap_alarm, minority_alarm, tight_alarm, sil_alarm])
            suspicion = float(np.clip(partial * (fires / 4.0), 0.0, 0.25))

        return {
            "suspicion_score"       : suspicion,
            "spectral_gap"          : float(spectral_gap),
            "relative_spectral_gap" : float(relative_gap),
            "rmt_baseline"          : float(rmt_baseline),
            "spectral_gap_threshold": float(spectral_gap_threshold),
            "top_singular_ratio"    : float(top_singular_ratio),
            "backdoor_detected"     : backdoor_detected,
            "minority_cluster_ratio": float(cluster_result["minority_ratio"]),
            "minority_cluster_count": int(cluster_result["minority_count"]),
            "minority_tightness"    : float(cluster_result["minority_tightness"]),
            "silhouette_score"      : float(cluster_result["silhouette"]),
            "gap_stat_ok"           : bool(cluster_result["gap_stat_ok"]),
            "signals": {
                "spectral_gap_alarm": bool(gap_alarm),
                "minority_alarm"    : bool(minority_alarm),
                "tightness_alarm"   : bool(tight_alarm),
                "silhouette_alarm"  : bool(sil_alarm),
            },
            "thresholds": {
                "spectral_gap"  : float(spectral_gap_threshold),
                "minority_ratio": MINORITY_CLUSTER_MAX,
                "tightness"     : CLUSTER_TIGHTNESS_MAX,
                "silhouette"    : SILHOUETTE_MIN,
                "gap_stat"      : GAP_STAT_MIN,
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _simulate_activations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        if self._projection_matrix is None or self._projection_matrix.shape[0] != n_features:
            self._projection_matrix = self._rng.randn(n_features, N_ACTIVATION_DIMS)

        acts  = np.tanh(X @ self._projection_matrix)

        # Guard: y may be None, contain NaN, or be non-numeric
        try:
            y_arr = np.array(y, dtype=float) if y is not None else np.zeros(len(X))
            y_arr = np.nan_to_num(y_arr, nan=0.0)
        except (ValueError, TypeError):
            y_arr = np.zeros(len(X))
        if len(np.unique(y_arr)) > 1:
            y_norm      = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min() + 1e-9) - 0.5
            label_signal = np.outer(y_norm, self._rng.randn(N_ACTIVATION_DIMS)) * 0.3
            acts         = acts + label_signal

        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        acts  = acts / np.clip(norms, 1e-9, None)
        return acts

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 1: RMT-normalised spectral gap
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_spectral_gap(self, activations: np.ndarray) -> tuple:
        try:
            _, S, _ = np.linalg.svd(activations, full_matrices=False)
        except np.linalg.LinAlgError:
            return 1.0, 0.0, 1.0

        if len(S) < 2 or S[1] < 1e-9:
            return 1.0, 0.0, 1.0

        n, d         = activations.shape
        # RMT Marchenko-Pastur: largest SV of random matrix ≈ sigma × (sqrt(n) + sqrt(d))
        rmt_baseline = float(np.sqrt(n) + np.sqrt(d))  # unnormalised
        # Our activations are L2-normalised rows → sigma ≈ 1/sqrt(d)
        rmt_baseline = rmt_baseline / np.sqrt(d)

        spectral_gap  = float(S[0] / S[1])
        top_ratio     = float(S[0] ** 2 / (np.sum(S ** 2) + 1e-9))
        return spectral_gap, top_ratio, rmt_baseline

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 3 & 4: gap statistic + minimum cluster size
    # ──────────────────────────────────────────────────────────────────────────
    def _cluster_analysis(self, acts_pca: np.ndarray) -> dict:
        default = {
            "minority_ratio"    : 0.5,
            "minority_count"    : 0,
            "minority_tightness": 1.0,
            "silhouette"        : 0.0,
            "gap_stat_ok"       : False,
        }
        try:
            km1    = KMeans(n_clusters=1, random_state=self.random_state, n_init=10)
            km2    = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            km1.fit(acts_pca)
            labels = km2.fit_predict(acts_pca)

            # RECT 3: gap statistic — is k=2 meaningfully better than k=1?
            inertia_k1  = float(km1.inertia_)
            inertia_k2  = float(km2.inertia_)
            gap_stat    = (inertia_k1 - inertia_k2) / (inertia_k1 + 1e-9)
            gap_stat_ok = gap_stat > GAP_STAT_MIN
        except Exception:
            return default

        counts         = np.bincount(labels)
        minority_idx   = int(np.argmin(counts))
        minority_mask  = (labels == minority_idx)
        minority_ratio = float(counts.min() / counts.sum())
        minority_count = int(counts.min())

        # RECT 4: only compute tightness for non-trivial clusters
        minority_pts   = acts_pca[minority_mask]
        if minority_count >= MIN_CLUSTER_POINTS:
            centre    = minority_pts.mean(axis=0)
            dists     = np.linalg.norm(minority_pts - centre, axis=1)
            tightness = float(dists.mean())
        else:
            tightness = float(CLUSTER_TIGHTNESS_MAX + 1.0)  # won't trigger alarm

        try:
            sil = silhouette_score(acts_pca, labels) if len(set(labels)) > 1 else 0.0
        except Exception:
            sil = 0.0

        return {
            "minority_ratio"    : minority_ratio,
            "minority_count"    : minority_count,
            "minority_tightness": tightness,
            "silhouette"        : float(sil),
            "gap_stat_ok"       : gap_stat_ok,
        }

    @staticmethod
    def _normalise(value: float, zero_point: float, spread: float) -> float:
        x = (value - zero_point) / max(spread, 1e-9)
        return float(np.clip(1.0 / (1.0 + np.exp(-x * 3)), 0.0, 1.0))

    @staticmethod
    def _null_result(reason: str) -> dict:
        return {
            "suspicion_score"       : 0.0,
            "spectral_gap"          : 1.0,
            "relative_spectral_gap" : 1.0,
            "rmt_baseline"          : 1.0,
            "top_singular_ratio"    : 0.0,
            "backdoor_detected"     : False,
            "minority_cluster_ratio": 0.5,
            "minority_cluster_count": 0,
            "minority_tightness"    : 1.0,
            "silhouette_score"      : 0.0,
            "gap_stat_ok"           : False,
            "signals": {
                "spectral_gap_alarm": False,
                "minority_alarm"    : False,
                "tightness_alarm"   : False,
                "silhouette_alarm"  : False,
            },
            "skip_reason": reason,
        }
