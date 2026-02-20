"""
Layer 2 — Spectral Activation Analysis  (RECTIFIED v3)
========================================================

NEW IN v3 — BLEND TRIGGER IMPROVEMENTS:

  BUG 4 ── Activation simulation doesn't amplify blend trigger structure
    OLD: tanh(X @ random_projection) — the random projection treats all
         input directions equally. A blend trigger that shifts samples along
         a specific direction in feature space gets diluted into all 32
         activation dimensions equally, weakening the spectral signal.
    FIX: Before projection, compute the top PCA direction of X and amplify
         it slightly in the activation. This preserves the backdoor's dominant
         direction through the simulated "network layer", making it visible
         in the singular value spectrum. We use a soft amplification (factor 2×)
         to avoid over-engineering a signal that isn't real.

  BUG 5 ── Spectral gap threshold is too conservative for low-poison-rate attacks
    OLD: SPECTRAL_GAP_BASE = 4.0. At 0.8% poison rate (8 of 1000 samples),
         the minority cluster signal barely moves the top singular value.
         The relative gap stays below 4.0 even with real poisoning.
    FIX: Lower BASE threshold to 2.5 and tighten the gap_stat requirement.
         The AND-gate (all 4 signals) still protects against false positives.
         Lower threshold catches real low-density attacks.

  BUG 6 ── Cluster analysis uses Euclidean distance in PCA space — poor for
            blend triggers which align samples along a single direction
    OLD: KMeans in PCA space — if poisoned samples are spread in 5D PCA
         space but concentrated in 1D (the trigger direction), KMeans may
         not find them as a tight cluster because tightness is computed as
         mean Euclidean distance in 5D.
    FIX: Before KMeans, project onto the top 2 PCA components only. In 2D,
         a 1D cluster is always tight. Also: fit PCA on incoming X (not
         on simulated activations) as an additional signal path.

All previously documented rectifications (RECT 1–4 from v2) are preserved.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── Constants ────────────────────────────────────────────────────────────────
SPECTRAL_GAP_BASE        = 2.5    # BUG 5: lowered from 4.0
REFERENCE_N_FEATURES     = 8
MINORITY_CLUSTER_MAX     = 0.15
CLUSTER_TIGHTNESS_MAX    = 0.5
SILHOUETTE_MIN           = 0.20   # BUG 5: lowered from 0.25 for low-density attacks
GAP_STAT_MIN             = 0.08   # BUG 5: slightly relaxed from 0.10
MIN_CLUSTER_POINTS       = 3
N_ACTIVATION_DIMS        = 32
N_PCA_COMPONENTS         = 5
MIN_SAMPLES              = 40
PCA_AMPLIFICATION        = 2.0    # BUG 4: amplify top PCA direction in activations


class SpectralActivationAnalyzer:
    """
    Detects backdoor attacks via spectral analysis of simulated activations.

    v3 improvements:
      - PCA-amplified activation simulation (catches blend triggers)
      - Lower spectral gap threshold (catches low-density attacks)
      - 2D cluster analysis (better for directional triggers)
      - Direct feature-space PCA cluster signal as additional path
    """

    def __init__(self, random_state: int = 42):
        self.random_state       = random_state
        self._rng               = np.random.RandomState(random_state)
        self._projection_matrix = None

    def analyze(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[0] < MIN_SAMPLES:
            return self._null_result("insufficient_samples")

        X = X.astype(float)
        n_features = X.shape[1]

        # Adaptive threshold (RECT 2)
        spectral_gap_threshold = SPECTRAL_GAP_BASE * np.sqrt(n_features / REFERENCE_N_FEATURES)

        # BUG 4: Compute top PCA direction of input for amplification
        try:
            n_comp_input = min(3, n_features, X.shape[0] - 1)
            input_pca    = PCA(n_components=n_comp_input, random_state=self.random_state)
            input_pca.fit(X)
            top_direction = input_pca.components_[0]   # shape (n_features,)
        except Exception:
            top_direction = None

        # Step 1: Simulate activations (BUG 4: PCA-amplified)
        activations = self._simulate_activations(X, y, top_direction)

        # Step 2: PCA reduction
        n_comp   = min(N_PCA_COMPONENTS, activations.shape[1], activations.shape[0] - 1)
        pca      = PCA(n_components=n_comp, random_state=self.random_state)
        acts_pca = pca.fit_transform(activations)

        # Step 3: RMT-normalised spectral gap (RECT 1)
        spectral_gap, top_singular_ratio, rmt_baseline = self._compute_spectral_gap(activations)
        relative_gap = spectral_gap / max(rmt_baseline, 1.0)
        gap_alarm    = relative_gap > spectral_gap_threshold

        # Step 4: Cluster analysis — use top 2 PCA dims (BUG 6)
        acts_2d        = acts_pca[:, :2] if acts_pca.shape[1] >= 2 else acts_pca
        cluster_result = self._cluster_analysis(acts_2d)

        # BUG 6: Also run cluster analysis directly on input feature space
        if n_features >= 2:
            feature_pca = input_pca.transform(X) if top_direction is not None else acts_2d
            feature_pca_2d = feature_pca[:, :2] if feature_pca.shape[1] >= 2 else feature_pca
            feature_cluster = self._cluster_analysis(feature_pca_2d)
            # Take the stronger cluster signal of the two
            if feature_cluster["minority_ratio"] < cluster_result["minority_ratio"]:
                cluster_result = feature_cluster

        # Step 5: Combine — all four signals required (lowered thresholds for v3)
        minority_alarm = (
            cluster_result["minority_ratio"] < MINORITY_CLUSTER_MAX
            and cluster_result["gap_stat_ok"]
            and cluster_result["minority_count"] >= MIN_CLUSTER_POINTS
        )
        tight_alarm    = (
            cluster_result["minority_tightness"] < CLUSTER_TIGHTNESS_MAX
            and cluster_result["minority_count"] >= MIN_CLUSTER_POINTS
        )
        sil_alarm      = cluster_result["silhouette"] > SILHOUETTE_MIN

        backdoor_detected = gap_alarm and minority_alarm and tight_alarm and sil_alarm

        # Partial credit: 3 of 4 signals is still suspicious
        signals_fired = sum([gap_alarm, minority_alarm, tight_alarm, sil_alarm])
        partial_backdoor = signals_fired >= 3

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
        elif partial_backdoor:
            # 3 signals fire → moderate suspicion (was 0 in v2!)
            suspicion = float(np.clip(
                (0.35 * sig_gap + 0.25 * sig_min + 0.25 * sig_tight + 0.15 * sig_sil) * 0.5,
                0.15, 0.50
            ))
        else:
            suspicion = float(min(signals_fired * 0.10, 0.35))

        return {
            "suspicion_score"       : suspicion,
            "spectral_gap"          : float(spectral_gap),
            "relative_spectral_gap" : float(relative_gap),
            "rmt_baseline"          : float(rmt_baseline),
            "spectral_gap_threshold": float(spectral_gap_threshold),
            "top_singular_ratio"    : float(top_singular_ratio),
            "backdoor_detected"     : backdoor_detected,
            "partial_backdoor"      : partial_backdoor,
            "signals_fired"         : signals_fired,
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
    # BUG 4: PCA-amplified activation simulation
    # ──────────────────────────────────────────────────────────────────────────
    def _simulate_activations(self, X: np.ndarray, y: np.ndarray,
                               top_direction: np.ndarray = None) -> np.ndarray:
        n_features = X.shape[1]
        if self._projection_matrix is None or self._projection_matrix.shape[0] != n_features:
            self._projection_matrix = self._rng.randn(n_features, N_ACTIVATION_DIMS)

        # BUG 4: Amplify the top PCA direction before projection
        X_amplified = X.copy()
        if top_direction is not None:
            # Project each sample onto top direction, then amplify that component
            projections   = X @ top_direction                      # (n,)
            X_amplified  += np.outer(projections, top_direction) * (PCA_AMPLIFICATION - 1.0)

        acts  = np.tanh(X_amplified @ self._projection_matrix)
        y_arr = np.array(y).astype(float)
        if len(np.unique(y_arr)) > 1:
            y_norm       = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min() + 1e-9) - 0.5
            label_signal = np.outer(y_norm, self._rng.randn(N_ACTIVATION_DIMS)) * 0.3
            acts         = acts + label_signal

        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        acts  = acts / np.clip(norms, 1e-9, None)
        return acts

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 1: RMT-normalised spectral gap (unchanged)
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_spectral_gap(self, activations: np.ndarray) -> tuple:
        try:
            _, S, _ = np.linalg.svd(activations, full_matrices=False)
        except np.linalg.LinAlgError:
            return 1.0, 0.0, 1.0

        if len(S) < 2 or S[1] < 1e-9:
            return 1.0, 0.0, 1.0

        n, d         = activations.shape
        rmt_baseline = float(np.sqrt(n) + np.sqrt(d))
        rmt_baseline = rmt_baseline / np.sqrt(d)

        spectral_gap  = float(S[0] / S[1])
        top_ratio     = float(S[0] ** 2 / (np.sum(S ** 2) + 1e-9))
        return spectral_gap, top_ratio, rmt_baseline

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 3 & 4 + BUG 6: Cluster analysis (now in 2D)
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

        minority_pts   = acts_pca[minority_mask]
        if minority_count >= MIN_CLUSTER_POINTS:
            centre    = minority_pts.mean(axis=0)
            dists     = np.linalg.norm(minority_pts - centre, axis=1)
            tightness = float(dists.mean())
        else:
            tightness = float(CLUSTER_TIGHTNESS_MAX + 1.0)

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
            "spectral_gap_threshold": float(SPECTRAL_GAP_BASE),
            "top_singular_ratio"    : 0.0,
            "backdoor_detected"     : False,
            "partial_backdoor"      : False,
            "signals_fired"         : 0,
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
            "thresholds": {
                "spectral_gap"  : float(SPECTRAL_GAP_BASE),
                "minority_ratio": MINORITY_CLUSTER_MAX,
                "tightness"     : CLUSTER_TIGHTNESS_MAX,
                "silhouette"    : SILHOUETTE_MIN,
                "gap_stat"      : GAP_STAT_MIN,
            },
            "skip_reason": reason,
        }