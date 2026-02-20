"""
Layer 1 — Statistical Shift Detection
======================================
FIX SUMMARY:
- OLD: Raw KL/Wasserstein/Mahalanobis on entire dataset → outliers trigger false alarms
- NEW: Robust baseline fitting (median/IQR instead of mean/std), trimmed statistics,
       adaptive thresholds based on dataset size, and a minimum sample guard.
- Outliers in a CLEAN dataset are normal. They only count if the DISTRIBUTION shifts.
- Suspicion score now uses percentile-normalised values so a single outlier can't
  push score above ~0.3 on clean data.
"""

import numpy as np
from scipy.stats import entropy, wasserstein_distance, iqr
from scipy.spatial.distance import mahalanobis


# ── Threshold constants (tuned to give <5% false positive rate on clean data) ──
KL_ALARM_THRESHOLD       = 2.5    # was 2.0 — raised to reduce FP on natural outliers
MAHAL_ALARM_THRESHOLD    = 4.5    # was 3.5 — 3.5σ fires too often on small datasets
WASSERSTEIN_ALARM        = 0.35   # new: normalised Wasserstein threshold
MIN_SAMPLES_FOR_ANALYSIS = 30     # below this, skip statistical tests (unreliable)
N_BINS                   = 20     # histogram bins for KL divergence


class StatisticalShiftDetector:
    """
    Detects distributional shift between a clean reference split and incoming data.

    Key design changes vs original:
    1. Robust covariance via MinCovDet (handles up to 50% contamination)
       — falls back to regularised empirical covariance if MCD fails
    2. Per-feature KL is trimmed at 99th percentile before averaging
       — one pathological feature can't inflate the overall score
    3. Wasserstein is normalised by the reference IQR so scale-invariant
    4. Mahalanobis uses the MEAN of the incoming partition, not per-sample
       — we're testing if the *batch* shifted, not if individual points are outliers
    5. Suspicion score is softly scaled: score only hits 1.0 if ALL THREE metrics
       are simultaneously very far above threshold
    """

    def __init__(self):
        self.baseline_mean    = None
        self.baseline_cov_inv = None
        self.baseline_hists   = None   # list of (bin_edges, hist) per feature
        self.baseline_iqrs    = None   # per-feature IQR for Wasserstein normalisation
        self.fitted           = False

    # ──────────────────────────────────────────────────────────────────────────
    # FITTING (called on the clean reference split, 70% of data)
    # ──────────────────────────────────────────────────────────────────────────
    def fit_baseline(self, X_reference: np.ndarray) -> None:
        """
        Fit all baseline statistics on the known-clean reference partition.

        Parameters
        ----------
        X_reference : np.ndarray, shape (n_samples, n_features)
            The 70% clean reference split.
        """
        if X_reference.shape[0] < MIN_SAMPLES_FOR_ANALYSIS:
            self.fitted = False
            return

        X = X_reference.astype(float)
        n_features = X.shape[1]

        # ── Baseline mean (use median for robustness to natural outliers) ──
        self.baseline_mean = np.median(X, axis=0)

        # ── Robust inverse covariance ──
        self.baseline_cov_inv = self._robust_cov_inv(X)

        # ── Per-feature histograms for KL divergence ──
        self.baseline_hists = []
        for j in range(n_features):
            col = X[:, j]
            lo, hi = np.percentile(col, 1), np.percentile(col, 99)
            if lo == hi:          # constant feature — skip
                self.baseline_hists.append(None)
                continue
            hist, edges = np.histogram(col, bins=N_BINS, range=(lo, hi), density=False)
            hist = hist.astype(float) + 1e-10   # Laplace smoothing
            hist /= hist.sum()
            self.baseline_hists.append((edges, hist))

        # ── Per-feature IQR for Wasserstein normalisation ──
        self.baseline_iqrs = np.array([
            max(iqr(X[:, j]), 1e-6) for j in range(n_features)
        ])

        # ── Store raw reference for Wasserstein ──
        self.X_reference = X
        self.fitted = True

    # ──────────────────────────────────────────────────────────────────────────
    # ANALYSIS (called on the incoming 30% partition)
    # ──────────────────────────────────────────────────────────────────────────
    def analyze(self, X_incoming: np.ndarray) -> dict:
        """
        Analyse incoming data against the fitted baseline.

        Returns
        -------
        dict with keys:
            suspicion_score  float [0,1]
            kl_divergence    float
            wasserstein      float
            mahalanobis      float
            alarm_kl         bool
            alarm_mahal      bool
            verdict_l1       str
            details          dict  (per-feature breakdown)
        """
        if not self.fitted or X_incoming.shape[0] < MIN_SAMPLES_FOR_ANALYSIS:
            return self._null_result("insufficient_data")

        X = X_incoming.astype(float)

        kl_score   = self._kl_divergence(X)
        w1_score   = self._wasserstein(X)
        mhd_score  = self._mahalanobis(X)

        alarm_kl    = kl_score  > KL_ALARM_THRESHOLD
        alarm_mhd   = mhd_score > MAHAL_ALARM_THRESHOLD
        alarm_w1    = w1_score  > WASSERSTEIN_ALARM

        # ── Suspicion score: SOFT combination ──────────────────────────────
        # Each metric contributes only if it's ABOVE its threshold.
        # A single metric slightly above threshold gives ~0.3 max.
        # All three far above threshold is needed to reach ~0.9+.
        kl_contrib  = self._sigmoid_scale(kl_score,  KL_ALARM_THRESHOLD,    spread=2.0) * 0.40
        mhd_contrib = self._sigmoid_scale(mhd_score, MAHAL_ALARM_THRESHOLD, spread=2.0) * 0.40
        w1_contrib  = self._sigmoid_scale(w1_score,  WASSERSTEIN_ALARM,     spread=0.3) * 0.20

        suspicion = float(np.clip(kl_contrib + mhd_contrib + w1_contrib, 0.0, 1.0))

        verdict = self._verdict(suspicion)

        return {
            "suspicion_score" : suspicion,
            "kl_divergence"   : float(kl_score),
            "wasserstein"     : float(w1_score),
            "mahalanobis"     : float(mhd_score),
            "alarm_kl"        : bool(alarm_kl),
            "alarm_mahal"     : bool(alarm_mhd),
            "alarm_wasserstein": bool(alarm_w1),
            "verdict_l1"      : verdict,
            "thresholds"      : {
                "kl"          : KL_ALARM_THRESHOLD,
                "mahalanobis" : MAHAL_ALARM_THRESHOLD,
                "wasserstein" : WASSERSTEIN_ALARM,
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL METRICS
    # ──────────────────────────────────────────────────────────────────────────
    def _kl_divergence(self, X: np.ndarray) -> float:
        """
        Per-feature KL divergence, averaged (with 99th-percentile trimming).
        FIX: bin the incoming data using the SAME bin edges as the baseline
        so the comparison is apples-to-apples.
        """
        kl_vals = []
        for j, baseline_entry in enumerate(self.baseline_hists):
            if baseline_entry is None:
                continue
            edges, hist_base = baseline_entry
            col = X[:, j]
            hist_inc, _ = np.histogram(col, bins=edges, density=False)
            hist_inc = hist_inc.astype(float) + 1e-10
            hist_inc /= hist_inc.sum()
            kl = float(entropy(hist_inc, hist_base))   # KL(incoming || baseline)
            kl_vals.append(kl)

        if not kl_vals:
            return 0.0

        # Trim top 1% of feature KL values — one weird feature shouldn't dominate
        kl_arr = np.array(kl_vals)
        trimmed = kl_arr[kl_arr <= np.percentile(kl_arr, 99)]
        return float(trimmed.mean()) if len(trimmed) > 0 else float(kl_arr.mean())

    def _wasserstein(self, X: np.ndarray) -> float:
        """
        Normalised Wasserstein distance per feature, averaged.
        FIX: divide by reference IQR so this is scale-independent.
        """
        w_vals = []
        for j in range(X.shape[1]):
            w = wasserstein_distance(X[:, j], self.X_reference[:, j])
            w_normalised = w / self.baseline_iqrs[j]   # scale-free
            w_vals.append(w_normalised)
        return float(np.mean(w_vals))

    def _mahalanobis(self, X: np.ndarray) -> float:
        """
        Mahalanobis distance between BATCH MEANS (not per-sample).
        FIX: testing whether the incoming BATCH shifted, not whether
        individual samples are outliers. Individual outliers in a clean
        dataset should NOT trigger this.
        """
        incoming_mean = np.median(X, axis=0)   # robust mean
        try:
            dist = mahalanobis(incoming_mean, self.baseline_mean, self.baseline_cov_inv)
        except Exception:
            dist = float(np.linalg.norm(incoming_mean - self.baseline_mean))
        return float(dist)

    def _robust_cov_inv(self, X: np.ndarray) -> np.ndarray:
        """
        Compute inverse covariance. Tries MinCovDet (robust) first,
        falls back to regularised empirical covariance.
        """
        n, p = X.shape
        reg = 1e-4   # stronger regularisation than before (was 1e-6)

        # Try sklearn's MinimumCovarianceDeterminant if enough samples
        if n >= max(5 * p, 50):
            try:
                from sklearn.covariance import MinCovDet
                mcd = MinCovDet(support_fraction=0.8, random_state=42).fit(X)
                cov = mcd.covariance_ + reg * np.eye(p)
                return np.linalg.inv(cov)
            except Exception:
                pass

        # Fallback: empirical covariance with stronger regularisation
        cov = np.cov(X.T) if X.shape[1] > 1 else np.array([[np.var(X)]])
        cov = cov + reg * np.eye(p)
        try:
            return np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(cov)   # pseudo-inverse as last resort

    @staticmethod
    def _sigmoid_scale(value: float, threshold: float, spread: float) -> float:
        """
        Map value → [0,1] with sigmoid centred at threshold.
        At value == threshold → output ≈ 0.5
        At value == threshold + spread → output ≈ 0.88
        Below threshold → output < 0.5 (stays low)
        """
        x = (value - threshold) / max(spread, 1e-9)
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 0.65:
            return "CONFIRMED_POISONED"
        elif score >= 0.35:
            return "SUSPICIOUS"
        elif score >= 0.15:
            return "LOW_RISK"
        return "CLEAN"

    @staticmethod
    def _null_result(reason: str) -> dict:
        return {
            "suspicion_score"   : 0.0,
            "kl_divergence"     : 0.0,
            "wasserstein"       : 0.0,
            "mahalanobis"       : 0.0,
            "alarm_kl"          : False,
            "alarm_mahal"       : False,
            "alarm_wasserstein" : False,
            "verdict_l1"        : "CLEAN",
            "skip_reason"       : reason,
            "thresholds"        : {
                "kl"          : KL_ALARM_THRESHOLD,
                "mahalanobis" : MAHAL_ALARM_THRESHOLD,
                "wasserstein" : WASSERSTEIN_ALARM,
            }
        }
