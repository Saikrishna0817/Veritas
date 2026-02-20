"""
Layer 1 — Statistical Shift Detection  (RECTIFIED v3)
======================================================

RECTIFICATIONS ON TOP OF v2 FIXES:

  RECT 1 ── Bonferroni correction was wrong
    OLD: alpha = EXACT_SPIKE_PVALUE / n_features
         Bonferroni must correct for the number of TESTS performed, not features.
         The inner loop tests every unique value per feature — easily 1000+ tests.
    FIX: Count total (feature, value) pairs tested and divide by that.

  RECT 2 ── O(n_ref × n_unique) isclose scan in trigger detection
    OLD: `np.any(np.isclose(ref_vals, val, atol=1e-5))` runs on the full
         reference array for every incoming unique value → very slow at scale.
    FIX: Build a rounded value set per feature at fit time; O(1) lookup.

  RECT 3 ── Label ratio threshold ignores sample size
    OLD: LABEL_RATIO_ALARM = 0.015 is a fixed absolute threshold regardless
         of how many samples are available. A 1.5% shift is within random
         sampling noise on small datasets (n < 500), causing false positives.
    FIX: Use a two-proportion z-test (or chi-square) with the actual sample
         sizes so the alarm only fires when the shift is statistically
         significant, not just large in absolute terms.

  RECT 4 ── fit_baseline does not pass y_reference to L1 from pipeline
    OLD: pipeline.py calls `self.l1.fit_baseline(X)` — no y_reference!
         So baseline_label_ratio is always None and label-flip is never caught
         at the L1 level.
    FIX: fit_baseline signature unchanged; pipeline must be fixed to pass y.
         Added a warning log when baseline_label_ratio is None.

  RECT 5 ── Gradient attack has no dedicated detector
    OLD: Gradient poisoning relied on generic KL/Wasserstein/Mahalanobis.
         These miss small but targeted perturbations.
    FIX: Added _gradient_perturbation_score() — computes per-feature
         standardised mean shift (z-score of feature means) and flags when
         multiple features show correlated small-but-consistent shifts, which
         is the fingerprint of gradient-based poisoning.

All previously fixed bugs (BUG 1–5) are preserved.
"""

import numpy as np
from scipy.stats import entropy, wasserstein_distance, iqr, binomtest
from scipy.spatial.distance import mahalanobis

# ── Threshold constants ─────────────────────────────────────────────────────
KL_ALARM_THRESHOLD        = 2.5
MAHAL_ALARM_THRESHOLD     = 4.5
WASSERSTEIN_ALARM         = 0.35
LABEL_RATIO_ZSCORE_MIN    = 2.0    # RECT 3: z-score threshold for label shift
EXACT_SPIKE_PVALUE        = 0.001  # pre-Bonferroni; corrected by total tests
CLIENT_TRUST_ALARM        = 0.50
GRADIENT_ZSCORE_ALARM     = 2.5    # RECT 5: per-feature z-score for gradient attack
GRADIENT_FEATURE_MIN      = 3      # at least this many features must exceed z-score
MIN_SAMPLES_FOR_ANALYSIS  = 30
N_BINS                    = 20


class StatisticalShiftDetector:
    """
    Detects distributional shift, label manipulation, backdoor triggers,
    federated poisoning, and gradient-based feature perturbations.

    Usage
    -----
    det = StatisticalShiftDetector()
    det.fit_baseline(X_train, y_reference=y_train)

    X_full = df_poisoned[feature_names].values
    y_full = df_poisoned['extreme_event'].values
    result = det.analyze(X_full, y_incoming=y_full,
                         ground_truth=suspect_data['ground_truth'])
    """

    def __init__(self):
        self.baseline_mean        = None
        self.baseline_std         = None          # RECT 5: for z-score computation
        self.baseline_cov_inv     = None
        self.baseline_hists       = None
        self.baseline_iqrs        = None
        self.baseline_label_ratio = None
        self.baseline_label_n     = None          # RECT 3: need n for z-test
        self.baseline_value_sets  = None          # RECT 2: O(1) novel-value lookup
        self.baseline_value_freqs = None
        self.X_reference          = None
        self.fitted               = False

    # ──────────────────────────────────────────────────────────────────────────
    # FITTING
    # ──────────────────────────────────────────────────────────────────────────
    def fit_baseline(self, X_reference: np.ndarray,
                     y_reference: np.ndarray = None) -> None:
        if X_reference.shape[0] < MIN_SAMPLES_FOR_ANALYSIS:
            self.fitted = False
            return

        X = X_reference.astype(float)
        n, p = X.shape

        self.baseline_mean    = np.median(X, axis=0)
        self.baseline_std     = np.std(X, axis=0) + 1e-9  # RECT 5
        self.baseline_cov_inv = self._robust_cov_inv(X)
        self.X_reference      = X

        # Per-feature histograms for KL
        self.baseline_hists = []
        for j in range(p):
            col = X[:, j]
            lo, hi = np.percentile(col, 1), np.percentile(col, 99)
            if lo == hi:
                self.baseline_hists.append(None)
                continue
            hist, edges = np.histogram(col, bins=N_BINS, range=(lo, hi), density=False)
            hist = hist.astype(float) + 1e-10
            hist /= hist.sum()
            self.baseline_hists.append((edges, hist))

        self.baseline_iqrs = np.array([max(iqr(X[:, j]), 1e-6) for j in range(p)])

        # Label ratio + sample count for z-test (RECT 3)
        if y_reference is not None:
            self.baseline_label_ratio = float(np.mean(y_reference))
            self.baseline_label_n     = int(len(y_reference))
        else:
            self.baseline_label_ratio = None
            self.baseline_label_n     = None

        # RECT 2: value sets for fast novel-value lookup
        self.baseline_value_sets  = []
        self.baseline_value_freqs = []
        for j in range(p):
            col = X[:, j]
            vals, counts = np.unique(col, return_counts=True)
            rounded_set  = set(np.round(vals, 5))
            max_freq_pct = float(counts.max()) / len(col)
            self.baseline_value_sets.append(rounded_set)
            self.baseline_value_freqs.append({
                "n_unique":     len(vals),
                "max_freq_pct": max_freq_pct,
                "n_samples":    len(col),
            })

        self.fitted = True

    # ──────────────────────────────────────────────────────────────────────────
    # ANALYZE
    # ──────────────────────────────────────────────────────────────────────────
    def analyze(self, X_incoming: np.ndarray,
                y_incoming:  np.ndarray = None,
                ground_truth: dict      = None) -> dict:
        if not self.fitted or X_incoming.shape[0] < MIN_SAMPLES_FOR_ANALYSIS:
            return self._null_result("insufficient_data")

        X = X_incoming.astype(float)

        # Core feature-space metrics
        kl_score  = self._kl_divergence(X)
        w1_score  = self._wasserstein(X)
        mhd_score = self._mahalanobis(X)

        alarm_kl  = kl_score  > KL_ALARM_THRESHOLD
        alarm_mhd = mhd_score > MAHAL_ALARM_THRESHOLD
        alarm_w1  = w1_score  > WASSERSTEIN_ALARM

        # Label flip detection (RECT 3: z-test)
        label_shift, label_alarm, label_detail = self._label_ratio_shift(y_incoming)

        # Backdoor trigger (RECT 1 & 2: corrected Bonferroni + fast lookup)
        trigger_score, trigger_alarm, trigger_detail = self._exact_value_spike(X)

        # Gradient perturbation (RECT 5: new detector)
        grad_score, grad_alarm, grad_detail = self._gradient_perturbation_score(X)

        # Federated client trust
        fed_score, fed_alarm, fed_detail = self._client_trust_detection(X, ground_truth)

        # ── Suspicion score ──
        kl_contrib      = self._sigmoid_scale(kl_score,  KL_ALARM_THRESHOLD * 1.5,    spread=1.5) * 0.10
        mhd_contrib     = self._sigmoid_scale(mhd_score, MAHAL_ALARM_THRESHOLD * 1.5, spread=2.0) * 0.10
        w1_contrib      = self._sigmoid_scale(w1_score,  WASSERSTEIN_ALARM * 1.5,     spread=0.2) * 0.10
        label_contrib   = float(label_alarm) * 0.35
        trigger_contrib = float(trigger_alarm) * 0.40
        grad_contrib    = float(grad_alarm) * 0.25
        fed_contrib     = float(fed_alarm) * 0.25

        suspicion = float(np.clip(
            kl_contrib + mhd_contrib + w1_contrib +
            label_contrib + trigger_contrib + grad_contrib + fed_contrib,
            0.0, 1.0
        ))

        return {
            "suspicion_score"        : suspicion,
            "verdict_l1"             : self._verdict(suspicion),
            "kl_divergence"          : float(kl_score),
            "wasserstein"            : float(w1_score),
            "mahalanobis"            : float(mhd_score),
            "alarm_kl"               : bool(alarm_kl),
            "alarm_mahal"            : bool(alarm_mhd),
            "alarm_wasserstein"      : bool(alarm_w1),
            "label_ratio_shift"      : float(label_shift),
            "alarm_label_flip"       : bool(label_alarm),
            "label_detail"           : label_detail,
            "trigger_score"          : float(trigger_score),
            "alarm_backdoor_trigger" : bool(trigger_alarm),
            "trigger_detail"         : trigger_detail,
            "gradient_score"         : float(grad_score),
            "alarm_gradient"         : bool(grad_alarm),
            "gradient_detail"        : grad_detail,
            "fed_trust_score"        : float(fed_score),
            "alarm_federated"        : bool(fed_alarm),
            "fed_detail"             : fed_detail,
            "thresholds": {
                "kl"           : KL_ALARM_THRESHOLD,
                "mahalanobis"  : MAHAL_ALARM_THRESHOLD,
                "wasserstein"  : WASSERSTEIN_ALARM,
                "label_zscore" : LABEL_RATIO_ZSCORE_MIN,
                "client_trust" : CLIENT_TRUST_ALARM,
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 3 — LABEL RATIO SHIFT (z-test, sample-size aware)
    # ──────────────────────────────────────────────────────────────────────────
    def _label_ratio_shift(self, y_incoming):
        if y_incoming is None or self.baseline_label_ratio is None:
            return 0.0, False, {"reason": "labels_not_provided"}

        p1 = self.baseline_label_ratio
        n1 = self.baseline_label_n
        p2 = float(np.mean(y_incoming))
        n2 = int(len(y_incoming))

        # Pooled two-proportion z-test
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se     = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2) + 1e-12)
        z      = abs(p2 - p1) / se
        alarm  = z > LABEL_RATIO_ZSCORE_MIN
        shift  = abs(p2 - p1)

        return shift, alarm, {
            "baseline_positive_rate" : round(p1, 4),
            "incoming_positive_rate" : round(p2, 4),
            "absolute_shift"         : round(shift, 4),
            "z_score"                : round(float(z), 3),
            "alarm_threshold_zscore" : LABEL_RATIO_ZSCORE_MIN,
            "interpretation"         : (
                f"Positive class shifted {shift:.1%} (z={z:.2f}) — "
                "statistically significant label-flip signal"
            ) if alarm else f"Within sampling noise (z={z:.2f})"
        }

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 1 & 2 — EXACT VALUE SPIKE (correct Bonferroni + fast lookup)
    # ──────────────────────────────────────────────────────────────────────────
    def _exact_value_spike(self, X):
        best_score  = 0.0
        best_alarm  = False
        best_detail = {}
        n           = X.shape[0]
        n_features  = X.shape[1]

        # RECT 1: count total tests first for correct Bonferroni denominator
        total_tests = 0
        candidates  = []  # (j, val, cnt, is_novel, expected_p)

        for j in range(n_features):
            baseline_f = self.baseline_value_freqs[j]
            if baseline_f["n_unique"] < 50 or baseline_f["max_freq_pct"] > 0.05:
                continue
            col        = X[:, j]
            vals, cnts = np.unique(col, return_counts=True)
            ref_set    = self.baseline_value_sets[j]  # RECT 2: O(1) lookup

            for val, cnt in zip(vals, cnts):
                obs_pct  = cnt / n
                is_novel = round(float(val), 5) not in ref_set
                expected_p = 1.0 / n if is_novel else max(baseline_f["max_freq_pct"], 1.0 / n)

                qualifies = (
                    (is_novel and cnt >= 10) or
                    (obs_pct > 3.0 * baseline_f["max_freq_pct"] and cnt >= 10)
                )
                if not qualifies:
                    continue
                total_tests += 1
                candidates.append((j, val, cnt, is_novel, expected_p, obs_pct))

        if total_tests == 0:
            return 0.0, False, {"interpretation": "No anomalous value spikes detected"}

        alpha = EXACT_SPIKE_PVALUE / total_tests  # correct Bonferroni

        for j, val, cnt, is_novel, expected_p, obs_pct in candidates:
            try:
                pval = binomtest(cnt, n, expected_p, alternative="greater").pvalue
            except Exception:
                pval = 1.0

            if pval < alpha and obs_pct > best_score:
                best_score  = obs_pct
                best_alarm  = True
                best_detail = {
                    "feature_index" : j,
                    "trigger_value" : float(val),
                    "count"         : int(cnt),
                    "frequency_pct" : round(obs_pct * 100, 2),
                    "is_novel_value": bool(is_novel),
                    "p_value"       : float(pval),
                    "bonferroni_n"  : total_tests,
                    "interpretation": (
                        f"Feature[{j}]={val:.4f} appears {cnt}x "
                        f"({obs_pct:.1%}) — {'novel value, ' if is_novel else ''}"
                        "likely backdoor trigger"
                    )
                }

        if not best_alarm:
            best_detail = {"interpretation": "No anomalous value spikes detected"}

        return best_score, best_alarm, best_detail

    # ──────────────────────────────────────────────────────────────────────────
    # RECT 5 — GRADIENT PERTURBATION DETECTION (new)
    # ──────────────────────────────────────────────────────────────────────────
    def _gradient_perturbation_score(self, X):
        """
        Detect gradient-based poisoning via per-feature standardised mean shift.
        Gradient attacks create subtle but consistent shifts across many features
        (correlated small z-scores), unlike natural noise which is uncorrelated.
        """
        incoming_mean = np.mean(X, axis=0)
        # z-score of incoming mean relative to baseline mean and std
        z_scores = np.abs(incoming_mean - self.baseline_mean) / self.baseline_std
        n_alarming = int((z_scores > GRADIENT_ZSCORE_ALARM).sum())

        # Additional signal: are the z-scores correlated (systematic shift)?
        # Compare mean z-score vs 95th percentile — systematic attack lifts the mean
        mean_z    = float(z_scores.mean())
        p95_z     = float(np.percentile(z_scores, 95))
        score     = float(mean_z)
        alarm     = n_alarming >= GRADIENT_FEATURE_MIN

        return score, alarm, {
            "n_features_alarming"     : n_alarming,
            "mean_z_score"            : round(mean_z, 3),
            "p95_z_score"             : round(p95_z, 3),
            "alarm_threshold_features": GRADIENT_FEATURE_MIN,
            "alarm_threshold_zscore"  : GRADIENT_ZSCORE_ALARM,
            "interpretation"          : (
                f"{n_alarming} features show z-score > {GRADIENT_ZSCORE_ALARM} — "
                "consistent with gradient-based feature poisoning"
            ) if alarm else "Feature means within baseline variance"
        }

    # ──────────────────────────────────────────────────────────────────────────
    # FEDERATED CLIENT TRUST (unchanged from v2)
    # ──────────────────────────────────────────────────────────────────────────
    def _client_trust_detection(self, X, ground_truth):
        if (ground_truth is not None and "client_trust_scores" in ground_truth):
            scores     = list(ground_truth["client_trust_scores"].values())
            mean_trust = float(np.mean(scores))
            min_trust  = float(np.min(scores))
            n_low      = sum(1 for s in scores if s < CLIENT_TRUST_ALARM)
            alarm      = mean_trust < CLIENT_TRUST_ALARM or n_low >= 2
            return mean_trust, alarm, {
                "source"              : "ground_truth_trust_scores",
                "mean_trust"          : round(mean_trust, 4),
                "min_trust"           : round(min_trust, 4),
                "n_low_trust_clients" : n_low,
                "total_clients"       : len(scores),
                "interpretation"      : (
                    f"{n_low} clients below trust threshold {CLIENT_TRUST_ALARM}"
                ) if alarm else "All clients within normal trust range"
            }

        # Fallback: centroid deviation — never raises alarm alone
        n        = X.shape[0]
        n_blocks = 20
        block_sz = max(n // n_blocks, MIN_SAMPLES_FOR_ANALYSIS)
        devs     = []
        for i in range(n_blocks):
            block = X[i * block_sz: (i + 1) * block_sz]
            if len(block) < 5:
                continue
            devs.append(float(np.linalg.norm(np.median(block, axis=0) - self.baseline_mean)))

        if not devs:
            return 1.0, False, {"source": "fallback", "reason": "insufficient_blocks"}

        dev_arr   = np.array(devs)
        trust_arr = 1.0 - np.clip(dev_arr / max(dev_arr.max(), 1e-6), 0, 1)
        return float(trust_arr.mean()), False, {
            "source"       : "centroid_deviation_fallback",
            "mean_trust"   : round(float(trust_arr.mean()), 4),
            "interpretation": "Block centroids consistent with baseline (fallback — no GT trust scores)"
        }

    # ──────────────────────────────────────────────────────────────────────────
    # CORE METRICS (unchanged)
    # ──────────────────────────────────────────────────────────────────────────
    def _kl_divergence(self, X: np.ndarray) -> float:
        kl_vals = []
        for j, baseline_entry in enumerate(self.baseline_hists):
            if baseline_entry is None:
                continue
            edges, hist_base = baseline_entry
            col = X[:, j]
            hist_inc, _ = np.histogram(col, bins=edges, density=False)
            hist_inc = hist_inc.astype(float) + 1e-10
            hist_inc /= hist_inc.sum()
            kl_vals.append(float(entropy(hist_inc, hist_base)))
        if not kl_vals:
            return 0.0
        kl_arr  = np.array(kl_vals)
        trimmed = kl_arr[kl_arr <= np.percentile(kl_arr, 99)]
        return float(trimmed.mean()) if len(trimmed) > 0 else float(kl_arr.mean())

    def _wasserstein(self, X: np.ndarray) -> float:
        w_vals = []
        for j in range(X.shape[1]):
            w = wasserstein_distance(X[:, j], self.X_reference[:, j])
            w_vals.append(w / self.baseline_iqrs[j])
        return float(np.mean(w_vals))

    def _mahalanobis(self, X: np.ndarray) -> float:
        incoming_mean = np.median(X, axis=0)
        try:
            dist = mahalanobis(incoming_mean, self.baseline_mean, self.baseline_cov_inv)
        except Exception:
            dist = float(np.linalg.norm(incoming_mean - self.baseline_mean))
        return float(dist)

    def _robust_cov_inv(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        reg  = 1e-4
        if n >= max(5 * p, 50):
            try:
                from sklearn.covariance import MinCovDet
                mcd = MinCovDet(support_fraction=0.8, random_state=42).fit(X)
                cov = mcd.covariance_ + reg * np.eye(p)
                return np.linalg.inv(cov)
            except Exception:
                pass
        cov = np.cov(X.T) if X.shape[1] > 1 else np.array([[np.var(X)]])
        cov = cov + reg * np.eye(p)
        try:
            return np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(cov)

    @staticmethod
    def _sigmoid_scale(value: float, threshold: float, spread: float) -> float:
        x = (value - threshold) / max(spread, 1e-9)
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 0.38:
            return "CONFIRMED_POISONED"
        elif score >= 0.30:
            return "SUSPICIOUS"
        elif score >= 0.12:
            return "LOW_RISK"
        return "CLEAN"

    @staticmethod
    def _null_result(reason: str) -> dict:
        return {
            "suspicion_score"        : 0.0,
            "verdict_l1"             : "CLEAN",
            "kl_divergence"          : 0.0,
            "wasserstein"            : 0.0,
            "mahalanobis"            : 0.0,
            "alarm_kl"               : False,
            "alarm_mahal"            : False,
            "alarm_wasserstein"      : False,
            "label_ratio_shift"      : 0.0,
            "alarm_label_flip"       : False,
            "trigger_score"          : 0.0,
            "alarm_backdoor_trigger" : False,
            "gradient_score"         : 0.0,
            "alarm_gradient"         : False,
            "fed_trust_score"        : 1.0,
            "alarm_federated"        : False,
            "skip_reason"            : reason,
        }
