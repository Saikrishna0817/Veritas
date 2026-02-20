"""
Layer 4 — Causal Proof Engine  (EXPANDED v4 — with Poisoning Fingerprint Detector)
=====================================================================================

NEW IN v4:
  - PoisoningFingerprintDetector injected and called at the end of run() when
    proof_valid=True. It analyses the flagged vs. clean split that L4 has
    already computed to classify the specific poisoning strategy used.

  POISONING FINGERPRINT DETECTOR
  ────────────────────────────────
  Answers: given that poisoning is proven, *which* strategy was it?

  The flagged sample set (anomalies from L3 that L4's causal proof confirmed
  are actually harmful) leaves characteristic geometric, statistical, and
  structural fingerprints:

    LABEL FLIP
      Flagged samples have correct-looking features but wrong-for-their-region
      labels. Signature: feature distributions of flagged and clean samples are
      similar (low Mahalanobis distance between group centroids) but label ratio
      inside flagged set differs sharply from the clean set.

    BACKDOOR TRIGGER
      Flagged samples cluster very tightly in feature space (they all share the
      trigger pattern). Signature: flagged set inertia ratio (1-cluster / 2-cluster
      KMeans) is high (one cluster explains well), AND at least one feature has a
      value spike (a value appearing ≥3× more often among flagged than clean).

    GRADIENT ATTACK
      Flagged samples are spread across feature space but shifted in specific
      dimensions. Signature: several features show large z-score deviation of
      their means (flagged vs. clean) but there is no single-feature spike and
      no label ratio anomaly.

    FEDERATED BYZANTINE
      Flagged samples arrive in contiguous blocks (from rogue clients, not
      interspersed uniformly). Signature: block homogeneity score — within-block
      variance is low relative to total flagged variance (the contaminated
      samples form coherent sub-batches, not individually crafted samples).

  Output is appended as ``poisoning_fingerprint`` in run()'s return dict.
  Only runs when proof_valid=True and at least MIN_FLAGGED_SAMPLES are present.

EXISTING MODELS (v3, unchanged):
  Three proxy classifiers (LR, RF, GBM), multi-proxy consensus, bootstrap CI,
  t-test, placebo, degradation projections. All v2 rectifications preserved.
"""

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.cluster import KMeans


# ── Constants ──────────────────────────────────────────────────────────────────
N_BOOTSTRAP          = 100
CI_LEVEL             = 0.95
P_VALUE_THRESHOLD    = 0.05
MIN_CAUSAL_EFFECT    = 0.01
PLACEBO_RATIO_MAX    = 0.20
MIN_FLAGGED_SAMPLES  = 5
MIN_TOTAL_SAMPLES    = 40
CV_FOLDS             = 3
GEOMETRIC_DECAY_RATE = 0.30

# ── Fingerprint thresholds ─────────────────────────────────────────────────────
FP_LABEL_RATIO_MIN_DELTA  = 0.15   # flagged label ratio differs ≥ 15pp from clean
FP_FEATURE_Z_ALARM        = 2.0    # per-feature z-score for gradient attack
FP_N_FEATURE_Z_ALARM      = 2      # ≥ 2 features alarming → gradient attack
FP_SPIKE_LIFT_MIN         = 3.0    # value appears ≥3× more often in flagged → trigger
FP_CLUSTER_INERTIA_RATIO  = 2.5    # 1-cluster / 2-cluster inertia ratio
FP_BLOCK_HOMOG_MIN        = 0.60   # within-block var ratio for federated detection
FP_CONFIRMED_CONF         = 0.70
FP_SUSPECTED_CONF         = 0.40


class PoisoningFingerprintDetector:
    """
    Injected into Layer 4.  Classifies the poisoning strategy from the
    flagged vs. clean split produced by CausalProofEngine.

    Only called when proof_valid=True (no point fingerprinting if
    the causal effect was not proven).
    """

    def analyze(
        self,
        X_flagged : np.ndarray,
        y_flagged : np.ndarray,
        X_clean   : np.ndarray,
        y_clean   : np.ndarray,
    ) -> dict:
        if len(X_flagged) < MIN_FLAGGED_SAMPLES:
            return {"fingerprint": "insufficient_flagged_samples",
                    "confidence": 0.0, "votes": {}, "signals": {}}

        signals = {}
        votes   = {}

        # ── Test A: Label Flip ────────────────────────────────────────────────
        lbl_ratio_flagged = float(np.mean(y_flagged)) if len(y_flagged) else 0.5
        lbl_ratio_clean   = float(np.mean(y_clean))   if len(y_clean)   else 0.5
        lbl_delta         = abs(lbl_ratio_flagged - lbl_ratio_clean)
        signals["label_ratio_flagged"] = round(lbl_ratio_flagged, 4)
        signals["label_ratio_clean"]   = round(lbl_ratio_clean, 4)
        signals["label_ratio_delta"]   = round(lbl_delta, 4)

        # Feature centroid distance between flagged and clean (normalised by clean std)
        feat_z_means = np.abs(
            np.mean(X_flagged, axis=0) - np.mean(X_clean, axis=0)
        ) / (np.std(X_clean, axis=0) + 1e-9)
        mean_feat_z = float(feat_z_means.mean())
        signals["mean_feature_z"] = round(mean_feat_z, 4)

        # Label flip = high label delta + low feature shift
        lf_score = float(np.clip(
            (lbl_delta / FP_LABEL_RATIO_MIN_DELTA) * 0.70 +
            max(0.0, 1.0 - mean_feat_z / 2.0) * 0.30,
            0.0, 1.0
        ))
        votes["LABEL_FLIP"] = lf_score

        # ── Test B: Backdoor Trigger ──────────────────────────────────────────
        # Sub-test 1: cluster tightness
        cluster_tight = False
        inertia_ratio  = 0.0
        if len(X_flagged) >= 6:
            try:
                km1 = KMeans(n_clusters=1, random_state=0, n_init=5).fit(X_flagged)
                km2 = KMeans(n_clusters=2, random_state=0, n_init=5).fit(X_flagged)
                inertia_ratio = km1.inertia_ / (km2.inertia_ + 1e-9)
                cluster_tight = inertia_ratio >= FP_CLUSTER_INERTIA_RATIO
            except Exception:
                pass
        signals["cluster_inertia_ratio"] = round(float(inertia_ratio), 3)
        signals["cluster_tight"]         = bool(cluster_tight)

        # Sub-test 2: single-feature value spike
        spike_features = []
        n_f, n_c = len(X_flagged), len(X_clean)
        for j in range(X_flagged.shape[1]):
            # Round to 2 dp for discrete value counting
            vals_f = np.round(X_flagged[:, j], 2)
            vals_c = np.round(X_clean[:, j],   2)
            unique_f = set(vals_f.tolist())
            for v in unique_f:
                freq_f = float((vals_f == v).sum()) / n_f
                freq_c = float((vals_c == v).sum()) / max(n_c, 1)
                if freq_f > 0.05 and freq_c < 1e-6:
                    # Value appears in flagged but not at all in clean
                    spike_features.append({"feature": j, "value": v,
                                           "flagged_freq": round(freq_f, 4)})
                elif freq_f > 0.05 and freq_c > 0 and freq_f / freq_c >= FP_SPIKE_LIFT_MIN:
                    spike_features.append({"feature": j, "value": v,
                                           "flagged_freq": round(freq_f, 4),
                                           "clean_freq": round(freq_c, 4),
                                           "lift": round(freq_f / freq_c, 2)})
        has_spike = len(spike_features) > 0
        signals["spike_features"] = spike_features[:5]   # top 5 for readability
        signals["has_value_spike"] = has_spike

        bk_score = float(np.clip(
            (0.55 if cluster_tight else 0.0) +
            (0.45 if has_spike else 0.0),
            0.0, 1.0
        ))
        votes["BACKDOOR_TRIGGER"] = bk_score

        # ── Test C: Gradient Attack ───────────────────────────────────────────
        # Many features shifted, no trigger spike, no label ratio anomaly
        n_alarming_features = int((feat_z_means > FP_FEATURE_Z_ALARM).sum())
        signals["n_alarming_features"] = n_alarming_features
        ga_score = float(np.clip(
            (min(n_alarming_features / FP_N_FEATURE_Z_ALARM, 1.0)) * 0.60 +
            (0.20 if not has_spike else 0.0) +
            (0.20 if lbl_delta < FP_LABEL_RATIO_MIN_DELTA else 0.0),
            0.0, 1.0
        ))
        votes["GRADIENT_ATTACK"] = ga_score

        # ── Test D: Federated Byzantine ───────────────────────────────────────
        # Flagged samples are homogeneous within contiguous blocks
        block_homog = 0.0
        if len(X_flagged) >= 20:
            block_size = max(len(X_flagged) // 5, 4)
            block_vars = []
            for i in range(0, len(X_flagged) - block_size, block_size):
                blk = X_flagged[i:i + block_size]
                block_vars.append(float(np.mean(np.var(blk, axis=0))))
            total_var = float(np.mean(np.var(X_flagged, axis=0)))
            if total_var > 1e-9:
                block_homog = 1.0 - float(np.mean(block_vars)) / total_var
        signals["block_homogeneity_score"] = round(block_homog, 4)
        fb_score = float(np.clip(block_homog / FP_BLOCK_HOMOG_MIN, 0.0, 1.0))
        votes["FEDERATED_BYZANTINE"] = fb_score

        # ── Verdict ───────────────────────────────────────────────────────────
        best_type = max(votes, key=votes.get)
        best_conf = votes[best_type]

        if best_conf >= FP_CONFIRMED_CONF:
            fingerprint = best_type
            status      = "CONFIRMED"
        elif best_conf >= FP_SUSPECTED_CONF:
            fingerprint = best_type
            status      = "SUSPECTED"
        else:
            fingerprint = "UNDETERMINED"
            status      = "UNDETERMINED"

        return {
            "fingerprint"      : fingerprint,
            "status"           : status,
            "confidence"       : round(float(best_conf), 3),
            "votes"            : {k: round(float(v), 3) for k, v in votes.items()},
            "signals"          : signals,
            "interpretation"   : (
                f"{status}: poisoning strategy appears to be {fingerprint} "
                f"(confidence={best_conf:.2f}). "
                f"Runner-up: {sorted(votes, key=votes.get)[-2]}"
                f"={votes[sorted(votes, key=votes.get)[-2]]:.2f}."
            ) if fingerprint != "UNDETERMINED" else
                "Poisoning strategy undetermined — mixed or novel attack signature.",
        }


# ── Constants ──────────────────────────────────────────────────────────────────
N_BOOTSTRAP          = 100
CI_LEVEL             = 0.95
P_VALUE_THRESHOLD    = 0.05
MIN_CAUSAL_EFFECT    = 0.01
PLACEBO_RATIO_MAX    = 0.20
MIN_FLAGGED_SAMPLES  = 5
MIN_TOTAL_SAMPLES    = 40
CV_FOLDS             = 3
GEOMETRIC_DECAY_RATE = 0.30


class CausalProofEngine:
    """
    Multi-proxy causal proof engine.

    Three classifiers (LR, RF, GBM) each estimate the causal effect of
    removing flagged samples. The MEDIAN effect is used for the validity
    check, making the result robust to any single proxy being mislead by
    label-flipped or gradient-perturbed training data.
    """

    def __init__(self, random_state: int = 42):
        self.random_state     = random_state
        self._fp_detector     = PoisoningFingerprintDetector()

    def run(self, X: np.ndarray, y: np.ndarray, flagged_indices: list) -> dict:
        X       = np.array(X, dtype=float)
        y       = np.array(y)
        flagged = list(flagged_indices)

        if len(flagged) < MIN_FLAGGED_SAMPLES or len(X) < MIN_TOTAL_SAMPLES:
            return self._null_result("insufficient_flagged_samples", len(flagged))

        # ── Step 1: Per-proxy accuracy WITH suspects ──
        proxy_with   = self._all_proxy_accuracies(X, y)

        # ── Step 2: Per-proxy accuracy WITHOUT suspects ──
        mask         = np.ones(len(X), dtype=bool); mask[flagged] = False
        X_wo, y_wo   = X[mask], y[mask]

        if len(X_wo) < MIN_TOTAL_SAMPLES // 2:
            return self._null_result("too_few_samples_after_removal", len(flagged))

        proxy_without = self._all_proxy_accuracies(X_wo, y_wo)

        # ── Step 3: Per-proxy causal effects + median consensus ──
        proxy_effects = {k: proxy_without[k] - proxy_with[k] for k in proxy_with}
        causal_effect = float(np.median(list(proxy_effects.values())))

        # ── Step 4: Bootstrap CI (using LR proxy for speed) ──
        bootstrap_effects = self._bootstrap(X, y, flagged)
        ci_lo = float(np.percentile(bootstrap_effects, (1-CI_LEVEL)/2*100))
        ci_hi = float(np.percentile(bootstrap_effects, (1+CI_LEVEL)/2*100))
        ci_excludes_zero = bool(ci_lo > 0)   # one-sided (RECT 3 from v2)

        # ── Step 5: t-test ──
        _, p_value  = ttest_1samp(bootstrap_effects, 0) if len(bootstrap_effects) > 1 else (None, 1.0)
        significant = bool(p_value < P_VALUE_THRESHOLD)

        # ── Step 6: Placebo ──
        placebo_effect = self._placebo_test(X, y, len(flagged))
        placebo_ratio  = abs(placebo_effect) / (abs(causal_effect) + 1e-9) if abs(causal_effect) > 1e-6 else 1.0
        placebo_valid  = bool(placebo_ratio < PLACEBO_RATIO_MAX)

        # ── Step 7: n_proxies_agree — require at least 2/3 proxies positive ──
        n_positive     = sum(1 for e in proxy_effects.values() if e > MIN_CAUSAL_EFFECT)
        proxies_agree  = n_positive >= 2

        # ── Step 8: Validity ──
        proof_valid = (
            causal_effect > MIN_CAUSAL_EFFECT
            and ci_excludes_zero
            and significant
            and placebo_valid
            and proxies_agree   # NEW: multi-proxy consensus required
        )

        # ── Step 9: Degradation + projections ──
        if proof_valid:
            degradation_score = float(np.clip(causal_effect, 0.0, 1.0))
            projections       = self._project_degradation(float(np.mean(list(proxy_with.values()))),
                                                          degradation_score)
        else:
            degradation_score = 0.0
            acc_with_mean     = float(np.mean(list(proxy_with.values())))
            projections       = {"day_30": acc_with_mean, "day_60": acc_with_mean, "day_90": acc_with_mean}

        suspicion = float(np.clip(degradation_score / 0.20, 0.0, 1.0)) if proof_valid else 0.0

        # ── Poisoning Fingerprint (injected sub-detector) ─────────────────────
        # Only runs when causal proof is valid — no point fingerprinting noise.
        if proof_valid and len(flagged) >= MIN_FLAGGED_SAMPLES:
            clean_mask   = np.ones(len(X), dtype=bool)
            clean_mask[flagged] = False
            X_flagged    = X[flagged]
            y_flagged    = y[flagged]
            X_clean_only = X[clean_mask]
            y_clean_only = y[clean_mask]
            fingerprint_result = self._fp_detector.analyze(
                X_flagged, y_flagged, X_clean_only, y_clean_only
            )
        else:
            fingerprint_result = {
                "fingerprint"   : "not_run",
                "status"        : "SKIPPED",
                "confidence"    : 0.0,
                "votes"         : {},
                "signals"       : {},
                "interpretation": "Fingerprinting skipped — causal proof not valid or too few flagged samples.",
            }

        return {
            "suspicion_score"      : suspicion,
            "causal_effect"        : causal_effect,
            "degradation_score"    : degradation_score,
            "proxy_effects"        : {k: round(v, 5) for k, v in proxy_effects.items()},
            "proxy_accuracies_with": {k: round(v, 5) for k, v in proxy_with.items()},
            "proxy_accuracies_wo"  : {k: round(v, 5) for k, v in proxy_without.items()},
            "n_proxies_agree"      : n_positive,
            "acc_with_poison"      : float(np.mean(list(proxy_with.values()))),
            "acc_without_poison"   : float(np.mean(list(proxy_without.values()))),
            "proof_valid"          : proof_valid,
            "bootstrap_ci"         : [ci_lo, ci_hi],
            "bootstrap_mean"       : float(np.mean(bootstrap_effects)),
            "p_value"              : float(p_value),
            "significant"          : significant,
            "placebo_effect"       : float(placebo_effect),
            "placebo_ratio"        : float(placebo_ratio),
            "placebo_valid"        : placebo_valid,
            "ci_excludes_zero"     : ci_excludes_zero,
            "n_flagged"            : len(flagged),
            "n_total"              : len(X),
            "projections"          : projections,
            "interpretation"       : self._interpret(proof_valid, causal_effect, degradation_score),
            # ── Injected sub-detector output ───────────────────────────────────
            "poisoning_fingerprint": fingerprint_result,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _all_proxy_accuracies(self, X, y) -> dict:
        """Run all three proxy classifiers and return their CV accuracies."""
        return {
            "logistic_regression" : self._cv_accuracy(X, y, "lr"),
            "random_forest"       : self._cv_accuracy(X, y, "rf"),
            "gradient_boosting"   : self._cv_accuracy(X, y, "gbm"),
        }

    def _cv_accuracy(self, X: np.ndarray, y: np.ndarray, model_type: str = "lr") -> float:
        if len(np.unique(y)) < 2:
            return 1.0
        try:
            if model_type == "lr":
                clf = LogisticRegression(max_iter=500, random_state=self.random_state, solver="lbfgs")
            elif model_type == "rf":
                clf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                             random_state=self.random_state, n_jobs=-1)
            else:  # gbm
                try:
                    from xgboost import XGBClassifier  # type: ignore
                    clf = XGBClassifier(n_estimators=50, max_depth=3, verbosity=0,
                                        use_label_encoder=False, eval_metric="logloss",
                                        random_state=self.random_state)
                except ImportError:
                    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                     random_state=self.random_state)

            min_cls = int(np.bincount(y.astype(int)).min())
            n_folds = min(CV_FOLDS, min_cls)
            if n_folds < 2:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                            random_state=self.random_state, stratify=y)
                clf.fit(X_tr, y_tr)
                return float((clf.predict(X_te) == y_te).mean())
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
        except Exception:
            return self._cv_accuracy(X, y, "lr") if model_type != "lr" else 0.5

    def _bootstrap(self, X, y, flagged):
        rng         = np.random.RandomState(self.random_state)
        effects     = []
        flagged_set = set(flagged)
        clean_idx   = [i for i in range(len(X)) if i not in flagged_set]
        for _ in range(N_BOOTSTRAP):
            bc  = rng.choice(clean_idx, size=len(clean_idx), replace=True)
            bf  = rng.choice(flagged,   size=len(flagged),   replace=True)
            ball = np.concatenate([bc, bf])
            Xb, yb = X[ball], y[ball]
            if len(np.unique(yb)) < 2: continue
            acc_with  = self._cv_accuracy(Xb, yb, "lr")
            Xc, yc    = X[bc], y[bc]
            if len(Xc) < 10 or len(np.unique(yc)) < 2: continue
            acc_wo    = self._cv_accuracy(Xc, yc, "lr")
            effects.append(acc_wo - acc_with)
        return np.array(effects) if effects else np.array([0.0])

    def _placebo_test(self, X, y, n_flagged):
        rng     = np.random.RandomState(self.random_state + 99)
        effects = []
        acc_w   = self._cv_accuracy(X, y, "lr")
        for _ in range(5):
            rm   = rng.choice(len(X), size=n_flagged, replace=False)
            mask = np.ones(len(X), dtype=bool); mask[rm] = False
            Xp, yp = X[mask], y[mask]
            if len(Xp) < 10 or len(np.unique(yp)) < 2: continue
            effects.append(self._cv_accuracy(Xp, yp, "lr") - acc_w)
        return float(np.mean(effects)) if effects else 0.0

    @staticmethod
    def _project_degradation(current_accuracy, degradation_score):
        floor = 0.50
        return {
            f"day_{d}": round(float(floor + max(current_accuracy-floor, 0.0) *
                              ((1.0-GEOMETRIC_DECAY_RATE)**(d//30))), 4)
            for d in [30, 60, 90]
        }

    @staticmethod
    def _interpret(proof_valid, causal_effect, degradation):
        if not proof_valid:
            return ("No causal harm proven. Suspicious samples may be natural outliers "
                    "or the effect is too small to distinguish from noise.")
        pct = round(degradation * 100, 1)
        return (f"CONFIRMED: Poison causes a proven {pct}% accuracy degradation "
                f"(causal effect = {round(causal_effect*100,1)} pp). "
                "Validated via multi-proxy consensus, bootstrap CI, placebo, and t-test.")

    @staticmethod
    def _null_result(reason, n_flagged=0):
        return {"suspicion_score":0.0,"causal_effect":0.0,"degradation_score":0.0,
                "proxy_effects":{},"proxy_accuracies_with":{},"proxy_accuracies_wo":{},
                "n_proxies_agree":0,"acc_with_poison":0.0,"acc_without_poison":0.0,
                "proof_valid":False,"bootstrap_ci":[0.0,0.0],"bootstrap_mean":0.0,
                "p_value":1.0,"significant":False,"placebo_effect":0.0,"placebo_ratio":1.0,
                "placebo_valid":False,"ci_excludes_zero":False,"n_flagged":n_flagged,
                "n_total":0,"projections":{"day_30":0.0,"day_60":0.0,"day_90":0.0},
                "interpretation":f"Skipped: {reason}","skip_reason":reason,
                "poisoning_fingerprint":{"fingerprint":"not_run","status":"SKIPPED",
                                         "confidence":0.0,"votes":{},"signals":{},
                                         "interpretation":"Skipped: insufficient data."}}
