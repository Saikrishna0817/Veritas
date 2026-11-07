# Benchmark v2 — baseline findings (2026-11-06)

Run `baseline-r1` of `veritas-tabular-poisoning-v2`: 2,520 batches on 18 real OpenML datasets.
Full tables: `backend/research/results/2026-11-06-benchmark-v2-baseline-r1.md` (JSON beside it).
Numbers below are the 12 held-out **test** datasets at 5% poison; nothing has been tuned on them yet.

**Caveat:** Veritas took ~37 s per batch (baselines 0.02–1.5 s), so it ran on 1 of 5 seeds
(`detector_seed_limits`): 12 test batches per cell instead of 60. Its confidence intervals are wide.

## Row level: AUPRC (random ranking = 0.05)

| attack | Veritas | best baseline |
|---|---|---|
| label flip (random) | 0.05 (chance) | cleanlab 0.67 |
| label flip (targeted) | 0.05 (chance) | cleanlab 0.73 |
| backdoor, out-of-range trigger | 0.38 | two-sample test 0.82 |
| backdoor, in-range trigger | 0.07 | two-sample test 0.64 |
| clean-label feature collision | 0.05 (chance) | two-sample test 0.86 |
| covariate shift | 0.07 | ECOD 0.28 |

## Batch level: share of attacked batches caught at 5% false alarms

Veritas: label flips 0–8%, backdoors 67%, collision 42%, shift 17%. Cleanlab: label flips 72–82%, backdoors ~64%.
Veritas's backdoor and collision numbers are on 12 batches each (CIs span ±25 points), so they are
not yet distinguishable from cleanlab's.

## What this says

1. **Veritas finds no label errors at all.** Its row detector never looks at labels, so label flips,
   the most common real-world data problem, score exactly at chance. cleanlab does this in ~1 s.
2. **Backdoors:** Veritas finds only out-of-range triggers (0.38); a plain classifier two-sample test
   finds both kinds far better (0.82 / 0.64).
3. **Speed is a product problem:** ~37 s per 3,000-row batch (up to ~130 s on 26-class data) is
   100–1,500× slower than the baselines it loses to.
4. **No attack type is best handled by any single method.** Label errors need a label-aware detector,
   backdoors and collisions need a two-sample/classifier test, shift needs a feature-level test.
   That matches the Phase 2 plan: combine cleanlab + a classifier two-sample test + outlier scores.

## Known benchmark weaknesses (fix before quoting publicly)

- The two-sample test (`c2st`) is strong on rows but weak on batch-level TPR; its batch statistic
  (AUC vs. clean batches) needs a permutation p-value instead.
- Dev and test datasets were chosen by size/diversity, not stratified; 12 datasets give wide CIs.
- (Resolved in the Phase 2 update below) real-noise track added; the old pipeline stays on 1 seed because of its runtime.
- Injected poisoning is still synthetic; only label noise has a real-data check.

---

# Phase 2 update (2026-11-07): `veritas2` and the real-noise track

## What `veritas2` is

`backend/research/baselines.py::veritas2` (frozen after dev-only tuning; see the comment there):

1. **Label check** (always on): gradient-boosted classifier trained on clean reference rows;
   incoming rows are ranked by cleanlab's label-quality score.
2. **Shift check** (gated): classifier two-sample test, "can a model tell incoming rows from
   held-out clean rows?". It joins the row ranking only when its own batch-level tail test
   reaches -log10 p >= 4, so it cannot dilute label rankings when nothing has shifted.
3. Evidence is converted to p-values against held-out clean rows; batch score is the strongest tail test.

Tuning used the 6 dev datasets only. What dev showed: label+shift beat label alone on backdoors and
collisions; adding ECOD/IsolationForest hurt (their noise lifted the clean-batch threshold); a plain
max over components diluted label rankings, which the gate fixes. Gate 3/4/6 tied; 4 was kept.

## Held-out test datasets (12 datasets, injected attacks, 5% poison)

| | old Veritas | **veritas2** | cleanlab | two-sample test |
|---|---|---|---|---|
| mean row AUPRC (random = 0.05) | 0.11 | **0.69** | 0.54 | 0.49 |
| mean batch TPR at 5% FPR | 0.33 | **0.78** | 0.55 | 0.14 |
| sec / batch | 37 | 1.8 | 0.8 | 1.5 |

Per attack, `veritas2` is at or above the best single method everywhere except random label flips
(0.66 vs cleanlab 0.67, inside the CI). Weakest: covariate shift (AUPRC 0.40 [0.24, 0.57]).
Full tables: `research/results/2026-11-06-benchmark-v2-final-r1.md` (old Veritas numbers are from
`baseline-r1`, 1 seed).

## Real human label noise (CIFAR-10N, nothing injected)

3,000-image batches, 20 seeds. Row AUPRC where random = the mistake rate (5% column shown):

| noise source (wrong labels) | old Veritas | veritas2 | cleanlab |
|---|---|---|---|
| aggregate (9%) | 0.05 | 0.45 [0.43, 0.47] | 0.46 [0.44, 0.48] |
| random1 (17%) | 0.05 | 0.54 [0.49, 0.58] | 0.56 [0.53, 0.59] |
| worse (40%) | 0.05 | 0.57 [0.53, 0.61] | 0.58 [0.55, 0.61] |

## Honest reading

- **The old pipeline is at chance on real label noise** and takes ~130 s per batch. Replacing it is justified.
- **On real human mistakes `veritas2` equals cleanlab, it does not beat it** (it trails by 0.01-0.02, mostly inside the CIs). The gain over cleanlab is on injected backdoors/collisions/shift, not on label noise.
- **Batch-level detection of small label-noise rates is weak for everyone.** At a 3-5% mistake rate, whole-batch TPR at 5% FPR is 0.05-0.8 with very wide CIs; at 10% it is 0.95-1.0. `veritas2` is behind cleanlab at 5% on the aggregate labels (0.25 vs 0.75, CIs overlap widely). Likely cause: the shift test adds noise to the clean-batch threshold. Not yet fixed; do not tune it on this track.
- The ranking is useful ("review these rows first"); the batch verdict is not yet reliable for small contamination.
- The real-noise data is one dataset family (CIFAR-10 embeddings); old Veritas ran on 5 of 20 seeds.

## Still open

- ~~Move `veritas2` into the app pipeline~~ done (see Phase 2b below).
- Explicit reference-set input in the UI/API instead of a file-order split.
- Fix batch-level power at small contamination rates; pick the fix on dev, report on test + real noise.
- Sandboxed model-file scanner; more real-noise datasets (CIFAR-100N, text embeddings).

---

# Phase 2b (2026-11-07): `veritas2` is now the app's default engine

`DetectionPipeline(engine="v2")` (the default) runs `app/detection/veritas2.py`; `engine="legacy"` keeps the
original five layers for comparison. The benchmark imports the shipped module, and re-running four real-noise
batches reproduced the stored scores exactly. End to end on a synthetic CSV upload (3,000 rows, 3 classes):
clean file -> CLEAN, 10% flipped labels -> CONFIRMED_POISONED, 58 rows flagged of which 55 were the flipped ones, 1.3 s.

Verdict cut-offs are batch evidence (-log10 p) read off held-out clean batches: LOW_RISK >= 4.0, SUSPICIOUS >= 5.5
(~5% of clean batches reach it), CONFIRMED >= 9.5 (~1%). They map onto the app's 0.15 / 0.35 / 0.65 suspicion
thresholds. **"CONFIRMED_POISONED" now means "evidence at the 1% false-alarm level", not a causal proof**: the
causal layer no longer runs. Rows are flagged only when the batch itself is at least SUSPICIOUS.

Not changed: the model-file scan still feeds extracted parameters through the same pipeline, and has no benchmark.

# What was wrong with the old pipeline (diagnostic run)

24 batches (3 dev datasets, 1 seed, 10% poison, 2 clean batches per dataset), every layer's score recorded.
Exploratory, not a benchmark: it is small and nothing here was used for tuning. Script output is not committed.

1. **The blend can never reach a verdict.** Layer 3 flagged *every* out-of-range backdoor row (31/31 and 55/55) and
   almost nothing else, yet the batch verdict was LOW_RISK (overall 0.21-0.34), because one layer firing is weighted
   at 0.20 and the rest sit near zero. Across all 18 attacked batches the overall score averaged 0.11 against 0.03-0.05
   on clean batches, and no attacked batch reached SUSPICIOUS (0.35); the highest was 0.34. A weighted average needs several
   layers to agree; these attacks trip one at a time.
2. **No layer looks at labels properly.** Label flips (random and targeted) scored 0.03-0.15, equal to clean batches.
   Layer 3 flagged 0-4 of 23-55 flipped rows, which is chance. Only the label-ratio shift in Layer 1 reacts, and only
   to targeted flips in 2 of 3 datasets.
3. **Layer 4 (causal) never fired and costs the most time.** `proof_valid` was false in all 24 batches and the causal
   effect was ~0.01 everywhere (including 10% poisoning). It took 2-4 s per batch on sets under 1,000 rows (about 60%
   of the run on qsar-biodeg), and it retrains models, so it grows with data size. Since `CONFIRMED_POISONED`
   required it, the top verdict was unreachable.
4. **Layer 2 adds a constant.** Spectral score is a fixed 0.10-0.20 on clean and attacked batches alike (it is 20% of
   the weight), which only lifts every batch's floor.
5. **Layer 1's KL and Mahalanobis alarms never fired** on any of the 24 batches; only the blend-trigger heuristic did,
   and it fired on in-range backdoors and targeted flips inconsistently (0.32-0.37 vs 0.02 clean).
6. **Layer 3 is a decent out-of-range row detector** (31/31 and 55/55 on `backdoor_ood`), but it is label-blind and
   missed in-range backdoors (1 of 31 and 1 of 55) and feature collisions (0). That is the one piece worth keeping
   in spirit; the shift check in `veritas2` does the same job with a trained discriminator and also finds in-range cases.
7. **Speed:** timing was only measured per layer on small datasets (wdbc 0.8 s, qsar-biodeg 4.5 s); the 37 s average in
   the full benchmark comes from larger sets and was not broken down per layer.

Net: the old design (hand-weighted average of five mostly label-blind layers, with the top verdict gated on a proof that
never validates) was not recoverable by re-weighting, which matches the earlier benchmark-v1 finding of 0.00 F1 per layer.
The legacy code is retained only so the benchmark can keep comparing against it; decide later whether to delete layers 1-5.
