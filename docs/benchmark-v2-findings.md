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
- Poisoning is synthetic. The planned real-noise sets (CIFAR-10N etc.) are not in yet.
- Veritas should be rerun on all 5 seeds once it is fast enough, or on a Phase 2 replacement.
