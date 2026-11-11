# Image benchmark findings

Held-out test datasets: CIFAR-10 and GTSRB. Tuning (dev) datasets: Fashion-MNIST and CIFAR-100. Attacks are applied to real pixels;
5 seeds, 3,000-image batches (1,200 incoming). Row AUPRC; a random ranking scores the poison rate. Raw results:
`backend/research/results/*benchmark-mm-images-*`.

| Attack | veritas2 1% | veritas_mm 1% | veritas2 5% | veritas_mm 5% |
|---|---|---|---|---|
| Random label flip | 0.45 | 0.44 | 0.64 | 0.59 |
| Targeted label flip | 0.41 | 0.41 | 0.49 | 0.49 |
| Patch backdoor | 0.39 | 0.49 | 0.63 | 0.95 |
| Blended backdoor | 0.20 | 0.20 | 0.39 | 0.64 |
| Clean-label backdoor | 0.02 | 0.20 | 0.04 | 0.70 |
| Duplicate flood | 0.08 | 0.50 | 0.82 | 0.82 |
| Class-noise | 0.02 | 0.02 | 0.08 | 0.14 |

`veritas_mm` = v3 (pixel gate 1, chosen on dev only; the dev gain was confirmed on test).

- A ResNet embedding does not see corner patches or faint noise; low-level pixel statistics do.
- At 1% the pixel check alone ranks patch (0.71), clean-label (0.57) and duplicates (0.79) better than the fused score: max-fusion
  with scores capped by the calibration-set size dilutes sharp channels. Fusion is the next thing to fix.
- Batch verdicts at 1% are weak for every detector (0.0-0.2 of attacked batches at 5% false alarms).
- Clean test batches (n = 40, noisy): 12.5% reach the lowest verdict cut-off (nominal 5%), 5% the middle one. Dev (n = 120): 1.7%.
- Class-noise and blended backdoors at 1% are at or near chance.
- Cost: a small drop on random label flips at 5% (0.64 -> 0.59).

## v4: shared-pattern checks and extrapolated row scores (2026-11-12)

Poisoned images share something: copies, a patch, a faint blend. Three checks score each image against all other pooled
(clean calibration + incoming) images: `dup` (nearest neighbour on 8x8 pixels), `cell` (3rd-highest correlation of high-pass
residuals within any 8x8 cell), `resid` (same over the whole image). Row scores beyond every clean row now keep their margin
(`extrapolated_neg_log_p`), so a sharp check can outrank the top 1% of clean rows of another check. Component set and gates
were chosen on dev only (`research/mm/shared_study_dev.py`; a noise-level check and three other variants were dropped).

Held-out test (CIFAR-10, GTSRB), `veritas_mm`, row AUPRC:

| Attack | v3 1% | v4 1% | v3 5% | v4 5% |
|---|---|---|---|---|
| Random label flip | 0.44 | 0.33 | 0.59 | 0.56 |
| Targeted label flip | 0.41 | 0.41 | 0.49 | 0.45 |
| Patch backdoor | 0.49 | 0.80 | 0.95 | 0.97 |
| Blended backdoor | 0.20 | 0.68 | 0.64 | 0.94 |
| Clean-label backdoor | 0.20 | 0.65 | 0.70 | 0.97 |
| Duplicate flood | 0.50 | 0.92 | 0.82 | 0.91 |
| Class-noise | 0.02 | 0.02 | 0.14 | 0.24 |
| Mean | 0.32 | 0.55 | 0.62 | 0.72 |

- Cost: random label flips at 1% fall from 0.44 to 0.33 (more checks compete for one ranked list). A per-check report would not have this cost.
- Batch verdicts at 1%: 0.1-0.4 of attacked batches at 5% false alarms. Still weak.
- Clean test batches (n = 120): 8.3% reach the lowest verdict cut-off (nominal 5%), 2.5% the middle one (nominal 1%), none the top one. Slightly loose; needs recalibration.
- Class-noise at 1% is still at chance.
- 16 s per batch.
