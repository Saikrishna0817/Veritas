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

## v5: per-check report (2026-11-12)

`detector.report(d)` returns, for each of the six checks, its evidence (raw and penalised), whether it joins the fused
ranking, and its own row scores. `veritas_mm` is built on it and is unchanged (v5 fused numbers equal v4). The benchmark
scores each check's own list (`mm_label` ... `mm_resid`) and `mm_lead` (the list of the check with the highest evidence).
No gate or penalty was changed: on 360 clean dev batches the fused evidence crosses 1.3 in 4.7% and 2.0 in 0.8%
(`research/mm/report_study_dev.py`).

Held-out test (CIFAR-10, GTSRB), row AUPRC at 1% poison:

| Attack | fused | lead | label | shift | pixel | dup | cell | resid |
|---|---|---|---|---|---|---|---|---|
| Random label flip | 0.33 | 0.40 | 0.45 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 |
| Targeted label flip | 0.41 | 0.41 | 0.41 | 0.01 | 0.02 | 0.03 | 0.01 | 0.01 |
| Patch backdoor | 0.80 | 0.51 | 0.39 | 0.02 | 0.71 | 0.01 | 0.88 | 0.89 |
| Blended backdoor | 0.68 | 0.51 | 0.20 | 0.02 | 0.05 | 0.01 | 0.50 | 0.70 |
| Clean-label backdoor | 0.65 | 0.64 | 0.02 | 0.02 | 0.57 | 0.01 | 0.79 | 0.77 |
| Duplicate flood | 0.92 | 0.79 | 0.08 | 0.91 | 0.79 | 0.93 | 0.92 | 0.92 |
| Class-noise | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 |
| Mean | 0.55 | 0.47 | 0.22 | 0.14 | 0.31 | 0.15 | 0.45 | 0.47 |

At 5%: fused 0.72, lead 0.72, label 0.33, shift 0.20, pixel 0.54, dup 0.17, cell 0.55, resid 0.58 (means).

- The label check's own list restores random label flips at 1% to 0.45 (fused 0.33; v3 fused 0.44) and at 5% to 0.64 (fused 0.56).
- For every attack except class-noise, some single check's list beats or matches the fused list (patch: resid 0.89; blended: resid 0.70;
  clean-label: cell 0.79; duplicates: dup 0.93). That choice is made with hindsight; the report shows all lists rather than picking one.
- `mm_lead` (0.47) is worse than the fused list (0.55) at 1%: at low rates the highest-evidence check is often not the right one. Do not use it as the single list.
- Clean test batches (n = 240): 7.5% reach the lowest cut-off (nominal 5%; 95% interval 4.8-11.5%), 1.3% the middle one (nominal 1%), none the top one.
  Dev says 4.7%, test 7.5%: slightly loose at worst, not changed.
- Batch verdict at the fixed 1.3 cut-off: 26% of attacked batches at 1% poison, 79% at 5%.
- Class-noise at 1% is at chance on every check.
- 14 s per batch for all eight outputs (components are computed once per batch).

## v6: full reference set, several tail sizes, chroma check (2026-11-13)

Why the batch verdict was weak at 1%: each batch has 1200 incoming rows but only 540 clean calibration rows. With 12
poisoned rows ranked above every clean row, the exact tail test tops out at 12 * log10(1740/1200) = 1.94, below the
1.3 cut-off after the 1.0 penalty. Only the label check needs held-out rows, so:

- every other check now compares incoming rows with all 1800 reference rows (ceiling 4.8);
- the label check is cross-fitted (5 folds) so every reference row has a held-out score; incoming rows get the mean of the fold models;
- batch evidence is the best exact tail test over the top 0.03%, 0.1%, 0.3% and 1%, minus log10(4);
- a seventh check, `chroma`: colour noise (high-pass residual of R-G and G-B) in the quietest 4x4 block of each image.
  Natural images have almost none; per-channel noise patterns and blends lift it.

All chosen on dev (`research/mm/weak_study_dev.py`, `research/mm/ref_study_dev.py`; dev fused AUPRC at 1% 0.46 -> 0.78,
batch flagged at 1% 27% -> 70%, clean flagged 4.7% -> 1.4%).

Held-out test (CIFAR-10, GTSRB), row AUPRC:

| Attack | v5 fused 1% | v6 fused 1% | v5 fused 5% | v6 fused 5% |
|---|---|---|---|---|
| Random label flip | 0.33 | 0.48 | 0.56 | 0.73 |
| Targeted label flip | 0.41 | 0.45 | 0.45 | 0.57 |
| Patch backdoor | 0.80 | 0.95 | 0.97 | 0.98 |
| Blended backdoor | 0.68 | 1.00 | 0.94 | 1.00 |
| Clean-label backdoor | 0.65 | 0.99 | 0.97 | 0.98 |
| Duplicate flood | 0.92 | 0.92 | 0.91 | 0.91 |
| Class-noise | 0.02 | 0.99 | 0.24 | 0.99 |
| Mean | 0.55 | 0.83 | 0.72 | 0.88 |

The label check's own list scores 0.58 (random) and 0.48 (targeted) at 1%.

Batch verdict (evidence >= 1.3), share of attacked batches flagged:

| Rate | v5 | v6 |
|---|---|---|
| 1% | 26% | 73% |
| 3% | 71% | 83% |
| 5% | 79% | 91% |
| 10% | 91% | 90% |

At 1%: patch, blended, duplicates, class-noise 100%; clean-label 90%; random and targeted label flips 10%.

- Clean test batches (n = 240): 2.1% reach 1.3 (nominal 5%), 0.8% reach 2.0 (nominal 1%), 0.4% (one batch, the shift check) reach 4.0 (nominal 0.01%).
- Label flips at batch level are still weak at 1% (10%) and slightly worse than v5 at 3% (random 0.50 vs 0.60). The cross-fitted label
  check never fired on a clean batch (dev or test), so it is over-cautious; scoring incoming rows with one fold model instead of the mean
  of five is the next thing to try.
- The chroma check is blind to noise that is identical across colour channels, and to greyscale data with greyscale noise.
- 36 s per batch (was 14 s); the cross-fitted label check is most of it (5 GBM fits, slow with 100 classes).

## v7: label-check batch evidence (2026-11-13)

In v6 the cross-fitted label check never fired on a clean batch: incoming rows were scored by the mean of five fold
models, which is less extreme than the single out-of-fold score each reference row gets. Two changes to its batch
evidence only (the row list still uses the mean, which ranks flipped rows better: dev AUPRC 0.45 vs 0.34):

- each incoming row is scored by one fold model (row i by fold i mod 5), on the same footing as the reference rows;
- evidence = best of the pooled tail test and the best per-claimed-class tail test (minus log10 of the class count),
  minus log10(2). A targeted flip puts all its rows in one claimed class.

Chosen on dev (`research/mm/label_study_dev.py`): clean flagged 1.4% -> 2.5% at 1.3, 0% -> 0.3% at 2.0; label flips
flagged at 1% random 10% -> 30%, targeted 30% -> 50%; all attacks at 1% 70% -> 77%.

Held-out test (CIFAR-10, GTSRB), share of attacked batches flagged (evidence >= 1.3). Row AUPRC is unchanged.

| Attack | v6 1% | v7 1% | v6 3% | v7 3% | v6 5% | v7 5% |
|---|---|---|---|---|---|---|
| Random label flip | 10% | 40% | 50% | 70% | 100% | 100% |
| Targeted label flip | 10% | 40% | 40% | 50% | 50% | 60% |
| All attacks | 73% | 81% | 83% | 87% | 91% | 93% |

At cut-off 2.0: all attacks at 1% 69% -> 74%, at 3% 77% -> 86%, at 5% 87% -> 93%; random flip at 1% 0% -> 40%, targeted at 1% still 0%.

- Clean test batches (n = 240): 4.2% reach 1.3 (nominal 5%; v6 2.1%), 1.3% reach 2.0 (nominal 1%; v6 0.8%), 0.4% reach 4.0.
  The label check caused 5 of the 10 clean alarms at 1.3 and one of the three at 2.0 (raw 2.24). Above nominal at 2.0
  by one batch; not retuned on test.
- Targeted flips are still the weakest: 40-60% of batches flagged at any rate, row AUPRC 0.45-0.57.
- 37 s per batch.
