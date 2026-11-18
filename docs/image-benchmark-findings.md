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

## v8: logistic-regression label check (2026-11-13)

The cross-fitted label check (5 gradient-boosting fits) was ~2/3 of the time per batch and grew with the class count
(dev: 11.6 s Fashion-MNIST, 67.1 s CIFAR-100). Its model is now a standardised logistic regression on the same 32 PCA
dims. Chosen on dev (`research/mm/speed_study_dev.py`, also tried GBM with 40 rounds and 25-nearest-neighbours):
0.1 s / 0.5 s; label-flip row AUPRC at 1% random 0.45 -> 0.57, targeted 0.50 -> 0.74; clean fused flagged at 1.3 2.2% -> 3.3%.

Held-out test (CIFAR-10, GTSRB):

| | v7 | v8 |
|---|---|---|
| Seconds per batch (whole detector) | 37.1 | 10.2 |
| Fused row AUPRC, mean at 1% / 5% | 0.83 / 0.88 | 0.86 / 0.91 |
| Random label flip AUPRC at 1% / 5% | 0.48 / 0.73 | 0.55 / 0.82 |
| Targeted label flip AUPRC at 1% / 5% | 0.45 / 0.57 | 0.62 / 0.71 |
| Attacked batches flagged at 1% / 3% / 5% / 10% | 81 / 87 / 93 / 90% | 81 / 94 / 96 / 93% |
| Random flip flagged at 1% | 40% | 20% |
| Targeted flip flagged at 1% / 5% | 40 / 60% | 60 / 80% |
| Clean flagged at 1.3 / 2.0 / 4.0 (n = 240) | 4.2 / 1.3 / 0.4% | 2.9 / 0.8 / 0.4% |

- Random flips at 1% flagged less often (2 of 10 batches vs 4); 10 batches per cell, so this is within noise, and their row AUPRC rose.
- Clean alarms now within nominal at every cut-off; the label check causes 2 of the 7 at 1.3.
- Batch time is now spread over checks: pixel and shift are the largest.

## v9: within-class signature check (2026-11-16)

Held-out numbers are in `benchmark-v2-findings.md` ("Clean-label triggers"). Mean row AUPRC 0.87 / 0.92 at 1% / 5%,
batches flagged 83 / 94 / 97 / 93% at 1 / 3 / 5 / 10%, clean 3.3% at 1.3 and 0.8% at 2.0 (n = 240), 13.6 s per batch.

## Attacks written by someone else (Phase 1, 2026-11-16)

`manifest-mm-images-ext-v1`, `results/2026-11-16-benchmark-mm-images-ext-v1.*`. Five attacks from BackdoorBench's
published code (commit f02e353, CC BY-NC 4.0, loaded at run time and not copied): BadNets, Blended (alpha 0.2), SIG
(clean-label), Low Frequency (their CIFAR-10 PreAct-ResNet18 trigger) and FTrojan (YUV frequency trigger). CIFAR-10 and
GTSRB test splits, 5 seeds, 4 rates, detector frozen at v9 and not tuned on these sets.

Share of batches flagged at 1.3 (n = 10 per cell, both datasets) and row AUPRC at 5%:

| Attack | 1% | 3% | 5% | 10% | row AUPRC at 5% | check that finds it |
|---|---|---|---|---|---|---|
| BadNets | 60% | 100% | 100% | 100% | 0.97 | label, pixel, signature |
| Blended | 40% | 100% | 100% | 100% | 0.94 | label, residual, signature |
| SIG (clean-label) | 80% | 90% | 80% | 90% | 0.85 | signature, pixel |
| Low Frequency | 50% | 100% | 100% | 100% | 0.77 | label only |
| FTrojan | 50% | 100% | 100% | 100% | 0.84 | label, pixel |

- Criterion set before the run: any attack below 50% of batches at 5% goes into the "not detected" list. None is.
  SIG on GTSRB is the weakest cell (60% at 5%, 3 of 5 batches; class size again).
- Clean batches: 8 of 240 at 1.3, 2 at 2.0 (same as v9).
- Low Frequency and FTrojan are found only because the attacker also relabels the poisoned images. The pattern
  checks see neither trigger (signature row AUPRC 0.08 and 0.07). A clean-label version of either would likely go
  unnoticed; that is tested in Phase 3.

## Real label errors (Phase 2, 2026-11-16)

`manifest-labelerrors-v1`, `results/2026-11-16-labelerrors-v1.*`. Test-set label errors confirmed by Mechanical Turk
reviewers (Northcutt et al. 2021, cleanlab/label-errors commit 6d5d6b3; an error = fewer than 3 of 5 reviewers kept
the given label: 54 on CIFAR-10, 585 on CIFAR-100, 15 on MNIST, as in the paper). With reference: 6,000 training images
as the reference, the 10,000 test images checked in batches. Without: the test set checked on its own. Nothing tuned.

Ranking of the confirmed errors (95% intervals in the result file):

| | errors | AUPRC (base rate) | in the top 5% | precision of the app's list |
|---|---|---|---|---|
| CIFAR-10, with reference | 54 | 0.03 (0.005) | 33% | 4% |
| CIFAR-10, no reference | | 0.03 | 33% | 4% |
| CIFAR-100, with reference | 585 | 0.12 (0.059) | 13% | 15% |
| CIFAR-100, no reference | | 0.12 | 14% | 16% |
| MNIST, with reference | 15 | 0.02 (0.002) | 53% | 2% |
| MNIST, no reference | | 0.02 | 53% | 2% |

- Real errors are ranked well above chance (AUROC 0.72-0.92) but the top of the list is mostly not confirmed errors:
  on CIFAR-10 about 1 in 25 listed images. Planted flips score far higher (row AUPRC 0.82) because they are random,
  while real errors are mostly ambiguous images the label model also finds hard.
- Precision is a lower bound: only cleanlab's 275 / 2,235 / 100 candidates were reviewed, so a listed image outside
  them may still be wrong. Against all candidates, CIFAR-100's list is 50% candidates.
- The label check alone ranks slightly better than the merged score (CIFAR-10 0.04 vs 0.03, MNIST 0.04 vs 0.02):
  the other checks add images that are unusual but correctly labelled.
- MNIST was checked at 28x28 here; the app resizes to 32x32.

## Attacks tuned to evade the checks (Phase 3, 2026-11-16)

`manifest-adaptive-images-v1`, `results/2026-11-16-benchmark-adaptive-images-v1.*`. CIFAR-10. Six attack families,
each over a strength grid at 1% and 5% of a 3,000-image incoming batch. The attacker picks the strongest setting
Veritas does not flag on their own seeds (0-2); every reported number comes from fresh seeds (10-12). A victim CNN
(~0.3M parameters, 15 epochs) is trained on the 6,000 reference + 3,000 incoming images to measure attack success
(ASR: share of non-target test images with the trigger classified as the target). Clean victim accuracy 0.68.

| Family | Flagged (all strengths, 1% / 5%) | Highest ASR | Evasive setting found |
|---|---|---|---|
| Smooth low-frequency trigger, relabelled | 100% / 100% | 0.21 | none |
| Warp (WaNet-style), relabelled | 100% / 100% | 0.16 | none |
| Faint 3x3 checkerboard, relabelled | 100% / 100% | 0.97 (opacity 0.4, 5%) | none |
| Low-frequency trigger, labels unchanged | 33% / 33% | 0.09 | yes, every strength |
| Boundary-aware label flips | 33% / 100% | 0.15 | yes at 1% |
| Random flips (reference) | 100% / 100% | 0.15 | none |

- Pre-set criterion: the claim is the flagged share among runs with ASR >= 0.9. Only the faint patch reached it
  (4 runs), and all 4 were flagged. **That is too little to support a claim either way.**
- The evasive attacks found (clean-label low-frequency, boundary flips at 1%) do evade Veritas, but they also do not
  work on this victim (ASR at most 0.09 and 0.08, against 0.04-0.08 with no poison).
- Why ASR is low: 5% of the incoming batch is 1.7% of the victim's training set, and the victim is small and briefly
  trained. A fair test needs a victim that learns backdoors from fewer images (longer training, a larger model, or
  poison spread over the whole training set): a new benchmark_id. Until then, Veritas makes no claim about informed
  attackers.
- The relabelled attacks are all caught through the label check, so the weakness to watch is a trigger that keeps
  labels correct; the low-frequency one shows the pattern checks miss it.

## New datasets, never used for tuning (Phase 4, 2026-11-16)

`manifest-mm-images-breadth-v1`, `results/2026-11-16-benchmark-mm-images-breadth-v1.*`. SVHN (test split) and
Tiny-ImageNet (200 classes, decoded to 32x32 with the app's decoder, ~15 images per class per batch). Detector frozen
at v9. Share of batches flagged at 1.3 at 1 / 3 / 5 / 10% (n = 5 each):

| Attack | SVHN | Tiny-ImageNet | CIFAR-10 / GTSRB (v9, for comparison) |
|---|---|---|---|
| Patch, blended, duplicates, class noise | 100% at every rate | 100% at every rate | 100% |
| Clean-label patch | 100% | 40% | 100% / 60-100% |
| Random label flip | 20 / 0 / 40 / 60% | 0 / 0 / 60 / 80% | 0-100% / 40-100% |
| Targeted label flip | 40 / 0 / 40 / 40% | 0% at every rate | 100% / 20-60% |
| Clean batches at 1.3 / 2.0 (n = 120) | 4.2% / 0% | **10.0% / 5.8%** | 4.2 / 1.7%, 2.5 / 0% |

- Row AUPRC at 5% across all four datasets (range): label flips 0.25-0.82 (here 0.30 random, 0.25 targeted), patch
  and blended 1.00, clean-label 0.74-1.00, duplicates 0.88-0.91, class noise 0.98-0.99.
- **Label flips do not carry over.** The label check works on the ImageNet embedding, which separates CIFAR-10 and
  GTSRB classes well, digits (SVHN) and 200 small classes (Tiny-ImageNet) poorly.
- **Clean alarms are not calibrated on Tiny-ImageNet**: 5.8% of clean batches reach SUSPICIOUS (nominal 1%). With
  ~15 items per class the per-class checks' p-values are too small. Many-class, few-per-class data gets false alarms.
- 14.3 s per batch.

## Attacks tuned to evade the checks, v2: a victim that learns (Phase 3, 2026-11-18)

`manifest-adaptive-images-v2`, `results/2026-11-18-benchmark-adaptive-images-v2.*`. Changed from v1 so the victim
can learn a backdoor, criterion unchanged and written before the run: reference 3,000 (poison is 2.5% / 5% of the
victim's training set at 5% / 10%), 40 epochs (clean accuracy 0.72), stronger strength grids, label-flip families
dropped (covered by v1). 3 eval seeds per cell; the attacker chooses on 3 other seeds.

| Family | Flagged | Runs with ASR >= 0.9 (flagged) | Highest ASR | Evasive setting |
|---|---|---|---|---|
| Faint 3x3 checkerboard, relabelled | 100% at every strength | 12 (12) | 1.00 | none |
| Smooth low-frequency trigger, relabelled | 100% at every strength | 1 (1) | 0.84 mean, 0.94 best | none |
| Warp (WaNet-style), relabelled | 100% at every strength | 1 (1) | 0.71 mean | none |
| Low-frequency trigger, labels unchanged | 0% up to amplitude 32 at 5%, 67% at amplitude 32 / 10% | 0 | 0.41 | yes: amplitude 32 at 5% (ASR 0.20, not flagged), 16 at 10% (ASR 0.15) |

- **Pre-set criterion:** every run where the attack worked (14 of 14) was flagged. That is the claim Veritas can make
  for attackers who relabel images, on this victim and CIFAR-10.
- **The clean-label low-frequency trigger gets past Veritas.** At 5% poison it is never flagged; on this victim it
  then makes 20% of triggered test images go to the target class (3% without poison). It did not reach 0.9; a
  stronger victim or longer training might, and Veritas would not see it. No pattern check reads smooth trigger
  energy: the shift and pixel checks start to react only at amplitude 32 and 10% poison.
- All relabelled attacks are caught by the label check (leading check at low strength); triggers strong enough to
  work are also caught by the patch and signature checks (faint patch from opacity 0.1).
