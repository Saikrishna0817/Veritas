# Tabular cell localisation, held-out test (manifest-v2 test split, 2026-11-13)

Produced by `SPLIT=test python -m research.tab.cell_study_dev --report` (variant frozen on dev; = `veritas2.cell_scores`).

Split test: 1440 attacked batches, datasets: PhishingWebsites, adult, credit-g, electricity, ilpd, kc1, letter, pendigits, phoneme, satimage, spambase, vehicle.

Cell AUPRC (random = the share of changed cells), cell-scores / row-only; then the column ranking inside truly poisoned rows.

| Attack | 1% cells | 3% cells | 5% cells | 10% cells | 1% in-row | 3% in-row | 5% in-row | 10% in-row |
|---|---|---|---|---|---|---|---|---|
| label_flip_random | 0.47 / 0.04 | 0.59 / 0.05 | 0.64 / 0.04 | 0.68 / 0.05 | 0.99 | 0.99 | 0.99 | 0.99 |
| label_flip_targeted | 0.55 / 0.06 | 0.66 / 0.05 | 0.71 / 0.05 | 0.73 / 0.05 | 0.99 | 0.99 | 0.99 | 0.99 |
| backdoor_ood | 0.39 / 0.08 | 0.72 / 0.20 | 0.81 / 0.22 | 0.89 / 0.24 | 0.98 | 0.99 | 0.99 | 0.98 |
| backdoor_inrange | 0.19 / 0.10 | 0.31 / 0.13 | 0.47 / 0.19 | 0.58 / 0.21 | 0.83 | 0.85 | 0.84 | 0.85 |
| feature_collision | 0.08 / 0.20 | 0.20 / 0.43 | 0.30 / 0.53 | 0.44 / 0.64 | 0.74 | 0.75 | 0.73 | 0.75 |
| covariate_shift | 0.01 / 0.01 | 0.09 / 0.06 | 0.15 / 0.10 | 0.26 / 0.16 | 0.57 | 0.55 | 0.55 | 0.56 |
| Mean | 0.28 / 0.08 | 0.43 / 0.15 | 0.51 / 0.19 | 0.60 / 0.23 | 0.85 | 0.85 | 0.85 | 0.85 |
