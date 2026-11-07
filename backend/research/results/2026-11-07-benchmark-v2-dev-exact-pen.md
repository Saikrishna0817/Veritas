# veritas-tabular-poisoning-v2 — dev-exact-pen

Profile `full`, 840 batches, split **dev**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| bp_exact_p06 | 0.65 [0.40, 0.86] | 0.59 [0.36, 0.83] | 0.72 [0.48, 0.89] | 0.58 [0.34, 0.81] | 0.61 [0.39, 0.81] | 0.15 [0.08, 0.25] | 1.162 |
| bp_exact_p10 | 0.65 [0.40, 0.86] | 0.59 [0.36, 0.83] | 0.72 [0.48, 0.89] | 0.58 [0.34, 0.81] | 0.61 [0.39, 0.81] | 0.15 [0.08, 0.25] | 1.156 |
| bp_exact_p15 | 0.65 [0.40, 0.86] | 0.59 [0.36, 0.83] | 0.72 [0.48, 0.89] | 0.58 [0.34, 0.81] | 0.61 [0.39, 0.81] | 0.15 [0.08, 0.25] | 1.152 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| bp_exact_p06 | 0.62 [0.40, 0.81] | 0.59 [0.38, 0.81] | 0.71 [0.47, 0.89] | 0.55 [0.32, 0.78] | 0.57 [0.31, 0.82] | 0.16 [0.07, 0.26] | 1.162 |
| bp_exact_p10 | 0.62 [0.40, 0.81] | 0.59 [0.38, 0.81] | 0.71 [0.47, 0.89] | 0.55 [0.32, 0.78] | 0.57 [0.31, 0.82] | 0.16 [0.07, 0.26] | 1.156 |
| bp_exact_p15 | 0.62 [0.40, 0.81] | 0.59 [0.38, 0.81] | 0.71 [0.47, 0.89] | 0.55 [0.32, 0.78] | 0.57 [0.31, 0.82] | 0.16 [0.07, 0.26] | 1.152 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| bp_exact_p06 | 0.57 [0.37, 0.83] | 0.63 [0.40, 0.87] | 0.67 [0.40, 1.00] | 0.57 [0.47, 0.77] | 0.87 [0.60, 1.00] | 0.23 [0.10, 0.40] | 1.162 |
| bp_exact_p10 | 0.63 [0.30, 0.90] | 0.70 [0.47, 0.90] | 0.77 [0.50, 1.00] | 0.63 [0.50, 0.80] | 0.83 [0.50, 1.00] | 0.30 [0.13, 0.43] | 1.156 |
| bp_exact_p15 | 0.63 [0.30, 0.90] | 0.70 [0.47, 0.90] | 0.77 [0.50, 1.00] | 0.63 [0.50, 0.80] | 0.63 [0.37, 0.83] | 0.23 [0.07, 0.37] | 1.152 |

