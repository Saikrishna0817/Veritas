# veritas-tabular-poisoning-v2 — final-r2

Profile `full`, 1680 batches, split **test**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| cleanlab_hgb | 0.67 [0.52, 0.79] | 0.73 [0.59, 0.84] | 0.65 [0.50, 0.77] | 0.62 [0.48, 0.74] | 0.47 [0.28, 0.66] | 0.11 [0.09, 0.14] | 0.789 |
| veritas2 | 0.67 [0.52, 0.79] | 0.73 [0.59, 0.84] | 0.83 [0.72, 0.93] | 0.78 [0.67, 0.88] | 0.72 [0.56, 0.85] | 0.35 [0.19, 0.52] | 1.822 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| cleanlab_hgb | 0.63 [0.50, 0.74] | 0.69 [0.58, 0.79] | 0.62 [0.48, 0.73] | 0.59 [0.48, 0.70] | 0.44 [0.23, 0.64] | 0.13 [0.09, 0.17] | 0.789 |
| veritas2 | 0.63 [0.50, 0.74] | 0.69 [0.58, 0.79] | 0.83 [0.71, 0.92] | 0.76 [0.66, 0.85] | 0.70 [0.53, 0.85] | 0.37 [0.21, 0.55] | 1.822 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| cleanlab_hgb | 0.72 [0.48, 0.90] | 0.82 [0.63, 0.95] | 0.63 [0.38, 0.82] | 0.65 [0.45, 0.83] | 0.42 [0.17, 0.65] | 0.08 [0.02, 0.17] | 0.789 |
| veritas2 | 0.70 [0.47, 0.88] | 0.75 [0.50, 0.93] | 0.82 [0.58, 1.00] | 0.87 [0.73, 1.00] | 0.87 [0.70, 0.98] | 0.48 [0.28, 0.68] | 1.822 |

