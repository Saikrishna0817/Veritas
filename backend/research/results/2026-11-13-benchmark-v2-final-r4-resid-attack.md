# veritas-tabular-poisoning-v2 — final-r4-resid-attack

Profile `full`, 1680 batches, split **test**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| veritas2 | 0.68 [0.54, 0.81] | 0.76 [0.63, 0.86] | 0.95 [0.90, 0.99] | 0.88 [0.76, 0.96] | 0.91 [0.77, 0.99] | 0.61 [0.40, 0.83] | 5.552 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| veritas2 | 0.64 [0.52, 0.75] | 0.72 [0.61, 0.81] | 0.94 [0.88, 0.99] | 0.86 [0.77, 0.94] | 0.89 [0.74, 0.97] | 0.61 [0.41, 0.81] | 5.552 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| veritas2 | 0.80 [0.58, 0.97] | 0.83 [0.62, 1.00] | 0.97 [0.90, 1.00] | 0.92 [0.77, 1.00] | 0.95 [0.85, 1.00] | 0.70 [0.45, 0.92] | 5.552 |

