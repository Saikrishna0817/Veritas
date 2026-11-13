# veritas-tabular-poisoning-v2 — final-r3-crossfit

Profile `full`, 1680 batches, split **test**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| veritas2 | 0.69 [0.55, 0.81] | 0.76 [0.64, 0.87] | 0.92 [0.80, 1.00] | 0.88 [0.77, 0.96] | 0.91 [0.77, 0.99] | 0.53 [0.34, 0.74] | 5.378 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| veritas2 | 0.65 [0.52, 0.76] | 0.72 [0.62, 0.82] | 0.91 [0.79, 0.99] | 0.85 [0.75, 0.93] | 0.89 [0.75, 0.97] | 0.51 [0.34, 0.71] | 5.378 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| veritas2 | 0.80 [0.58, 0.97] | 0.83 [0.62, 1.00] | 0.92 [0.77, 1.00] | 0.92 [0.73, 1.00] | 0.95 [0.85, 1.00] | 0.63 [0.40, 0.85] | 5.378 |

