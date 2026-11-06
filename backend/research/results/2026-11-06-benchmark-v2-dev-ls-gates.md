# veritas-tabular-poisoning-v2 — dev-ls-gates

Profile `full`, 840 batches, split **dev**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| v2_ls_gate3 | 0.61 [0.37, 0.83] | 0.56 [0.36, 0.80] | 0.73 [0.55, 0.88] | 0.59 [0.35, 0.80] | 0.76 [0.67, 0.85] | 0.19 [0.12, 0.31] | 1.087 |
| v2_ls_gate4 | 0.64 [0.39, 0.86] | 0.57 [0.36, 0.80] | 0.74 [0.55, 0.88] | 0.59 [0.35, 0.80] | 0.76 [0.67, 0.85] | 0.19 [0.12, 0.31] | 1.086 |
| v2_ls_gate6 | 0.64 [0.39, 0.86] | 0.59 [0.36, 0.83] | 0.75 [0.56, 0.90] | 0.59 [0.35, 0.81] | 0.75 [0.64, 0.85] | 0.19 [0.11, 0.29] | 1.103 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| v2_ls_gate3 | 0.58 [0.36, 0.78] | 0.56 [0.37, 0.79] | 0.72 [0.55, 0.88] | 0.55 [0.31, 0.77] | 0.76 [0.64, 0.86] | 0.21 [0.12, 0.33] | 1.087 |
| v2_ls_gate4 | 0.61 [0.39, 0.81] | 0.57 [0.37, 0.79] | 0.74 [0.57, 0.89] | 0.55 [0.31, 0.77] | 0.76 [0.64, 0.86] | 0.21 [0.12, 0.33] | 1.086 |
| v2_ls_gate6 | 0.61 [0.39, 0.81] | 0.59 [0.38, 0.81] | 0.75 [0.58, 0.89] | 0.55 [0.32, 0.77] | 0.75 [0.61, 0.86] | 0.21 [0.13, 0.31] | 1.103 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | backdoor_ood | backdoor_inrange | feature_collision | covariate_shift | sec/batch |
|---|---|---|---|---|---|---|---|
| v2_ls_gate3 | 0.63 [0.40, 0.87] | 0.60 [0.47, 0.73] | 0.90 [0.67, 1.00] | 0.57 [0.40, 0.73] | 0.93 [0.80, 1.00] | 0.33 [0.07, 0.53] | 1.087 |
| v2_ls_gate4 | 0.63 [0.40, 0.87] | 0.60 [0.47, 0.73] | 0.90 [0.67, 1.00] | 0.57 [0.40, 0.73] | 0.93 [0.80, 1.00] | 0.33 [0.07, 0.53] | 1.086 |
| v2_ls_gate6 | 0.63 [0.40, 0.87] | 0.60 [0.47, 0.73] | 0.90 [0.67, 1.00] | 0.57 [0.40, 0.73] | 0.93 [0.80, 1.00] | 0.33 [0.07, 0.53] | 1.103 |

