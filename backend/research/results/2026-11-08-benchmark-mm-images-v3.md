# veritas-image-poisoning-v3 — v3

Profile `full`, 320 batches, split **test**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| veritas2 | 0.64 [0.60, 0.69] | 0.49 [0.39, 0.59] | 0.63 [0.58, 0.67] | 0.39 [0.37, 0.41] | 0.04 [0.02, 0.05] | 0.82 [0.79, 0.85] | 0.08 [0.08, 0.08] | 8.806 |
| veritas_mm | 0.59 [0.59, 0.60] | 0.49 [0.39, 0.59] | 0.95 [0.94, 0.95] | 0.64 [0.56, 0.72] | 0.70 [0.45, 0.95] | 0.82 [0.78, 0.85] | 0.14 [0.09, 0.18] | 15.269 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| veritas2 | 0.60 [0.56, 0.64] | 0.45 [0.35, 0.55] | 0.56 [0.52, 0.61] | 0.37 [0.35, 0.39] | 0.02 [0.00, 0.03] | 0.82 [0.80, 0.84] | 0.07 [0.06, 0.08] | 8.806 |
| veritas_mm | 0.57 [0.56, 0.58] | 0.45 [0.35, 0.55] | 0.93 [0.92, 0.94] | 0.61 [0.53, 0.70] | 0.70 [0.46, 0.95] | 0.84 [0.81, 0.87] | 0.14 [0.11, 0.17] | 15.269 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| veritas2 | 0.90 [0.80, 1.00] | 0.60 [0.60, 0.60] | 0.90 [0.90, 1.00] | 0.60 [0.60, 0.60] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 8.806 |
| veritas_mm | 0.90 [0.80, 0.90] | 0.50 [0.50, 0.60] | 1.00 [1.00, 1.00] | 0.90 [0.60, 1.00] | 0.80 [0.60, 1.00] | 1.00 [1.00, 1.00] | 0.10 [0.00, 0.20] | 15.269 |

