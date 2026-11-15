# veritas-image-poisoning-v4 — v4

Profile `full`, 400 batches, split **test**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| veritas_mm | 0.56 [0.53, 0.60] | 0.45 [0.37, 0.52] | 0.97 [0.95, 1.00] | 0.94 [0.88, 0.99] | 0.97 [0.95, 1.00] | 0.91 [0.90, 0.91] | 0.24 [0.20, 0.27] | 16.133 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| veritas_mm | 0.55 [0.53, 0.56] | 0.41 [0.34, 0.48] | 0.96 [0.92, 0.99] | 0.87 [0.78, 0.96] | 0.95 [0.90, 1.00] | 0.91 [0.90, 0.92] | 0.26 [0.22, 0.29] | 16.133 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| veritas_mm | 0.90 [0.80, 1.00] | 0.50 [0.40, 0.60] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.80 [0.60, 1.00] | 1.00 [1.00, 1.00] | 0.20 [0.20, 0.20] | 16.133 |

