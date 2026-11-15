# veritas-image-poisoning-v2 — smoke2

Profile `smoke`, 18 batches, split **test**, poison rate 0.05. Cells: mean [95% CI, cluster bootstrap over datasets].

### Row level: AUPRC

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| cleanlab_hgb | 0.72 [0.72, 0.72] | 0.73 [0.73, 0.73] | 0.67 [0.67, 0.67] | 0.55 [0.55, 0.55] | 0.07 [0.07, 0.07] | 0.04 [0.04, 0.04] | 0.07 [0.07, 0.07] | 3.246 |
| c2st | 0.05 [0.05, 0.05] | 0.05 [0.05, 0.05] | 0.06 [0.06, 0.06] | 0.14 [0.14, 0.14] | 0.07 [0.07, 0.07] | 0.92 [0.92, 0.92] | 0.08 [0.08, 0.08] | 1.84 |
| veritas2 | 0.72 [0.72, 0.72] | 0.73 [0.73, 0.73] | 0.67 [0.67, 0.67] | 0.55 [0.55, 0.55] | 0.07 [0.07, 0.07] | 0.92 [0.92, 0.92] | 0.07 [0.07, 0.07] | 5.423 |

### Row level: precision@k (k = number of poisoned rows)

Random ranking scores 0.05.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| cleanlab_hgb | 0.70 [0.70, 0.70] | 0.65 [0.65, 0.65] | 0.60 [0.60, 0.60] | 0.50 [0.50, 0.50] | 0.05 [0.05, 0.05] | 0.00 [0.00, 0.00] | 0.05 [0.05, 0.05] | 3.246 |
| c2st | 0.08 [0.08, 0.08] | 0.02 [0.02, 0.02] | 0.10 [0.10, 0.10] | 0.20 [0.20, 0.20] | 0.03 [0.03, 0.03] | 0.90 [0.90, 0.90] | 0.12 [0.12, 0.12] | 1.84 |
| veritas2 | 0.70 [0.70, 0.70] | 0.65 [0.65, 0.65] | 0.60 [0.60, 0.60] | 0.50 [0.50, 0.50] | 0.05 [0.05, 0.05] | 0.90 [0.90, 0.90] | 0.05 [0.05, 0.05] | 5.423 |

### Batch level: TPR at 5% clean-batch FPR

Share of attacked batches scored above the 95% quantile of clean batches.

| detector | label_flip_random | label_flip_targeted | patch_backdoor | blended_backdoor | clean_label_backdoor | duplicate_flood | class_noise | sec/batch |
|---|---|---|---|---|---|---|---|---|
| cleanlab_hgb | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 3.246 |
| c2st | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.84 |
| veritas2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5.423 |

