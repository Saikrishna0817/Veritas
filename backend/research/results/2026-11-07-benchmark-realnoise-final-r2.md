# veritas-realnoise-v1 — final-r2

220 batches, 20 seeds, mistake rate 0.05. Cells: mean [95% CI, bootstrap over seeds]. Random ranking scores the mistake rate.

### Row level: AUPRC

| detector | aggre_label | random_label1 | worse_label | sec/batch |
|---|---|---|---|---|
| cleanlab_hgb | 0.46 [0.44, 0.48] | 0.56 [0.53, 0.59] | 0.58 [0.55, 0.61] | 3.383 |
| veritas2 | 0.46 [0.44, 0.48] | 0.55 [0.51, 0.59] | 0.58 [0.55, 0.61] | 5.314 |

### Row level: precision@k

| detector | aggre_label | random_label1 | worse_label | sec/batch |
|---|---|---|---|---|
| cleanlab_hgb | 0.44 [0.42, 0.46] | 0.53 [0.50, 0.55] | 0.55 [0.53, 0.58] | 3.383 |
| veritas2 | 0.44 [0.42, 0.46] | 0.52 [0.49, 0.55] | 0.55 [0.53, 0.58] | 5.314 |

### Batch level: TPR at 5% clean-batch FPR

| detector | aggre_label | random_label1 | worse_label | sec/batch |
|---|---|---|---|---|
| cleanlab_hgb | 0.75 [0.15, 0.95] | 0.80 [0.60, 1.00] | 0.80 [0.40, 1.00] | 3.383 |
| veritas2 | 0.55 [0.30, 1.00] | 0.90 [0.70, 1.00] | 0.70 [0.50, 1.00] | 5.314 |

