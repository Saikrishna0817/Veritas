# Image benchmark findings

Held-out test datasets: CIFAR-10 and GTSRB. Tuning (dev) datasets: Fashion-MNIST and CIFAR-100. Attacks are applied to real pixels;
5 seeds, 3,000-image batches (1,200 incoming). Row AUPRC; a random ranking scores the poison rate. Raw results:
`backend/research/results/*benchmark-mm-images-*`.

| Attack | veritas2 1% | veritas_mm 1% | veritas2 5% | veritas_mm 5% |
|---|---|---|---|---|
| Random label flip | 0.45 | 0.44 | 0.64 | 0.59 |
| Targeted label flip | 0.41 | 0.41 | 0.49 | 0.49 |
| Patch backdoor | 0.39 | 0.49 | 0.63 | 0.95 |
| Blended backdoor | 0.20 | 0.20 | 0.39 | 0.64 |
| Clean-label backdoor | 0.02 | 0.20 | 0.04 | 0.70 |
| Duplicate flood | 0.08 | 0.50 | 0.82 | 0.82 |
| Class-noise | 0.02 | 0.02 | 0.08 | 0.14 |

`veritas_mm` = v3 (pixel gate 1, chosen on dev only; the dev gain was confirmed on test).

- A ResNet embedding does not see corner patches or faint noise; low-level pixel statistics do.
- At 1% the pixel check alone ranks patch (0.71), clean-label (0.57) and duplicates (0.79) better than the fused score: max-fusion
  with scores capped by the calibration-set size dilutes sharp channels. Fusion is the next thing to fix.
- Batch verdicts at 1% are weak for every detector (0.0-0.2 of attacked batches at 5% false alarms).
- Clean test batches (n = 40, noisy): 12.5% reach the lowest verdict cut-off (nominal 5%), 5% the middle one. Dev (n = 120): 1.7%.
- Class-noise and blended backdoors at 1% are at or near chance.
- Cost: a small drop on random label flips at 5% (0.64 -> 0.59).
