# veritas-adaptive-images-v3 (v3)

Flagged = veritas_mm evidence >= 1.3 (app.detection.veritas2.LOW_RISK_AT, the app's lowest flagged level). Eval seeds [10, 11, 12]; evasive strength picked on attacker seeds [0, 1, 2]. Clean-data victim accuracy: 0.719.

## lowfreq (pixel units of 255, amplitude of a sigma-4 smooth pattern)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.05 | 1.00 | 23.18 | label | 0.75 | 0.07 | 0.707 | 0.03 |  |
| 8 | 0.05 | 1.00 | 22.18 | label | 0.83 | 0.09 | 0.712 | 0.03 |  |
| 16 | 0.05 | 1.00 | 21.41 | label | 0.74 | 0.33 | 0.716 | 0.03 |  |
| 32 | 0.05 | 1.00 | 32.14 | sig_smooth | 0.90 | 0.77 | 0.710 | 0.03 |  |
| 4 | 0.1 | 1.00 | 44.87 | label | 0.89 | 0.09 | 0.699 | 0.03 |  |
| 8 | 0.1 | 1.00 | 45.45 | label | 0.87 | 0.20 | 0.689 | 0.03 |  |
| 16 | 0.1 | 1.00 | 45.36 | label | 0.80 | 0.65 | 0.709 | 0.03 |  |
| 32 | 0.1 | 1.00 | 73.57 | sig_smooth | 0.94 | 0.84 | 0.701 | 0.03 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: 1.00 (n=1)

## warp (max pixel displacement of a smooth flow field)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.05 | 1.00 | 19.78 | label | 0.73 | 0.09 | 0.703 | 0.05 |  |
| 1 | 0.05 | 1.00 | 19.31 | label | 0.68 | 0.17 | 0.713 | 0.05 |  |
| 2 | 0.05 | 1.00 | 17.88 | label | 0.64 | 0.22 | 0.703 | 0.06 |  |
| 4 | 0.05 | 1.00 | 14.11 | label | 0.53 | 0.47 | 0.713 | 0.08 |  |
| 0.5 | 0.1 | 1.00 | 42.99 | label | 0.86 | 0.21 | 0.706 | 0.05 |  |
| 1 | 0.1 | 1.00 | 39.80 | label | 0.83 | 0.32 | 0.695 | 0.05 |  |
| 2 | 0.1 | 1.00 | 34.45 | label | 0.75 | 0.39 | 0.700 | 0.06 |  |
| 4 | 0.1 | 1.00 | 26.03 | label | 0.69 | 0.71 | 0.709 | 0.08 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: 1.00 (n=1)

## faint_patch (opacity of the 3x3 checkerboard)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 1.00 | 22.08 | label | 0.82 | 0.03 | 0.717 | 0.03 |  |
| 0.1 | 0.05 | 1.00 | 30.30 | pixel | 0.94 | 0.16 | 0.705 | 0.03 |  |
| 0.2 | 0.05 | 1.00 | 41.96 | sig | 1.00 | 0.96 | 0.705 | 0.03 |  |
| 0.4 | 0.05 | 1.00 | 44.19 | sig | 1.00 | 1.00 | 0.709 | 0.04 |  |
| 0.05 | 0.1 | 1.00 | 43.58 | label | 0.86 | 0.08 | 0.697 | 0.03 |  |
| 0.1 | 0.1 | 1.00 | 69.02 | pixel | 0.98 | 0.47 | 0.695 | 0.03 |  |
| 0.2 | 0.1 | 1.00 | 92.26 | sig | 1.00 | 1.00 | 0.707 | 0.03 |  |
| 0.4 | 0.1 | 1.00 | 92.26 | sig | 1.00 | 1.00 | 0.709 | 0.04 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: 1.00 (n=12)

## cl_lowfreq (as lowfreq, on target-class images, labels unchanged)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.05 | 0.00 | 0.19 | sig_smooth | 0.05 | 0.03 | 0.725 | 0.03 |  |
| 8 | 0.05 | 0.00 | 0.23 | sig_smooth | 0.06 | 0.04 | 0.712 | 0.03 | yes |
| 16 | 0.05 | 0.67 | 3.79 | sig_smooth | 0.22 | 0.08 | 0.718 | 0.03 |  |
| 32 | 0.05 | 1.00 | 32.98 | sig_smooth | 0.85 | 0.20 | 0.727 | 0.03 |  |
| 4 | 0.1 | 0.00 | 0.31 | sig_smooth | 0.11 | 0.03 | 0.719 | 0.03 |  |
| 8 | 0.1 | 0.00 | 0.36 | sig_smooth | 0.12 | 0.05 | 0.712 | 0.03 | yes |
| 16 | 0.1 | 1.00 | 13.64 | sig_smooth | 0.44 | 0.15 | 0.709 | 0.03 |  |
| 32 | 0.1 | 1.00 | 72.41 | sig_smooth | 0.89 | 0.41 | 0.714 | 0.03 |  |

Evasive strength (flagged share, ASR): rate 0.05: s=8 (0.00, 0.04); rate 0.1: s=8 (0.00, 0.05)
Headline, flagged share among runs with ASR >= 0.9: no run reached ASR 0.9

Policy: Veritas is not tuned on these results. Never edit this file in place: create a new benchmark_id for any change. Results are quoted once.
