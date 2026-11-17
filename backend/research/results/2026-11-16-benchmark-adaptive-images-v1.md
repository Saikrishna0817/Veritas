# veritas-adaptive-images-v1 (v1)

Flagged = veritas_mm evidence >= 1.3 (app.detection.veritas2.LOW_RISK_AT, the app's lowest flagged level). Eval seeds [10, 11, 12]; evasive strength picked on attacker seeds [0, 1, 2]. Clean-data victim accuracy: 0.681.

## lowfreq (pixel units of 255, amplitude of a sigma-4 smooth pattern)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.01 | 1.00 | 6.74 | label | 0.66 | 0.06 | 0.689 | 0.04 |  |
| 4 | 0.01 | 1.00 | 7.22 | label | 0.70 | 0.04 | 0.678 | 0.04 |  |
| 8 | 0.01 | 1.00 | 6.02 | label | 0.58 | 0.06 | 0.688 | 0.05 |  |
| 16 | 0.01 | 1.00 | 5.85 | label | 0.46 | 0.07 | 0.680 | 0.06 |  |
| 2 | 0.05 | 1.00 | 35.33 | label | 0.81 | 0.06 | 0.686 | 0.04 |  |
| 4 | 0.05 | 1.00 | 33.09 | label | 0.79 | 0.05 | 0.686 | 0.04 |  |
| 8 | 0.05 | 1.00 | 34.02 | label | 0.79 | 0.07 | 0.677 | 0.05 |  |
| 16 | 0.05 | 1.00 | 29.34 | label | 0.74 | 0.21 | 0.679 | 0.06 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: no run reached ASR 0.9

## warp (max pixel displacement of a smooth flow field)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.01 | 1.00 | 5.57 | label | 0.53 | 0.05 | 0.682 | 0.05 |  |
| 0.5 | 0.01 | 1.00 | 7.01 | label | 0.64 | 0.06 | 0.677 | 0.05 |  |
| 1 | 0.01 | 1.00 | 5.23 | label | 0.53 | 0.06 | 0.679 | 0.05 |  |
| 2 | 0.01 | 1.00 | 5.35 | label | 0.53 | 0.05 | 0.684 | 0.04 |  |
| 0.25 | 0.05 | 1.00 | 33.88 | label | 0.80 | 0.08 | 0.680 | 0.05 |  |
| 0.5 | 0.05 | 1.00 | 28.64 | label | 0.73 | 0.10 | 0.679 | 0.05 |  |
| 1 | 0.05 | 1.00 | 29.10 | label | 0.72 | 0.15 | 0.678 | 0.05 |  |
| 2 | 0.05 | 1.00 | 24.00 | label | 0.65 | 0.16 | 0.676 | 0.04 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: no run reached ASR 0.9

## faint_patch (opacity of the 3x3 checkerboard)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.01 | 1.00 | 6.14 | label | 0.63 | 0.05 | 0.683 | 0.04 |  |
| 0.1 | 0.01 | 1.00 | 6.77 | label | 0.64 | 0.05 | 0.683 | 0.04 |  |
| 0.2 | 0.01 | 1.00 | 7.84 | pixel | 0.87 | 0.05 | 0.687 | 0.05 |  |
| 0.4 | 0.01 | 1.00 | 11.53 | sig | 1.00 | 0.28 | 0.683 | 0.05 |  |
| 0.05 | 0.05 | 1.00 | 34.35 | label | 0.78 | 0.05 | 0.675 | 0.04 |  |
| 0.1 | 0.05 | 1.00 | 53.85 | pixel | 0.95 | 0.07 | 0.682 | 0.04 |  |
| 0.2 | 0.05 | 1.00 | 66.18 | sig | 0.98 | 0.42 | 0.683 | 0.05 |  |
| 0.4 | 0.05 | 1.00 | 69.25 | sig | 1.00 | 0.97 | 0.683 | 0.05 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: 1.00 (n=4)

## cl_lowfreq (as lowfreq, on target-class images, labels unchanged)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.01 | 0.33 | 0.74 | resid | 0.01 | 0.05 | 0.683 | 0.04 |  |
| 4 | 0.01 | 0.33 | 0.74 | resid | 0.01 | 0.05 | 0.683 | 0.04 |  |
| 8 | 0.01 | 0.33 | 0.74 | resid | 0.01 | 0.06 | 0.681 | 0.05 |  |
| 16 | 0.01 | 0.33 | 0.74 | resid | 0.01 | 0.06 | 0.691 | 0.06 | yes |
| 2 | 0.05 | 0.33 | 0.74 | resid | 0.05 | 0.05 | 0.681 | 0.04 |  |
| 4 | 0.05 | 0.33 | 0.74 | resid | 0.05 | 0.04 | 0.692 | 0.04 |  |
| 8 | 0.05 | 0.33 | 0.74 | resid | 0.05 | 0.06 | 0.682 | 0.05 |  |
| 16 | 0.05 | 0.33 | 0.80 | resid | 0.05 | 0.09 | 0.684 | 0.06 | yes |

Evasive strength (flagged share, ASR): rate 0.01: s=16 (0.33, 0.06); rate 0.05: s=16 (0.33, 0.09)
Headline, flagged share among runs with ASR >= 0.9: no run reached ASR 0.9

## boundary_flip (none; flips the source-class images an attacker-side logistic regression rates most target-like)

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.01 | 0.33 | 0.74 | shift | 0.11 | 0.08 | 0.693 | 0.08 | yes |
| 1 | 0.05 | 1.00 | 17.99 | label | 0.54 | 0.15 | 0.683 | 0.08 |  |

Evasive strength (flagged share, ASR): rate 0.01: s=1 (0.33, 0.08)
Headline, flagged share among runs with ASR >= 0.9: no run reached ASR 0.9

## random_flip (none; reference: research.mm.attacks.label_flip_targeted (random source-class picks))

| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.01 | 1.00 | 5.78 | label | 0.49 | 0.10 | 0.683 | 0.08 |  |
| 1 | 0.05 | 1.00 | 34.04 | label | 0.76 | 0.15 | 0.680 | 0.08 |  |

Evasive strength (flagged share, ASR): none (every strength is flagged on the attacker seeds)
Headline, flagged share among runs with ASR >= 0.9: no run reached ASR 0.9

Policy: Veritas is not tuned on these results. Never edit this file in place: create a new benchmark_id for any change. Results are quoted once.
