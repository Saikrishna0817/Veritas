# Audio benchmark findings

## Setup (manifest-audio-v1 / v2, 2026-11-14)

- **Data:** Speech Commands v0.02 (35 words, 1 s clips) and ESC-50 (50 environmental sound classes, 5 s). Downloads
  are SHA-256 pinned in the manifest. Dev = Speech Commands validation speakers + ESC-50 folds 1-3; test = testing
  speakers + folds 4-5. Dev and test share no speakers or clips, but come from the same two datasets, so this is a
  weaker held-out test than the image track's separate datasets.
- **Attacks** on raw 16 kHz waveforms (`research/audio/attacks.py`): random and targeted label flips, a 100 ms tone
  trigger with relabelling, a blended noise track with relabelling, the tone on correctly labelled target-class clips
  (clean-label), duplicate floods, a faint per-class noise pattern.
- **Detector:** the shipped image detector (`veritas_mm`) unchanged. The pixel and shared-pattern checks see 32x32
  log-mel spectrogram images; the label and shift checks see a clip embedding.
- Batches: 3,000 Speech Commands clips (1,200 incoming), all of ESC-50's split (dev 1,200, test 800: 320 incoming,
  so 1% poison is 3 clips). 5 seeds, 4 rates, 60 clean batches per dataset.

## Chosen on dev

| Dev variant | Speech Commands mean row AUPRC 1% / 5% | ESC-50 1% / 5% | Clean flagged at 1.3 (SC / ESC) | s / batch |
|---|---|---|---|---|
| v1: ResNet-18 on the spectrogram image | 0.40 / 0.55 | 0.34 / 0.48 | 1.7 / 1.7% | 15 / 8 |
| v2: HuBERT-base layer 6 embedding | 0.61 / 0.79 | 0.42 / 0.57 | 5.0 / 3.3% | 12 / 6 |
| v3: v2 + 64x64 check images | 0.64 / 0.78 | 0.42 / 0.58 | 3.3 / 6.7% | 38 / 32 |

ImageNet ResNet-18 separates sound classes poorly (cross-validated accuracy on Speech Commands 25%; HuBERT layer 6
86%, ESC-50 44% vs 64%). Layer 6 had the best mean accuracy of layers 6, 9 and 12. 64x64 check images gave +0.03 at a
3-5x cost and more clean alarms on ESC-50, so v2 is the configuration (`manifest-audio-v2.json`).

## Held-out test (v2, `results/2026-11-14-benchmark-audio-test-v2.*`)

Row AUPRC (random = the poison rate) / share of attacked batches flagged at 1.3:

| Attack | Speech Commands 1% | Speech Commands 5% | ESC-50 1% | ESC-50 5% |
|---|---|---|---|---|
| Random label flip | 0.54 / 20% | 0.83 / 100% | 0.27 / 0% | 0.58 / 20% |
| Targeted label flip | 0.57 / 40% | 0.78 / 100% | 0.25 / 0% | 0.34 / 0% |
| Tone trigger | 0.69 / 60% | 0.80 / 100% | 0.31 / 0% | 0.72 / 20% |
| Blended noise | 0.91 / 100% | 1.00 / 100% | 0.40 / 0% | 0.89 / 100% |
| Clean-label tone | 0.02 / 0% | 0.03 / 0% | 0.02 / 0% | 0.04 / 20% |
| Duplicate flood | 0.92 / 100% | 0.91 / 100% | 0.67 / 20% | 0.94 / 100% |
| Class noise | 0.51 / 40% | 1.00 / 100% | 0.09 / 0% | 0.22 / 0% |
| Mean | 0.59 / 51% | 0.76 / 86% | 0.29 / 3% | 0.53 / 37% |

- Clean test batches flagged at 1.3: 3.3% on both (nominal 5%, n = 60 each); none at 2.0.
- **Clean-label triggers are not found** (row AUPRC near the 1-5% base rate): a short tone on correctly labelled clips
  changes neither the label fit nor anything the pooled checks compare. A trigger-specific check would be needed.
- ESC-50 is small (320 incoming clips per test batch); at 1% the batch verdict almost never fires there.
- About 15 s per 3,000-clip batch on one CPU thread (HuBERT embedding excluded; it is computed once per clip).

## In the app

`POST /analyze/audio` (zip of PCM WAV files in class folders, optional reference zip, background job) runs this
configuration: clips mixed to mono 16 kHz, padded or trimmed to the median length (at most 10 s), HuBERT layer 6
for the label and shift checks, 32x32 spectrograms for the rest. The benchmark imports the app's code
(`app/detection/audio.py`, `app/detection/veritas_mm.py`).
