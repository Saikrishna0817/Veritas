# Video benchmark findings

## Setup (manifest-video-v1 / v2, 2026-11-14)

- **Data:** UCF-101 (101 human actions, 13,320 YouTube clips). The official server ran at ~40 KB/s, so the videos come
  from the Hugging Face mirror MichiganNLP/ucf-101 (`UCF101.zip`, a ZIP repack of the official RAR). Its SHA-256
  matches the mirror's published hash, its split file is byte-identical to the official one from crcv.ucf.edu, and all
  13,320 clips of the official split lists are present (none missing, none extra). Official split 1 separates clips by
  group (actors and scene): dev = trainlist01 (9,537 clips), test = testlist01 (3,783).
- **Decoding:** ffmpeg, 8 evenly spaced frames per clip at 64x64 (`app/detection/video.py`).
- **Attacks** on the decoded frames (`research/video/attacks.py`), the image recipes on every frame: random and
  targeted label flips, a checkerboard patch with relabelling, a blended noise pattern with relabelling, the patch on
  correctly labelled target-class clips (clean-label), duplicate floods, a fixed per-class noise pattern.
- **Detector:** the shipped image detector (`veritas_mm`) unchanged; the label and shift checks see the mean ResNet-18
  embedding of the 8 frames (cross-validated accuracy over 101 classes on dev: 58%, chance 1%).
- Batches: 3,000 clips (1,200 incoming), 5 seeds, 4 rates, 120 clean batches per split.

## Chosen on dev: what the pixel and shared-pattern checks see

| Dev view (32x32) | Mean row AUPRC 1% / 3% / 5% / 10% | Clean flagged at 1.3 |
|---|---|---|
| v1: 2x2 mosaic of 4 frames | 0.37 / 0.43 / 0.48 / 0.58 | 5.8% |
| v2: temporal mean of the 8 frames | 0.57 / 0.62 / 0.70 / 0.75 | 6.7% |
| v3: the middle frame | 0.49 / 0.61 / 0.68 / 0.73 | 7.5% |

The mosaic shrinks each frame to 16x16, so a patch becomes 2 pixels and faint noise averages away. The temporal mean
keeps anything stamped identically on every frame (a trigger, a noise mask) and blurs the moving content around it,
so the colour-noise and pixel checks see it: blended noise at 1% 0.40 -> 1.00, class noise 0.03 -> 0.87. v2 is the
configuration (`manifest-video-v2.json`).

## Held-out test (v2, `results/2026-11-14-benchmark-video-test-v2.*`)

Row AUPRC (random = the poison rate) / share of attacked batches flagged at 1.3:

| Attack | 1% | 5% |
|---|---|---|
| Random label flip | 0.50 / 0% | 0.85 / 100% |
| Targeted label flip | 0.43 / 40% | 0.66 / 80% |
| Patch trigger | 0.53 / 20% | 0.91 / 100% |
| Blended noise | 1.00 / 100% | 1.00 / 100% |
| Clean-label patch | 0.01 / 0% | 0.02 / 20% |
| Duplicate flood | 0.90 / 100% | 0.93 / 100% |
| Class noise | 0.35 / 60% | 0.88 / 100% |
| Mean | 0.53 / 46% | 0.75 / 86% |

At 3% the mean is 0.66 / 69%, at 10% 0.75 / 80%.

- Clean test batches (n = 120): 4.2% reach 1.3 (nominal 5%), 2.5% reach 2.0 (nominal 1%; 3 batches), none 4.0.
- **Clean-label patches are not found**, as for images and audio: correct labels plus a trigger the pooled checks do
  not single out. A trigger-specific check would be needed.
- A trigger on only some frames, or a moving one, would be blurred by the temporal mean; only static every-frame
  triggers were benchmarked.
- About 16 s per 3,000-clip batch on one CPU thread (frame embedding excluded; computed once per clip).

## New dataset, never used for tuning: HMDB-51 (2026-11-22)

`manifest-video-breadth-v1`, `results/2026-11-22-benchmark-video-breadth-v1.*`, detector at ccd8e9e (video code as in
v2; a rerun of v2 at fb3ab71 reproduced its numbers exactly). HMDB-51 official split 1 test clips (1,530, 30 per class
in 51 classes), from the Hugging Face mirror MichiganNLP/hmdb the Serre lab page links (the lab's own links returned
404), SHA-256 pinned. The test pool is small, so batches hold 612 clips (1,200 on UCF-101): 1% poison is 6 clips.

| Attack | row AUPRC 1% / 5% | flagged 1% / 5% | UCF-101 row AUPRC 1% / 5% | UCF-101 flagged 1% / 5% |
|---|---|---|---|---|
| Random label flips | 0.13 / 0.46 | 0 / 0% | 0.50 / 0.85 | 0 / 100% |
| Targeted label flips | 0.47 / 0.24 | 20 / 20% | 0.43 / 0.66 | 40 / 80% |
| Patch trigger | 0.16 / 0.77 | 0 / 100% | 0.53 / 0.91 | 20 / 100% |
| Blended trigger | 1.00 / 1.00 | 0 / 100% | 1.00 / 1.00 | 100 / 100% |
| Clean-label trigger | 0.03 / 0.10 | 0 / 0% | 0.01 / 0.02 | 0 / 20% |
| Duplicate flood | 0.89 / 0.88 | 60 / 100% | 0.90 / 0.93 | 100 / 100% |
| Class noise | 0.70 / 0.97 | 20 / 100% | 0.35 / 0.88 | 60 / 100% |
| **Mean** | **0.48 / 0.63** | **14 / 60%** | 0.53 / 0.75 | 46 / 86% |

- Clean batches: 3.3% at LOW_RISK (4 of 120, nominal 5%), 1.7% at SUSPICIOUS (nominal 1%). Calibration carries over.
- Label flips carry over worst: random flips at 5% were flagged in none of 5 batches (UCF-101: all). With 30 clips per
  class and ImageNet frame features, the label check cannot tell HMDB-51's actions apart well; the same failure as
  images on SVHN and Tiny-ImageNet.
- Triggers and duplicates carry over at 5% (all batches flagged); at 1% (6 clips) only duplicates are flagged reliably.
  Part of the drop at 1% is the smaller batch: 6 poisoned clips give the tail test less to work with.
- Clean-label triggers: not detected, as on UCF-101.

## In the app

`POST /analyze/video` (zip of video files in class folders: mp4, avi, mov, mkv, webm, mpg; optional reference zip;
background job) runs this configuration through the same zip reader and detector step as images and audio. ffmpeg
decodes each upload from a private temp file with a 60 s timeout and a frame cap; ffmpeg parsing untrusted files is
the main remaining attack surface, so the worker should run isolated in production. The benchmark imports the app's
code (`app/detection/video.py`).
