# Veritas model card

Last updated 2026-11-07. Covers the two things Veritas does: **training-data checks** (`veritas2`) and
**model-file scanning**. Numbers come from `docs/benchmark-v2-findings.md` and `docs/scanner-benchmark.md`;
raw results are in `backend/research/results/`.

## What it is for

- **Training-data check.** You give Veritas a labelled tabular CSV (and ideally a trusted reference set).
  It ranks the incoming rows by how likely their label is wrong or their features have been tampered with,
  and gives one batch-level verdict.
- **Model-file scan.** You give it a pickle, joblib, PyTorch, NumPy, safetensors, GGUF, ONNX or zip file.
  It reports whether loading the file could run code. The file is read as bytes and never loaded.

It is a triage aid: "review these rows first", "do not load this file". It is not a certification, not proof of an
attack, and not a compliance (NIST AI RMF, EU AI Act) claim.

## Training-data check (`veritas2`)

**Method** (since 2026-11-13). (1) *Label check*: gradient-boosted classifiers cross-fitted over all clean reference
rows (5 folds) rank rows by cleanlab's label-quality score; every reference row gets an out-of-fold score. (2) *Shift
check*: a classifier two-sample test asks whether a model can tell incoming rows from the clean reference rows; it joins
the ranking only when its own tail test is strong (GATE = 3.0). (3) Evidence is the best exact hypergeometric tail test
over the top 0.03%, 0.1%, 0.3% and 1%, minus log10(4), against all reference rows; penalties label 0.1, shift 1.0.
Verdict cut-offs: p <= 0.05 LOW_RISK, 0.01 SUSPICIOUS, 1e-4 CONFIRMED. (4) *Cell scores*: row score plus how unusual
each value is in its column (label cell: the label check), so a report can name the poisoned values. (5) *Residual
check* (gated): rows whose columns no longer fit together (cross-fitted ridge residuals). (6) *Attack profile*: for a
batch that is not CLEAN, the closest of six benchmarked attacks, from a classifier trained on dev batches only.

**Evaluation.** Benchmark v2: 18 real OpenML datasets (6 dev, 12 held-out test), six injected attacks, four poison
rates, five seeds, plus clean batches. Detector design and tuning used the dev datasets only. CIFAR-10N (real human
label mistakes, nothing injected) is a second held-out track.

| Held-out result | old Veritas | veritas2 (Nov 7) | veritas2 (Nov 13) | cleanlab | two-sample test |
|---|---|---|---|---|---|
| row AUPRC at 5% poison (random = 0.05) | 0.11 | 0.69 | **0.80** | 0.54 | 0.49 |
| batch detection rate at 5% false alarms | 0.33 | 0.78 | **0.86** | 0.55 | 0.14 |

By poison rate (batch detection at 5% false alarms), Nov 13: 1% 0.47, 3% 0.79, 5% 0.86, 10% 0.94 (Nov 7: 0.14, 0.61,
0.74, 0.87). Cell AUPRC (finding the changed values and labels) 0.38 at 1%, 0.60 at 5%. Attack profile on held-out
flagged batches: exact attack 74%, right family 86% (random vs targeted label flips are often confused). The CIFAR-10N comparison below
was run on the Nov 7 detector. On real CIFAR-10N
label noise `veritas2` matches cleanlab (row AUPRC 0.45 to 0.57 vs 0.46 to 0.58); it does not beat it.

**Calibration.** On held-out clean batches 3.7% reached LOW_RISK (nominal 5%, n = 240), 1.3% reached SUSPICIOUS (nominal 1%)
and none reached CONFIRMED (Nov 7 detector: 7.1%, 0.8%, 0).

**Dependency check.** After the numpy 2.5 / pandas 3.1 upgrade the 12,000 per-batch results were identical to the
earlier run (see findings doc).

## Image-data check (`veritas_mm`)

`POST /analyze/images` takes a zip with one folder per class (optionally a second zip of trusted reference images)
and runs as a background job. Images are resized to 32x32 RGB and embedded with ImageNet ResNet-18. Seven checks:
label (cross-fitted logistic regression on the embedding), shift and pixel two-sample tests, and near-duplicates,
a shared patch, a shared faint pattern and colour noise. Evidence and verdict cut-offs are the same as for tables.
The result lists the flagged images (file, class, score, the check that flagged it) and the leading check.

Held-out benchmark (CIFAR-10, GTSRB; docs/image-benchmark-findings.md, v8): row AUPRC 0.86 at 1% poison and 0.91 at
5%; 81% of attacked batches flagged at 1%; clean batches flagged 2.9% at LOW_RISK. About 4 s for 420 images, 10 s
for 3,000 on one CPU thread.

## Audio-data check

`POST /analyze/audio` takes a zip of PCM WAV files in class folders (optionally a reference zip). The image detector
runs on 32x32 log-mel spectrograms, with HuBERT-base (layer 6) embeddings for the label and shift checks. Held-out
(Speech Commands testing speakers, ESC-50 folds 4-5; docs/audio-benchmark-findings.md): mean row AUPRC 0.59 at 1% and
0.76 at 5% on Speech Commands, 0.29 and 0.53 on ESC-50; clean batches flagged 3.3% at LOW_RISK. Clean-label audio
triggers are not detected.

## Video-data check

`POST /analyze/video` takes a zip of video files in class folders. ffmpeg decodes 8 frames per clip; the image detector
runs on the frames' temporal mean, with the mean ResNet-18 frame embedding for the label and shift checks. Held-out
(UCF-101 testlist01 groups; docs/video-benchmark-findings.md): mean row AUPRC 0.53 at 1% and 0.75 at 5%; 46% / 86% of
attacked batches flagged; clean batches flagged 4.2% at LOW_RISK, 2.5% at SUSPICIOUS (nominal 1%). Clean-label video
triggers are not detected, and only static every-frame triggers were tested.

## Model-file scanner

Abstract pickle machine over the opcode stream (no import, no execution), deny list plus safe list plus
deny-by-default for unknown globals, nested-loader and dotted-name handling, per-format parsers, and a sandboxed
subprocess (`python -I`, resource limits) for the production path.

Verdicts: `DANGEROUS`, `REVIEW`, `NO_CODE_EXECUTION_FOUND`, `SAFE_FORMAT`.

Benchmark: 710 generated malicious files and 221 benign files (156 pinned Hugging Face files), compared with
picklescan 1.0.5 and modelscan 0.8.8. On the held-out split, first run: Veritas flagged 73% DANGEROUS (83% including
REVIEW) with 0 of 74 benign files DANGEROUS, picklescan 20%, modelscan 16%. The scanner was fixed after that run, so the
post-fix 339/339 is no longer a held-out number.

## Known limits

- **Small contamination is hard.** At 1% poison (about 12 rows in 1,200) veritas2 flags 42% of batches at LOW_RISK
  (cleanlab-style and the Nov 7 detector about 15-19% at a calibrated cut-off). Reliable batch verdicts need about 3-5%
  contamination; below that, use the ranked row and cell lists.
- **Weakest attacks at 1%:** covariate shift (row AUPRC 0.18; 0.61 at 5%) and feature collision (0.39).
- **Attack profile** names the closest of six known attacks; an attack of another kind will be given one of those names.
- **Data types:** tables (numeric and categorical columns, up to 200,000 rows validated) and labelled images (up to
  6,000 per check) audio clips (PCM WAV, up to 6,000, at most 10 s used) and video clips (8 frames each, up to 6,000). Not free text yet.
- **Images are checked at 32x32.** A trigger smaller than about 1/32 of the image side can vanish in the resize. Image
  results name the leading check, not an attack type; label flips at 1% are the weakest case (20% of batches flagged).
  Without a reference zip the reference is a random 70% of each class, so poison anywhere in the zip also lands in it.
- **You must supply a trusted reference.** With no reference file or split column, Veritas treats the first 70% of rows
  in file order as trusted and warns you. If the reference is itself poisoned, the check is meaningless.
- **Real-noise evidence is one dataset family** (CIFAR-10 embeddings).
- **The scanner's corpus was written by the same person who wrote the scanner.** No real-world malicious models are
  in it. Third-party-class pickles (for example `dabl`, `autogluon`) come back REVIEW (22% of held-out Hub files),
  because loading them runs that library. A gadget inside a safe-listed library is caught only if the deny list or a
  dangerous name catches it. A different red team would find different gaps.
- **A clean scan is not a guarantee.** `NO_CODE_EXECUTION_FOUND` means no known pattern was found; load models from
  untrusted sources in a sandbox anyway.

## Data handling

Uploaded files wait in blob storage only until the worker has analysed them, then are deleted. Results and history are
kept for `VERITAS_RETENTION_DAYS` (default 30) and are visible only to the owning account; an account can delete all
of its data at any time. See `docs/legal/privacy-policy.md` (draft).
