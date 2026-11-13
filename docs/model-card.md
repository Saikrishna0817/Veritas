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
each value is in its column (label cell: the label check), so a report can name the poisoned values.

**Evaluation.** Benchmark v2: 18 real OpenML datasets (6 dev, 12 held-out test), six injected attacks, four poison
rates, five seeds, plus clean batches. Detector design and tuning used the dev datasets only. CIFAR-10N (real human
label mistakes, nothing injected) is a second held-out track.

| Held-out result | old Veritas | veritas2 (Nov 7) | veritas2 (Nov 13) | cleanlab | two-sample test |
|---|---|---|---|---|---|
| row AUPRC at 5% poison (random = 0.05) | 0.11 | 0.69 | **0.78** | 0.54 | 0.49 |
| batch detection rate at 5% false alarms | 0.33 | 0.78 | **0.84** | 0.55 | 0.14 |

By poison rate (batch detection at 5% false alarms), Nov 13: 1% 0.44, 3% 0.76, 5% 0.84, 10% 0.94 (Nov 7: 0.14, 0.61,
0.74, 0.87). Cell AUPRC (finding the changed values and labels) 0.35 at 1%, 0.60 at 5%. The CIFAR-10N comparison below
was run on the Nov 7 detector. On real CIFAR-10N
label noise `veritas2` matches cleanlab (row AUPRC 0.45 to 0.57 vs 0.46 to 0.58); it does not beat it.

**Calibration.** On held-out clean batches 3.3% reached LOW_RISK (nominal 5%, n = 240), 1.3% reached SUSPICIOUS (nominal 1%)
and none reached CONFIRMED (Nov 7 detector: 7.1%, 0.8%, 0).

**Dependency check.** After the numpy 2.5 / pandas 3.1 upgrade the 12,000 per-batch results were identical to the
earlier run (see findings doc).

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

- **Small contamination is hard.** At 1% poison (about 12 rows in 1,200) veritas2 flags 37% of batches at LOW_RISK
  (cleanlab-style and the Nov 7 detector about 15-19% at a calibrated cut-off). Reliable batch verdicts need about 3-5%
  contamination; below that, use the ranked row and cell lists.
- **Weakest attack:** covariate shift (row AUPRC 0.04 at 1%, 0.53 at 5%).
- **Data types:** structured numeric/categorical tables with a label column. Not images, audio, video or text
  (except as pre-computed embeddings), and not validated beyond 200,000 rows.
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
