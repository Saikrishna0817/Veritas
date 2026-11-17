# Plan: making Veritas trustworthy (2026-11-15)

Today every number comes from attacks written by the same person who wrote the detector, planted in a few public
datasets, with no attacker trying to evade the checks. This plan closes those gaps in order of how much they could
change what Veritas is allowed to claim. Rules carried over from the benchmark work: success criteria and analysis
are written down before a run, tuning happens only on dev, held-out numbers are quoted once, and a manifest is never
edited in place (a change gets a new benchmark_id).

## Where things stand

| | With a reference | Without a reference |
|---|---|---|
| Use | New data checked against data you trust | Ranked lists of what to look at; no verdict |
| Finds | Label flips (from ~3%), triggers, duplicates, class noise; images also at 1% | Label flips, shared triggers |
| Misses | Clean-label triggers in audio/video, covariate shift in tables at 1%, moving video triggers | Anything that looks like ordinary data |
| Evidence | 12 table, 2 image, 2 audio, 1 video datasets; one real-noise family (CIFAR-10N) | Images held-out; tables in progress |

## Phase 0: finish what is running (done 2026-11-16)

Held-out runs for the within-class check (images v9, audio v3) and the rest of the no-reference table test, in
`~/.cache/veritas-logs/heldout_chain.sh`. Write up, adjust the app's limit texts only if the numbers support it.

**Result:** clean-label audio triggers now flagged on Speech Commands (19 of 20 batches), not on ESC-50; images
unchanged (already caught). App texts updated. Details in `benchmark-v2-findings.md`.

## Phase 1: attacks written by someone else

**Why first:** cheapest way to remove the "same author" bias, and it reuses the existing harnesses.

- Images: generate poisoned sets with BackdoorBench's published attack code (BadNets, Blended, SIG, Label-consistent,
  WaNet, Input-aware, SSBA, LF) on CIFAR-10 and GTSRB, at 1/3/5/10%.
- Tables: noise and poisoning from published generators (for example cleanlab's noise matrices for label flips; the
  feature-collision and backdoor recipes from published tabular-backdoor papers).
- New manifests (`manifest-mm-images-ext-v1`, `manifest-v2-ext-v1`); the detector is frozen at its current version and
  not tuned on these sets at all: they are a test, nothing more.

**Success criterion (set now):** report row AUPRC and batch detection per attack and rate with 95% intervals. Any
attack where batch detection at 5% poison is below 50% goes into the model card's "not detected" list by name.

**Needs:** permission to download BackdoorBench code and datasets (~1-2 GB). CPU only.

**Result (images, 2026-11-16):** five BackdoorBench attacks (BadNets, Blended, SIG, Low Frequency, FTrojan), all
80-100% of batches flagged at 5%, so none joins the "not detected" list. Low Frequency and FTrojan are found only
through their relabelling. WaNet, Input-aware, SSBA and label-consistent need trained generators and were not run;
tables not started. Details in `image-benchmark-findings.md`.

## Phase 2: real errors, not planted ones

**Why:** planted attacks may look nothing like real bad data.

- Human-verified label errors in published test sets (labelerrors.com, Northcutt et al. 2021): CIFAR-10, CIFAR-100,
  MNIST, AudioSet, QuickDraw. These are real mistakes confirmed by reviewers.
- Real noisy labels with clean ground truth: CIFAR-100N (alongside CIFAR-10N, already used).
- Measure: does the label list rank the verified errors first (precision in the top 1-5%, AUPRC), with and without a
  reference.

**Success criterion:** published as-is; this phase cannot be tuned on (no dev split exists for it).

**Needs:** permission for downloads (~0.5 GB for the image sets; AudioSet clips are on YouTube and only partly
available, so audio may be limited to the subset still online).

**Result (CIFAR-10, CIFAR-100, MNIST, 2026-11-16):** confirmed errors ranked above chance (AUROC 0.72-0.92), but the
app's list is 2-16% confirmed errors, with or without a reference. Model card updated. AudioSet, QuickDraw and
CIFAR-100N not run.

## Phase 3: an attacker who knows the checks (the biggest gap)

**Why:** a determined attacker reads the docs. A detector that only stops standard recipes gives false comfort.

1. **Victim model first.** Train a small CNN (CIFAR-10, 32x32, CPU, about 20-40 minutes per run) and a gradient-boosted
   model (tables) on poisoned data, to measure attack success rate (ASR): does the backdoor or targeted flip work?
   An evasive attack that no longer works is not a failure of the detector.
2. **Adaptive attacks, one per check family:**
   - label check: flip only rows the label model already finds ambiguous (near the decision boundary);
   - shared-pattern / within-class checks: low-frequency or warping triggers (low high-pass energy), spread over
     several classes; trigger strength reduced until the check stops seeing it;
   - shift and pixel checks: match the reference's per-feature statistics after poisoning;
   - tables: in-range triggers that respect the columns' correlations.
3. **Report** detection against ASR: for each attack, the trade-off curve (how much detection drops as the attack is
   tuned to evade, and what ASR remains).

**Success criterion (set now):** the claim Veritas may make is the detection rate at ASR >= 90%. If adaptive
attacks reach ASR >= 90% with detection near chance, the model card says so plainly for that attack family.

**Needs:** no downloads; most compute of all phases. Run as the capped background service (see the memory note), one
victim at a time.

**Result (images v1, 2026-11-16):** inconclusive. Only the faint patch reached ASR >= 0.9 (4 runs, all flagged). The
evasive settings found (clean-label low-frequency trigger, boundary flips at 1%) evade Veritas but do not work on the
small victim (poison is 1.7% of its training set). Next: a victim that learns backdoors from fewer images, as a new
benchmark_id. Tables not started.

## Phase 4: breadth

**Why:** 1-2 datasets per media type cannot show the numbers carry over.

- Images: SVHN, STL-10 (resized), a Tiny-ImageNet subset.
- Audio: UrbanSound8K, and a larger Speech Commands slice; separate dev and test datasets (today they come from the
  same two).
- Video: HMDB-51 (UCF-101 is the only one today); moving and brief triggers added to the attack set.
- Tables: 6-8 more OpenML datasets with different shapes (many categories, high dimension, heavy class imbalance).

**Success criterion:** headline numbers are re-quoted as ranges across datasets, not single values.

**Result so far (2026-11-16):** images (SVHN, Tiny-ImageNet): triggers, duplicates and class noise carry over; label
flips do not (row AUPRC 0.25-0.30 vs 0.72-0.82), and Tiny-ImageNet gives 5.8% clean alarms at SUSPICIOUS. Audio
(UrbanSound8K): between Speech Commands and ESC-50, clean alarms at nominal. Tables: run paused until the machine is
free. Video: HMDB-51 waits for the archive (official link down).

**Needs:** permission for downloads (several GB; HMDB-51 ~2 GB, UrbanSound8K ~6 GB).

## Phase 5: real users

**Why:** benchmarks do not show whether the lists are useful to a person deciding what to fix.

- A short pilot with 3-5 people who check their own data with Veritas.
- In the results view, a per-item "was this actually bad?" mark (the review endpoint already stores decisions);
  report precision of the top of each list from those marks, with the users' consent.
- Collect what users expected Veritas to find and it did not.

**Success criterion:** at least one dataset where users confirm items Veritas found; every reported miss becomes a
named limit or a new attack in the benchmark.

## Phase 6: make the claims checkable

- One command reproduces every headline number from the manifests (`make reproduce`), with the expected outputs and
  their intervals committed.
- Every result records the detector version and manifest it was measured against, so a report can be traced to the
  evidence behind it.
- The model card lists each claim with the run that supports it and links to the result files.

## Order and rough effort

| Phase | Effort | Downloads | Changes the claims |
|---|---|---|---|
| 0 Finish held-out runs | running | none | audio/image clean-label text |
| 1 Others' attacks | 2-3 sessions | yes, ~2 GB | likely adds named misses |
| 2 Real errors | 1-2 sessions | yes, ~0.5 GB+ | first real-world evidence |
| 3 Adaptive attacks | 4-6 sessions | none | the largest: what an informed attacker can do |
| 4 Breadth | 3-4 sessions | yes, several GB | ranges instead of single numbers |
| 5 Real users | ongoing | none | usefulness, real misses |
| 6 Checkable claims | 1 session | none | anyone can verify |

Phases 1 and 2 can start as soon as downloads are approved. Phase 3 needs a free machine (no other training) or a
remote one; it is the step that decides whether Veritas can be trusted against someone trying to beat it.
