# Veritas detection model card

## Scope

Veritas v2.2 produces experimental, batch-level risk signals for structured
numeric CSV data. It is not a certified security product, a compliance
certification mechanism, or evidence that an individual row/model is poisoned.

## Benchmark v1 results

`backend/research/results/2026-09-02-benchmark-v1-eval-r3.json` is detector fix
and calibration revision 3. It follows the revision-2 harness correction: its
ground truth remains benchmark-owned, and its reference partition remains
independent of the upload path's row-order split.

Revision 3 scores Layer 3 only on incoming rows after fitting it to the clean
reference set. It also removes the unsupported extra 0.02 Layer-3 gate margin:
the Step 1 benchmark measured 0.00 clean Layer-3 flags and a 0.06 incoming
boiling-frog flag rate, which the former gate suppressed by 80%. The corrected
boiling-frog Layer-3 score is 0.0333 and its combined score is 0.0523; neither
reaches the 0.15 LOW_RISK verdict threshold. Accordingly, the final benchmark
still reports precision, recall and F1 of 0.00 for every incident-level layer
and for the combined pipeline. The final clean-batch false-positive rate is
0.00.

The pre-registered revision-3 policy holds existing thresholds and weights
fixed, rejects any configuration with clean-batch false-positive rate above
2%, and reports F1 without post-hoc threshold selection. It meets that safety
constraint, but it does **not** establish effective incident detection: label
flip, backdoor, clean-label, gradient poisoning, and boiling frog all have zero
incident-level recall at this operating point. Layer 3 retains row-level signal
for backdoor (precision 1.00, recall 0.20), gradient poisoning (precision 1.00,
recall 0.083), and boiling frog (precision 1.00, recall 0.48), which is not yet
strong enough for the batch aggregate.

## Limitations

The benchmark is synthetic, contains one incident batch per attack, and does
not establish performance on real-world data, models, images, audio or video.
NIST AI RMF is a voluntary risk-management framework; this report is not a
claim of NIST or EU AI Act compliance.
