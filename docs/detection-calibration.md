# Detection thresholds and calibration status

The thresholds in `backend/app/detection/` and aggregation weights in
`backend/app/detection/pipeline.py` are **experimental heuristic defaults**.
They are not calibrated accuracy guarantees and must not be used to claim a
specific detection rate or regulatory compliance.

Before changing a threshold or making a performance claim:

1. Version a labelled clean/poisoned benchmark and record its provenance.
2. Keep a test split that is never used to select weights or thresholds.
3. Evaluate each layer and the combined verdict against documented baselines.
4. Report precision, recall, F1, PR-AUC, false-positive rate, random seed and
   confidence intervals per supported attack type.
5. Record the selected parameters, data version and code revision with the
   release/report.

Benchmark contract: `backend/research/benchmark/manifest.json` defines the
immutable CSV poisoning benchmark v1, its random seed, generator, package
versions, reference/evaluation split and attack injection counts. The evaluation
harness and checked-in metrics report must be completed before any threshold is
changed.

Until that process is completed, UI scores are risk signals for analyst review,
not proof that data or a model is poisoned.

The initial v1 report is a negative result (zero combined recall/F1). It is a
baseline for detector improvement, **not** authorization to lower thresholds
without a larger held-out benchmark and an explicitly selected operating policy.
