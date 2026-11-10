# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
- **ML / MLOps engineers** checking a labelled tabular dataset against a trusted reference before training, and scanning model files (pickle, joblib, PyTorch, safetensors, ONNX, GGUF, NumPy) before loading them.
- **Security analysts** triaging suspected data poisoning: reviewing flagged rows, the human-in-the-loop queue, incidents and evidence reports.

## Product Purpose
Veritas is a triage aid. It ranks incoming rows by how likely their label is wrong or their features were tampered with, gives one calibrated batch verdict (CLEAN / LOW_RISK / SUSPICIOUS / CONFIRMED_POISONED), and statically scans model files for anything that can run code on load (DANGEROUS / REVIEW / NO_CODE_EXECUTION_FOUND / SAFE_FORMAT). Success: the user knows which rows to review first and which files not to load.

## Positioning
Benchmarked, not claimed: on 12 held-out OpenML datasets at 5% poison, row AUPRC 0.69 vs cleanlab 0.54; batch detection 0.78 at 5% false alarms. Model files are never loaded, only read as bytes in a sandbox.

## Constraints
- Name in UI: **Veritas**.
- Frontend only; the backend API (`/api/v1`, `/ws/v1/detection-stream`, cookie session + CSRF) is fixed.
- Honesty: it is not a certification or proof of attack. Simulated/heuristic features (SOC, forensics narratives, attack-type classification) are labelled research preview.
- Small contamination (~1%) is hard: verdicts need ~5%+; below that use the ranked rows.
