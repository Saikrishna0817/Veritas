# Model-file scanner benchmark (2026-11-07)

Compares the Veritas scanner (`backend/app/scanner/model_scan.py`) with **picklescan 1.0.5** and **modelscan 0.8.8**.
Everything is reproducible: `python -m research.scanner_bench.build_corpus`, then
`python -m research.scanner_bench.run_scanner_bench --comparator-python <python with modelscan+picklescan>`.
Raw results: `backend/research/results/2026-11-07-scanner-bench-*.{md,json}`. Files are scanned, never loaded.

## Corpus
- **Malicious (710 files, generated, never downloaded):** 100+ payload families (code execution, file/network/system access,
  library gadgets, dotted-attribute bypasses, nested pickles, hand-built opcode shapes), each wrapped in up to 7 containers
  (raw pickle protocol 0 and 4, trojaned sklearn model, payload appended after a clean pickle, gzip/xz-wrapped, PyTorch zip),
  plus object-dtype `.npy/.npz`, zips with executables, Keras Lambda, hostile ONNX and malformed safetensors. All payload
  commands are inert strings.
- **Benign (221 files):** locally trained sklearn (25 estimator types, joblib in 6 compressions), torch (state dicts, checkpoints,
  full modules, TorchScript), numpy, safetensors, ONNX, and 156 real public files from the Hugging Face Hub pinned by revision
  and SHA-256 in `hf_selection.json` (sklearn pickles/joblib, ONNX, GGUF, PyTorch `.bin/.pt`, safetensors).
- **Split:** `dev` = used while building the scanner. `heldout` = 13 payload families I wrote **after** the scanner's rules were
  fixed on the dev set (dotted paths through allowed libraries, file/exit hooks, a nested loader, ~40 more stdlib operations)
  plus 74 other Hugging Face files chosen further down the download ranking. The held-out set was run once before any
  rule was changed for it (`r2-before-heldout-fixes`).

## Results

| | malicious caught | benign flagged |
|---|---|---|
| **Held-out, first run (honest number)** | | |
| Veritas, DANGEROUS | 248/339 (73%) | 0/74 |
| Veritas, DANGEROUS or REVIEW | 283/339 (83%) | 17/74 (23%) |
| picklescan | 68/339 (20%) | 0/74 |
| modelscan | 53/339 (16%) | 0/74 |
| **Dev (rules were tuned on this)** | | |
| Veritas, DANGEROUS | 368/371 (99%) | 1/147 |
| Veritas, DANGEROUS or REVIEW | 371/371 (100%) | 5/147 (3%) |
| picklescan | 177/371 (48%) | 1/147 |
| modelscan | 135/371 (36%) | 1/147 |

Held-out misses on the first run (all fixed afterwards, tests added): dotted/aliased names through allowed libraries
(`sklearn.utils` + `os.system`, 5 families, 35 files, **not even flagged for review**), `torch.storage._load_from_bytes` wrapping
a nested pickle, `atexit.register`, `codecs.open`, `io.open`. After the fixes the same held-out run is 339/339, but that is
**not** a held-out number any more. The only dev misses are by design: a custom ONNX operator domain and two malformed
safetensors are REVIEW, not DANGEROUS.

Like for like (raw pickle variants only, where the other tools are strongest): Veritas 212/212 dev and 144/196 held-out on the
first run; picklescan 144/212 and 56/196; modelscan 112/212 and 44/196. Much of the gap on the full corpus comes from formats
the other tools do not look inside (gzip/xz-wrapped joblib, ONNX, safetensors, zips with executables); on raw pickles it is
mostly their deny lists (they miss `importlib`, `platform.popen`, `marshal`, `torch.hub`, `numpy.load`, ...).

Speed: 931 files in about 1.8 s for Veritas in-process (0.9 s picklescan, 0.8 s modelscan); the production path adds a subprocess.

## What this does and does not show
- I wrote the malicious corpus and the scanner, so the payload ideas overlap. The held-out split reduces that bias, it does
  not remove it: a different red team would find different holes. The first held-out run found a serious one (dotted names).
- No **real-world** malicious models are in the corpus (the public ones are mostly gone from the Hub and handling them adds
  risk). Real attacks are mostly os/subprocess/builtins payloads, which every tool catches; the harder cases here are synthetic.
- **Benign REVIEW rate is the cost of a deny-by-default design:** 22% of held-out Hub files are REVIEW, almost all sklearn
  pickles that import third-party classes (`dabl`, `autogluon`, `imodels`, `networkx`, `ultralytics`, `__main__`). A pickle of a
  third-party class runs that library on load, so the scanner cannot vouch for it. Zero benign files were called DANGEROUS on the
  held-out set; one dev file was (an Ultralytics YOLOv8 `.pt` importing `__builtin__.getattr`, which picklescan and modelscan also flag).
- Safe-list gaps remain by construction: a gadget inside a library on the safe list (sklearn, scipy, torch.nn, xgboost, pandas
  internals) is only caught if it is on the deny list or reaches a dangerous name. The held-out run shows that list is incomplete.
- Not covered: HDF5/Keras `.h5` (reported REVIEW), TensorFlow SavedModel, pickles larger than the sandbox limits, and attacks on the
  scanner itself beyond the resource limits (a crash or timeout is DANGEROUS).

Roadmap exit criteria for the scanner ("100% of generated malicious files flagged, 0 false positives on a benign corpus, at least
parity with modelscan") are met only on the post-fix run. On the honest first held-out run: 73% DANGEROUS, 0 benign DANGEROUS,
better than modelscan and picklescan by a wide margin.
