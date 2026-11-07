# Benchmarks

| Benchmark | Data | Entry point |
|---|---|---|
| v1 (`manifest.json`) | 1 synthetic dataset, one batch per attack. Kept for history; too small to rank detectors | (harness removed with the five-layer pipeline; see git tag `legacy-five-layer-pipeline`) |
| v2 (`manifest-v2.json`) | 18 real OpenML datasets (6 dev, 12 held-out test), 6 attacks × 4 poison rates × 5 seeds, plus clean batches | `python -m research.bench_v2` |

## Running v2

```bash
cd backend
pip install -r research/requirements.txt
PYTHONPATH=. python -m research.bench_v2 --profile smoke   # ~20 s, 2 datasets
PYTHONPATH=. python -m research.bench_v2 --label my-run    # full, ~15-30 min on 12 cores
```

Datasets download once to `~/.cache/veritas-bench` (override with `VERITAS_BENCH_CACHE`).
Results land in `results/<date>-benchmark-v2-<label>.{json,md}`.

## Rules

- Tune thresholds or detectors on **dev** datasets only; quote **test** numbers.
- Never edit `manifest-v2.json`; create a new `benchmark_id` instead.
- A new detector goes into `baselines.DETECTORS` and must beat the existing
  baselines on test before the app uses it.
