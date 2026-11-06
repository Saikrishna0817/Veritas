"""Real-noise track: CIFAR-10N human labelling mistakes, scored like benchmark v2.

Each batch draws 3,000 images. The reference set keeps the original (trusted)
labels. The incoming set carries worker labels, with exactly ``rate`` of its rows
being genuine worker mistakes (worker label != original label) and the rest rows
the workers got right. Nothing is injected: the errors are the ones humans made.

Run from ``backend/``:  python -m research.bench_realnoise [--profile smoke]
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import date  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from research import noise_data  # noqa: E402
from research.baselines import DETECTORS  # noqa: E402
from research.bench_v2 import _row_metrics, detector_inputs  # noqa: E402

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "benchmark" / "manifest-realnoise.json"


def load_manifest(profile: str = "full") -> dict:
    m = json.loads(MANIFEST_PATH.read_text())
    if profile != "full":
        m.update(m["profiles"][profile])
    m["profile"] = profile
    return m


@lru_cache(maxsize=1)
def _data() -> dict:
    return noise_data.load()


def run_batch(spec: dict) -> dict:
    m, data = spec["manifest"], _data()
    rng = np.random.default_rng([m["random_seed"], spec["seed"], spec["source_index"] + 1,
                                 int(spec["rate"] * 1000), spec["rep"]])
    clean, worker = data["clean"], data[spec["source"]] if spec["source"] else data["clean"]
    wrong = np.flatnonzero(worker != clean)
    right = np.flatnonzero(worker == clean)

    n_rows = m["batch_rows"]
    n_ref = int(n_rows * m["reference_fraction"])
    n_inc = n_rows - n_ref
    n_wrong = round(spec["rate"] * n_inc) if spec["source"] else 0

    picked = rng.choice(len(clean), n_ref, replace=False)  # reference: any rows, trusted labels
    pool = np.setdiff1d(np.arange(len(clean)), picked)
    inc = np.concatenate([rng.choice(np.intersect1d(wrong, pool), n_wrong, replace=False),
                          rng.choice(np.intersect1d(right, pool), n_inc - n_wrong, replace=False)])
    inc = inc[rng.permutation(n_inc)]

    X_ref, y_ref = data["X"][picked].astype(float), clean[picked]
    X_inc, y_inc = data["X"][inc].astype(float), worker[inc]
    mask = y_inc != clean[inc]
    assert mask.sum() == n_wrong

    d = detector_inputs(X_ref, y_ref, X_inc, y_inc, 10, m["calibration_fraction_of_reference"])
    out = {k: spec[k] for k in ("source", "seed", "rate", "rep")}
    out.update({"n_incoming": n_inc, "n_poisoned": int(mask.sum()), "detectors": {}})
    limits = m.get("detector_seed_limits", {})
    for name, detect in DETECTORS.items():
        if spec["seed"] >= limits.get(name, m["seeds"]) or name not in (spec.get("only") or DETECTORS):
            continue  # too slow to run on every seed
        started = time.perf_counter()
        rows, batch = detect(d)
        res = {"batch_score": batch, "seconds": round(time.perf_counter() - started, 3)}
        if mask.any():
            res.update(_row_metrics(np.asarray(rows, dtype=float), mask))
        out["detectors"][name] = res
    return out


def batch_specs(m: dict) -> list[dict]:
    specs = []
    for seed in range(m["seeds"]):
        base = {"manifest": m, "seed": seed}
        specs += [{**base, "source": None, "source_index": -1, "rate": 0.0, "rep": r}
                  for r in range(m["clean_batches_per_seed"])]
        specs += [{**base, "source": src, "source_index": i, "rate": rate, "rep": 0}
                  for i, src in enumerate(m["noise_sources"]) for rate in m["poison_rates"]]
    return specs


def _ci(records: list[dict], stat, m: dict) -> list[float]:
    """[point, lo, hi], bootstrapping over seeds."""
    seeds = sorted({r["seed"] for r in records})
    by_seed = {s: [r for r in records if r["seed"] == s] for s in seeds}
    rng = np.random.default_rng(0)
    boots = [stat([r for s in rng.choice(seeds, len(seeds)) for r in by_seed[s]])
             for _ in range(m["metrics"]["ci"]["resamples"])]
    boots = [b for b in boots if b == b]
    a = (1 - m["metrics"]["ci"]["level"]) / 2
    return [round(stat(records), 4), round(float(np.quantile(boots, a)), 4), round(float(np.quantile(boots, 1 - a)), 4)]


def aggregate(records: list[dict], m: dict) -> dict:
    fpr = m["metrics"]["batch"]["tpr_at_clean_fpr"]
    summary = {}
    for det in [d for d in DETECTORS if any(d in r["detectors"] for r in records)]:
        ran = [r for r in records if det in r["detectors"]]
        clean = [r["detectors"][det]["batch_score"] for r in ran if r["source"] is None]
        summary[det] = {"_clean_batch_threshold": round(float(np.quantile(clean, 1 - fpr)), 4),
                        "_n_batches": len(ran),
                        "_mean_seconds": round(float(np.mean([r["detectors"][det]["seconds"] for r in ran])), 3)}
        for src in m["noise_sources"]:
            for rate in m["poison_rates"]:
                cell = [r for r in ran if r["source"] == src and r["rate"] == rate]

                def mean_of(metric, det=det):
                    return lambda rs: float(np.mean([r["detectors"][det][metric] for r in rs
                                                     if r["source"] == src and r["rate"] == rate]))

                def tpr(rs, det=det, src=src, rate=rate):
                    pooled = [r["detectors"][det]["batch_score"] for r in rs if r["source"] is None]
                    hit = [r["detectors"][det]["batch_score"] for r in rs if r["source"] == src and r["rate"] == rate]
                    return float(np.mean(np.asarray(hit) > np.quantile(pooled, 1 - fpr)))

                summary[det].setdefault(src, {})[str(rate)] = {
                    **{mt: _ci(ran, mean_of(mt), m) for mt in m["metrics"]["row"]},
                    "batch_tpr": _ci(ran, tpr, m), "n_batches": len(cell)}
    return summary


def to_markdown(report: dict) -> str:
    m, summary = report["manifest"], report["summary"]
    rate = "0.05" if "0.05" in str(m["poison_rates"]) else str(m["poison_rates"][0])
    lines = [f"# {m['benchmark_id']} — {report['label']}", "",
             f"{report['n_batches']} batches, {m['seeds']} seeds, mistake rate {rate}. "
             "Cells: mean [95% CI, bootstrap over seeds]. Random ranking scores the mistake rate.", ""]
    for title, metric in (("Row level: AUPRC", "auprc"), ("Row level: precision@k", "precision_at_k"),
                          ("Batch level: TPR at 5% clean-batch FPR", "batch_tpr")):
        lines += [f"### {title}", "", "| detector | " + " | ".join(m["noise_sources"]) + " | sec/batch |",
                  "|" + "---|" * (len(m["noise_sources"]) + 2)]
        for det, cells in summary.items():
            vals = [cells[s][rate][metric] for s in m["noise_sources"]]
            lines.append(f"| {det} | " + " | ".join(f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for v in vals)
                         + f" | {cells['_mean_seconds']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="full")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--label", default=None)
    parser.add_argument("--only", nargs="+", choices=list(DETECTORS), help="score just these detectors")
    args = parser.parse_args()

    m = load_manifest(args.profile)
    _data()  # build/load the embedding cache once before forking workers
    specs = [{**spec, "only": args.only} for spec in batch_specs(m)]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        records = []
        for i, rec in enumerate(pool.map(run_batch, specs, chunksize=2), 1):
            records.append(rec)
            if i % 10 == 0 or i == len(specs):
                print(f"{i}/{len(specs)} batches, {time.perf_counter() - started:.0f}s", flush=True)
    label = args.label or args.profile
    report = {"manifest": m, "label": label, "n_batches": len(records),
              "runtime_seconds": round(time.perf_counter() - started, 1),
              "summary": aggregate(records, m), "batches": records}
    out = ROOT / "results" / f"{date.today().isoformat()}-benchmark-realnoise-{label}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(to_markdown(report) + "\n")
    print(out.with_suffix(".md"))


if __name__ == "__main__":
    main()
