"""Benchmark v2: real OpenML datasets, injected attacks, baselines, CIs.

Each batch shuffles one dataset, splits it into reference / incoming rows,
injects one attack into the incoming rows (or none, for clean batches) and
scores it with every detector in ``research.baselines``. Ground truth is the
injection mask, owned here and never shown to a detector.

Run from ``backend/``:  python -m research.bench_v2 --profile smoke
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import date  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research.baselines import DETECTORS  # noqa: E402

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "benchmark" / "manifest-v2.json"
CACHE = Path(os.getenv("VERITAS_BENCH_CACHE", Path.home() / ".cache" / "veritas-bench"))
MIN_CLASS_ROWS = 10


def load_manifest(profile: str = "full") -> dict:
    m = json.loads(MANIFEST_PATH.read_text())
    if profile != "full":
        p = m["profiles"][profile]
        keep = set(p["datasets"])
        m["datasets"] = {s: [d for d in ds if d["data_id"] in keep] for s, ds in m["datasets"].items()}
        m.update({k: v for k, v in p.items() if k != "datasets"})
    m["profile"] = profile
    return m


# ── Data ─────────────────────────────────────────────────────────────────────

def _openml(data_id: int):
    """fetch_openml's default ARFF parser breaks on categorical columns under pandas 3.1; liac-arff does not."""
    try:
        return fetch_openml(data_id=data_id, data_home=str(CACHE), as_frame=True, parser="auto")
    except TypeError:
        return fetch_openml(data_id=data_id, data_home=str(CACHE), as_frame=True, parser="liac-arff")


@lru_cache(maxsize=None)
def load_dataset(data_id: int, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Numeric matrix + integer labels. Categoricals become codes, NaNs the median."""
    bunch = _openml(data_id)
    X = bunch.data.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    X = X.astype(float)
    X = X.fillna(X.median()).loc[:, lambda f: f.nunique() > 1]
    y = pd.Series(pd.factorize(bunch.target)[0])
    keep = y.map(y.value_counts()).to_numpy() >= MIN_CLASS_ROWS
    X, y = X.to_numpy()[keep], pd.factorize(y[keep])[0]
    if len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X, y, bunch.details["name"]


def data_hash(X: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X).tobytes() + y.astype(np.int64).tobytes()).hexdigest()[:16]


# ── Attacks: mutate the incoming (X, y) in place, return poisoned row indices ─

def _pick(rng, candidates: np.ndarray, n: int) -> np.ndarray:
    return rng.choice(candidates, size=min(n, len(candidates)), replace=False)


def label_flip_random(X, y, rng, n, ctx):
    idx = _pick(rng, np.arange(len(y)), n)
    k = ctx["n_classes"]
    y[idx] = (y[idx] + rng.integers(1, k, len(idx))) % k
    return idx


def label_flip_targeted(X, y, rng, n, ctx):
    idx = _pick(rng, np.flatnonzero(y == ctx["major"]), n)
    y[idx] = ctx["minor"]
    return idx


def _backdoor(X, y, rng, n, ctx, values):
    idx = _pick(rng, np.flatnonzero(y != ctx["minor"]), n)
    X[np.ix_(idx, ctx["trigger"])] = values[ctx["trigger"]]
    y[idx] = ctx["minor"]
    return idx


def backdoor_ood(X, y, rng, n, ctx):
    """Trigger values outside the reference range, label set to the target class."""
    return _backdoor(X, y, rng, n, ctx, ctx["ref_max"] + ctx["ref_std"])


def backdoor_inrange(X, y, rng, n, ctx):
    """Trigger at each feature's 95th percentile: every value looks normal on its own."""
    return _backdoor(X, y, rng, n, ctx, ctx["ref_p95"])


def feature_collision(X, y, rng, n, ctx):
    """Clean-label: correctly labelled majority rows moved next to one victim row."""
    victims = np.flatnonzero(y != ctx["major"])
    victim = X[rng.choice(victims)].copy()
    idx = _pick(rng, np.flatnonzero(y == ctx["major"]), n)
    X[idx] = victim + 0.1 * (X[idx] - victim)
    return idx


def covariate_shift(X, y, rng, n, ctx):
    """A corrupted source: 30% of features shifted by one reference std dev."""
    idx = _pick(rng, np.arange(len(y)), n)
    cols = ctx["shift_cols"]
    X[np.ix_(idx, cols)] += ctx["ref_std"][cols]
    return idx


ATTACKS = {f.__name__: f for f in (label_flip_random, label_flip_targeted, backdoor_ood,
                                   backdoor_inrange, feature_collision, covariate_shift)}


# ── One batch ────────────────────────────────────────────────────────────────

def _row_metrics(scores: np.ndarray, mask: np.ndarray) -> dict:
    k = int(mask.sum())
    top = np.argsort(-scores, kind="stable")[:k]
    if np.ptp(scores) == 0:  # constant scores carry no ranking: chance level
        return {"auroc": 0.5, "auprc": float(mask.mean()), "precision_at_k": float(mask.mean())}
    return {"auroc": float(roc_auc_score(mask, scores)),
            "auprc": float(average_precision_score(mask, scores)),
            "precision_at_k": float(mask[top].mean())}


def detector_inputs(X_ref, y_ref, X_inc, y_inc, n_classes: int, cal_fraction: float) -> dict:
    """What every detector sees: fit/calibration slices of the reference set plus the incoming batch."""
    n_cal = int(len(X_ref) * cal_fraction)
    scaler = StandardScaler().fit(X_ref[n_cal:])
    return {"X_fit": scaler.transform(X_ref[n_cal:]), "y_fit": y_ref[n_cal:],
            "X_cal": scaler.transform(X_ref[:n_cal]), "y_cal": y_ref[:n_cal],
            "X_inc": scaler.transform(X_inc), "y_inc": y_inc, "n_classes": n_classes,
            "X_ref_raw": X_ref, "y_ref": y_ref, "X_inc_raw": X_inc}


def run_batch(spec: dict) -> dict:
    m = spec["manifest"]
    X, y, name = load_dataset(spec["data_id"], m["max_rows"], m["random_seed"] + spec["seed"])
    rng = np.random.default_rng([m["random_seed"], spec["data_id"], spec["seed"],
                                 spec["attack_index"] + 1, int(spec["rate"] * 1000), spec["rep"]])
    perm = rng.permutation(len(y))
    n_ref = int(len(y) * m["reference_fraction"])
    ref, inc = perm[:n_ref], perm[n_ref:]
    X_ref, y_ref = X[ref], y[ref]
    X_inc, y_inc = X[inc].copy(), y[inc].copy()

    counts = np.bincount(y_ref, minlength=y.max() + 1)
    order = np.argsort(-counts, kind="stable")
    n_feat = X.shape[1]
    ctx = {"n_classes": int(y.max() + 1), "major": int(order[0]), "minor": int(order[1]),
           "ref_max": X_ref.max(axis=0), "ref_std": X_ref.std(axis=0), "ref_p95": np.percentile(X_ref, 95, axis=0),
           "trigger": rng.choice(n_feat, size=min(3, n_feat), replace=False),
           "shift_cols": rng.choice(n_feat, size=max(1, int(0.3 * n_feat)), replace=False)}

    mask = np.zeros(len(inc), dtype=bool)
    if spec["attack"]:
        n_poison = max(1, round(spec["rate"] * len(inc)))
        mask[ATTACKS[spec["attack"]](X_inc, y_inc, rng, n_poison, ctx)] = True

    d = detector_inputs(X_ref, y_ref, X_inc, y_inc, ctx["n_classes"], m["calibration_fraction_of_reference"])

    out = {k: spec[k] for k in ("split", "data_id", "seed", "attack", "rate", "rep")}
    out.update({"dataset": name, "data_hash": data_hash(X, y), "n_incoming": len(inc),
                "n_poisoned": int(mask.sum()), "detectors": {}})
    limits = m.get("detector_seed_limits", {})
    for det_name, detect in DETECTORS.items():
        if spec["seed"] >= limits.get(det_name, m["seeds"]) or det_name not in (spec.get("only") or DETECTORS):
            continue  # too slow to run on every seed; see detector_seed_limits in the manifest
        started = time.perf_counter()
        rows, batch = detect(d)
        res = {"batch_score": batch, "seconds": round(time.perf_counter() - started, 3)}
        if mask.any():
            res.update(_row_metrics(np.asarray(rows, dtype=float), mask))
        out["detectors"][det_name] = res
    return out


def batch_specs(m: dict) -> list[dict]:
    specs = []
    for split, datasets in m["datasets"].items():
        for ds in datasets:
            for seed in range(m["seeds"]):
                base = {"manifest": m, "split": split, "data_id": ds["data_id"], "seed": seed}
                specs += [{**base, "attack": None, "attack_index": -1, "rate": 0.0, "rep": r}
                          for r in range(m["clean_batches_per_seed"])]
                specs += [{**base, "attack": a, "attack_index": i, "rate": rate, "rep": 0}
                          for i, a in enumerate(m["attacks"]) for rate in m["poison_rates"]]
    return specs


# ── Aggregation ──────────────────────────────────────────────────────────────

def _cluster_ci(records: list[dict], stat, resamples: int, level: float, seed: int = 0) -> list[float]:
    """[point, lo, hi]: resample whole datasets, since batches of one dataset are correlated."""
    by_ds: dict[int, list[dict]] = {}
    for r in records:
        by_ds.setdefault(r["data_id"], []).append(r)
    keys = list(by_ds)
    rng = np.random.default_rng(seed)
    point = stat(records)
    boots = [stat([r for k in rng.choice(keys, len(keys)) for r in by_ds[k]]) for _ in range(resamples)]
    boots = [b for b in boots if b == b]  # drop NaN resamples
    alpha = (1 - level) / 2
    return [round(point, 4), round(float(np.quantile(boots, alpha)), 4), round(float(np.quantile(boots, 1 - alpha)), 4)]


def _tpr_at_fpr(fpr: float, det: str, attack: str, rate: float):
    def stat(records):
        clean = [r["detectors"][det]["batch_score"] for r in records if r["attack"] is None and det in r["detectors"]]
        hit = [r["detectors"][det]["batch_score"] for r in records
               if r["attack"] == attack and r["rate"] == rate and det in r["detectors"]]
        if not clean or not hit:
            return float("nan")
        return float(np.mean(np.asarray(hit) > np.quantile(clean, 1 - fpr)))
    return stat


def _mean_metric(det: str, attack: str, rate: float, metric: str):
    def stat(records):
        vals = [r["detectors"][det][metric] for r in records
                if r["attack"] == attack and r["rate"] == rate and det in r["detectors"]]
        return float(np.mean(vals)) if vals else float("nan")
    return stat


def aggregate(records: list[dict], m: dict) -> dict:
    ci = m["metrics"]["ci"]
    fpr = m["metrics"]["batch"]["tpr_at_clean_fpr"]
    dets = list(dict.fromkeys(d for r in records for d in r["detectors"]))  # whatever the run scored, first-seen order
    summary = {}
    for split in m["datasets"]:
        recs = [r for r in records if r["split"] == split]
        if not recs:
            continue
        s = summary[split] = {}
        for det in dets:
            s[det] = {}
            for attack in m["attacks"]:
                for rate in m["poison_rates"]:
                    cell = {mt: _cluster_ci(recs, _mean_metric(det, attack, rate, mt), ci["resamples"], ci["level"])
                            for mt in m["metrics"]["row"]}
                    cell["batch_tpr"] = _cluster_ci(recs, _tpr_at_fpr(fpr, det, attack, rate),
                                                    ci["resamples"], ci["level"])
                    s[det].setdefault(attack, {})[str(rate)] = cell
            ran = [r for r in recs if det in r["detectors"]]
            s[det]["_n_batches"] = len(ran)
            clean = [r["detectors"][det]["batch_score"] for r in ran if r["attack"] is None]
            s[det]["_clean_batch_threshold"] = round(float(np.quantile(clean, 1 - fpr)), 4)
            s[det]["_mean_seconds"] = round(float(np.mean([r["detectors"][det]["seconds"] for r in ran])), 3)
    return summary


def to_markdown(report: dict) -> str:
    m, summary = report["manifest"], report["summary"]
    split = "test" if "test" in summary else next(iter(summary))
    rate = "0.05" if "0.05" in summary[split][next(iter(summary[split]))][m["attacks"][0]] else str(m["poison_rates"][0])
    fpr = m["metrics"]["batch"]["tpr_at_clean_fpr"]
    lines = [f"# {m['benchmark_id']} — {report['label']}", "",
             f"Profile `{m['profile']}`, {report['n_batches']} batches, split **{split}**, poison rate {rate}. "
             f"Cells: mean [95% CI, cluster bootstrap over datasets].", ""]

    def table(title, metric, note):
        head = "| detector | " + " | ".join(m["attacks"]) + " | sec/batch |"
        rows = [f"### {title}", "", note, "", head, "|" + "---|" * (len(m["attacks"]) + 2)]
        for det, cells in summary[split].items():
            vals = [cells[a][rate][metric] for a in m["attacks"]]
            rows.append(f"| {det} | " + " | ".join(f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for v in vals)
                        + f" | {cells['_mean_seconds']} |")
        return rows + [""]

    lines += table("Row level: AUPRC", "auprc", f"Random ranking scores {rate}.")
    lines += table("Row level: precision@k (k = number of poisoned rows)", "precision_at_k",
                   f"Random ranking scores {rate}.")
    lines += table(f"Batch level: TPR at {fpr:.0%} clean-batch FPR", "batch_tpr",
                   f"Share of attacked batches scored above the {1 - fpr:.0%} quantile of clean batches.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="full")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--label", default=None)
    parser.add_argument("--only", nargs="+", choices=list(DETECTORS), help="score just these detectors")
    parser.add_argument("--splits", nargs="+", default=None, help="restrict to these dataset splits (dev/test)")
    args = parser.parse_args()

    m = load_manifest(args.profile)
    for datasets in m["datasets"].values():  # download once, before workers race for the cache
        for ds in datasets:
            _openml(ds["data_id"])
    if args.splits:
        m["datasets"] = {k: v for k, v in m["datasets"].items() if k in args.splits}
    specs = [{**spec, "only": args.only} for spec in batch_specs(m)]
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(pool.map(run_batch, specs, chunksize=4), 1):
            records.append(rec)
            if i % 50 == 0 or i == len(specs):
                print(f"{i}/{len(specs)} batches, {time.perf_counter() - started:.0f}s", flush=True)

    label = args.label or args.profile
    report = {"manifest": m, "label": label, "n_batches": len(records),
              "runtime_seconds": round(time.perf_counter() - started, 1),
              "summary": aggregate(records, m), "batches": records}
    out = ROOT / "results" / f"{date.today().isoformat()}-benchmark-v2-{label}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(to_markdown(report) + "\n")
    print(out.with_suffix(".md"))


if __name__ == "__main__":
    main()
