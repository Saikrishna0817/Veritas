"""Image poisoning benchmark: real datasets, attacks on raw pixels, detectors on embeddings.

Batches are prepared in this process (attacks are cheap, only altered images are re-embedded, on the GPU if there
is one); the detectors then run in a process pool. Aggregation and the report are shared with ``bench_v2``.

Run from ``backend/``:  python -m research.bench_mm --profile smoke
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research.baselines import DETECTORS  # noqa: E402
from research.bench_v2 import _row_metrics, aggregate, detector_inputs, to_markdown  # noqa: E402
from research.mm import datasets  # noqa: E402
from research.mm import ext_attacks  # noqa: E402
from research.mm.attacks import ATTACKS, make_context  # noqa: E402
from research.mm.detector import check_detector, mm_lead, pixel_c2st, veritas_mm  # noqa: E402
from research.mm.embed import clean_embeddings, embed_images  # noqa: E402
from research.mm.pixel import pixel_features  # noqa: E402
from research import detector_commit  # noqa: E402

ALL_DETECTORS = {**DETECTORS, "pixel_c2st": pixel_c2st, "veritas_mm": veritas_mm, "mm_lead": mm_lead,
                 **{f"mm_{n}": check_detector(n) for n in ("label", "shift", "pixel", "dup", "cell", "resid", "chroma", "sig", "sig_smooth")}}

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "benchmark" / os.getenv("VERITAS_MM_MANIFEST", "manifest-mm-images-v2.json")


def load_manifest(profile: str = "full") -> dict:
    m = json.loads(MANIFEST_PATH.read_text())
    if profile != "full":
        p = m["profiles"][profile]
        keep = set(p["datasets"])
        m["datasets"] = {s: [d for d in ds if d["data_id"] in keep] for s, ds in m["datasets"].items()}
        m.update({k: v for k, v in p.items() if k != "datasets"})
    m["profile"] = profile
    return m


def prepare_batch(spec: dict, ds: datasets.ImageSet, emb: np.ndarray, pix: np.ndarray) -> dict:
    """Attack the incoming pixels, re-embed what changed, return what a detector (and the scorer) needs."""
    m = spec["manifest"]
    rng = np.random.default_rng([m["random_seed"], zlib_id(spec["data_id"]), spec["seed"],
                                 spec["attack_index"] + 1, int(spec["rate"] * 1000), spec["rep"]])
    sel = rng.permutation(len(ds.y))[:m["batch_rows"]]
    n_ref = int(m["batch_rows"] * m["reference_fraction"])
    ref, inc = sel[:n_ref], sel[n_ref:]
    raw, y = ds.raw[inc].copy(), ds.y[inc].copy()
    E_inc, P_inc = emb[inc].copy(), pix[inc].copy()
    mask = np.zeros(len(inc), dtype=bool)
    if spec["attack"]:
        ctx = make_context(raw, ds.n_classes, rng)
        n_poison = max(1, round(spec["rate"] * len(inc)))
        idx = {**ATTACKS, **ext_attacks.ATTACKS}[spec["attack"]](raw, y, rng, n_poison, ctx)
        mask[idx] = True
        E_inc[idx], P_inc[idx] = embed_images(raw[idx]), pixel_features(raw[idx])
    pca = PCA(m["pca_dims"], random_state=0).fit(emb[ref])
    d = detector_inputs(pca.transform(emb[ref]), ds.y[ref], pca.transform(E_inc), y, ds.n_classes,
                        m["calibration_fraction_of_reference"])
    n_cal = int(len(ref) * m["calibration_fraction_of_reference"])
    scaler = StandardScaler().fit(pix[ref][n_cal:])  # pixel views follow the same fit/cal split as the embedding
    d.update({"P_cal": scaler.transform(pix[ref][:n_cal]), "P_inc": scaler.transform(P_inc),
              "R_cal": ds.raw[ref][:n_cal], "R_inc": raw,
              "P_fit": scaler.transform(pix[ref][n_cal:]), "R_fit": ds.raw[ref][n_cal:]})
    return {"d": d, "mask": mask}


def zlib_id(name: str) -> int:
    import zlib
    return zlib.crc32(name.encode())


def score_batch(job: dict) -> dict:
    spec, d, mask = job["spec"], job["d"], job["mask"]
    out = {k: spec[k] for k in ("split", "data_id", "seed", "attack", "rate", "rep")}
    out.update({"n_incoming": len(mask), "n_poisoned": int(mask.sum()), "detectors": {}})
    for name in spec["manifest"]["detectors"]:
        started = time.perf_counter()
        rows, batch = ALL_DETECTORS[name](d)
        res = {"batch_score": batch, "seconds": round(time.perf_counter() - started, 3)}
        if mask.any():
            res.update(_row_metrics(np.asarray(rows, dtype=float), mask))
        out["detectors"][name] = res
    return out


def batch_specs(m: dict) -> list[dict]:
    specs = []
    for split, dss in m["datasets"].items():
        for ds in dss:
            for seed in range(m["seeds"]):
                base = {"manifest": m, "split": split, "data_id": ds["data_id"], "seed": seed}
                specs += [{**base, "attack": None, "attack_index": -1, "rate": 0.0, "rep": r}
                          for r in range(m["clean_batches_per_seed"])]
                specs += [{**base, "attack": a, "attack_index": i, "rate": rate, "rep": 0}
                          for i, a in enumerate(m["attacks"]) for rate in m["poison_rates"]]
    return specs


def spec_key(s: dict) -> tuple:
    return tuple(s[k] for k in ("split", "data_id", "seed", "attack", "rate", "rep"))


def score_all(m: dict, jobs_fn, total: int, workers: int, ckpt: Path) -> list[dict]:
    """Score prepared batches in a small window (Executor.map would prepare every batch before scoring the first) and
    append each record to ckpt, so a killed run resumes: batches already in ckpt are skipped before being prepared."""
    records = []
    if ckpt.exists():
        with ckpt.open("rb") as f:   # this run's own local checkpoint
            while True:
                try: records.append(pickle.load(f))
                except (EOFError, pickle.UnpicklingError): break   # a record cut short by the kill is redone
        print(f"resuming: {len(records)}/{total} batches in {ckpt}", flush=True)
    started, done = time.perf_counter(), {spec_key(r) for r in records}
    with ProcessPoolExecutor(workers) as pool, ckpt.open("ab") as f:
        pending = deque()
        def take():
            records.append(pending.popleft().result()); pickle.dump(records[-1], f); f.flush()
            if len(records) % 25 == 0:
                print(f"{len(records)}/{total} batches, {time.perf_counter() - started:.0f}s", flush=True)
        for job in jobs_fn(m, done):
            pending.append(pool.submit(score_batch, job))
            while len(pending) > 2 * workers or (pending and pending[0].done()):
                take()
        while pending:
            take()
    return records


def checkpoint_path(kind: str, label: str) -> Path:
    path = Path(os.path.expanduser(f"~/.cache/veritas-logs/ckpt-{kind}-{label}.pkl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def jobs(m: dict, done=frozenset()):
    """Yield prepared batches one dataset at a time, so only one dataset's pixels are in memory. Skips specs in done."""
    specs = [s for s in batch_specs(m) if spec_key(s) not in done]
    for data_id in dict.fromkeys(s["data_id"] for s in specs):
        ds = datasets.load(data_id)
        emb, pix = clean_embeddings(ds), pixel_features(ds.raw)
        for spec in (s for s in specs if s["data_id"] == data_id):
            yield {"spec": spec, **prepare_batch(spec, ds, emb, pix)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="full")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--label", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()

    m = load_manifest(args.profile)
    if args.splits:
        m["datasets"] = {k: v for k, v in m["datasets"].items() if k in args.splits}
    label = args.label or args.profile
    started, ckpt = time.perf_counter(), checkpoint_path("mm-images", label)
    records = score_all(m, jobs, len(batch_specs(m)), args.workers, ckpt)
    report = {"manifest": m, "label": label, "detector_commit": detector_commit(), "n_batches": len(records),
              "runtime_seconds": round(time.perf_counter() - started, 1),
              "summary": aggregate(records, m), "batches": records}
    out = ROOT / "results" / f"{date.today().isoformat()}-benchmark-mm-images-{label}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(to_markdown(report) + "\n")
    print(out.with_suffix(".md"))
    ckpt.unlink()


if __name__ == "__main__":
    main()
