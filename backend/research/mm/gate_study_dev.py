"""Dev-only gate study: collect per-component (cal, inc) scores once, then compare gating rules offline.
Run: PYTHONPATH=. python -m research.mm.gate_study_dev (writes ~/.cache/veritas-logs/gate_dev.pkl). Never point it at the test split."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import pickle, time
from concurrent.futures import ProcessPoolExecutor
from research import bench_mm
from research.mm import datasets
from research.mm.detector import pixel_component
from research.mm.embed import clean_embeddings
from research.mm.pixel import pixel_features
from app.detection import veritas2 as v2

def comps(job):
    d = job["d"]
    out = {n: f(d) for n, f in (("label", v2.label_component), ("shift", v2.shift_component), ("pixel", pixel_component))}
    s = job["spec"]
    return {"data_id": s["data_id"], "attack": s["attack"], "rate": s["rate"], "seed": s["seed"], "mask": job["mask"], "comps": out}

def jobs(m):
    specs = bench_mm.batch_specs(m)
    for data_id in dict.fromkeys(s["data_id"] for s in specs):
        ds = datasets.load(data_id); emb, pix = clean_embeddings(ds), pixel_features(ds.raw)
        for spec in (s for s in specs if s["data_id"] == data_id):
            yield {"spec": spec, **bench_mm.prepare_batch(spec, ds, emb, pix)}

if __name__ == "__main__":
    m = bench_mm.load_manifest("full")
    m["datasets"] = {"dev": m["datasets"]["dev"]}
    m["poison_rates"] = [0.01, 0.05]; m["clean_batches_per_seed"] = 12
    t = time.perf_counter(); out = []
    with ProcessPoolExecutor(6) as pool:
        for i, r in enumerate(pool.map(comps, jobs(m), chunksize=1), 1):
            out.append(r)
            if i % 20 == 0: print(i, f"{time.perf_counter()-t:.0f}s", flush=True)
    pickle.dump(out, open(os.path.expanduser("~/.cache/veritas-logs/gate_dev.pkl"), "wb"))
    print("done", len(out))
