"""Dev-only study of shared-structure components and the extrapolated fusion.
Collect:  PYTHONPATH=. python -m research.mm.shared_study_dev            (-> ~/.cache/veritas-logs/shared_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.mm.shared_study_dev --analyze
Dev datasets only; never point at the test split."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score
from research import bench_mm
from research.mm import datasets
from research.mm.detector import pixel_component, GATE, PENALTY
from research.mm.embed import clean_embeddings
from research.mm.pixel import pixel_features
from research.mm.shared import NEW, extrapolated_neg_log_p
from app.detection import veritas2 as v2

PKL = os.path.expanduser("~/.cache/veritas-logs/shared_dev.pkl")
OLD = {"label": v2.label_component, "shift": v2.shift_component, "pixel": pixel_component}

def comps(job):
    d, s, out, secs = job["d"], job["spec"], {}, {}
    for n, f in {**OLD, **NEW}.items():
        t = time.perf_counter(); out[n] = f(d); secs[n] = time.perf_counter() - t
    return {"data_id": s["data_id"], "attack": s["attack"], "rate": s["rate"], "seed": s["seed"],
            "mask": job["mask"], "comps": out, "secs": secs}

def jobs(m):
    specs = bench_mm.batch_specs(m)
    for data_id in dict.fromkeys(s["data_id"] for s in specs):
        ds = datasets.load(data_id); emb, pix = clean_embeddings(ds), pixel_features(ds.raw)
        for spec in (s for s in specs if s["data_id"] == data_id):
            yield {"spec": spec, **bench_mm.prepare_batch(spec, ds, emb, pix)}

def collect(seeds=5):
    m = bench_mm.load_manifest("full")
    m["datasets"] = {"dev": m["datasets"]["dev"]}
    m["poison_rates"] = [0.01, 0.05]; m["clean_batches_per_seed"] = 12; m["seeds"] = seeds
    t = time.perf_counter(); out = []
    with ProcessPoolExecutor(6) as pool:
        for i, r in enumerate(pool.map(comps, jobs(m), chunksize=1), 1):
            out.append(r)
            if i % 20 == 0: print(i, f"{time.perf_counter()-t:.0f}s", flush=True)
    pickle.dump(out, open(PKL, "wb")); print("done", len(out))

# ---- analysis ----
ADDED = list(NEW)
VARIANTS = {  # name -> (components, gate overrides, scorer)
    "current": (["label", "shift", "pixel"], v2.neg_log_p),
    "current+new": (["label", "shift", "pixel"] + ADDED, v2.neg_log_p),
    "current_extrap": (["label", "shift", "pixel"], extrapolated_neg_log_p),
    "current+new_extrap": (["label", "shift", "pixel"] + ADDED, extrapolated_neg_log_p),
}
G = {**GATE, **{n: 1.0 for n in ADDED}}
PEN = {**PENALTY, **{n: 1.0 for n in ADDED}}

def fuse(r, names, scorer):
    rows, ev = [], []
    for n in names:
        cal, inc = r["comps"][n]
        e = v2.exact_tail_test(cal, inc); ev.append(e - PEN[n])
        if n == "label" or e >= G[n]: rows.append(scorer(cal, inc))
    return np.max(rows, axis=0), max(ev)

def analyze():
    R = pickle.load(open(PKL, "rb"))
    atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]]
    attacks = list(dict.fromkeys(r["attack"] for r in atk))
    print(f"batches: {len(R)} ({len(cln)} clean), seeds: {len({r['seed'] for r in R})}")
    ap = lambda r, s: average_precision_score(r["mask"], s)
    def table(title, cols):  # cols: name -> f(r)->row scores
        print(f"\n{title}"); print("attack".ljust(24) + "".join(f"{c[:14]:>15}" for c in cols))
        for rate in (0.01, 0.05):
            for a in attacks:
                rs = [r for r in atk if r["attack"] == a and r["rate"] == rate]
                print(f"{a[:20]:<20}{rate:4.2f}" + "".join(f"{np.mean([ap(r, f(r)) for r in rs]):15.2f}" for f in cols.values()))
            print(f"{'MEAN':<20}{rate:4.2f}" + "".join(
                f"{np.mean([np.mean([ap(r, f(r)) for r in atk if r['attack'] == a and r['rate'] == rate]) for a in attacks]):15.2f}" for f in cols.values()))
    table("5a row AUPRC, each component alone (raw inc score)", {n: (lambda r, n=n: r["comps"][n][1]) for n in OLD.keys() | set(ADDED)})
    table("5b row AUPRC, fused", {k: (lambda r, k=k: fuse(r, *VARIANTS[k])[0]) for k in VARIANTS})
    print("\n5c clean batches: share with evidence >= 1.3 / >= 2.0")
    for k in VARIANTS:
        ev = np.array([fuse(r, *VARIANTS[k])[1] for r in cln]); print(f"  {k:<22}{(ev>=1.3).mean():.3f} {(ev>=2.0).mean():.3f}")
    print("  per component share >= 1.3 (raw evidence, no penalty):")
    for n in list(OLD) + ADDED:
        e = np.array([v2.exact_tail_test(*r["comps"][n]) for r in cln]); print(f"  {n:<22}{(e>=1.3).mean():.3f}")
    print("\n5d batch TPR at 5% clean FPR")
    for k in VARIANTS:
        thr = np.quantile([fuse(r, *VARIANTS[k])[1] for r in cln], 0.95)
        out = [np.mean([fuse(r, *VARIANTS[k])[1] > thr for r in atk if r["rate"] == rate]) for rate in (0.01, 0.05)]
        print(f"  {k:<22}1%: {out[0]:.3f}  5%: {out[1]:.3f}  (thr {thr:.2f})")
    print("\n5e mean seconds/batch")
    for n in list(OLD) + ADDED: print(f"  {n:<12}{np.mean([r['secs'][n] for r in R]):.2f}")
    # extrapolation sanity: top-12 rows of plain vs extrapolated, per component and fused, on clean batches
    for k in ("current+new", "current"):
        ch = np.mean([set(np.argsort(-fuse(r, *VARIANTS[k])[0])[:12]) != set(np.argsort(-fuse(r, *VARIANTS[k + "_extrap"])[0])[:12]) for r in cln])
        print(f"\nsanity: clean batches whose fused top-12 changes with extrapolation ({k}): {ch:.3f}")
    bad = 0
    for r in cln:
        for n in R[0]["comps"]:
            cal, inc = r["comps"][n]; a, b = v2.neg_log_p(cal, inc), extrapolated_neg_log_p(cal, inc)
            inside = inc <= cal.max(); bad += not np.allclose(a[inside], b[inside]) or not (np.argsort(-a, kind="stable") == np.argsort(-b, kind="stable")).all()
    print(f"sanity: single-component (batch,component) pairs where extrapolation changed inside values or order: {bad}")

if __name__ == "__main__":
    if "--analyze" in sys.argv: analyze()
    else: collect(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5)
