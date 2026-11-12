"""Dev-only study for the per-check report: row AUPRC per check / fused / lead / oracle, lead counts, firing rates,
clean calibration under three penalty schemes, batch TPR.  Attacked batches come from shared_dev.pkl.
Collect extra clean dev batches:  PYTHONPATH=. python -m research.mm.report_study_dev   (-> report_dev_clean.pkl)
Analyse:                          PYTHONPATH=. python -m research.mm.report_study_dev --analyze
Dev datasets only; never point at the test split."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score
from research import bench_mm
from research.mm import datasets
from research.mm.detector import components, GATE, PENALTY
from research.mm.embed import clean_embeddings
from research.mm.pixel import pixel_features
from research.mm.shared import extrapolated_neg_log_p
from app.detection import veritas2 as v2

LOG = os.path.expanduser("~/.cache/veritas-logs/")
OLD_PKL, PKL = LOG + "shared_dev.pkl", LOG + "report_dev_clean.pkl"
NAMES = ["label", "shift", "pixel", "dup", "cell", "resid"]
OLD_KEY = {"cell": "cell8_k3", "resid": "resid_k3"}  # names in shared_dev.pkl
FIRST_NEW_SEED, MIN_CLEAN = 5, 360  # seeds 0-4 are in shared_dev.pkl (they hold 120 clean dev batches)

def comps(job):
    s = job["spec"]
    return {"data_id": s["data_id"], "seed": s["seed"], "rep": s["rep"], "comps": components(job["d"])}

def jobs(m):
    specs = [s for s in bench_mm.batch_specs(m) if s["seed"] >= FIRST_NEW_SEED]
    for data_id in dict.fromkeys(s["data_id"] for s in specs):
        ds = datasets.load(data_id); emb, pix = clean_embeddings(ds), pixel_features(ds.raw)
        for spec in (s for s in specs if s["data_id"] == data_id):
            yield {"spec": spec, **bench_mm.prepare_batch(spec, ds, emb, pix)}

def collect():
    m = bench_mm.load_manifest("full")
    m["datasets"] = {"dev": m["datasets"]["dev"]}; m["attacks"] = []; m["poison_rates"] = []
    n_old = sum(not r["attack"] for r in pickle.load(open(OLD_PKL, "rb")))
    per_seed = len(m["datasets"]["dev"]) * m["clean_batches_per_seed"]
    m["seeds"] = FIRST_NEW_SEED + -(-(MIN_CLEAN - n_old) // per_seed)  # seeds FIRST_NEW_SEED..seeds-1
    t = time.perf_counter(); out = []
    with ProcessPoolExecutor(6) as pool:
        for i, r in enumerate(pool.map(comps, jobs(m), chunksize=1), 1):
            out.append(r)
            if i % 20 == 0: print(i, f"{time.perf_counter()-t:.0f}s", flush=True)
    pickle.dump(out, open(PKL, "wb")); print("done", len(out), "new clean; total", n_old + len(out))

# ---- analysis ----
CUT = 1.3
SCHEMES = {"A": {"label": 0.1, **{n: 1.0 for n in NAMES[1:]}}, "B": {"label": 0.301, **{n: 1.0 for n in NAMES[1:]}},
           "C": {n: float(np.log10(6)) for n in NAMES}}

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / 2 / n) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / (1 + z * z / n)
    return c - h, c + h

def ev(r):  # raw evidence per check
    if "ev" not in r: r["ev"] = {n: v2.exact_tail_test(*r["comps"][n]) for n in NAMES}
    return r["ev"]

def rows(r):
    if "rows" not in r: r["rows"] = {n: extrapolated_neg_log_p(*r["comps"][n]) for n in NAMES}
    return r["rows"]

def pen(r, s): return {n: ev(r)[n] - SCHEMES[s][n] for n in NAMES}
def lead(r): p = pen(r, "A"); return max(NAMES, key=lambda n: p[n])
def fused(r): return np.max([rows(r)[n] for n in NAMES if n == "label" or ev(r)[n] >= GATE[n]], axis=0)

def analyze():
    R = pickle.load(open(OLD_PKL, "rb")); N = pickle.load(open(PKL, "rb"))
    for r in R: r["comps"] = {n: r["comps"][OLD_KEY.get(n, n)] for n in NAMES}
    atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]] + N
    attacks = list(dict.fromkeys(r["attack"] for r in atk)); rates = (0.01, 0.05)
    print(f"attacked batches: {len(atk)}; clean batches: {len(cln)} ({len(cln) - len(N)} from shared_dev.pkl + {len(N)} new); "
          f"clean per dataset: { {d: sum(r['data_id'] == d for r in cln) for d in dict.fromkeys(r['data_id'] for r in cln)} }")
    ap = lambda r, s: average_precision_score(r["mask"], s)
    cols = {**{n: (lambda r, n=n: rows(r)[n]) for n in NAMES}, "fused": fused, "lead": lambda r: rows(r)[lead(r)]}
    def score(r): return {**{c: ap(r, f(r)) for c, f in cols.items()}, "oracle": max(ap(r, rows(r)[n]) for n in NAMES)}
    for r in atk: r["ap"] = score(r)
    C = list(cols) + ["oracle"]
    print("\n1. Row AUPRC (mean over batches); lead = check with highest penalised evidence (scheme A); oracle = best single check per batch")
    print("attack".ljust(26) + "".join(f"{c:>9}" for c in C))
    for rate in rates:
        for a in attacks:
            rs = [r for r in atk if r["attack"] == a and r["rate"] == rate]
            print(f"{a[:21]:<21}{rate:5.2f}" + "".join(f"{np.mean([r['ap'][c] for r in rs]):9.2f}" for c in C))
        print(f"{'MEAN':<21}{rate:5.2f}" + "".join(
            f"{np.mean([np.mean([r['ap'][c] for r in atk if r['attack'] == a and r['rate'] == rate]) for a in attacks]):9.2f}" for c in C))
    print("\n2. Lead check (highest penalised evidence, scheme A): number of batches per check")
    print("attack".ljust(26) + "".join(f"{n:>8}" for n in NAMES) + f"{'n':>6}")
    groups = [(f"{a[:21]:<21}{rate:5.2f}", [r for r in atk if r["attack"] == a and r["rate"] == rate]) for rate in rates for a in attacks]
    for lab, rs in groups + [(f"{'CLEAN':<21}{'':5}", cln)]:
        print(lab + "".join(f"{sum(lead(r) == n for r in rs):8d}" for n in NAMES) + f"{len(rs):6d}")
    print(f"\n3. Share of batches where a check's penalised evidence (scheme A) >= {CUT}")
    print("attack".ljust(26) + "".join(f"{n:>8}" for n in NAMES))
    for lab, rs in groups + [(f"{'CLEAN':<21}{'':5}", cln)]:
        print(lab + "".join(f"{np.mean([pen(r, 'A')[n] >= CUT for r in rs]):8.2f}" for n in NAMES))
    print("\n4. Clean calibration (share of clean batches; nominal 0.05 / 0.01 / 0.0001; [95% Wilson interval])")
    sets = [("POOLED", cln)] + [(d, [r for r in cln if r["data_id"] == d]) for d in dict.fromkeys(r["data_id"] for r in cln)]
    def line(lab, vals):
        s = f"  {lab:<26}"
        for t in (1.3, 2.0, 4.0):
            k = int(sum(v >= t for v in vals)); lo, hi = wilson(k, len(vals)); s += f"  >={t}: {k/len(vals):.3f} [{lo:.3f},{hi:.3f}]"
        print(s)
    for lab, rs in sets:
        print(f"{lab} (n={len(rs)})")
        for n in NAMES: line(f"raw {n}", [ev(r)[n] for r in rs])
        for s in SCHEMES: line(f"max penalised, scheme {s}", [max(pen(r, s).values()) for r in rs])
    print(f"\n5. Batch TPR at fixed cutoff {CUT} on max penalised evidence")
    print("attack".ljust(26) + "".join(f"{'scheme ' + s:>10}" for s in SCHEMES))
    for lab, rs in groups:
        print(lab + "".join(f"{np.mean([max(pen(r, s).values()) >= CUT for r in rs]):10.2f}" for s in SCHEMES))
    for rate in rates:
        print(f"{'ALL attacks':<21}{rate:5.2f}" + "".join(
            f"{np.mean([max(pen(r, s).values()) >= CUT for r in atk if r['rate'] == rate]):10.2f}" for s in SCHEMES))

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
