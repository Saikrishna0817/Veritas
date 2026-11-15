"""Dev-only study: compare incoming rows with ALL reference rows (fit + cal), multi-tail tests, cross-fit label, chroma check.
Collect (same dev batches as weak_study_dev):  PYTHONPATH=. python -m research.mm.ref_study_dev   (-> ~/.cache/veritas-logs/ref_dev.pkl)
Analyse:                                       PYTHONPATH=. python -m research.mm.ref_study_dev --analyze
Dev datasets only; never point at the test split."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score
from research import bench_mm
from research.mm import report_study_dev as rs, shared_study_dev as ss
from research.mm.detector import components_v6
from research.mm.shared import extrapolated_neg_log_p, multi_tail_test

PKL = rs.LOG + "ref_dev.pkl"
ALL = ["label", "label_cf", "shift", "pixel", "dup", "cell", "resid", "chroma"]

def comps(job):
    s, secs = job["spec"], {}
    c = components_v6(job["d"], secs)
    return {"data_id": s["data_id"], "attack": s["attack"], "rate": s["rate"], "seed": s["seed"], "mask": job["mask"], "comps": c, "secs": secs}

def collect(fn=None, path=PKL):
    """Run ``fn`` (default comps) on the dev batches, appending each record to ``path`` (resumes where it stopped)."""
    m = bench_mm.load_manifest("full"); m["datasets"] = {"dev": m["datasets"]["dev"]}
    m1 = {**m, "poison_rates": [0.01, 0.05], "clean_batches_per_seed": 12, "seeds": 5}      # as shared_study_dev.collect
    n_new = rs.MIN_CLEAN - 120; per_seed = len(m["datasets"]["dev"]) * m["clean_batches_per_seed"]
    m2 = {**m, "attacks": [], "poison_rates": [], "seeds": rs.FIRST_NEW_SEED + -(-n_new // per_seed)}  # as report_study_dev.collect
    t = time.perf_counter(); done = len(_load(path)) if os.path.exists(path) else 0  # resume: records are appended one by one
    gen = itertools.islice(itertools.chain(ss.jobs(m1), rs.jobs(m2)), done, None)
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 6))) as pool, open(path, "ab") as f:
        for i, r in enumerate(pool.map(fn or comps, gen, chunksize=1), done + 1):
            pickle.dump(r, f); f.flush()
            if i % 20 == 0: print(i, f"{time.perf_counter()-t:.0f}s", flush=True)
    print("done", i)

def _load(path=PKL):
    out = []
    with open(path, "rb") as f:   # this study's own local output
        while True:
            try: out.append(pickle.load(f))
            except EOFError: return out

# ---- analysis ----
GRIDS = {"1%": (0.01,), "0.1-1%": (0.001, 0.003, 0.01), "0.03-1%": (0.0003, 0.001, 0.003, 0.01)}
PEN = {"label": 0.1, "label_cf": 0.1}; GATE = {"shift": 3.0}   # others: penalty 1.0, gate 1.0; label checks always admitted
S0 = ["label", "shift", "pixel", "dup", "cell", "resid"]; S1 = S0 + ["chroma"]; S2 = ["label_cf"] + S1[1:]
SETS = {"S0": S0, "S1": S1, "S2": S2}
VARIANTS = [(s, g) for s in SETS for g in GRIDS]

def analyze():
    R = _load()
    atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]]
    attacks = list(dict.fromkeys(r["attack"] for r in atk)); rates = (0.01, 0.05)
    ap = lambda r, s: average_precision_score(r["mask"], s)
    groups = [(f"{a[:21]:<21}{rate:5.2f}", [r for r in atk if r["attack"] == a and r["rate"] == rate]) for rate in rates for a in attacks]
    dss = list(dict.fromkeys(r["data_id"] for r in cln))
    def raw(r, n, g):
        k = (n, g)
        if k not in r.setdefault("_e", {}): r["_e"][k] = multi_tail_test(*r["comps"][n], GRIDS[g])
        return r["_e"][k]
    def pen(r, s, g): return {n: raw(r, n, g) - PEN.get(n, 1.0) for n in SETS[s]}
    def best(r, s, g): return max(pen(r, s, g).values())
    def fused(r, s, g):
        return np.max([extrapolated_neg_log_p(*r["comps"][n]) for n in SETS[s]
                       if n.startswith("label") or raw(r, n, g) >= GATE.get(n, 1.0)], axis=0)
    V = "".join(f"{'v' + str(i):>7}" for i in range(len(VARIANTS)))
    print(f"attacked {len(atk)}, clean {len(cln)}: { {d: sum(r['data_id'] == d for r in cln) for d in dss} }")
    print("variants:"); [print(f"  v{i}: {s} ({'+'.join(SETS[s])}) / tail grid {g}") for i, (s, g) in enumerate(VARIANTS)]
    print("\n1. Row AUPRC of each component alone (raw inc score)"); print(" " * 26 + "".join(f"{c:>9}" for c in ALL))
    for lab, g in groups: print(lab + "".join(f"{np.mean([ap(r, r['comps'][c][1]) for r in g]):9.2f}" for c in ALL))
    for rate in rates:
        print(f"{'MEAN':<21}{rate:5.2f}" + "".join(f"{np.mean([np.mean([ap(r, r['comps'][c][1]) for r in g]) for lab, g in groups if lab.endswith(f'{rate:5.2f}')]):9.2f}" for c in ALL))
    print("   chroma AUPRC per dataset (class_noise, blended_backdoor)")
    for a in ("class_noise", "blended_backdoor"):
        for ds in dss:
            for rate in rates:
                g = [r for r in atk if r["attack"] == a and r["rate"] == rate and r["data_id"] == ds]
                print(f"   {a:<18}{ds:<14}{rate:5.2f}  {np.mean([ap(r, r['comps']['chroma'][1]) for r in g]) if g else float('nan'):.2f}  (n={len(g)})")
    print("\n2a. Clean: share of batches with raw evidence >= 1.3 / >= 2.0 per component and tail grid")
    for lab, rr in [("POOLED", cln)] + [(d, [r for r in cln if r["data_id"] == d]) for d in dss]:
        print(f"{lab} (n={len(rr)})"); print(" " * 12 + "".join(f"{c:>13}" for c in ALL))
        for g in GRIDS: print(f"  {g:<10}" + "".join(f"{np.mean([raw(r, c, g) >= 1.3 for r in rr]):7.3f}/{np.mean([raw(r, c, g) >= 2.0 for r in rr]):.3f}" for c in ALL))
    print("\n2b. Clean: share with max penalised evidence >= t, per variant"); print(" " * 26 + V)
    for lab, rr in [("POOLED", cln)] + [(d, [r for r in cln if r["data_id"] == d]) for d in dss]:
        for t in (1.3, 2.0, 4.0):
            print(f"{('>= ' + str(t) + ' ' + lab):<26}" + "".join(f"{np.mean([best(r, s, g) >= t for r in rr]):7.3f}" for s, g in VARIANTS) + f"   n={len(rr)}")
    for t in (1.3, 2.0):
        print(f"\n3. Batch TPR at cut-off {t} (max penalised evidence)"); print(" " * 26 + V)
        for lab, g_ in groups + [(f"{'ALL':<21}{rate:5.2f}", [r for r in atk if r["rate"] == rate]) for rate in rates]:
            print(lab + "".join(f"{np.mean([best(r, s, g) >= t for r in g_]):7.2f}" for s, g in VARIANTS))
    print("\n4. Fused row AUPRC"); print(" " * 26 + V)
    for lab, g_ in groups: print(lab + "".join(f"{np.mean([ap(r, fused(r, s, g)) for r in g_]):7.2f}" for s, g in VARIANTS))
    for rate in rates:
        print(f"{'MEAN':<21}{rate:5.2f}" + "".join(f"{np.mean([np.mean([ap(r, fused(r, s, g)) for r in g_]) for lab, g_ in groups if lab.endswith(f'{rate:5.2f}')]):7.2f}" for s, g in VARIANTS))
    print("\n5. Lead check (highest penalised evidence), variant (S2, 0.03-1%)")
    print("attack".ljust(26) + "".join(f"{n:>9}" for n in S2) + f"{'n':>6}")
    for lab, g_ in groups + [(f"{'CLEAN':<21}{'':5}", cln)]:
        print(lab + "".join(f"{sum(max(S2, key=lambda n: pen(r, 'S2', '0.03-1%')[n]) == n for r in g_):9d}" for n in S2) + f"{len(g_):6d}")
    print("\nmean seconds per component per batch:", {n: round(float(np.mean([r['secs'][n] for r in R])), 1) for n in ALL})

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
