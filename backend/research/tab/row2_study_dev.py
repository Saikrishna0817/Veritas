"""Dev-only study, round 2 of the tabular row checks: covariate shift, 1% batch verdicts, speed.
Collect:  PYTHONPATH=. python -m research.tab.row2_study_dev            (-> ~/.cache/veritas-logs/row2_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.tab.row2_study_dev --analyze
New candidate checks, each scored the same way for reference and incoming rows (cross-fitted, so the in-sample bias that
sank the marginal-outlier check cannot recur): ridge residuals (rows that break the columns' correlations), Mahalanobis
distance, k-th nearest-neighbour distance (reference rows leave themselves out). Speed: GBM label check with 50 rounds.
Same batches as row_study_dev (paired with row_dev.pkl for the current checks). Dev datasets only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from research import bench_v2
from research.tab import row_study_dev as r1
from app.detection import veritas2 as v2

PKL = os.path.expanduser("~/.cache/veritas-logs/row2_dev.pkl")

def _folds(n):
    return list(KFold(5, shuffle=True, random_state=0).split(np.arange(n)))

def crossfit_rows(d, fit_score):
    """(ref, inc_one): ``fit_score(train) -> score(X)``; reference rows out-of-fold, incoming rows one fold model each."""
    R = np.vstack([d["X_fit"], d["X_cal"]]); I = d["X_inc"]
    ref, one = np.zeros(len(R)), np.zeros(len(I)); fold = np.arange(len(I)) % 5
    for k, (tr, te) in enumerate(_folds(len(R))):
        score = fit_score(R[tr]); ref[te] = score(R[te]); one[fold == k] = score(I[fold == k])
    return ref, one

def ridge_fit(R):
    """Sum over columns of squared residuals, each predicted from the other columns, scaled by its training spread."""
    models = []
    for j in range(R.shape[1]):
        keep = np.arange(R.shape[1]) != j; m = Ridge(1.0).fit(R[:, keep], R[:, j])
        models.append((keep, m, max(np.std(R[:, j] - m.predict(R[:, keep])), 1e-6)))
    return lambda X: sum(((X[:, j] - m.predict(X[:, keep])) / s) ** 2 for j, (keep, m, s) in enumerate(models))

def maha_fit(R):
    lw = LedoitWolf().fit(R); return lambda X: lw.mahalanobis(X)

def knn_fit(R, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(R); return lambda X: nn.kneighbors(X)[0][:, -1]

def collect_one(spec):
    d, mask, cells, out = bench_v2.make_batch(spec); secs = {}; c = {}
    for n, f in (("ridge", ridge_fit), ("maha", maha_fit), ("knn", knn_fit)):
        t = time.perf_counter(); c[n] = crossfit_rows(d, f); secs[n] = time.perf_counter() - t
    t = time.perf_counter(); c["cf_gbm50"] = r1.crossfit(d, lambda: HistGradientBoostingClassifier(max_iter=50, random_state=0)); secs["cf_gbm50"] = time.perf_counter() - t
    return {**{k: out[k] for k in ("data_id", "dataset", "seed", "attack", "rate", "rep")}, "mask": mask, "comps": c, "secs": secs}

def collect():
    S = r1.specs(); done = len(_load()) if os.path.exists(PKL) else 0; t = time.perf_counter()
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 4))) as pool, open(PKL, "ab") as f:
        for i, r in enumerate(pool.map(collect_one, itertools.islice(S, done, None), chunksize=2), done + 1):
            pickle.dump(r, f); f.flush()
            if i % 50 == 0: print(i, len(S), f"{time.perf_counter()-t:.0f}s", flush=True)
    print("done", len(S))

def _load():
    out = []
    with open(PKL, "rb") as f:   # this study's own local output
        while True:
            try: out.append(pickle.load(f))
            except EOFError: return out

# ---- analysis ----
def analyze():
    A, B = r1._load(), _load(); assert len(A) == len(B)
    for a, b in zip(A, B):
        assert (a["data_id"], a["seed"], a["attack"], a["rate"], a["rep"]) == (b["data_id"], b["seed"], b["attack"], b["rate"], b["rep"])
        a["comps"].update(b["comps"]); a["secs"].update(b["secs"])
    T = v2.TAILS; E = v2.extrapolated_neg_log_p
    def check(r, n, pen=1.0, gate=1.0):
        c = r["comps"][n]
        if n.startswith("cf_"): ref, inc, one = c; return v2.multi_tail_test(ref, one, T) - 0.1, E(ref, inc), True
        e = v2.multi_tail_test(*c, T); return e - pen, E(*c), e >= gate
    VARIANTS = {"current": [], "+ridge": ["ridge"], "+maha": ["maha"], "+knn": ["knn"], "+ridge+knn": ["ridge", "knn"],
                "+ridge g2": [("ridge", 1.0, 2.0)], "+ridge p0.7": [("ridge", 0.7, 1.0)], "gbm50": "gbm50"}
    def score(r, v):
        lab = "cf_gbm50" if v == "gbm50" else "cf_gbm"
        ch = [check(r, lab), check(r, "shift_full", 1.0, 3.0)]
        for x in ([] if v == "gbm50" else v):
            ch.append(check(r, *x) if isinstance(x, tuple) else check(r, x))
        return max(e for e, _, _ in ch), np.max([s for _, s, a in ch if a], axis=0)
    for r in A: r["v"] = {n: score(r, VARIANTS[n] if n != "gbm50" else "gbm50") for n in VARIANTS}
    atk = [r for r in A if r["attack"]]; cln = [r for r in A if not r["attack"]]
    attacks = list(dict.fromkeys(r["attack"] for r in atk)); rates = sorted({r["rate"] for r in atk}); V = list(VARIANTS); W = 13
    print("seconds per batch:", {k: round(np.mean([r["secs"][k] for r in A]), 2) for k in A[0]["secs"]})
    print(" " * 28 + "".join(f"{n:>{W}}" for n in V))
    for t in (1.3, 2.0, 4.0):
        print(f"{'clean >= ' + str(t):<28}" + "".join(f"{np.mean([r['v'][n][0] >= t for r in cln]):{W}.3f}" for n in V))
    for ds in dict.fromkeys(r["dataset"] for r in cln):
        print(f"{'  >= 1.3 ' + ds[:18]:<28}" + "".join(f"{np.mean([r['v'][n][0] >= 1.3 for r in cln if r['dataset'] == ds]):{W}.3f}" for n in V))
    for title, f in (("batch flagged at 1.3", lambda r, n: r["v"][n][0] >= 1.3),
                     ("row AUPRC", lambda r, n: average_precision_score(r["mask"], r["v"][n][1]))):
        print(f"\n{title}"); print(" " * 28 + "".join(f"{n:>{W}}" for n in V))
        for rate in rates:
            for a in attacks + ["MEAN"]:
                g = [r for r in atk if r["rate"] == rate and a in ("MEAN", r["attack"])]
                print(f"{a[:22]:<23}{rate:5.2f}" + "".join(f"{np.mean([f(r, n) for r in g]):{W}.3f}" for n in V))
    print("\nalone: row AUPRC of each new check by attack at 0.01 / 0.05")
    for n in ("ridge", "maha", "knn"):
        print(f"  {n:<6}" + "  ".join(f"{a[:12]} " + "/".join(f"{np.mean([average_precision_score(r['mask'], E(*r['comps'][n])) for r in atk if r['attack'] == a and r['rate'] == rate]):.2f}" for rate in (0.01, 0.05)) for a in attacks))

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
