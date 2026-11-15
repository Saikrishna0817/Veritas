"""Dev-only study: tables at 1% poison. Does a neighbour label check (the share of a row's nearest reference rows
with a different label) add to the boosted-tree label check? Same cross-fitting as label_crossfit_component.
Collect:  PYTHONPATH=. python -m research.tab.knn_study_dev           (-> ~/.cache/veritas-logs/knn_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.tab.knn_study_dev --analyze
Label variants: gbm (today), knn, avg (1 - mean of the two probabilities of the given label). Batch evidence =
max(label - 0.1, shift - 1, resid - 1), as in veritas2.detect. Dev datasets only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from research import bench_v2
from app.detection import veritas2 as v2

PKL = os.path.expanduser("~/.cache/veritas-logs/knn_dev.pkl")
RATES, SEEDS, CLEAN_PER_SEED, K = (0.01, 0.05), 3, 8, 10
VARIANTS = ("gbm", "knn", "avg")


def label_variants(d):
    """variant -> (ref, inc, one), cross-fitted as in veritas2.label_crossfit_component."""
    X, y = np.vstack([d["X_fit"], d["X_cal"]]), np.concatenate([d["y_fit"], d["y_cal"]])
    cv = (StratifiedKFold if np.bincount(y)[np.unique(y)].min() >= 5 else KFold)(5, shuffle=True, random_state=0)
    n, m, C = len(y), len(d["y_inc"]), d["n_classes"]
    out = {v: (np.zeros(n), np.zeros(m), np.zeros(m)) for v in VARIANTS}
    fold = np.arange(m) % 5
    for k, (tr, te) in enumerate(cv.split(X, y)):
        gbm = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        sc = StandardScaler().fit(X[tr]); nn = NearestNeighbors(n_neighbors=K).fit(sc.transform(X[tr]))
        def probs(X_, y_):
            pg = v2._probs(gbm, X_, C)[np.arange(len(y_)), y_]
            idx = nn.kneighbors(sc.transform(X_), return_distance=False)
            pk = ((y[tr][idx] == y_[:, None]).sum(1) + 1) / (K + C)   # Laplace-smoothed share with the same label
            return {"gbm": pg, "knn": pk, "avg": (pg + pk) / 2}
        pr, pi = probs(X[te], y[te]), probs(d["X_inc"], d["y_inc"])
        for v in VARIANTS:
            ref, inc, one = out[v]
            ref[te] = 1 - pr[v]; inc += (1 - pi[v]) / 5; one[fold == k] = (1 - pi[v])[fold == k]
    return out


def one(spec):
    d, mask, _, _ = bench_v2.make_batch(spec)
    lab = label_variants(d)
    ev = {v: v2.multi_tail_test(r, o) for v, (r, _, o) in lab.items()}
    rows = {v: v2.extrapolated_neg_log_p(r, i) for v, (r, i, _) in lab.items()}
    other = max(v2.multi_tail_test(*v2.shift_component(v2.full_reference(d))) - v2.SHIFT_PENALTY,
                v2.multi_tail_test(*v2.residual_component(d)) - v2.RESID_PENALTY if d["X_inc"].shape[1] > 1 else 0.0, 0.0)
    return {"data_id": spec["data_id"], "attack": spec["attack"], "rate": spec["rate"], "mask": mask, "ev": ev, "rows": rows, "other": other}


def specs():
    m = bench_v2.load_manifest("full"); m["datasets"] = {"dev": m["datasets"]["dev"]}
    m.update(poison_rates=list(RATES), seeds=SEEDS, clean_batches_per_seed=CLEAN_PER_SEED)
    return bench_v2.batch_specs(m)


def collect():
    S = specs(); t = time.perf_counter()
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 3))) as pool, open(PKL, "wb") as f:
        for i, r in enumerate(pool.map(one, S, chunksize=1), 1):
            pickle.dump(r, f); f.flush()
            if i % 20 == 0: print(i, len(S), f"{time.perf_counter() - t:.0f}s", flush=True)
    print("done", len(S))


def analyze():
    R = []
    with open(PKL, "rb") as f:   # this study's own local output
        while True:
            try: R.append(pickle.load(f))
            except EOFError: break
    cln = [r for r in R if not r["attack"]]; atk = [r for r in R if r["attack"]]
    batch = lambda r, v: max(r["ev"][v] - v2.LABEL_PENALTY, r["other"])  # noqa: E731
    print(f"{len(atk)} attacked, {len(cln)} clean")
    for t in (1.3, 2.0):
        print(f"clean flagged at {t}: " + "  ".join(f"{v} {np.mean([batch(r, v) >= t for r in cln]):.3f}" for v in VARIANTS))
    print(f"\n{'attack':<22}{'rate':>5}" + "".join(f"{v + ' AUPRC':>12}{'flag1.3':>9}" for v in VARIANTS))
    for rate in RATES:
        for a in list(dict.fromkeys(r["attack"] for r in atk)) + ["MEAN"]:
            g = [r for r in atk if r["rate"] == rate and a in ("MEAN", r["attack"])]
            print(f"{a[:21]:<22}{rate:5.2f}" + "".join(
                f"{np.mean([average_precision_score(r['mask'], np.maximum(r['rows'][v], 0)) for r in g]):12.2f}{np.mean([batch(r, v) >= 1.3 for r in g]):9.2f}" for v in VARIANTS))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
