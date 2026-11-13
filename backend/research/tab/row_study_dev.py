"""Dev-only study: bring the image track's upgrades to the tabular row/batch checks (veritas2).
Collect:  PYTHONPATH=. python -m research.tab.row_study_dev            (-> ~/.cache/veritas-logs/row_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.tab.row_study_dev --analyze
Candidates: label check cross-fitted over all reference rows (GBM or logistic regression; one fold model per incoming
row for the batch test, the mean of folds for the row list; pooled and per-claimed-class tests), shift check against
all reference rows, a marginal-outlier check, several tail sizes. Dev datasets of manifest-v2 only, plus extra clean
dev batches (seeds 5-14) so clean false-alarm rates are measured on 360 batches."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research import bench_v2
from app.detection import veritas2 as v2

PKL = os.path.expanduser("~/.cache/veritas-logs/row_dev.pkl")
MODELS = {"gbm": lambda: HistGradientBoostingClassifier(random_state=0),
          "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))}

def full_ref(d):
    """Detector inputs whose calibration slot holds every reference row (fit + cal)."""
    return {**d, "X_cal": np.vstack([d["X_fit"], d["X_cal"]]), "y_cal": np.concatenate([d["y_fit"], d["y_cal"]])}

def crossfit(d, make):
    """(ref, inc mean of folds, inc one fold per row): label suspicion 1 - P(label), out-of-fold for reference rows."""
    X, y = np.vstack([d["X_fit"], d["X_cal"]]), np.concatenate([d["y_fit"], d["y_cal"]])
    cv = (StratifiedKFold if np.bincount(y)[np.unique(y)].min() >= 5 else KFold)(5, shuffle=True, random_state=0)
    ref, inc, one = np.zeros(len(y)), np.zeros(len(d["y_inc"])), np.zeros(len(d["y_inc"]))
    fold = np.arange(len(d["y_inc"])) % 5
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = make().fit(X[tr], y[tr])
        score = lambda X_, y_: 1.0 - v2._probs(model, X_, d["n_classes"])[np.arange(len(y_)), y_]
        ref[te] = score(X[te], y[te]); s = score(d["X_inc"], d["y_inc"])
        inc += s / 5; one[fold == k] = s[fold == k]
    return ref, inc, one

def marginal_component(d):
    """Row = its most unusual column (-log10 share of reference values at least as far from the mean); reference rows
    are scored against the reference itself, so both sides follow the same rule."""
    R, I = np.abs(np.vstack([d["X_fit"], d["X_cal"]])), np.abs(d["X_inc"])
    col = lambda A: np.stack([v2.neg_log_p(R[:, j], A[:, j]) for j in range(R.shape[1])], axis=1).max(1)
    return col(R), col(I)

def collect_one(spec):
    d, mask, cells, out = bench_v2.make_batch(spec); f = full_ref(d)
    t = time.perf_counter(); secs = {}
    def timed(n, fn):
        t = time.perf_counter(); r = fn(); secs[n] = time.perf_counter() - t; return r
    c = {"label": timed("label", lambda: v2.label_component(d)), "shift": timed("shift", lambda: v2.shift_component(d)),
         "shift_full": timed("shift_full", lambda: v2.shift_component(f)), "marg": timed("marg", lambda: marginal_component(d)),
         **{f"cf_{n}": timed(f"cf_{n}", lambda m=m: crossfit(d, m)) for n, m in MODELS.items()}}
    return {**{k: out[k] for k in ("data_id", "dataset", "seed", "attack", "rate", "rep")}, "mask": mask, "comps": c, "secs": secs,
            "y_ref": f["y_cal"], "y_inc": d["y_inc"], "k": d["n_classes"]}

def specs():
    m = bench_v2.load_manifest("full"); m["datasets"] = {"dev": m["datasets"]["dev"]}
    extra = {**m, "seeds": 15, "attacks": [], "poison_rates": [], "clean_batches_per_seed": 4}
    return bench_v2.batch_specs(m) + [s for s in bench_v2.batch_specs(extra) if s["seed"] >= m["seeds"]]

def collect():
    S = specs(); done = len(_load()) if os.path.exists(PKL) else 0; t = time.perf_counter()
    for ds in bench_v2.load_manifest("full")["datasets"]["dev"]: bench_v2._openml(ds["data_id"])
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 5))) as pool, open(PKL, "ab") as f:
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
from research.mm.shared import extrapolated_neg_log_p, multi_tail_test  # noqa: E402
TAILS = {"1%": (0.01,), "multi": (0.0003, 0.001, 0.003, 0.01)}

def per_class(ref, inc, yr, yi, k, tails):
    return max([multi_tail_test(ref[yr == c], inc[yi == c], tails) for c in np.unique(yi) if (yr == c).sum() >= 20] or [0.0]) - np.log10(k)

def checks(r, v):
    """name -> (penalised evidence, row scores, admitted) for variant v."""
    c, T = r["comps"], TAILS[v["tails"]]
    if v["label"] == "cur":
        lab = c["label"]; ev = multi_tail_test(*lab, T); rows = v2.neg_log_p(*lab)
    else:
        ref, inc, one = c[f"cf_{v['label']}"]; ev = multi_tail_test(ref, one, T)
        if v.get("class"): ev = max(max(ev, per_class(ref, one, r["y_ref"], r["y_inc"], r["k"], T)) - np.log10(2), 0.0)
        rows = extrapolated_neg_log_p(ref, inc)
    out = {"label": (ev - v["pen"]["label"], rows, True)}
    names = ["shift_full" if v["full"] else "shift"] + (["marg"] if v.get("marg") else [])
    for n in names:
        e = multi_tail_test(*c[n], T)
        out[n] = (e - v["pen"][n.split("_")[0]], (extrapolated_neg_log_p if v["full"] else v2.neg_log_p)(*c[n]), e >= v["gate"][n.split("_")[0]])
    return out

CUR_PEN = {"label": 0.0, "shift": float(np.log10(4))}
NEW_PEN = {"label": 0.1, "shift": 1.0, "marg": 1.0}
G = {"shift": 3.0, "marg": 1.0}
VARIANTS = {"v2 (current)": dict(label="cur", full=False, tails="1%", pen=CUR_PEN, gate=G),
            "v2 multi-tail": dict(label="cur", full=False, tails="multi", pen=CUR_PEN, gate=G),
            "cf gbm": dict(label="gbm", full=True, tails="multi", pen=NEW_PEN, gate=G),
            "cf lr": dict(label="lr", full=True, tails="multi", pen=NEW_PEN, gate=G),
            "cf gbm +class": dict(label="gbm", full=True, tails="multi", pen=NEW_PEN, gate=G, **{"class": True}),
            "cf gbm +marg": dict(label="gbm", full=True, tails="multi", pen=NEW_PEN, gate=G, marg=True),
            "cf gbm +class+marg": dict(label="gbm", full=True, tails="multi", pen=NEW_PEN, gate=G, marg=True, **{"class": True})}

def analyze():
    R = _load(); atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]]
    attacks = list(dict.fromkeys(r["attack"] for r in atk)); rates = sorted({r["rate"] for r in atk})
    for r in R:
        r["v"] = {}
        for n, v in VARIANTS.items():
            ch = checks(r, v)
            r["v"][n] = (max(e for e, _, _ in ch.values()), np.max([s for _, s, a in ch.values() if a], axis=0))
    V = list(VARIANTS); W = max(len(n) for n in V) + 2
    print(f"attacked {len(atk)}, clean {len(cln)}; seconds per batch:",
          {k: round(np.mean([r["secs"][k] for r in R]), 2) for k in R[0]["secs"]})
    print("\n" + " " * 28 + "".join(f"{n:>{W}}" for n in V))
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

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
