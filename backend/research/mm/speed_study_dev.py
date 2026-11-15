"""Dev-only study: cheaper models for the cross-fitted label check (it is ~2/3 of the time per batch; 5 GBM fits).
Collect:  PYTHONPATH=. python -m research.mm.speed_study_dev            (-> ~/.cache/veritas-logs/speed_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.mm.speed_study_dev --analyze
Same batches as label_study_dev (GBM baseline read from label_dev.pkl). Dev datasets only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research.mm import ref_study_dev as rf, label_study_dev as ls
from research.mm.detector import PENALTY, TAILS
from research.mm.shared import extrapolated_neg_log_p, multi_tail_test
from app.detection import veritas2 as v2

PKL = rf.rs.LOG + "speed_dev.pkl"
MODELS = {"lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)),
          "hgb40": lambda: HistGradientBoostingClassifier(max_iter=40, random_state=0),
          "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(25, weights="distance"))}

def crossfit(d, make):
    """As detector.label_crossfit_component with another model: (ref, inc mean of folds, one fold per incoming row)."""
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

def collect_one(job):
    d, s, out = job["d"], job["spec"], {}
    for n, make in MODELS.items():
        t = time.perf_counter(); out[n] = crossfit(d, make) + (time.perf_counter() - t,)
    return {"data_id": s["data_id"], "attack": s["attack"], "rate": s["rate"], "seed": s["seed"], "mask": job["mask"], "m": out}

def analyze():
    S, L, R = rf._load(PKL), rf._load(ls.PKL), rf._load(rf.PKL)
    assert len(S) == len(L) == len(R)
    for a, b, c in zip(S, L, R):
        assert (a["data_id"], a["seed"], a["attack"]) == (b["data_id"], b["seed"], b["attack"]) and (a["mask"] == b["mask"]).all()
        one = b["inc"][np.arange(b["inc"].shape[1]) % 5, np.arange(b["inc"].shape[1])]
        a["m"]["gbm"] = (b["ref"], b["inc"].mean(0), one, c["secs"]["label_cf"])
        a["oth"] = max(multi_tail_test(*c["comps"][n], TAILS) - PENALTY[n] for n in ("shift", "pixel", "dup", "cell", "resid", "chroma"))
        ev = lambda ref, one: max(max(multi_tail_test(ref, one, TAILS), ls.per_class(ref, one, b["y_ref"], b["y_inc"], b["k"])) - np.log10(2), 0.0)
        a["ev"] = {n: ev(m[0], m[2]) - PENALTY["label"] for n, m in a["m"].items()}
    V = ["gbm"] + list(MODELS); atk = [r for r in S if r["attack"]]; cln = [r for r in S if not r["attack"]]
    print(" " * 28 + "".join(f"{v:>10}" for v in V))
    for ds in dict.fromkeys(r["data_id"] for r in S):
        print(f"{'seconds ' + ds:<28}" + "".join(f"{np.mean([r['m'][v][3] for r in S if r['data_id'] == ds]):10.1f}" for v in V))
    for t in (1.3, 2.0):
        print(f"{'clean label >= ' + str(t):<28}" + "".join(f"{np.mean([r['ev'][v] >= t for r in cln]):10.3f}" for v in V))
        print(f"{'clean fused >= ' + str(t):<28}" + "".join(f"{np.mean([max(r['ev'][v], r['oth']) >= t for r in cln]):10.3f}" for v in V))
    for rate in (0.01, 0.05):
        for a in list(dict.fromkeys(r["attack"] for r in atk)) + ["ALL"]:
            g = [r for r in atk if r["rate"] == rate and a in ("ALL", r["attack"])]
            print(f"{a[:20]:<20}{rate:5.2f} TPR" + "".join(f"{np.mean([max(r['ev'][v], r['oth']) >= 1.3 for r in g]):10.2f}" for v in V))
            if "label" in a:
                print(f"{'':<20}{rate:5.2f} AP " + "".join(f"{np.mean([average_precision_score(r['mask'], extrapolated_neg_log_p(r['m'][v][0], r['m'][v][1])) for r in g]):10.2f}" for v in V))

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else rf.collect(collect_one, PKL)
