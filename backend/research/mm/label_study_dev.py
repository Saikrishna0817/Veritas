"""Dev-only study of the label check's batch evidence (it never fired on a clean batch in v6, so it is over-cautious).
Collect per-fold label scores:  PYTHONPATH=. python -m research.mm.label_study_dev   (-> ~/.cache/veritas-logs/label_dev.pkl)
Analyse:                        PYTHONPATH=. python -m research.mm.label_study_dev --analyze
Variants: incoming rows scored by the mean of the 5 fold models (v6) or by one fold model each (same footing as the
out-of-fold reference rows); pooled tail test, per-claimed-class tail test, or both. Dev datasets only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from research.mm import ref_study_dev as rf
from research.mm.detector import GATE, PENALTY, TAILS
from research.mm.shared import extrapolated_neg_log_p, multi_tail_test
from app.detection import veritas2 as v2

PKL = rf.rs.LOG + "label_dev.pkl"

def label_folds(job):
    """As detector.label_crossfit_component, but keeps each fold model's incoming scores."""
    d, s = job["d"], job["spec"]
    X, y = np.vstack([d["X_fit"], d["X_cal"]]), np.concatenate([d["y_fit"], d["y_cal"]])
    cv = (StratifiedKFold if np.bincount(y)[np.unique(y)].min() >= 5 else KFold)(5, shuffle=True, random_state=0)
    ref, inc = np.zeros(len(y)), np.zeros((5, len(d["y_inc"])))
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        score = lambda X_, y_: 1.0 - v2._probs(model, X_, d["n_classes"])[np.arange(len(y_)), y_]
        ref[te] = score(X[te], y[te]); inc[k] = score(d["X_inc"], d["y_inc"])
    return {"data_id": s["data_id"], "attack": s["attack"], "rate": s["rate"], "seed": s["seed"], "mask": job["mask"],
            "ref": ref, "inc": inc, "y_ref": y, "y_inc": d["y_inc"], "k": d["n_classes"]}

# ---- analysis ----
def per_class(ref, inc, yr, yi, k):
    """Best per-claimed-class tail test, Bonferroni over classes (targeted flips all claim one class)."""
    return max(max(multi_tail_test(ref[yr == c], inc[yi == c], TAILS) for c in range(k) if (yr == c).any() and (yi == c).any())
               - np.log10(k), 0.0)

def analyze():
    L, R = rf._load(PKL), rf._load(rf.PKL)
    assert len(L) == len(R)
    for a, b in zip(L, R):
        assert (a["data_id"], a["seed"], a["attack"], a["rate"]) == (b["data_id"], b["seed"], b["attack"], b["rate"]) and (a["mask"] == b["mask"]).all()
        assert np.allclose(a["inc"].mean(0), b["comps"]["label_cf"][1]) and np.allclose(a["ref"], b["comps"]["label_cf"][0])
        a["other"] = {n: b["comps"][n] for n in ("shift", "pixel", "dup", "cell", "resid", "chroma")}
        a["one"] = a["inc"][np.random.default_rng(len(a["y_inc"]) + a["seed"]).integers(0, 5, a["inc"].shape[1]), np.arange(a["inc"].shape[1])]
    atk = [r for r in L if r["attack"]]; cln = [r for r in L if not r["attack"]]
    attacks = list(dict.fromkeys(r["attack"] for r in atk)); rates = (0.01, 0.05); dss = list(dict.fromkeys(r["data_id"] for r in L))
    groups = [(f"{a[:21]:<21}{rate:5.2f}", [r for r in atk if r["attack"] == a and r["rate"] == rate]) for rate in rates for a in attacks]
    for r in L:
        r["oth_ev"] = {n: multi_tail_test(*c, TAILS) for n, c in r["other"].items()}
        r["ev"] = {}
        for sc in ("mean", "one"):
            inc = r["inc"].mean(0) if sc == "mean" else r["one"]
            p, c = multi_tail_test(r["ref"], inc, TAILS), per_class(r["ref"], inc, r["y_ref"], r["y_inc"], r["k"])
            r["ev"][(sc, "pooled")], r["ev"][(sc, "class")], r["ev"][(sc, "both")] = p, c, max(p, c) - np.log10(2)
    V = list(L[0]["ev"]); hdr = " " * 26 + "".join(f"{s + '/' + t:>13}" for s, t in V)
    lab_pen = lambda r, v: r["ev"][v] - PENALTY["label"]
    fused = lambda r, v: max([lab_pen(r, v)] + [e - PENALTY[n] for n, e in r["oth_ev"].items()])
    print(f"attacked {len(atk)}, clean {len(cln)}")
    print("\n1. Clean: label check raw evidence >= 1.3 / >= 2.0, and fused max penalised evidence >= 1.3 / >= 2.0"); print(hdr)
    for lab, rr in [("POOLED", cln)] + [(d, [r for r in cln if r["data_id"] == d]) for d in dss]:
        print(f"{'label ' + lab:<26}" + "".join(f"{np.mean([r['ev'][v] >= 1.3 for r in rr]):7.3f}/{np.mean([r['ev'][v] >= 2.0 for r in rr]):.3f}" for v in V) + f"  n={len(rr)}")
        print(f"{'fused ' + lab:<26}" + "".join(f"{np.mean([fused(r, v) >= 1.3 for r in rr]):7.3f}/{np.mean([fused(r, v) >= 2.0 for r in rr]):.3f}" for v in V))
    for t in (1.3, 2.0):
        print(f"\n2. Batch TPR at {t}: label check alone (penalised) | fused"); print(hdr)
        for lab, g in groups + [(f"{'ALL':<21}{rate:5.2f}", [r for r in atk if r["rate"] == rate]) for rate in rates]:
            print(lab + "".join(f"{np.mean([lab_pen(r, v) >= t for r in g]):6.2f}|{np.mean([fused(r, v) >= t for r in g]):.2f}" for v in V))
    print("\n3. Row AUPRC of the label list: mean-of-folds vs one fold per incoming row")
    for lab, g in groups:
        if "label" in lab: print(lab + "".join(f"{np.mean([average_precision_score(r['mask'], extrapolated_neg_log_p(r['ref'], r['inc'].mean(0) if sc == 'mean' else r['one'])) for r in g]):8.2f}" for sc in ("mean", "one")))
    print("\n4. Clean label raw evidence quantiles (50/90/95/99%)")
    for v in V: print(f"  {v}: " + " ".join(f"{q:.2f}" for q in np.quantile([r['ev'][v] for r in cln], [0.5, 0.9, 0.95, 0.99])))

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else rf.collect(label_folds, PKL)
