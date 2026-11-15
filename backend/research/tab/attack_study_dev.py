"""Dev-only study: name the attack behind a suspicious tabular batch from veritas2.attack_features.
Collect:  PYTHONPATH=. python -m research.tab.attack_study_dev              (-> ~/.cache/veritas-logs/attack_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.tab.attack_study_dev --analyze    (leave-one-dataset-out on dev)
Freeze:   PYTHONPATH=. python -m research.tab.attack_study_dev --freeze     (-> app/detection/attack_profile.json)
Trained on the dev attacked batches only; held-out accuracy comes from bench_v2 on the test split."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, json, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research import bench_v2
from app.detection import veritas2 as v2

PKL = os.path.expanduser("~/.cache/veritas-logs/attack_dev.pkl")
OUT = Path(v2.__file__).with_name("attack_profile.json")

def collect_one(spec):
    d, mask, cells, out = bench_v2.make_batch(spec); o = v2.detect(d)
    return {**{k: out[k] for k in ("dataset", "seed", "attack", "rate")}, "x": v2.attack_features(d, o), "evidence": o["batch_evidence"]}

def collect():
    m = bench_v2.load_manifest("full"); m["datasets"] = {"dev": m["datasets"]["dev"]}
    S = [s for s in bench_v2.batch_specs(m) if s["attack"]]
    done = len(_load()) if os.path.exists(PKL) else 0; t = time.perf_counter()
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

def _xy(R):
    return np.array([[r["x"][k] for k in v2.FEATURES] for r in R]), np.array([r["attack"] for r in R])

MODELS = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
          "gbm": lambda: HistGradientBoostingClassifier(max_iter=100, random_state=0)}

def analyze():
    R = _load(); dss = sorted({r["dataset"] for r in R}); X, y = _xy(R)
    flagged = np.array([r["evidence"] >= v2.LOW_RISK_AT for r in R]); rate = np.array([r["rate"] for r in R])
    for name, make in MODELS.items():
        pred = np.empty(len(y), dtype=object)
        for ds in dss:   # leave one dataset out: the attribution must transfer to unseen tables
            te = np.array([r["dataset"] == ds for r in R])
            pred[te] = make().fit(X[~te], y[~te]).predict(X[te])
        ok = pred == y
        print(f"\n{name}: accuracy all {ok.mean():.2f}, flagged batches {ok[flagged].mean():.2f} (n={flagged.sum()})")
        print("  by rate (flagged):", {r_: f"{ok[flagged & (rate == r_)].mean():.2f}" for r_ in sorted(set(rate))})
        for a in v2.ATTACKS:
            s = (y == a) & flagged
            print(f"  {a:<22} {ok[s].mean():.2f}  most confused with: {dict(zip(*np.unique(pred[s & ~ok], return_counts=True)))}")

def freeze():
    """The dev feature table as plain JSON (no pickle); veritas2.attack_profile fits its GBM on it at first use."""
    X, y = _xy(_load())
    OUT.write_text(json.dumps({"features": list(v2.FEATURES), "X": np.round(X, 5).tolist(), "y": y.tolist(),
                               "trained_on": "manifest-v2 dev split, attacked batches (research/tab/attack_study_dev.py); "
                                             "leave-one-dataset-out accuracy on flagged dev batches 0.76"}) + "\n")
    print(OUT, OUT.stat().st_size)

if __name__ == "__main__":
    {"--analyze": analyze, "--freeze": freeze}.get(next((a for a in sys.argv[1:]), ""), collect)()
