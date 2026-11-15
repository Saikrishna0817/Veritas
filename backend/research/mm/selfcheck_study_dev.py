"""Dev-only study: images with no trusted reference (poison anywhere in the set): which list finds what.
Collect:  PYTHONPATH=. python -m research.mm.selfcheck_study_dev            (-> ~/.cache/veritas-logs/mm_selfcheck_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.mm.selfcheck_study_dev --analyze
1,500 images per set, poison planted across all of them, veritas_mm.self_check (5 folds). Dev datasets only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score
from app.detection import veritas_mm
from research import bench_mm
from research.mm import datasets
from research.mm.attacks import ATTACKS, make_context
from research.mm.embed import clean_embeddings, embed_images

SPLIT = os.getenv("SPLIT", "dev")
PKL = os.path.expanduser(f"~/.cache/veritas-logs/mm_selfcheck_{SPLIT}.pkl")
N, RATES, SEEDS, CLEAN = 1500, (0.01, 0.05), 2, 4
SETS = {"dev": ("fashion_mnist", "cifar100"), "test": ("cifar10", "gtsrb")}

def specs():
    out = []
    for ds in SETS[SPLIT]:
        for seed in range(SEEDS):
            out += [{"data_id": ds, "seed": seed, "attack": None, "rate": 0.0, "rep": r} for r in range(CLEAN)]
            out += [{"data_id": ds, "seed": seed, "attack": a, "rate": rate, "rep": 0} for a in ATTACKS for rate in RATES]
    return out

_cache = {}
def one(spec):
    if spec["data_id"] not in _cache:
        ds = datasets.load(spec["data_id"]); _cache.clear(); _cache[spec["data_id"]] = (ds, clean_embeddings(ds))
    ds, emb = _cache[spec["data_id"]]
    rng = np.random.default_rng([20261115, bench_mm.zlib_id(spec["data_id"]), spec["seed"], list(ATTACKS).index(spec["attack"]) + 1 if spec["attack"] else 0, int(spec["rate"] * 1000), spec["rep"]])
    sel = rng.permutation(len(ds.y))[:N]
    raw, y, E = ds.raw[sel].copy(), ds.y[sel].copy(), emb[sel].copy()
    mask = np.zeros(N, bool)
    if spec["attack"]:
        idx = ATTACKS[spec["attack"]](raw, y, rng, max(1, round(spec["rate"] * N)), make_context(raw, ds.n_classes, rng))
        mask[idx] = True
        E[idx] = embed_images(raw[idx])
    t = time.perf_counter()
    out = veritas_mm.self_check(raw, E, y, ds.n_classes)
    return {**spec, "mask": mask, "secs": time.perf_counter() - t, "rows": {k: v["rows"] for k, v in out.items()}, "ev": {k: v["evidence"] for k, v in out.items()}}

def collect():
    S = specs(); done = len(_load()) if os.path.exists(PKL) else 0; t = time.perf_counter()
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 3))) as pool, open(PKL, "ab") as f:
        for i, r in enumerate(pool.map(one, itertools.islice(S, done, None), chunksize=4), done + 1):
            pickle.dump(r, f); f.flush()
            if i % 10 == 0: print(i, len(S), f"{time.perf_counter()-t:.0f}s", flush=True)
    print("done", len(S))

def _load():
    out = []
    with open(PKL, "rb") as f:   # this study's own local output
        while True:
            try: out.append(pickle.load(f))
            except EOFError: return out

CONTENT = ("shift", "pixel", "dup", "cell", "resid", "chroma")

def analyze():
    R = _load(); atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]]
    screen = lambda r: max(r["ev"].values())
    print(f"attacked {len(atk)}, clean {len(cln)}; {np.mean([r['secs'] for r in R]):.0f} s per set; clean screened >=1.3: {np.mean([screen(r) >= 1.3 for r in cln]):.3f}")
    lists = {**{c: (lambda r, c=c: r["rows"][c]) for c in ("label",) + CONTENT},
             "content (max)": lambda r: np.max([r["rows"][c] for c in CONTENT], 0),
             "content (gated)": lambda r: np.max([r["rows"][c] for c in CONTENT if r["ev"][c] + veritas_mm.PENALTY[c] >= veritas_mm.GATE[c]] or [np.zeros(len(r["mask"]))], 0)}
    print(f"{'attack':<22}rate " + "".join(f"{k[:13]:>14}" for k in lists) + "   flagged")
    for rate in RATES:
        for a in list(ATTACKS) + ["MEAN"]:
            g = [r for r in atk if r["rate"] == rate and a in ("MEAN", r["attack"])]
            print(f"{a[:21]:<22}{rate:4.2f}" + "".join(f"{np.mean([average_precision_score(r['mask'], f(r)) for r in g]):14.2f}" for f in lists.values())
                  + f"{np.mean([screen(r) >= 1.3 for r in g]):10.2f}")

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
