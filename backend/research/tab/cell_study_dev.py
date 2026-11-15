"""Dev-only study: which cells of a tabular batch were poisoned (feature values and labels, not just rows).
Collect:  PYTHONPATH=. python -m research.tab.cell_study_dev            (-> ~/.cache/veritas-logs/cell_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.tab.cell_study_dev --analyze
Ground truth is bench_v2.make_batch's cell mask: feature columns, then the label. Dev datasets of manifest-v2 only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from research import bench_v2
from app.detection import veritas2 as v2

SPLIT = os.getenv("SPLIT", "dev")   # "test" only for the one held-out run of the frozen variant (report())
PKL = os.path.expanduser(f"~/.cache/veritas-logs/cell_{SPLIT}.pkl")

def _per_col(cal, inc):
    """Column-wise -log10 share of clean reference cells at least as extreme."""
    return np.stack([v2.neg_log_p(cal[:, j], inc[:, j]) for j in range(inc.shape[1])], axis=1)

def conditional_residuals(d):
    """|x_j - prediction from the other features and the one-hot label| (ridge fit on X_fit), for cal and inc."""
    k = d["n_classes"]; onehot = lambda y: np.eye(k)[y]
    Zf, Zc, Zi = (np.c_[d[f"X_{s}"], onehot(d[f"y_{s}"])] for s in ("fit", "cal", "inc"))
    n = d["X_fit"].shape[1]; rc, ri = np.zeros((len(Zc), n)), np.zeros((len(Zi), n))
    for j in range(n):
        keep = np.arange(Zf.shape[1]) != j
        m = Ridge(1.0).fit(Zf[:, keep], Zf[:, j])
        rc[:, j], ri[:, j] = np.abs(Zc[:, j] - m.predict(Zc[:, keep])), np.abs(Zi[:, j] - m.predict(Zi[:, keep]))
    return rc, ri

def collect_one(spec):
    d, mask, cells, out = bench_v2.make_batch(spec)
    lab, shf = v2.label_component(d), v2.shift_component(d)
    rc, ri = conditional_residuals(d)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return {**{k: out[k] for k in ("data_id", "dataset", "seed", "attack", "rate", "rep")}, "mask": mask, "cells": cells,
            "label": v2.neg_log_p(*lab), "label_ev": v2.exact_tail_test(*lab),
            "shift": v2.neg_log_p(*shf), "shift_ev": v2.exact_tail_test(*shf),
            "marg": f32(_per_col(np.abs(d["X_cal"]), np.abs(d["X_inc"]))), "cond": f32(_per_col(rc, ri))}

def collect():
    m = bench_v2.load_manifest("full"); m["datasets"] = {SPLIT: m["datasets"][SPLIT]}
    specs = bench_v2.batch_specs(m)
    for ds in m["datasets"][SPLIT]: bench_v2._openml(ds["data_id"])
    done = len(_load()) if os.path.exists(PKL) else 0; t = time.perf_counter()
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 5))) as pool, open(PKL, "ab") as f:
        for i, r in enumerate(pool.map(collect_one, itertools.islice(specs, done, None), chunksize=2), done + 1):
            pickle.dump(r, f); f.flush()
            if i % 50 == 0: print(i, len(specs), f"{time.perf_counter()-t:.0f}s", flush=True)
    print("done", len(specs))

def _load():
    out = []
    with open(PKL, "rb") as f:   # this study's own local output
        while True:
            try: out.append(pickle.load(f))
            except EOFError: return out

# ---- analysis ----
def row_score(r):
    """veritas2's row ranking: label check, plus the shift check when its batch test passes the gate."""
    return np.maximum(r["label"], r["shift"]) if r["shift_ev"] >= v2.GATE else r["label"]

def attribution(r, feat):
    """(n, n_features + 1) column evidence: feature columns from ``feat``, the label column from the label check."""
    F = {"marg": r["marg"], "max": np.maximum(r["marg"], r["cond"]) - np.log10(2)}[feat.removesuffix("-pen")]
    if feat.endswith("-pen"): F = F - np.log10(F.shape[1])  # the best of d feature columns vs the one label column
    return np.c_[F, r["label"]]

VARIANTS = {"row only": lambda r, A: np.repeat(row_score(r)[:, None], A.shape[1], 1),
            "column only": lambda r, A: A,
            "row + column": lambda r, A: row_score(r)[:, None] + A,
            "min(row, column)": lambda r, A: np.minimum(row_score(r)[:, None], A),
            "row2 + column": lambda r, A: np.maximum(row_score(r), A[:, :-1].max(1))[:, None] + A}

def analyze():
    R = [r for r in _load() if r["attack"]]; rng = np.random.default_rng(0)
    attacks = list(dict.fromkeys(r["attack"] for r in R)); rates = sorted({r["rate"] for r in R})
    print(f"attacked batches {len(R)}; datasets {sorted({r['dataset'] for r in R})}")
    print("share of changed cells that are labels / features, per attack:",
          {a: f"{np.mean([r['cells'][:, -1].sum() / max(r['cells'].sum(), 1) for r in R if r['attack'] == a]):.2f}" for a in attacks})
    jitter = lambda S: S + 1e-9 * rng.random(S.shape)  # ties (e.g. row only) at random, not in file order
    cell_ap = lambda r, S: average_precision_score(r["cells"].ravel(), jitter(S).ravel())
    def within(r, A):   # column ranking inside the truly poisoned rows that have both changed and unchanged cells
        rows = [i for i in np.flatnonzero(r["mask"]) if 0 < r["cells"][i].sum() < r["cells"].shape[1]]
        return np.mean([average_precision_score(r["cells"][i], jitter(A[i])) for i in rows]) if rows else np.nan
    for feat in sys.argv[2:] or ("marg", "max", "marg-pen", "max-pen"):
        print(f"\n== feature attribution: {feat} ==")
        print(f"{'cell AUPRC':<30}" + "".join(f"{v:>18}" for v in VARIANTS) + f"{'col AP in bad rows':>20}{'base rate':>11}")
        for rate in rates:
            for a in attacks + ["MEAN"]:
                g = [r for r in R if r["rate"] == rate and a in ("MEAN", r["attack"])]
                if a == "MEAN":
                    vals = [np.mean([np.mean([cell_ap(r, f(r, attribution(r, feat))) for r in R if r["rate"] == rate and r["attack"] == b]) for b in attacks]) for f in VARIANTS.values()]
                else:
                    vals = [np.mean([cell_ap(r, f(r, attribution(r, feat))) for r in g]) for f in VARIANTS.values()]
                w = np.nanmean([within(r, attribution(r, feat)) for r in g])
                print(f"{a[:22]:<24}{rate:5.2f} " + "".join(f"{v:18.3f}" for v in vals) + f"{w:20.2f}{np.mean([r['cells'].mean() for r in g]):11.4f}")

def report():
    """Frozen variant (row + column, marg-pen; = veritas2.cell_scores) against row-only, as a markdown table."""
    R = [r for r in _load() if r["attack"]]; rng = np.random.default_rng(0)
    attacks = list(dict.fromkeys(r["attack"] for r in R)); rates = sorted({r["rate"] for r in R})
    ap = lambda r, S: average_precision_score(r["cells"].ravel(), (S + 1e-9 * rng.random(S.shape)).ravel())
    def within(r, A):
        rows = [i for i in np.flatnonzero(r["mask"]) if 0 < r["cells"][i].sum() < r["cells"].shape[1]]
        return [average_precision_score(r["cells"][i], A[i] + 1e-9 * rng.random(A.shape[1])) for i in rows]
    lines = [f"Split {SPLIT}: {len(R)} attacked batches, datasets: {', '.join(sorted({r['dataset'] for r in R}))}.", "",
             "Cell AUPRC (random = the share of changed cells), cell-scores / row-only; then the column ranking inside truly poisoned rows.", "",
             "| Attack | " + " | ".join(f"{x:.0%} cells" for x in rates) + " | " + " | ".join(f"{x:.0%} in-row" for x in rates) + " |",
             "|---|" + "---|" * (2 * len(rates))]
    for a in attacks + ["Mean"]:
        cl, wr = [], []
        for rate in rates:
            per = [b for b in attacks if a in ("Mean", b)]
            c = [np.mean([ap(r, VARIANTS["row + column"](r, attribution(r, "marg-pen"))) for r in R if r["attack"] == b and r["rate"] == rate]) for b in per]
            o = [np.mean([ap(r, VARIANTS["row only"](r, attribution(r, "marg-pen"))) for r in R if r["attack"] == b and r["rate"] == rate]) for b in per]
            w = [np.nanmean(sum((within(r, attribution(r, "marg-pen")) for r in R if r["attack"] == b and r["rate"] == rate), []) or [np.nan]) for b in per]
            cl.append(f"{np.mean(c):.2f} / {np.mean(o):.2f}"); wr.append(f"{np.nanmean(w):.2f}")
        lines.append(f"| {a} | " + " | ".join(cl) + " | " + " | ".join(wr) + " |")
    print("\n".join(lines))

if __name__ == "__main__":
    report() if "--report" in sys.argv else analyze() if "--analyze" in sys.argv else collect()
