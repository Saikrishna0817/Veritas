"""Dev-only study: checking a table when there is no trusted reference (poison may be anywhere in the file).
Collect:  PYTHONPATH=. python -m research.tab.selfcheck_study_dev            (-> ~/.cache/veritas-logs/selfcheck_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.tab.selfcheck_study_dev --analyze
Poison is planted in the WHOLE dataset (the "reference" is contaminated too). Two ways to check it:
  A  fallback (today): first 70% (shuffled order) is the reference, only the last 30% is scored;
  B  self-check: 5 folds, each fold checked against the other four, so every row is scored; batch evidence is the
     strongest fold's evidence minus log10(5) (Bonferroni over folds).
Rows A never scores rank last, so the whole-file AUPRC shows what a user would miss. Dev datasets only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import itertools, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from research import bench_v2
from app.detection import veritas2 as v2

PKL = os.path.expanduser("~/.cache/veritas-logs/selfcheck_dev.pkl")
PKL2 = os.path.expanduser("~/.cache/veritas-logs/selfcheck2_dev.pkl")   # round 2: per-check row scores in every fold
SPLIT = os.getenv("SPLIT", "dev")   # "test" only for the one held-out run of the chosen design (--test)
PKL_TEST = os.path.expanduser("~/.cache/veritas-logs/selfcheck_test.pkl")
CHECKS = ("label", "shift", "resid")
K = 5
RATES, SEEDS, CLEAN_PER_SEED = (0.01, 0.05), 3, 4

def poisoned_dataset(spec):
    """The whole dataset with an attack planted at ``rate`` across all rows; returns X, y, mask, n_classes."""
    m = spec["manifest"]
    X, y, _ = bench_v2.load_dataset(spec["data_id"], m["max_rows"], m["random_seed"] + spec["seed"])
    X, y = X.copy(), y.copy()
    rng = np.random.default_rng([m["random_seed"], spec["data_id"], spec["seed"], spec["attack_index"] + 1, int(spec["rate"] * 1000), spec["rep"], 7])
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    counts = np.bincount(y)
    order = np.argsort(-counts, kind="stable")
    ctx = {"n_classes": int(y.max() + 1), "major": int(order[0]), "minor": int(order[1]),
           "ref_max": X.max(axis=0), "ref_std": X.std(axis=0), "ref_p95": np.percentile(X, 95, axis=0),
           "trigger": rng.choice(X.shape[1], size=min(3, X.shape[1]), replace=False),
           "shift_cols": rng.choice(X.shape[1], size=max(1, int(0.3 * X.shape[1])), replace=False)}
    mask = np.zeros(len(y), bool)
    if spec["attack"]:
        mask[bench_v2.ATTACKS[spec["attack"]](X, y, rng, max(1, round(spec["rate"] * len(y))), ctx)] = True
    return X, y, mask, ctx["n_classes"]

def run(X_ref, y_ref, X_inc, y_inc, k, cal):
    o = v2.detect(bench_v2.detector_inputs(X_ref, y_ref, X_inc, y_inc, k, cal))
    return o["row_scores"], o["batch_evidence"]

def collect_one(spec):
    X, y, mask, k = poisoned_dataset(spec)
    cal = spec["manifest"]["calibration_fraction_of_reference"]
    n_ref = int(len(y) * 0.7)
    t = time.perf_counter()
    rows_a, ev_a = run(X[:n_ref], y[:n_ref], X[n_ref:], y[n_ref:], k, cal)
    sec_a = time.perf_counter() - t
    t = time.perf_counter()
    rows_b, evs = np.zeros(len(y)), []
    for tr, te in KFold(K, shuffle=True, random_state=0).split(X):
        r, e = run(X[tr], y[tr], X[te], y[te], k, cal)
        rows_b[te] = r
        evs.append(e)
    return {"data_id": spec["data_id"], "seed": spec["seed"], "attack": spec["attack"], "rate": spec["rate"], "mask": mask,
            "n_ref": n_ref, "rows_a": rows_a, "ev_a": ev_a, "rows_b": rows_b, "evs_b": evs,
            "sec_a": sec_a, "sec_b": time.perf_counter() - t}

def checks_one(spec):
    """Self-check with each check's own row scores and evidence per fold (no gating), for ranking variants."""
    X, y, mask, k = poisoned_dataset(spec)
    cal = spec["manifest"]["calibration_fraction_of_reference"]
    rows = {c: np.zeros(len(y)) for c in CHECKS}
    ev = {c: [] for c in CHECKS}
    for tr, te in KFold(K, shuffle=True, random_state=0).split(X):
        d = bench_v2.detector_inputs(X[tr], y[tr], X[te], y[te], k, cal)
        ref, inc, one = v2.label_crossfit_component(d)
        rows["label"][te] = v2.extrapolated_neg_log_p(ref, inc); ev["label"].append(v2.multi_tail_test(ref, one))
        for c, (cal_s, inc_s) in (("shift", v2.shift_component(v2.full_reference(d))), ("resid", v2.residual_component(d))):
            rows[c][te] = v2.extrapolated_neg_log_p(cal_s, inc_s); ev[c].append(v2.multi_tail_test(cal_s, inc_s))
    return {"data_id": spec["data_id"], "seed": spec["seed"], "attack": spec["attack"], "rate": spec["rate"], "mask": mask, "rows": rows, "ev": ev}


def test_one(spec):
    """Held-out record: today's fallback (A) and the chosen self-check per-check lists, same poisoned dataset."""
    r = checks_one(spec)
    X, y, mask, k = poisoned_dataset(spec)
    n_ref = int(len(y) * 0.7)
    rows_a, ev_a = run(X[:n_ref], y[:n_ref], X[n_ref:], y[n_ref:], k, spec["manifest"]["calibration_fraction_of_reference"])
    return {**r, "n_ref": n_ref, "rows_a": rows_a, "ev_a": ev_a}


def specs():
    m = bench_v2.load_manifest("full"); m["datasets"] = {SPLIT: m["datasets"][SPLIT]}
    m.update(poison_rates=list(RATES), seeds=SEEDS, clean_batches_per_seed=CLEAN_PER_SEED)
    return bench_v2.batch_specs(m)

def collect(fn=None, path=PKL):
    S = specs(); done = len(_load(path)) if os.path.exists(path) else 0; t = time.perf_counter()
    with ProcessPoolExecutor(int(os.getenv("WORKERS", 4))) as pool, open(path, "ab") as f:
        for i, r in enumerate(pool.map(fn or collect_one, itertools.islice(S, done, None), chunksize=1), done + 1):
            pickle.dump(r, f); f.flush()
            if i % 20 == 0: print(i, len(S), f"{time.perf_counter()-t:.0f}s", flush=True)
    print("done", len(S))

def _load(path=PKL):
    out = []
    with open(path, "rb") as f:   # this study's own local output
        while True:
            try: out.append(pickle.load(f))
            except EOFError: return out

def analyze():
    R = _load(); atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]]
    ev_b = lambda r: max(r["evs_b"]) - np.log10(K)
    def full_a(r):   # rows A never scored rank below every scored row
        s = np.full(len(r["mask"]), -1e9); s[r["n_ref"]:] = r["rows_a"]; return s
    print(f"attacked {len(atk)}, clean {len(cln)}; seconds per dataset A {np.mean([r['sec_a'] for r in R]):.1f}, B {np.mean([r['sec_b'] for r in R]):.1f}")
    for t in (1.3, 2.0):
        print(f"clean flagged at {t}:  A {np.mean([r['ev_a'] >= t for r in cln]):.3f}   B {np.mean([ev_b(r) >= t for r in cln]):.3f}")
    print(f"\n{'attack':<22}rate  share scored A/B   whole-file AUPRC A/B   flagged at 1.3 A/B")
    for rate in RATES:
        for a in list(dict.fromkeys(r["attack"] for r in atk)) + ["MEAN"]:
            g = [r for r in atk if r["rate"] == rate and a in ("MEAN", r["attack"])]
            sc = np.mean([r["mask"][r["n_ref"]:].sum() / max(r["mask"].sum(), 1) for r in g])
            apa = np.mean([average_precision_score(r["mask"], full_a(r)) for r in g])
            apb = np.mean([average_precision_score(r["mask"], r["rows_b"]) for r in g])
            print(f"{a[:21]:<22}{rate:4.2f}  {sc:.2f} / 1.00        {apa:.2f} / {apb:.2f}            "
                  f"{np.mean([r['ev_a'] >= 1.3 for r in g]):.2f} / {np.mean([ev_b(r) >= 1.3 for r in g]):.2f}")

def analyze2():
    """Ranking variants on the per-check self-check scores; batch evidence = best check's best fold, Bonferroni."""
    R = _load(PKL2); atk = [r for r in R if r["attack"]]; cln = [r for r in R if not r["attack"]]
    pen = {"label": v2.LABEL_PENALTY, "shift": v2.SHIFT_PENALTY, "resid": v2.RESID_PENALTY}
    batch = lambda r: max(max(r["ev"][c]) - pen[c] for c in CHECKS) - np.log10(K)
    V = {"label only": lambda r: r["rows"]["label"],
         "max of all (ungated)": lambda r: np.max([r["rows"][c] for c in CHECKS], 0),
         "label + resid": lambda r: np.maximum(r["rows"]["label"], r["rows"]["resid"]),
         "gated as today": lambda r: np.max([r["rows"][c] for c in CHECKS if c == "label" or max(r["ev"][c]) >= (v2.GATE if c == "shift" else v2.RESID_GATE)], 0)}
    print(f"clean flagged at 1.3 / 2.0: {np.mean([batch(r) >= 1.3 for r in cln]):.3f} / {np.mean([batch(r) >= 2 for r in cln]):.3f}  (n={len(cln)})")
    print(f"\n{'attack':<22}rate " + "".join(f"{n:>22}" for n in V) + "   flagged 1.3")
    for rate in RATES:
        for a in list(dict.fromkeys(r["attack"] for r in atk)) + ["MEAN"]:
            g = [r for r in atk if r["rate"] == rate and a in ("MEAN", r["attack"])]
            print(f"{a[:21]:<22}{rate:4.2f}" + "".join(f"{np.mean([average_precision_score(r['mask'], f(r)) for r in g]):22.2f}" for f in V.values())
                  + f"{np.mean([batch(r) >= 1.3 for r in g]):12.2f}")


if __name__ == "__main__":
    if "--analyze2" in sys.argv: analyze2()
    elif "--analyze" in sys.argv: analyze()
    elif "--checks" in sys.argv: collect(checks_one, PKL2)
    elif "--test" in sys.argv:
        assert SPLIT == "test"
        collect(test_one, PKL_TEST)
    else: collect()
