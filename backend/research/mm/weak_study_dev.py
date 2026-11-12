"""Dev-only study of the two weak spots at 1%: batch verdicts (multi-tail test) and class-noise (noise-floor checks).
Collect noise-floor scores for the batches of shared_dev.pkl and report_dev_clean.pkl (regenerated in the same order):
          PYTHONPATH=. python -m research.mm.weak_study_dev            (-> ~/.cache/veritas-logs/weak_dev.pkl)
Analyse:  PYTHONPATH=. python -m research.mm.weak_study_dev --analyze
Dev datasets only; never point at the test split."""
import pickle, sys, time
import numpy as np
from sklearn.metrics import average_precision_score
from research.mm import report_study_dev as rs, shared_study_dev as ss  # sets thread env, loads bench_mm
from research.mm.shared import _pool, _residual, _split, extrapolated_neg_log_p
from research.mm.detector import GATE, PENALTY
from research import bench_mm
from app.detection import veritas2 as v2

PKL = rs.LOG + "weak_dev.pkl"
NAMES = rs.NAMES

def noise_floor(d, block=4):
    """Per image: low quantiles over 4x4 blocks of the mean squared high-pass residual, for luminance and for
    colour differences. Added noise lifts the quietest blocks; independent per-channel noise lifts the colour ones."""
    r = _residual(_pool(d)); n, h, w, _ = r.shape; H, W = h // block * block, w // block * block
    blocks = lambda a: (a[:, :H, :W] ** 2).reshape(n, H // block, block, W // block, block).mean(axis=(2, 4)).reshape(n, -1)
    lum = blocks(r.mean(-1)); chr_ = (blocks(r[..., 0] - r[..., 1]) + blocks(r[..., 1] - r[..., 2])) / 2
    out = {"lum_q10": np.percentile(lum, 10, axis=1), "lum_min": lum.min(1), "lum_q25": np.percentile(lum, 25, axis=1),
           "chr_q10": np.percentile(chr_, 10, axis=1), "chr_min": chr_.min(1), "chr_q25": np.percentile(chr_, 25, axis=1)}
    return {k: _split(d, v) for k, v in out.items()}

def collect():
    m = bench_mm.load_manifest("full"); m["datasets"] = {"dev": m["datasets"]["dev"]}
    m1 = {**m, "poison_rates": [0.01, 0.05], "clean_batches_per_seed": 12, "seeds": 5}      # as shared_study_dev.collect
    n_new = rs.MIN_CLEAN - 120; per_seed = len(m["datasets"]["dev"]) * m["clean_batches_per_seed"]
    m2 = {**m, "attacks": [], "poison_rates": [], "seeds": rs.FIRST_NEW_SEED + -(-n_new // per_seed)}  # as report_study_dev.collect
    t = time.perf_counter(); out = {"old": [], "new": []}
    for key, gen in (("old", ss.jobs(m1)), ("new", rs.jobs(m2))):
        for i, job in enumerate(gen, 1):
            s = job["spec"]
            out[key].append({"data_id": s["data_id"], "attack": s["attack"], "rate": s["rate"], "seed": s["seed"],
                             "mask": job["mask"], "comps": noise_floor(job["d"])})
            if i % 50 == 0: print(key, i, f"{time.perf_counter()-t:.0f}s", flush=True)
    pickle.dump(out, open(PKL, "wb")); print("done", len(out["old"]), len(out["new"]))

# ---- analysis ----
GRIDS = {"1%": (0.01,), "0.1-1%": (0.001, 0.003, 0.01), "0.03-1%": (0.0003, 0.001, 0.003, 0.01),
         "0.03-3%": (0.0003, 0.001, 0.003, 0.01, 0.03)}

def multi_tail(cal, inc, fracs):
    """Largest exact tail-test evidence over several tail sizes, Bonferroni-corrected for the number tried."""
    return max(max(v2.exact_tail_test(cal, inc, 1 - f) for f in fracs) - np.log10(len(fracs)), 0.0)

def load():
    # pickles here are this study's own local output
    R = pickle.load(open(rs.OLD_PKL, "rb")); N = pickle.load(open(rs.PKL, "rb")); W = pickle.load(open(PKL, "rb"))
    for r in R: r["comps"] = {n: r["comps"][rs.OLD_KEY.get(n, n)] for n in NAMES}
    assert len(R) == len(W["old"]) and len(N) == len(W["new"])
    for a, b in list(zip(R, W["old"])) + list(zip(N, W["new"])):
        assert a["data_id"] == b["data_id"] and a["seed"] == b["seed"] and a.get("attack") == b["attack"]
        if a.get("attack"): assert (a["mask"] == b["mask"]).all()
        assert len(a["comps"]["label"][1]) == len(b["comps"]["lum_q10"][1])
        a["comps"].update(b["comps"]); a.setdefault("attack", None)
    return [r for r in R if r["attack"]], [r for r in R if not r["attack"]] + N

def analyze():
    atk, cln = load(); attacks = list(dict.fromkeys(r["attack"] for r in atk)); rates = (0.01, 0.05)
    groups = [(f"{a[:21]:<21}{rate:5.2f}", [r for r in atk if r["attack"] == a and r["rate"] == rate]) for rate in rates for a in attacks]
    ap = lambda r, s: average_precision_score(r["mask"], s)
    cand = [k for k in cln[-1]["comps"] if k not in NAMES]
    print(f"attacked {len(atk)}, clean {len(cln)}")
    print("\n1. Row AUPRC of each noise candidate alone (raw score)"); print(" " * 26 + "".join(f"{c:>9}" for c in cand))
    for lab, rs_ in groups: print(lab + "".join(f"{np.mean([ap(r, r['comps'][c][1]) for r in rs_]):9.2f}" for c in cand))
    print("   clean share with raw evidence >= 1.3 / >= 2.0 (1% tail), per dataset")
    for ds in dict.fromkeys(r["data_id"] for r in cln):
        print(f"   {ds:<14}" + "".join(f"  {c} {np.mean([v2.exact_tail_test(*r['comps'][c]) >= 1.3 for r in cln if r['data_id'] == ds]):.3f}/"
                                      f"{np.mean([v2.exact_tail_test(*r['comps'][c]) >= 2.0 for r in cln if r['data_id'] == ds]):.3f}" for c in cand))
    print("   class_noise AUPRC per dataset")
    for ds in dict.fromkeys(r["data_id"] for r in atk):
        for rate in rates:
            rs_ = [r for r in atk if r["attack"] == "class_noise" and r["rate"] == rate and r["data_id"] == ds]
            print(f"   {ds:<14}{rate:5.2f}" + "".join(f"{np.mean([ap(r, r['comps'][c][1]) for r in rs_]):9.2f}" for c in cand))

    def evid(r, names, grid):  # penalised evidence per check
        k = (tuple(names), grid)
        if k not in r.setdefault("_e", {}):
            r["_e"][k] = {n: multi_tail(*r["comps"][n], GRIDS[grid]) - PENALTY.get(n, 1.0) for n in names}
        return r["_e"][k]
    def fused(r, names, grid):
        e = evid(r, names, grid)
        return np.max([extrapolated_neg_log_p(*r["comps"][n]) for n in names
                       if n == "label" or e[n] + PENALTY.get(n, 1.0) >= GATE.get(n, 1.0)], axis=0)
    SETS = {"six": NAMES, **{f"six+{c}": NAMES + [c] for c in ("lum_q10", "chr_q10", "lum_min", "chr_min")},
            "six+lum+chr_q10": NAMES + ["lum_q10", "chr_q10"]}
    variants = [("six", g) for g in GRIDS] + [(s, g) for s in list(SETS)[1:] for g in ("1%", "0.03-1%")]
    hdr = "".join(f"{s[4:] or 'six':>10}" if s != "six" else f"{'six':>10}" for s, _ in variants)
    print("\nvariants (check set / tail grid):"); [print(f"  v{i}: {s} / {g}") for i, (s, g) in enumerate(variants)]
    V = "".join(f"{'v' + str(i):>7}" for i in range(len(variants)))
    print("\n2. Clean batches: share with max penalised evidence >= 1.3, >= 2.0, >= 4.0"); print(" " * 26 + V)
    for t in (1.3, 2.0, 4.0):
        print(f"{'>= ' + str(t):<26}" + "".join(f"{np.mean([max(evid(r, SETS[s], g).values()) >= t for r in cln]):7.3f}" for s, g in variants))
    for ds in dict.fromkeys(r["data_id"] for r in cln):
        print(f"{'>= 1.3 ' + ds:<26}" + "".join(f"{np.mean([max(evid(r, SETS[s], g).values()) >= 1.3 for r in cln if r['data_id'] == ds]):7.3f}" for s, g in variants))
    print("\n3. Batch TPR at cut-off 1.3"); print(" " * 26 + V)
    for lab, rs_ in groups + [(f"{'ALL':<21}{rate:5.2f}", [r for r in atk if r["rate"] == rate]) for rate in rates]:
        print(lab + "".join(f"{np.mean([max(evid(r, SETS[s], g).values()) >= 1.3 for r in rs_]):7.2f}" for s, g in variants))
    print("\n4. Batch TPR at cut-off 2.0"); print(" " * 26 + V)
    for lab, rs_ in [(f"{'ALL':<21}{rate:5.2f}", [r for r in atk if r["rate"] == rate]) for rate in rates]:
        print(lab + "".join(f"{np.mean([max(evid(r, SETS[s], g).values()) >= 2.0 for r in rs_]):7.2f}" for s, g in variants))
    print("\n5. Fused row AUPRC (gates use the variant's evidence)"); print(" " * 26 + V)
    for lab, rs_ in groups: print(lab + "".join(f"{np.mean([ap(r, fused(r, SETS[s], g)) for r in rs_]):7.2f}" for s, g in variants))
    for rate in rates:
        print(f"{'MEAN':<21}{rate:5.2f}" + "".join(f"{np.mean([np.mean([ap(r, fused(r, SETS[s], g)) for r in rs_]) for lab, rs_ in groups if lab.endswith(f'{rate:5.2f}')]):7.2f}" for s, g in variants))

if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
