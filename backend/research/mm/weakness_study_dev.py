"""Dev-only study of three image weaknesses found on held-out data (docs/image-benchmark-findings.md, 2026-11-16/18):
1. clean alarms with many classes and few images each (the within-class check `sig` on Tiny-ImageNet);
2. a smooth low-frequency trigger on correctly labelled images is not detected (adaptive v2);
3. label flips on data the ImageNet embedding separates poorly (SVHN, Tiny-ImageNet).
Candidates are scored side by side on dev batches; nothing here touches a test dataset.

Collect:  CONFIG=fmnist|c100|c100small PYTHONPATH=. python -m research.mm.weakness_study_dev
Analyse:  PYTHONPATH=. python -m research.mm.weakness_study_dev --analyze
c100small uses 1,500-image batches (about 9 reference and 6 incoming images per class, as Tiny-ImageNet's 200 classes
in 3,000-image batches)."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "2")
import pickle, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score
os.environ.setdefault("VERITAS_MM_MANIFEST", "manifest-mm-images-v9.json")
from research import bench_mm
from research.mm import ext_attacks
from research.adaptive.attacks import _backdoor, lowfreq_trigger
from app.detection import veritas2 as v2, veritas_mm as vmm

CONFIGS = {"fmnist": ("fashion_mnist", 3000), "c100": ("cifar100", 3000), "c100small": ("cifar100", 1500)}
ATTACKS = ["cl_lowfreq16", "cl_lowfreq32", "clean_label_backdoor", "label_flip_random", "label_flip_targeted"]
RATES, SEEDS, CLEAN_PER_SEED = [0.01, 0.05], 3, 20
LOG = os.path.expanduser("~/.cache/veritas-logs")
STAGE = int(os.getenv("STAGE", "1"))   # 2: smooth-trigger candidates only, on fewer batches


def _cl_lowfreq(s):
    """The adaptive v2 clean-label smooth trigger as a bench_mm attack (its pattern drawn from the batch's rng)."""
    def attack(raw, y, rng, n, ctx):
        field = gaussian_filter(rng.normal(size=raw.shape[1:]), sigma=(4, 4, 0), mode="wrap")
        ctx["lowfreq"] = (field / np.abs(field).max()).astype(np.float32)
        return _backdoor(lowfreq_trigger, clean_label=True)(raw, y, rng, n, ctx, s)
    return attack


ext_attacks.ATTACKS.update({"cl_lowfreq16": _cl_lowfreq(16), "cl_lowfreq32": _cl_lowfreq(32)})


def lowpass(R: np.ndarray, k: int = 4) -> np.ndarray:
    """(N, 32, 32, 3) -> (N, 8, 8, 3) block means: only the smooth part of the image."""
    n, h, w, c = R.shape
    return R.astype(np.float32).reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4))


def project_out(view, ref, k):
    """view with the top-k principal directions of the reference's view removed: natural images vary mostly at low
    frequencies, which swamps a smooth trigger's direction in the within-class SVD."""
    def f(a):
        x = view(a).reshape(len(a), -1)
        R = view(ref).reshape(len(ref), -1)
        mu = R.mean(0)
        V = np.linalg.svd(R - mu, full_matrices=False)[2][:k]
        return (x - mu) - (x - mu) @ V.T @ V
    return f


def sig_variant(r, view, min_class):
    """class_signature_component on another view of the pixels and another minimum class size."""
    saved = vmm._residual
    if view == "low":
        vmm._residual = lambda a: lowpass(a)  # noqa: E731
    elif view.startswith("lowproj"):
        vmm._residual = project_out(lowpass, r["R_cal"], int(view[7:]))
    elif view.startswith("hpproj"):
        vmm._residual = project_out(saved, r["R_cal"], int(view[6:]))
    try:
        return vmm.class_signature_component(r, min_class=min_class)
    finally:
        vmm._residual = saved


def label_concat(d):
    """The label check on the embedding joined with the pixel features."""
    j = {**d, **{f"X_{s}": np.hstack([d[f"X_{s}"], d[f"P_{s}"]]) for s in ("fit", "cal", "inc")}}
    ref, inc, one = vmm.label_crossfit_component(j)
    return vmm.label_evidence(j, ref, one), v2.extrapolated_neg_log_p(ref, inc)


def one(job):
    d, mask = job["d"], job["mask"]
    rep = vmm.report(d)
    r = vmm.full_reference(d)
    out = {"base": {n: (c["raw"], c["rows"]) for n, c in rep.items()},
           "veritas": vmm.detect(d)["batch_evidence"]}
    for view, mcs in {1: (("hp", (6, 10, 15, 20)), ("low", (6, 10, 15, 20))),
                      2: (("hp", (6,)), ("lowproj5", (6,)), ("lowproj20", (6,)), ("lowproj50", (6,)), ("hpproj20", (6,))),
                      3: (("hp", (6,)), ("lowproj50", (6,)), ("lowproj100", (6,)), ("lowproj150", (6,)))}[STAGE]:
        for mc in mcs:
            cal, inc = sig_variant(r, view, mc)
            out[f"sig_{view}{mc}"] = (v2.multi_tail_test(cal, inc, vmm.TAILS), v2.extrapolated_neg_log_p(cal, inc))
    if STAGE == 1:
        out["label_concat"] = label_concat(d)
    return {"spec": {k: job["spec"][k] for k in ("data_id", "seed", "attack", "rate", "rep")}, "mask": mask, "scores": out}


def collect(config):
    data_id, rows = CONFIGS[config]
    m = bench_mm.load_manifest()
    m["datasets"] = {"dev": [{"data_id": data_id}]}
    m.update(attacks=ATTACKS, poison_rates=RATES, seeds=SEEDS, clean_batches_per_seed=CLEAN_PER_SEED, batch_rows=rows)
    if STAGE == 2:
        m.update(attacks=["cl_lowfreq16", "cl_lowfreq32", "clean_label_backdoor"], clean_batches_per_seed=10)
    if STAGE == 3:
        m.update(attacks=["cl_lowfreq16", "cl_lowfreq32"], clean_batches_per_seed=20)
    path, t, n = f"{LOG}/weakness{'' if STAGE == 1 else STAGE}_{config}_dev.pkl", time.perf_counter(), 0
    total = len(bench_mm.batch_specs(m))
    with open(path, "wb") as f:
        for job in bench_mm.jobs(m):
            pickle.dump(one(job), f); f.flush(); n += 1
            if n % 10 == 0: print(n, total, f"{time.perf_counter() - t:.0f}s", flush=True)
    print("done", n, flush=True)


def load(config):
    R = []
    path = f"{LOG}/weakness{'' if STAGE == 1 else STAGE}_{config}_dev.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:   # this study's own local output
            while True:
                try: R.append(pickle.load(f))
                except EOFError: break
    return R


def analyze():
    cands = {1: ["sig_hp6", "sig_hp10", "sig_hp15", "sig_hp20", "sig_low6", "sig_low10", "sig_low15", "sig_low20"],
             2: ["sig_hp6", "sig_lowproj56", "sig_lowproj206", "sig_lowproj506", "sig_hpproj206"],
             3: ["sig_hp6", "sig_lowproj506", "sig_lowproj1006", "sig_lowproj1506"]}[STAGE]
    for config in CONFIGS:
        R = load(config)
        if not R: continue
        cln = [r for r in R if not r["spec"]["attack"]]
        print(f"\n== {config}: {len(R) - len(cln)} attacked, {len(cln)} clean")
        print("clean, raw evidence >= penalty + 1.3 (sig: 2.6, label 1.4):",
              "  ".join(f"{c} {np.mean([r['scores'][c][0] >= 2.6 for r in cln]):.3f}" for c in cands),
              f"  label {np.mean([r['scores']['base']['label'][0] >= 1.4 for r in cln]):.3f}",
              f"  veritas {np.mean([r['scores']['veritas'] >= 1.3 for r in cln]):.3f}")
        print("which check alarms on clean (raw >= penalty + 1.3):",
              {n: int(sum(r["scores"]["base"][n][0] >= vmm.PENALTY[n] + 1.3 for r in cln)) for n in cln[0]["scores"]["base"]})
        for a in sorted({r["spec"]["attack"] for r in R if r["spec"]["attack"]}):
            for rate in RATES:
                g = [r for r in R if r["spec"]["attack"] == a and r["spec"]["rate"] == rate]
                if not g: continue
                au = lambda c, sub=None: np.mean([average_precision_score(r["mask"], (r["scores"][c][1] if sub is None else r["scores"][c][sub][1])) for r in g])  # noqa: E731
                fl = lambda c, thr: np.mean([r["scores"][c][0] >= thr for r in g])  # noqa: E731
                line = f"{a:<22}{rate:5.2f} veritas flagged {np.mean([r['scores']['veritas'] >= 1.3 for r in g]):.2f}"
                if a.startswith("label"):
                    line += (f" | label AUPRC {np.mean([average_precision_score(r['mask'], r['scores']['base']['label'][1]) for r in g]):.2f}"
                             f" fl {np.mean([r['scores']['base']['label'][0] >= 1.4 for r in g]):.2f}"
                             f" | concat AUPRC {au('label_concat'):.2f} fl {fl('label_concat', 1.4):.2f}")
                else:
                    line += " | " + "  ".join(f"{c} {au(c):.2f}/{fl(c, 2.6):.2f}" for c in cands)
                print(line)


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect(os.environ["CONFIG"])
