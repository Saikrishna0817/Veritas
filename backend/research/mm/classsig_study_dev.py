"""Dev-only study: a within-class trigger check (spectral signature, Tran et al. 2018) for clean-label backdoors,
where the poison keeps its correct label and sits in one class, so pooled checks see it as ordinary class similarity.
Collect:  KIND=audio|images|video PYTHONPATH=. python -m research.mm.classsig_study_dev   (shipped manifests, dev split)
Analyse:  KIND=... PYTHONPATH=. python -m research.mm.classsig_study_dev --analyze
Per class c: deviations of the incoming items from the class's reference mean; the top singular vector of one half
of them scores the other half (cross-fitted), and the class's reference items on the same vector are the clean
comparison. Each class's scores are divided by its reference mean (leave-one-out for reference items), then one
tail test over all classes: per-class tail tests on ~14 reference items (ESC-50) were badly anti-conservative.
Dev split only."""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "2")
import pickle, sys, time
import numpy as np
from sklearn.metrics import average_precision_score
KIND = os.getenv("KIND", "audio")
os.environ.setdefault("VERITAS_AUDIO_MANIFEST", "manifest-audio-v2.json"); os.environ.setdefault("VERITAS_VIDEO_MANIFEST", "manifest-video-v2.json")
os.environ.setdefault("VERITAS_MM_MANIFEST", "manifest-mm-images-v8.json")
from research import bench_audio, bench_mm, bench_video
from app.detection import veritas2 as v2, veritas_mm as vmm

PKL = os.path.expanduser(f"~/.cache/veritas-logs/classsig_{KIND}_dev.pkl")
TRIGGER = "tone_backdoor" if KIND == "audio" else "patch_backdoor"
ATTACKS, RATES, SEEDS, CLEAN_PER_SEED = ["clean_label_backdoor", TRIGGER, "class_noise"], [0.01, 0.05], 3, 20
HARNESS = {"audio": (bench_audio.load_manifest, bench_audio.jobs, bench_audio.batch_specs),
           "images": (bench_mm.load_manifest, bench_mm.jobs, bench_mm.batch_specs),
           "video": (bench_video.load_manifest, bench_video.jobs, bench_audio.batch_specs)}[KIND]
DEV_SETS = {"images": [{"data_id": "fashion_mnist"}, {"data_id": "cifar100"}]}   # the image v8 manifest lists only test
MIN_CLASS = 6


def views(d):
    """name -> (reference features, incoming features), reference = fit + cal rows."""
    R = lambda a: a.reshape(len(a), -1).astype(np.float32)  # noqa: E731
    Rref, Rinc = np.concatenate([d["R_fit"], d["R_cal"]]), d["R_inc"]
    return {"resid": (R(vmm._residual(Rref.astype(np.float32))), R(vmm._residual(Rinc.astype(np.float32)))),
            "embed": (np.vstack([d["X_fit"], d["X_cal"]]), d["X_inc"])}   # "view" (raw pixels) ranked at chance on audio dev


def class_signature(Fr, Fi, yr, yi):
    """Per class: (cal scores of the class's reference items, inc scores, incoming indices)."""
    sd = Fr.std(0) + 1e-6
    out = {}
    for c in np.unique(yi):
        rc, ic = np.flatnonzero(yr == c), np.flatnonzero(yi == c)
        if len(rc) < MIN_CLASS or len(ic) < MIN_CLASS:
            continue
        mu = Fr[rc].mean(0)
        # Reference items leave themselves out of the mean (x_i - mean_-i = (x_i - mean) n/(n-1)), as incoming items are
        # out of it: otherwise their deviations shrink and every class looks unusual when classes are small.
        A, B = (Fi[ic] - mu) / sd, (Fr[rc] - mu) / sd * (len(rc) / (len(rc) - 1))
        half = np.arange(len(ic)) % 2
        inc, cal = np.zeros(len(ic)), np.zeros(len(rc))
        for h in (0, 1):
            v = np.linalg.svd(A[half != h], full_matrices=False)[2][0]
            inc[half == h] = (A[half == h] @ v) ** 2
            cal[np.arange(len(rc)) % 2 == h] = (B[np.arange(len(rc)) % 2 == h] @ v) ** 2
        out[int(c)] = (cal, inc, ic)
    return out


def one(job):
    d, mask = job["d"], job["mask"]
    yr = np.concatenate([d["y_fit"], d["y_cal"]])
    return {"spec": {k: job["spec"][k] for k in ("data_id", "seed", "attack", "rate", "rep")}, "mask": mask, "n_classes": d["n_classes"],
            "sig": {name: class_signature(Fr, Fi, yr, d["y_inc"]) for name, (Fr, Fi) in views(d).items()}}


def collect():
    load, jobs, specs = HARNESS
    m = load()
    m["datasets"] = {"dev": DEV_SETS.get(KIND) or m["datasets"]["dev"]}
    m.update(attacks=ATTACKS, poison_rates=RATES, seeds=SEEDS, clean_batches_per_seed=CLEAN_PER_SEED)
    t, n = time.perf_counter(), 0
    with open(PKL, "wb") as f:
        for job in jobs(m):
            pickle.dump(one(job), f); f.flush(); n += 1
            if n % 10 == 0: print(n, len(specs(m)), f"{time.perf_counter() - t:.0f}s", flush=True)
    print("done", n)


def scores(r, name):
    """Row scores and batch evidence after scaling each class by its reference mean, one tail test over all classes."""
    inc, C, I = np.zeros(len(r["mask"])), [], []
    for cal, s, ic in r["sig"][name].values():
        n, tot = len(cal), cal.sum()
        C.append(cal / ((tot - cal) / (n - 1)))
        I.append(s / (tot / n)); inc[ic] = I[-1]
    if not C:
        return inc, 0.0
    C = np.concatenate(C)
    return v2.extrapolated_neg_log_p(C, inc), max(v2.multi_tail_test(C, np.concatenate(I), vmm.TAILS), 0.0)


def analyze():
    R = []
    with open(PKL, "rb") as f:   # this study's own local output
        while True:
            try: R.append(pickle.load(f))
            except EOFError: break
    cln = [r for r in R if not r["spec"]["attack"]]; atk = [r for r in R if r["spec"]["attack"]]
    names = list(R[0]["sig"])
    print(f"{len(atk)} attacked, {len(cln)} clean (dev: {sorted({r['spec']['data_id'] for r in R})})")
    for ds in sorted({r["spec"]["data_id"] for r in cln}):
        c = [r for r in cln if r["spec"]["data_id"] == ds]
        print(f"clean {ds} (n={len(c)}), share with raw evidence >= 1.3 / 2.0 / 2.3 / 3.0: " + "  ".join(
            f"{n} " + " / ".join(f"{np.mean([scores(r, n)[1] >= t for r in c]):.3f}" for t in (1.3, 2.0, 2.3, 3.0)) for n in names))
    print(f"\n{'attack':<22}{'rate':>5}{'data':>16}" + "".join(f"{n + ' AUPRC':>13}{'ev>=2':>7}" for n in names))
    for a in ATTACKS:
        for rate in RATES:
            for ds in sorted({r["spec"]["data_id"] for r in R}):
                g = [r for r in atk if r["spec"]["attack"] == a and r["spec"]["rate"] == rate and r["spec"]["data_id"] == ds]
                s = {n: [scores(r, n) for r in g] for n in names}
                print(f"{a:<22}{rate:5.2f}{ds:>16}" + "".join(
                    f"{np.mean([average_precision_score(r['mask'], x[0]) for r, x in zip(g, s[n])]):13.2f}{np.mean([x[1] >= 2 for x in s[n]]):7.2f}" for n in names))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else collect()
