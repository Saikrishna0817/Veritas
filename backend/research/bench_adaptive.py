"""Adaptive attacks against the image checks: does Veritas still flag attacks tuned to evade it, and does the attack
still work on a model trained on the poisoned data?  One process (the victim CNN trains on CPU).

Run from ``backend/``:  python -m research.bench_adaptive --label v1 [--smoke]
(another manifest: VERITAS_ADAPTIVE_MANIFEST=manifest-adaptive-images-v2.json; victim threads: VERITAS_VICTIM_THREADS)
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.detection.veritas_mm import CHECKS  # noqa: E402
from research.adaptive import victim  # noqa: E402
from research.adaptive.attacks import FAMILIES, FLIPS, make_context  # noqa: E402
from research.bench_mm import checkpoint_path  # noqa: E402
from research.bench_v2 import _row_metrics, detector_inputs  # noqa: E402
from research.mm import datasets  # noqa: E402
from research.mm.detector import report  # noqa: E402
from research.mm.embed import clean_embeddings, embed_images  # noqa: E402
from research.mm.pixel import pixel_features  # noqa: E402

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "benchmark" / os.getenv("VERITAS_ADAPTIVE_MANIFEST", "manifest-adaptive-images-v1.json")


def load_manifest(smoke: bool) -> dict:
    m = json.loads(MANIFEST_PATH.read_text())
    m["sizes"] = {**m["sizes"], **m["sizes"]["smoke"]} if smoke else m["sizes"]
    if smoke:
        m.update({"families": {f: {**m["families"][f], "strengths": m["families"][f]["strengths"][:1]}
                               for f in list(m["families"])[:2]},
                  "rates": [0.05], "attacker_seeds": [0], "eval_seeds": [10], "victim": {**m["victim"], "epochs": 1}})
    return m


def specs(m: dict) -> list[dict]:
    """Eval seeds outermost so the clean victim of a seed is trained once."""
    out = []
    for kind, seeds in (("eval", m["eval_seeds"]), ("attacker", m["attacker_seeds"])):
        for seed in seeds:
            out += [{"kind": kind, "family": f, "s": s, "rate": r, "seed": seed}
                    for f, fam in m["families"].items() for s in fam["strengths"] for r in m["rates"]]
    return out


def spec_key(s: dict) -> tuple:
    return tuple(s[k] for k in ("family", "s", "rate", "seed"))


def pool_split(m: dict, ds, seed: int):
    """Seeded: reference, incoming and the attacker's own images (not in the reference), as pool indices."""
    sz = m["sizes"]
    perm = np.random.default_rng([m["random_seed"], seed]).permutation(len(ds.y))
    ref, rest = perm[:sz["reference"]], perm[sz["reference"]:]
    return ref, rest[:sz["incoming"]], rest[-sz["attacker_own"]:]


def seed_context(m: dict, ds, seed: int) -> dict:
    """Source/target and the fixed trigger elements, the same for every run of a seed."""
    return make_context(ds.raw[:1], ds.n_classes, np.random.default_rng([m["random_seed"], seed, 7]))


def prepare(m, spec, ds, emb, pix, ctx):
    """Attack the incoming pixels, re-embed what changed; returns detector inputs, poison mask and the training data."""
    ref, inc, own = pool_split(m, ds, spec["seed"])
    pca = PCA(m["pca_dims"], random_state=0).fit(emb[ref])
    fam = spec["family"]
    ctx = dict(ctx)
    if fam == "boundary_flip":
        clf = LogisticRegression(max_iter=300).fit(pca.transform(emb[own]), ds.y[own])
        ctx["target_score"] = clf.predict_proba(pca.transform(emb[inc]))[:, ctx["target"]]
    raw, y = ds.raw[inc].copy(), ds.y[inc].copy()
    E_inc, P_inc = emb[inc].copy(), pix[inc].copy()
    rng = np.random.default_rng([m["random_seed"], spec["seed"], list(FAMILIES).index(fam),
                                 int(spec["s"] * 100), int(spec["rate"] * 1000)])
    idx = FAMILIES[fam][0](raw, y, rng, max(1, round(spec["rate"] * len(inc))), ctx, spec["s"])
    mask = np.zeros(len(inc), dtype=bool); mask[idx] = True
    changed = idx[(raw[idx] != ds.raw[inc][idx]).any(axis=(1, 2, 3))]   # flips leave the pixels (and embeddings) alone
    if len(changed):
        E_inc[changed], P_inc[changed] = embed_images(raw[changed]), pixel_features(raw[changed])
    d = detector_inputs(pca.transform(emb[ref]), ds.y[ref], pca.transform(E_inc), y, ds.n_classes,
                        m["reference_calibration_fraction"])
    n_cal = int(len(ref) * m["reference_calibration_fraction"])
    scaler = StandardScaler().fit(pix[ref][n_cal:])
    d.update({"P_cal": scaler.transform(pix[ref][:n_cal]), "P_inc": scaler.transform(P_inc),
              "R_cal": ds.raw[ref][:n_cal], "R_inc": raw,
              "P_fit": scaler.transform(pix[ref][n_cal:]), "R_fit": ds.raw[ref][n_cal:]})
    train = (np.concatenate([ds.raw[ref], raw]), np.concatenate([ds.y[ref], y]))
    return d, mask, train, ctx


def detect(d: dict, mask: np.ndarray) -> dict:
    """The same numbers as research/mm/detector.py's veritas_mm and check_detector, from one report (each of those
    recomputes the whole report)."""
    r = report(d)
    rows = np.max([c["rows"] for c in r.values() if c["admitted"]], axis=0)
    return {"evidence": float(max(c["evidence"] for c in r.values())), "checks": {n: float(c["raw"]) for n, c in r.items()},
            "auprc": _row_metrics(np.asarray(rows, dtype=float), mask)["auprc"]}


def test_set(m: dict, seed: int):
    ds = datasets._tv("CIFAR10", train=False)
    sel = np.random.default_rng([m["random_seed"], seed, 3]).permutation(len(ds.targets))[:m["sizes"]["test"]]
    return ds.data[sel], np.asarray(ds.targets)[sel]


class Victims:
    """Test set and the clean-data victim of the current seed (trained once, for the baseline)."""
    def __init__(self, m: dict, ds):
        self.m, self.ds, self.seed = m, ds, None

    def start(self, seed: int) -> None:
        if seed != self.seed:
            self.seed, self.test = seed, test_set(self.m, seed)
            ref, inc, _ = pool_split(self.m, self.ds, seed)
            both = np.concatenate([ref, inc])
            self.clean = victim.train(self.ds.raw[both], self.ds.y[both], self.m["victim"]["epochs"], seed)

    def run(self, spec: dict, train, ctx) -> dict:
        self.start(spec["seed"])
        epochs, (x, y), (tx, ty) = self.m["victim"]["epochs"], train, self.test
        trigger, flip = FAMILIES[spec["family"]][1], spec["family"] in FLIPS
        net = victim.train(x, y, epochs, spec["seed"])
        got = victim.evaluate(net, tx, ty, ctx, trigger, spec["s"], flip)
        base = victim.evaluate(self.clean, tx, ty, ctx, trigger, spec["s"], flip)
        return {"acc": got["acc"], "asr": got["asr"], "base_acc": base["acc"], "base_asr": base["asr"]}


def read_ckpt(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with path.open("rb") as f:   # this run's own local checkpoint
            while True:
                try: records.append(pickle.load(f))
                except (EOFError, pickle.UnpicklingError): break   # a record cut short by the kill is redone
    return records


def run(m: dict, ckpt: Path) -> list[dict]:
    records = read_ckpt(ckpt)
    todo = [s for s in specs(m) if spec_key(s) not in {spec_key(r) for r in records}]
    print(f"{len(records)} done, {len(todo)} to run", flush=True)
    ds = datasets.load(m["dataset"])
    emb, pix = clean_embeddings(ds), pixel_features(ds.raw)
    victims, started = Victims(m, ds), time.perf_counter()
    with ckpt.open("ab") as f:
        for i, spec in enumerate(todo, 1):
            d, mask, train, ctx = prepare(m, spec, ds, emb, pix, seed_context(m, ds, spec["seed"]))
            rec = {**spec, **detect(d, mask)}
            if spec["kind"] == "eval":
                rec.update(victims.run(spec, train, ctx))
            records.append(rec); pickle.dump(rec, f); f.flush()
            print(f"{i}/{len(todo)} {spec['family']} s={spec['s']} r={spec['rate']} seed={spec['seed']} "
                  f"ev={rec['evidence']:.2f} asr={rec.get('asr', float('nan')):.2f} {time.perf_counter() - started:.0f}s", flush=True)
    return records


def evasive(records: list[dict], family: str, rate: float, thr: float):
    """Largest strength whose mean evidence over the attacker seeds is below the flagged threshold, else None."""
    by_s = {}
    for r in records:
        if r["kind"] == "attacker" and r["family"] == family and r["rate"] == rate:
            by_s.setdefault(r["s"], []).append(r["evidence"])
    ok = [s for s, v in by_s.items() if np.mean(v) < thr]
    return max(ok) if ok else None


def summarize(records: list[dict], m: dict) -> dict:
    thr, ev = m["flagged"]["threshold"], [r for r in records if r["kind"] == "eval"]
    out = {}
    for fam, spec in m["families"].items():
        rows = []
        for rate in m["rates"]:
            s_ev = evasive(records, fam, rate, thr)
            for s in spec["strengths"]:
                g = [r for r in ev if (r["family"], r["s"], r["rate"]) == (fam, s, rate)]
                if not g:
                    continue
                means = {c: np.mean([r["checks"][c] for r in g]) for c in CHECKS}
                rows.append({"s": s, "rate": rate, "n": len(g), "flagged": float(np.mean([r["evidence"] >= thr for r in g])),
                             "evidence": float(np.mean([r["evidence"] for r in g])), "lead": max(means, key=means.get),
                             "auprc": float(np.mean([r["auprc"] for r in g])), "asr": float(np.mean([r["asr"] for r in g])),
                             "acc": float(np.mean([r["acc"] for r in g])), "base_asr": float(np.mean([r["base_asr"] for r in g])),
                             "evasive": s == s_ev})
        hit = [r for r in ev if r["family"] == fam and r["asr"] >= 0.9]
        out[fam] = {"rows": rows, "headline_n": len(hit),
                    "headline_flagged": float(np.mean([r["evidence"] >= thr for r in hit])) if hit else None}
    out["baseline_acc"] = float(np.mean([r["base_acc"] for r in ev])) if ev else None
    return out


def to_markdown(report: dict) -> str:
    m, summ = report["manifest"], report["summary"]
    lines = [f"# {m['benchmark_id']} ({report['label']})", "",
             f"Flagged = veritas_mm evidence >= {m['flagged']['threshold']} ({m['flagged']['source']}). Eval seeds {m['eval_seeds']}; "
             f"evasive strength picked on attacker seeds {m['attacker_seeds']}. Clean-data victim accuracy: {summ['baseline_acc']:.3f}.", ""]
    for fam, spec in m["families"].items():
        s = summ[fam]
        lines += [f"## {fam} ({spec['unit']})", "",
                  "| strength | rate | flagged | evidence | lead check | row AUPRC | ASR | clean acc | ASR w/o poison | evasive |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['s']} | {r['rate']} | {r['flagged']:.2f} | {r['evidence']:.2f} | {r['lead']} | {r['auprc']:.2f} | "
                  f"{r['asr']:.2f} | {r['acc']:.3f} | {r['base_asr']:.2f} | {'yes' if r['evasive'] else ''} |" for r in s["rows"]]
        ev = [r for r in s["rows"] if r["evasive"]]
        lines += ["", "Evasive strength (flagged share, ASR): " + ("; ".join(
            f"rate {r['rate']}: s={r['s']} ({r['flagged']:.2f}, {r['asr']:.2f})" for r in ev) or "none (every strength is flagged on the attacker seeds)")]
        lines += [f"Headline, flagged share among runs with ASR >= 0.9: " + (
            f"{s['headline_flagged']:.2f} (n={s['headline_n']})" if s["headline_n"] else "no run reached ASR 0.9"), ""]
    lines += [f"Policy: {m['policy']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    label = "smoke" if args.smoke else args.label
    m, started, ckpt = load_manifest(args.smoke), time.perf_counter(), checkpoint_path("adaptive-images", label)
    records = run(m, ckpt)
    report = {"manifest": m, "label": label, "runtime_seconds": round(time.perf_counter() - started, 1),
              "summary": summarize(records, m), "runs": records}
    out = ROOT / "results" / f"{date.today().isoformat()}-benchmark-adaptive-images-{label}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(to_markdown(report) + "\n")
    print(out.with_suffix(".md"))
    ckpt.unlink()


if __name__ == "__main__":
    main()
