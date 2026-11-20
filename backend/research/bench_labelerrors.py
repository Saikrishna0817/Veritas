"""Real, human-verified label errors (Northcutt et al. 2021, labelerrors.com): how well does Veritas rank them?

Mode A: a trusted reference (sample of the training set), the test set scored in chunks. Mode B: no reference, 5-fold
self-check. Nothing is planted. Run from ``backend/``:
    python -m research.bench_labelerrors --datasets cifar10 cifar100 mnist [--limit N]
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.detection import veritas_mm  # noqa: E402
from research.bench_v2 import detector_inputs  # noqa: E402
from research.mm import datasets  # noqa: E402
from research.mm.embed import embed_images  # noqa: E402
from research.mm.pixel import pixel_features  # noqa: E402
from research import detector_commit  # noqa: E402

ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "benchmark" / "manifest-labelerrors-v1.json").read_text())
GT_DIR = Path.home() / ".cache" / "veritas-ext" / "label-errors"
CACHE = Path.home() / ".cache" / "veritas-bench" / "labelerrors"
TV = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "mnist": "MNIST"}
SCORERS = ("veritas_mm", "label")


def load_split(name: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    ds = datasets._tv(TV[name], train=train)
    data = ds.data.numpy() if hasattr(ds.data, "numpy") else ds.data
    return datasets._to_rgb(data), np.asarray(ds.targets, dtype=np.int64)


def ground_truth(name: str, y_test: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks over test images. Asserts the ids index the same images as the published labels."""
    original = np.load(GT_DIR / "original_test_labels" / f"{name}_test_set_original_labels.npy")
    assert np.array_equal(original.astype(np.int64), y_test), f"{name}: test labels differ from original_test_labels"
    cand = json.loads((GT_DIR / "mturk" / f"{name}_mturk.json").read_text())
    ids = np.array([c["id"] for c in cand])
    confirmed = np.array([c["mturk"]["given"] < 3 for c in cand])
    alt = np.array([c["mturk"]["given"] + c["mturk"].get("both", 0) < 3 for c in cand])
    masks = {"confirmed": np.zeros(len(y_test), bool), "candidates": np.zeros(len(y_test), bool)}
    masks["confirmed"][ids[confirmed]] = True
    masks["candidates"][ids] = True
    masks["_given_plus_both_lt_3"] = int(alt.sum())
    return masks


def embeddings(name: str, tag: str, raw: np.ndarray) -> np.ndarray:
    path = CACHE / f"{name}-{tag}-resnet18.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embed_images(raw))
    return np.load(path)


def chunks(n: int, size: int, rng) -> list[np.ndarray]:
    order = rng.permutation(n)
    return [order[i:i + size] for i in range(0, n, size)]


def mode_a(m, raw, y, E, tr_raw, tr_y, tr_E, n_classes, rng) -> tuple[dict, list]:
    A = m["mode_a_reference"]
    pca = PCA(A["pca_dims"], random_state=0).fit(tr_E)
    cal = A["calibration_fraction_of_reference"]
    n_cal = int(len(tr_y) * cal)
    P_ref, Z_ref = pixel_features(tr_raw), pca.transform(tr_E)
    scaler = StandardScaler().fit(P_ref[n_cal:])
    P = pixel_features(raw)
    scores = {s: np.zeros(len(y)) for s in SCORERS}
    parts = chunks(len(y), A["batch_rows"], rng)
    for idx in parts:
        d = detector_inputs(Z_ref, tr_y, pca.transform(E[idx]), y[idx], n_classes, cal)
        d.update({"P_cal": scaler.transform(P_ref[:n_cal]), "P_inc": scaler.transform(P[idx]), "R_cal": tr_raw[:n_cal], "R_inc": raw[idx],
                  "P_fit": scaler.transform(P_ref[n_cal:]), "R_fit": tr_raw[n_cal:]})
        out = veritas_mm.detect(d)
        scores["veritas_mm"][idx], scores["label"][idx] = out["row_scores"], out["report"]["label"]["rows"]
    return scores, parts


def mode_b(m, raw, y, E, n_classes, rng) -> tuple[dict, list]:
    scores, parts = {"label": np.zeros(len(y))}, chunks(len(y), m["mode_b_no_reference"]["chunk_rows"], rng)
    for idx in parts:
        scores["label"][idx] = veritas_mm.self_check(raw[idx], E[idx], y[idx], n_classes)["label"]["rows"]
    return scores, parts


def list_size(n: int) -> int:
    return min(200, max(20, round(0.05 * n)))


def in_app_list(scores: np.ndarray, parts: list) -> np.ndarray:
    mask = np.zeros(len(scores), bool)
    for idx in parts:
        mask[idx[np.argsort(-scores[idx], kind="stable")[:list_size(len(idx))]]] = True
    return mask


def point_metrics(scores, pos, listed) -> dict:
    n, npos = len(scores), int(pos.sum())
    if npos == 0 or npos == n:
        return {}
    out = {"base_rate": npos / n, "auprc": average_precision_score(pos, scores), "auroc": roc_auc_score(pos, scores)}
    order = np.argsort(-scores, kind="stable")
    for frac in MANIFEST["metrics"]["top_fractions"]:
        hit = pos[order[:max(1, int(np.ceil(frac * n)))]].sum()
        out[f"precision_top{frac:.0%}"], out[f"recall_top{frac:.0%}"] = hit / max(1, int(np.ceil(frac * n))), hit / npos
    out["recall_app_list"], out["precision_app_list"] = pos[listed].sum() / npos, pos[listed].mean()
    return out


def with_ci(scores, pos, listed, rng, resamples: int) -> dict:
    point = point_metrics(scores, pos, listed)
    if not point:
        return {}
    boots = [point_metrics(scores[i], pos[i], listed[i]) for i in (rng.integers(0, len(scores), len(scores)) for _ in range(resamples))]
    out = {}
    for k, v in point.items():
        b = [r[k] for r in boots if r]
        out[k] = [float(v), *map(float, np.percentile(b, [2.5, 97.5]))] if b else [float(v), None, None]
    return out


def run_dataset(name: str, limit: int | None, n_ref: int, resamples: int) -> dict:
    m, seed = MANIFEST, MANIFEST["seed"]
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    raw, y = load_split(name, train=False)
    gt = ground_truth(name, y)
    alt_count = gt.pop("_given_plus_both_lt_3")
    keep = slice(0, limit)
    raw, y, gt = raw[keep], y[keep], {k: v[keep] for k, v in gt.items()}
    n_classes = int(y.max()) + 1
    E = embeddings(name, f"test{len(y)}", raw)
    tr_raw, tr_y = load_split(name, train=True)
    ref = np.sort(train_test_split(np.arange(len(tr_y)), train_size=n_ref, stratify=tr_y, random_state=seed)[0])
    tr_raw, tr_y = tr_raw[ref], tr_y[ref]
    tr_E = embeddings(name, f"trainref{n_ref}", tr_raw)
    out = {"n_test": len(y), "n_classes": n_classes, "n_reference": n_ref, "order_check": "passed",
           "ground_truth_counts": {"confirmed": int(gt["confirmed"].sum()), "candidates": int(gt["candidates"].sum()),
                                   "confirmed_if_given_plus_both_lt_3": alt_count}, "modes": {}}
    for mode, scores_parts in (("A_with_reference", lambda: mode_a(m, raw, y, E, tr_raw, tr_y, tr_E, n_classes, rng)),
                               ("B_no_reference", lambda: mode_b(m, raw, y, E, n_classes, rng))):
        started = time.perf_counter()
        scores, parts = scores_parts()
        out["modes"][mode] = {"seconds": round(time.perf_counter() - started, 1), "scorers": {
            s: {g: with_ci(v, gt[g], in_app_list(v, parts), np.random.default_rng(seed), resamples) for g in gt}
            for s, v in scores.items()}}
        print(f"{name} {mode} {out['modes'][mode]['seconds']}s", flush=True)
    return out


def to_markdown(report: dict) -> str:
    lines = [f"# {report['manifest']['benchmark_id']} — {report['label']}", "",
             f"Detector commit `{report['detector_commit']}`. Cells: estimate [95% CI, bootstrap over test images]. "
             "Positives are only found among cleanlab's candidates: precision is a lower bound.", ""]
    keys = ["base_rate", "auprc", "auroc", "precision_top1%", "recall_top1%", "precision_top5%", "recall_top5%",
            "precision_app_list", "recall_app_list"]
    for name, r in report["datasets"].items():
        lines += [f"## {name}", "", f"Ground truth: {r['ground_truth_counts']}; {r['n_test']} test images, {r['n_reference']} reference.", ""]
        for mode, md in r["modes"].items():
            lines += [f"### {mode}", "", "| scorer | truth | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
            for s, by_gt in md["scorers"].items():
                for g, met in by_gt.items():
                    cell = lambda k: "-" if k not in met else (f"{met[k][0]:.3f}" if met[k][1] is None else f"{met[k][0]:.3f} [{met[k][1]:.3f}, {met[k][2]:.3f}]")  # noqa: E731
                    lines.append(f"| {s} | {g} | " + " | ".join(cell(k) for k in keys) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=MANIFEST["datasets"])
    parser.add_argument("--label", default="v1")
    parser.add_argument("--limit", type=int, default=None, help="cap the test set (smoke run)")
    args = parser.parse_args()
    n_ref = MANIFEST["mode_a_reference"]["n_reference"]
    if "VERITAS_LE_NREF" in os.environ:
        if args.limit is None:
            parser.error("VERITAS_LE_NREF is only allowed with --limit")
        n_ref = int(os.environ["VERITAS_LE_NREF"])
    label = args.label + ("-smoke" if args.limit else "")
    started = time.perf_counter()
    commit = detector_commit()
    res = {n: run_dataset(n, args.limit, n_ref, 1000) for n in args.datasets}
    report = {"manifest": MANIFEST, "label": label, "detector_commit": commit, "limit": args.limit, "n_reference_used": n_ref,
              "runtime_seconds": round(time.perf_counter() - started, 1), "datasets": res}
    out = ROOT / "results" / f"{date.today().isoformat()}-labelerrors-{label}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    out.with_suffix(".md").write_text(to_markdown(report) + "\n")
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
