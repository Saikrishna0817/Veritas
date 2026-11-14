"""Audio poisoning benchmark: Speech Commands and ESC-50, attacks on raw waveforms, the image detector on log-mel
spectrogram images. Batch preparation, scoring and the report are shared with ``bench_mm``.

Run from ``backend/``:  VERITAS_AUDIO_MANIFEST=manifest-audio-v1.json python -m research.bench_audio --splits dev
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import date  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research import bench_mm  # noqa: E402
from research.audio import datasets  # noqa: E402
from research.audio.attacks import ATTACKS, make_context  # noqa: E402
from research.audio.spectro import spectrogram_images  # noqa: E402
from research.bench_v2 import aggregate, detector_inputs, to_markdown  # noqa: E402
from research.mm.embed import embed_images  # noqa: E402
from research.mm.pixel import pixel_features  # noqa: E402

MANIFEST_PATH = bench_mm.ROOT / "benchmark" / os.getenv("VERITAS_AUDIO_MANIFEST", "manifest-audio-v1.json")
CACHE = datasets.ROOT / "cache"


def load_manifest() -> dict:
    return {**json.loads(MANIFEST_PATH.read_text()), "profile": "full"}


def clean_views(ds: datasets.AudioSet, size: int):
    """(spectrogram images, embeddings, pixel features) of every clean clip, cached per dataset split."""
    path = CACHE / f"{ds.name}-views{size}.npz"
    if not path.exists():
        img = spectrogram_images(ds.wave, size)
        np.savez(path, img=img, emb=embed_images(img), pix=pixel_features(img))
    v = np.load(path)
    return v["img"], v["emb"], v["pix"]


def prepare_batch(spec: dict, ds: datasets.AudioSet, img: np.ndarray, emb: np.ndarray, pix: np.ndarray) -> dict:
    """Attack the incoming waveforms, redraw and re-embed only what changed, build the detector inputs as bench_mm does."""
    m = spec["manifest"]
    rng = np.random.default_rng([m["random_seed"], bench_mm.zlib_id(spec["data_id"] + spec["split"]), spec["seed"],
                                 spec["attack_index"] + 1, int(spec["rate"] * 1000), spec["rep"]])
    sel = rng.permutation(len(ds.y))[:min(spec["batch_rows"], len(ds.y))]
    n_ref = int(len(sel) * m["reference_fraction"])
    ref, inc = sel[:n_ref], sel[n_ref:]
    y, R_inc, E_inc, P_inc = ds.y[inc].copy(), img[inc].copy(), emb[inc].copy(), pix[inc].copy()
    mask = np.zeros(len(inc), dtype=bool)
    if spec["attack"]:
        x = np.asarray(ds.wave[inc]).astype(np.float32) / 32768.0
        ctx = make_context(x, ds.n_classes, rng)
        idx = ATTACKS[spec["attack"]](x, y, rng, max(1, round(spec["rate"] * len(inc))), ctx)
        mask[idx] = True
        R_inc[idx] = spectrogram_images(np.clip(x[idx], -1, 1), m["image_size"])
        E_inc[idx], P_inc[idx] = embed_images(R_inc[idx]), pixel_features(R_inc[idx])
    pca = PCA(m["pca_dims"], random_state=0).fit(emb[ref])
    d = detector_inputs(pca.transform(emb[ref]), ds.y[ref], pca.transform(E_inc), y, ds.n_classes,
                        m["calibration_fraction_of_reference"])
    n_cal = int(len(ref) * m["calibration_fraction_of_reference"])
    scaler = StandardScaler().fit(pix[ref][n_cal:])
    d.update({"P_cal": scaler.transform(pix[ref][:n_cal]), "P_inc": scaler.transform(P_inc), "R_cal": img[ref][:n_cal],
              "R_inc": R_inc, "P_fit": scaler.transform(pix[ref][n_cal:]), "R_fit": img[ref][n_cal:]})
    return {"d": d, "mask": mask}


def batch_specs(m: dict) -> list[dict]:
    rows = {(s, d["data_id"]): d.get("batch_rows", m["batch_rows"]) for s, ds in m["datasets"].items() for d in ds}
    return [{**s, "batch_rows": rows[(s["split"], s["data_id"])]} for s in bench_mm.batch_specs(m)]


def jobs(m: dict):
    """Prepared batches, one dataset split at a time."""
    specs = batch_specs(m)
    for key in dict.fromkeys((s["split"], s["data_id"]) for s in specs):
        ds = datasets.load(key[1], key[0])
        img, emb, pix = clean_views(ds, m["image_size"])
        for spec in (s for s in specs if (s["split"], s["data_id"]) == key):
            yield {"spec": spec, **prepare_batch(spec, ds, img, emb, pix)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", required=True)
    parser.add_argument("--splits", nargs="+", required=True, help="dev for tuning, test once for the headline")
    args = parser.parse_args()
    m = load_manifest()
    m["datasets"] = {k: v for k, v in m["datasets"].items() if k in args.splits}
    total, started, records = len(batch_specs(m)), time.perf_counter(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(pool.map(bench_mm.score_batch, jobs(m), chunksize=1), 1):
            records.append(rec)
            if i % 25 == 0 or i == total:
                print(f"{i}/{total} batches, {time.perf_counter() - started:.0f}s", flush=True)
    report = {"manifest": m, "label": args.label, "n_batches": len(records), "runtime_seconds": round(time.perf_counter() - started, 1),
              "summary": aggregate(records, m), "batches": records}
    out = bench_mm.ROOT / "results" / f"{date.today().isoformat()}-benchmark-audio-{args.label}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(to_markdown(report) + "\n")
    print(out.with_suffix(".md"))


if __name__ == "__main__":
    main()
