"""Video poisoning benchmark: UCF-101, attacks on decoded frames, the image detector on a 2x2 frame mosaic with the
mean ResNet-18 frame embedding. Batching, scoring and the report are shared with bench_audio / bench_mm.

Run from ``backend/``:  python -m research.bench_video --splits dev --label dev-v1
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.detection import video  # noqa: E402
from research import bench_audio, bench_mm  # noqa: E402
from research.bench_v2 import detector_inputs  # noqa: E402
from research.mm.pixel import pixel_features  # noqa: E402
from research.video import datasets  # noqa: E402
from research.video.attacks import ATTACKS, make_context  # noqa: E402

MANIFEST_PATH = bench_mm.ROOT / "benchmark" / os.getenv("VERITAS_VIDEO_MANIFEST", "manifest-video-v1.json")


def load_manifest() -> dict:
    return {**json.loads(MANIFEST_PATH.read_text()), "profile": "full"}


def views(frames: np.ndarray, kind: str, threads: int | None = None):
    """(check images of the given ``kind``, mean frame embeddings, pixel features of the check images)."""
    img = video.VIEWS[kind](frames)
    return img, video.embed_clips(frames, threads=threads), pixel_features(img)


def clean_views(ds: datasets.VideoSet, kind: str):
    path = datasets.ROOT / "cache" / f"{ds.name}-views{'' if kind == 'mosaic' else '-' + kind}.npz"
    if not path.exists():
        img, emb, pix = [], [], []
        for i in range(0, len(ds.y), 512):
            a, b, c = views(np.asarray(ds.frames[i:i + 512]), kind, threads=4)
            img.append(a), emb.append(b), pix.append(c)
        np.savez(path, img=np.concatenate(img), emb=np.concatenate(emb), pix=np.concatenate(pix))
    v = np.load(path)
    return v["img"], v["emb"], v["pix"]


def prepare_batch(spec: dict, ds: datasets.VideoSet, img, emb, pix) -> dict:
    """Attack the incoming frames, rebuild the views of what changed, detector inputs as bench_mm builds them."""
    m = spec["manifest"]
    rng = np.random.default_rng([m["random_seed"], bench_mm.zlib_id(spec["data_id"] + spec["split"]), spec["seed"],
                                 spec["attack_index"] + 1, int(spec["rate"] * 1000), spec["rep"]])
    sel = rng.permutation(len(ds.y))[:min(spec["batch_rows"], len(ds.y))]
    n_ref = int(len(sel) * m["reference_fraction"])
    ref, inc = sel[:n_ref], sel[n_ref:]
    y, R_inc, E_inc, P_inc = ds.y[inc].copy(), img[inc].copy(), emb[inc].copy(), pix[inc].copy()
    mask = np.zeros(len(inc), dtype=bool)
    if spec["attack"]:
        frames = np.asarray(ds.frames[inc]).copy()
        ctx = make_context(frames, ds.n_classes, rng)
        idx = ATTACKS[spec["attack"]](frames, y, rng, max(1, round(spec["rate"] * len(inc))), ctx)
        mask[idx] = True
        R_inc[idx], E_inc[idx], P_inc[idx] = views(frames[idx], m.get("view", "mosaic"), threads=4)
    pca = PCA(m["pca_dims"], random_state=0).fit(emb[ref])
    d = detector_inputs(pca.transform(emb[ref]), ds.y[ref], pca.transform(E_inc), y, ds.n_classes,
                        m["calibration_fraction_of_reference"])
    n_cal = int(len(ref) * m["calibration_fraction_of_reference"])
    scaler = StandardScaler().fit(pix[ref][n_cal:])
    d.update({"P_cal": scaler.transform(pix[ref][:n_cal]), "P_inc": scaler.transform(P_inc), "R_cal": img[ref][:n_cal],
              "R_inc": R_inc, "P_fit": scaler.transform(pix[ref][n_cal:]), "R_fit": img[ref][n_cal:]})
    return {"d": d, "mask": mask}


def jobs(m: dict):
    specs = bench_audio.batch_specs(m)
    for split in dict.fromkeys(s["split"] for s in specs):
        ds = datasets.load(split)
        img, emb, pix = clean_views(ds, m.get("view", "mosaic"))
        for spec in (s for s in specs if s["split"] == split):
            yield {"spec": spec, **prepare_batch(spec, ds, img, emb, pix)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", required=True)
    parser.add_argument("--splits", nargs="+", required=True, help="dev for tuning, test once for the headline")
    args = parser.parse_args()
    bench_audio.run(load_manifest(), jobs, args.splits, args.label, "video", args.workers)


if __name__ == "__main__":
    main()
