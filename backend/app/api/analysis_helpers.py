"""Shared analysis orchestration used by upload, model scan, and dataset routes.

All pipeline orchestration lives here — routes call these helpers rather than
duplicating the ingest → detect → classify → forensics → defense sequence.

Changes:
  - Centralised (H2): routes now call run_csv_analysis / run_model_analysis
  - Uses shared thread pool (M4)
  - asyncio.get_running_loop() (L2)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import numpy as np

from app.api import dependencies as deps
from app.scanner.model_scan import scan_sandboxed

logger = logging.getLogger(__name__)


def _orchestrate_csv(csv_bytes: bytes, filename: str, owner_id: str, split: dict | None = None) -> tuple[dict[str, Any], str]:
    """Run the full CSV ingestion + detection + forensics pipeline (blocking).

    Returns (full_result_dict, dataset_id).
    This function is designed to be called inside a thread pool executor.
    """
    started = time.perf_counter()

    engine = deps.CSVIngestionEngine()
    ingested = engine.ingest(csv_bytes, filename=filename, **(split or {}))

    upload_pipeline = deps.DetectionPipeline()
    detection_result = upload_pipeline.run_on_upload(ingested)

    samples = ingested["samples"]
    incoming_samples = samples[ingested["reference_split"]:]

    attack_class = deps.classifier.classify(detection_result)

    # Tag the rows Layer 3 flagged as suspected
    row_flags = detection_result["layer_results"]["row_flags"]
    flagged = set(row_flags.get("flagged_indices_incoming", []))
    for i, s in enumerate(incoming_samples):
        s["row_score"] = row_flags["row_scores"][i]
        if i in flagged:
            s["poison_status"] = "suspected"

    suspicious_cells = _suspicious_cells(ingested, row_flags)
    pattern = deps.reconstructor.reconstruct(incoming_samples, attack_class, detection_result["layer_results"])
    blast = deps.blast_mapper.map(incoming_samples, detection_result["layer_results"])
    scope = ingested["dataset_id"]
    defense_action = deps.defense.decide_action(
        incoming_samples,
        detection_result["overall_suspicion_score"],
        detection_result["verdict"],
        scope=scope,
        owner_id=owner_id,
    )
    hitl_case = None
    if detection_result.get("requires_human_review"):
        hitl_case = deps.hitl.enqueue(
            incoming_samples,
            {**detection_result["layer_results"], "attack_classification": attack_class},
            detection_result["overall_suspicion_score"],
            scope=scope,
            owner_id=owner_id,
        )

    full_result: dict[str, Any] = {
        **detection_result,
        "dataset_info": {
            "dataset_id": ingested["dataset_id"],
            "filename": ingested["filename"],
            "n_rows": ingested["n_rows"],
            "n_features": ingested["n_features"],
            "feature_names": ingested["feature_names"],
            "label_column": ingested["label_column"],
            "has_labels": ingested["has_labels"],
            "detection_mode": ingested["detection_mode"],
            "reference_split": ingested["reference_split"],
            "schema": ingested["schema"],
            "warnings": ingested["warnings"],
            "created_at": ingested["created_at"],
        },
        "attack_classification": attack_class,
        "suspicious_cells": suspicious_cells,
        "injection_pattern": pattern,
        "blast_radius": blast,
        "defense_action": defense_action,
        "hitl_case": hitl_case,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return full_result, ingested["dataset_id"]


MAX_REPORTED_ROWS = 500


def _suspicious_cells(ingested: dict, row_flags: dict) -> list[dict]:
    """Flagged rows, most suspicious first, each with its line in the user's file and its top cells as the user wrote
    them (column name, original value; text categories decoded). Capped at MAX_REPORTED_ROWS rows."""
    split, df, cats = ingested["reference_split"], ingested["original_df"], ingested.get("categories", {})
    names = list(ingested["feature_names"]) + [ingested["label_column"] or "label"]
    scores = row_flags["row_scores"]

    def value(i: int, col: str):
        v = df[col].iloc[split + i]          # column first: a whole row would upcast ints to floats
        if col in cats and v == v and 0 <= int(v) < len(cats[col]):   # an encoded text value
            return cats[col][int(v)]
        return v.item() if hasattr(v, "item") else v

    rows = sorted(row_flags.get("top_cells", {}).items(), key=lambda kv: -scores[kv[0]])[:MAX_REPORTED_ROWS]
    samples = ingested["samples"]
    return [{"row": i, "csv_line": samples[split + i]["csv_line"], "row_score": scores[i],
             "cells": [{"column": names[j], "value": value(i, names[j]), "score": s, "is_label": j == len(names) - 1}
                       for j, s in cells]}
            for i, cells in rows]


async def run_csv_analysis(
    csv_bytes: bytes,
    filename: str,
    owner_id: str,
    source: str = "upload",
    extra: dict | None = None,
    split: dict | None = None,
) -> tuple[dict[str, Any], str]:
    """Async wrapper: runs _orchestrate_csv in the shared thread pool."""
    loop = asyncio.get_running_loop()
    full_result, dataset_id = await loop.run_in_executor(
        deps.get_thread_pool(), _orchestrate_csv, csv_bytes, filename, owner_id, split
    )
    full_result["source"] = source
    full_result["owner_id"] = owner_id
    if extra:
        full_result.update(extra)
    return deps.to_serializable(full_result), dataset_id


IMAGE_PCA_DIMS, IMAGE_CAL_FRACTION = 32, 0.3   # as benchmarked (manifest-mm-images-v8.json)


def _mm_threads() -> int:
    import os
    return min(4, os.cpu_count() or 1)


def _run_mm(R, E, ing: dict, started: float, source: str, noun: str, extra_info: dict) -> dict[str, Any]:
    """Run veritas_mm on views R (N, 32, 32, 3) uint8 and embeddings E, with the reference first in ``ing``. Builds
    the detector inputs exactly as the benchmarks do: PCA of E fitted on reference rows, fit/calibration split."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from app.detection import veritas2, veritas_mm

    y, s, names = ing["labels"], ing["reference_split"], ing["class_names"]
    with threadpool_limits(_mm_threads()):
        P = veritas_mm.pixel_features(R)
        n_cal = int(s * IMAGE_CAL_FRACTION)
        Z = PCA(min(IMAGE_PCA_DIMS, s - 1), random_state=0).fit(E[:s]).transform(E)
        zs, ps = StandardScaler().fit(Z[n_cal:s]), StandardScaler().fit(P[n_cal:s])
        part = lambda A, f: (f(A[n_cal:s]), f(A[:n_cal]), f(A[s:]))  # noqa: E731  (fit, cal, incoming)
        (Xf, Xc, Xi), (Pf, Pc, Pi), (Rf, Rc, Ri) = part(Z, zs.transform), part(P, ps.transform), part(R, lambda a: a)
        d = {"X_fit": Xf, "X_cal": Xc, "X_inc": Xi, "y_fit": y[n_cal:s], "y_cal": y[:n_cal], "y_inc": y[s:],
             "n_classes": len(names), "P_fit": Pf, "P_cal": Pc, "P_inc": Pi, "R_fit": Rf, "R_cal": Rc, "R_inc": Ri}
        out = veritas_mm.detect(d)
    verdict, suspicion = veritas2.verdict_for(out["batch_evidence"])
    cut = min(2.0, float(np.log10(s + 1)))
    flagged = (np.flatnonzero(out["row_scores"] >= cut) if verdict in ("SUSPICIOUS", "CONFIRMED_POISONED") else np.array([], int))
    flagged = flagged[np.argsort(-out["row_scores"][flagged], kind="stable")]
    checks = {n: {"evidence": round(max(c["evidence"], 0.0), 3), "in_row_ranking": bool(c["admitted"]), "description": veritas_mm.CHECKS[n]}
              for n, c in out["report"].items()}
    leading = max(checks, key=lambda n: checks[n]["evidence"]) if verdict != "CLEAN" else None
    return {
        "verdict": verdict, "overall_suspicion_score": suspicion, "batch_evidence": round(out["batch_evidence"], 3),
        "source": source, "detection_mode": noun, "n_samples": len(y) - s,
        "checks": checks, "leading_check": leading,
        "attack_classification": {"attack_type": leading or "unknown", "classification_status": "check_based",
                                  "description": veritas_mm.CHECKS[leading] if leading else "The batch is not suspicious."},
        f"flagged_{noun}s": [{"file": ing["paths"][s + i], "class": names[y[s + i]] if ing["has_labels"] else None,
                              "score": round(float(out["row_scores"][i]), 3), "check": out["row_check"][i]}
                             for i in flagged[:MAX_REPORTED_ROWS]],
        "n_flagged": int(len(flagged)),
        "dataset_info": {"dataset_id": ing["dataset_id"], "filename": ing["filename"], f"n_{noun}s": len(y), "n_reference": s,
                         "n_incoming": len(y) - s, "class_names": names, "has_labels": ing["has_labels"], **extra_info,
                         "warnings": ing["warnings"], "created_at": ing["created_at"]},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _orchestrate_images(zip_bytes: bytes, filename: str, owner_id: str, reference_bytes: bytes | None = None) -> dict[str, Any]:
    """Image zip -> veritas_mm result (blocking; runs in a background job)."""
    from app.detection import image_embed
    from app.ingestion.image_engine import ImageIngestionEngine

    started = time.perf_counter()
    ing = ImageIngestionEngine().ingest(zip_bytes, filename, reference_bytes)
    E = image_embed.embed_images(ing["images"], threads=_mm_threads())
    return _run_mm(ing["images"], E, ing, started, "image_upload", "image", {"image_size": "32x32 RGB (resized)"})


def _orchestrate_audio(zip_bytes: bytes, filename: str, owner_id: str, reference_bytes: bytes | None = None) -> dict[str, Any]:
    """WAV zip -> veritas_mm on spectrogram images, HuBERT layer-6 embeddings for the label and shift checks."""
    import torch
    from app.detection import audio
    from app.ingestion.audio_engine import RATE, AudioIngestionEngine

    started = time.perf_counter()
    ing = AudioIngestionEngine().ingest(zip_bytes, filename, reference_bytes)
    torch.set_num_threads(_mm_threads())
    E = audio.embed(ing["clips"], layers=(audio.EMBED_LAYER,))[audio.EMBED_LAYER]
    return _run_mm(audio.spectrogram_images(ing["clips"]), E, ing, started, "audio_upload", "clip",
                   {"clip_seconds": round(ing["clips"].shape[1] / RATE, 2), "view": "32x32 log-mel spectrogram"})


SCAN_SCORE = {"DANGEROUS": 1.0, "REVIEW": 0.5}


def _orchestrate_model(model_bytes: bytes, model_filename: str) -> dict[str, Any]:
    """Statically scan a model file in a sandboxed worker (blocking, for the thread pool)."""
    started = time.perf_counter()
    scan = scan_sandboxed(model_bytes, model_filename)
    metadata = None
    if scan["format"] == "pickle" and scan["verdict"] == "NO_CODE_EXECUTION_FOUND":
        try:   # restricted loader only after the scanner found nothing to run
            _, metadata = deps.model_engine.load_and_validate(model_bytes, model_filename)
        except ValueError:
            metadata = None   # a pickle, but not one of the supported sklearn estimators
    return {
        "scan_id": str(uuid.uuid4()),
        "model_filename": model_filename,
        "source": "model_scan",
        "verdict": scan["verdict"],
        "overall_suspicion_score": SCAN_SCORE.get(scan["verdict"], 0.0),
        "scan": scan,
        "model_type": (metadata or {}).get("model_type") or scan["format"],
        "model_metadata": metadata,
        "interpretation": (
            "Static analysis only: the file was never loaded or run. DANGEROUS means it imports something that can run code "
            "or access the system when loaded. NO_CODE_EXECUTION_FOUND means every import is a known data container; that is "
            "not a guarantee of safety. This does not tell you whether the model was trained on poisoned data."
        ),
        "attack_classification": {},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


async def run_model_analysis(model_bytes: bytes, model_filename: str) -> dict[str, Any]:
    """Async wrapper: runs _orchestrate_model in the shared thread pool."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(deps.get_thread_pool(), _orchestrate_model, model_bytes, model_filename)
    return deps.to_serializable(result)
