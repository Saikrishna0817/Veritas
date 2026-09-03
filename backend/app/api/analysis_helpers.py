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
from typing import Any

from app.api import dependencies as deps

logger = logging.getLogger(__name__)


def _orchestrate_csv(csv_bytes: bytes, filename: str) -> tuple[dict[str, Any], str]:
    """Run the full CSV ingestion + detection + forensics pipeline (blocking).

    Returns (full_result_dict, dataset_id).
    This function is designed to be called inside a thread pool executor.
    """
    started = time.perf_counter()

    engine = deps.CSVIngestionEngine()
    ingested = engine.ingest(csv_bytes, filename=filename)

    upload_pipeline = deps.DetectionPipeline()
    detection_result = upload_pipeline.run_on_upload(ingested)

    samples = ingested["samples"]
    incoming_samples = samples[ingested["reference_split"]:]

    attack_class = deps.classifier.classify(detection_result["layer_results"], incoming_samples)

    # Tag samples exceeding ensemble threshold as suspected
    ensemble_scores = detection_result["layer_results"]["layer3_ensemble"].get("ensemble_scores", [])
    if ensemble_scores:
        for i, s in enumerate(incoming_samples):
            if i < len(ensemble_scores) and ensemble_scores[i] > 0.6:
                s["poison_status"] = "suspected"

    pattern = deps.reconstructor.reconstruct(incoming_samples, attack_class, detection_result["layer_results"])
    sophistication = deps.sophistication.score(attack_class, pattern, detection_result)
    blast = deps.blast_mapper.map(incoming_samples, detection_result["layer_results"])
    counterfactual = deps.counterfactual.simulate(detection_result["layer_results"], blast)
    defense_action = deps.defense.decide_action(
        incoming_samples,
        detection_result["overall_suspicion_score"],
        detection_result["verdict"],
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
        "injection_pattern": pattern,
        "sophistication": sophistication,
        "blast_radius": blast,
        "counterfactual": counterfactual,
        "defense_action": defense_action,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return full_result, ingested["dataset_id"]


async def run_csv_analysis(
    csv_bytes: bytes,
    filename: str,
    source: str = "upload",
    extra: dict | None = None,
) -> tuple[dict[str, Any], str]:
    """Async wrapper: runs _orchestrate_csv in the shared thread pool."""
    loop = asyncio.get_running_loop()
    full_result, dataset_id = await loop.run_in_executor(
        deps.get_thread_pool(), _orchestrate_csv, csv_bytes, filename
    )
    full_result["source"] = source
    if extra:
        full_result.update(extra)
    return deps.to_serializable(full_result), dataset_id


def _orchestrate_model(
    model_bytes: bytes,
    model_filename: str,
    dataset_bytes: bytes | None,
    dataset_filename: str | None,
) -> dict[str, Any]:
    """Run model scan pipeline (blocking — for thread pool)."""
    started = time.perf_counter()
    ingested = deps.model_engine.ingest(model_bytes, model_filename, dataset_bytes, dataset_filename)

    scan_pipeline = deps.DetectionPipeline()
    detection_result = scan_pipeline.run_on_upload(ingested)

    samples = ingested["samples"]
    incoming = samples[ingested["reference_split"]:]

    attack_class = deps.classifier.classify(detection_result["layer_results"], incoming)
    pattern = deps.reconstructor.reconstruct(incoming, attack_class, detection_result["layer_results"])
    sophistication = deps.sophistication.score(attack_class, pattern, detection_result)
    blast = deps.blast_mapper.map(incoming, detection_result["layer_results"])
    defense_action = deps.defense.decide_action(
        incoming, detection_result["overall_suspicion_score"], detection_result["verdict"]
    )

    return {
        **detection_result,
        "scan_id": ingested["scan_id"],
        "model_filename": model_filename,
        "dataset_filename": dataset_filename,
        "model_type": ingested["model_type"],
        "model_metadata": ingested["model_metadata"],
        "extraction_info": ingested["extraction_info"],
        "attack_classification": attack_class,
        "injection_pattern": pattern,
        "sophistication": sophistication,
        "blast_radius": blast,
        "defense_action": defense_action,
        "source": "model_scan",
        "interpretation": (
            "Parameters extracted from the model's learned weights/trees were "
            "analyzed for statistical anomalies consistent with poisoning. "
            "A high suspicion score suggests the model was trained on poisoned data."
        ),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


async def run_model_analysis(
    model_bytes: bytes,
    model_filename: str,
    dataset_bytes: bytes | None,
    dataset_filename: str | None,
) -> dict[str, Any]:
    """Async wrapper: runs _orchestrate_model in the shared thread pool."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        deps.get_thread_pool(),
        _orchestrate_model,
        model_bytes,
        model_filename,
        dataset_bytes,
        dataset_filename,
    )
    return deps.to_serializable(result)
