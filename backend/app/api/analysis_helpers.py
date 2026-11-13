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
        "injection_pattern": pattern,
        "blast_radius": blast,
        "defense_action": defense_action,
        "hitl_case": hitl_case,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return full_result, ingested["dataset_id"]


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
