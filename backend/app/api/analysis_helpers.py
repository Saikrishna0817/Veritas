"""Shared analysis orchestration used by upload, model scan, and dataset routes."""

from __future__ import annotations

import time
from typing import Any

from app.api import dependencies as deps


def run_csv_analysis(csv_bytes: bytes, filename: str, source: str = "upload", extra: dict | None = None) -> dict[str, Any]:
    """Run the full CSV ingestion + detection + forensics pipeline."""
    started = time.perf_counter()
    engine = deps.CSVIngestionEngine()
    ingested = engine.ingest(csv_bytes, filename=filename)

    pipeline = deps.DetectionPipeline()
    detection_result = pipeline.run_on_upload(ingested)

    samples = ingested["samples"]
    incoming_samples = samples[ingested["reference_split"] :]

    attack_class = deps.classifier.classify(detection_result["layer_results"], incoming_samples)

    ensemble_scores = detection_result["layer_results"]["layer3_ensemble"].get("ensemble_scores", [])
    if ensemble_scores:
        for index, sample in enumerate(incoming_samples):
            if index < len(ensemble_scores) and ensemble_scores[index] > 0.6:
                sample["poison_status"] = "suspected"

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
        "source": source,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    if extra:
        full_result.update(extra)
    return full_result, ingested["dataset_id"]


def run_model_analysis(
    model_bytes: bytes,
    model_filename: str,
    dataset_bytes: bytes | None,
    dataset_filename: str | None,
) -> dict[str, Any]:
    """Run model scan pipeline."""
    started = time.perf_counter()
    ingested = deps.model_engine.ingest(model_bytes, model_filename, dataset_bytes, dataset_filename)

    pipeline = deps.DetectionPipeline()
    detection_result = pipeline.run_on_upload(ingested)

    samples = ingested["samples"]
    incoming = samples[ingested["reference_split"] :]

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
