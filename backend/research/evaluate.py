"""Deterministic, incident-level evaluation for the CSV poisoning benchmark.

Layers 1, 2, 4 and 5 emit one signal per analysed batch rather than one label
per row. Metrics are therefore incident-level: each clean or injected batch is
one evaluation example. This deliberately avoids fabricating row-level metrics.
"""
from __future__ import annotations

import copy
import argparse
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from app.demo import data_generator as generator
from app.detection.pipeline import DetectionPipeline, THRESHOLD_SUSPICIOUS

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "benchmark" / "manifest.json"
LAYER_NAMES = ("l1_statistical", "l2_spectral", "l3_ensemble", "l4_causal", "l5_federated", "combined")
POSITIVE_VERDICTS = frozenset({"CONFIRMED_POISONED", "SUSPICIOUS", "LOW_RISK"})
PRE_REGISTERED_OPERATING_POINT = {
    "selection_rule": (
        "Keep the pre-existing verdict thresholds and layer weights; do not "
        "select a new threshold from this single-scenario-per-attack benchmark."
    ),
    "maximum_clean_batch_false_positive_rate": 0.02,
    "primary_measurement": "combined incident-level false_positive_rate",
}


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def _samples_to_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray([sample["feature_vector"] for sample in samples], dtype=float),
            np.asarray([sample["label"] for sample in samples], dtype=int))


def is_positive_verdict(verdict: str) -> bool:
    """Map every non-clean pipeline risk verdict to an incident alarm."""
    return verdict in POSITIVE_VERDICTS


def _binary_metrics(predictions: list[bool], truth: list[bool]) -> dict[str, float]:
    tp = sum(pred and actual for pred, actual in zip(predictions, truth))
    fp = sum(pred and not actual for pred, actual in zip(predictions, truth))
    fn = sum(not pred and actual for pred, actual in zip(predictions, truth))
    tn = sum(not pred and not actual for pred, actual in zip(predictions, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0}


def evaluate_scenario(attack_name: str | None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one immutable benchmark scenario; ``None`` denotes clean data."""
    manifest = manifest or load_manifest()
    seed = manifest["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    clean = generator.generate_clean_dataset(manifest["clean_source"]["samples"])
    reference_n = manifest["clean_source"]["split"]["reference"]
    reference = copy.deepcopy(clean[:reference_n])
    incoming = copy.deepcopy(clean[reference_n:])
    if attack_name:
        attack = next(item for item in manifest["attacks"] if item["name"] == attack_name)
        getattr(generator, attack["injector"])(incoming, n_poison=attack["poison_count"])
    # These labels belong to benchmark generation. They are captured before
    # any pipeline call and never read from ingestion/pipeline output, whose
    # `poison_status` field is deliberately an unknown/suspected placeholder.
    poisoned_incoming_indices = [index for index, sample in enumerate(incoming)
                                 if sample.get("attack_type") == attack_name]

    x_ref, y_ref = _samples_to_arrays(reference)
    x_incoming, y_incoming = _samples_to_arrays(incoming)
    scale_min = np.minimum(x_ref.min(axis=0), x_incoming.min(axis=0))
    scale_span = np.maximum(x_ref.max(axis=0), x_incoming.max(axis=0)) - scale_min
    scale_span[scale_span == 0] = 1.0
    pipeline = DetectionPipeline(random_state=seed)
    pipeline.fit_baseline((x_ref - scale_min) / scale_span, y_ref)
    result = pipeline.analyze((x_incoming - scale_min) / scale_span, y_incoming)
    scores = result["layer_scores"]
    l3_flagged_full = result["details"]["layer3"].get("flagged_indices", [])
    l3_flagged_incoming = sorted(index - len(reference) for index in l3_flagged_full if index >= len(reference))
    return {"attack_type": attack_name or "clean", "ground_truth": bool(attack_name),
            "poisoned_incoming_indices": poisoned_incoming_indices,
            "incoming_count": len(incoming),
            "scores": {**scores, "combined": result["overall_suspicion"]},
            "predictions": {"combined": is_positive_verdict(result["verdict"])},
            "layer3_flagged_incoming_indices": l3_flagged_incoming,
            "result": result}


def _metrics(rows: list[dict], detector: str) -> dict[str, float]:
    predictions = ([row["predictions"]["combined"] for row in rows] if detector == "combined"
                   else [row["scores"][detector] >= THRESHOLD_SUSPICIOUS for row in rows])
    truth = [row["ground_truth"] for row in rows]
    return _binary_metrics(predictions, truth)


def _layer3_row_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Evaluate Layer 3 row flags against benchmark-owned incoming labels."""
    per_attack: dict[str, dict[str, float]] = {}
    for row in rows:
        n_rows = row["incoming_count"]
        truth = [index in set(row["poisoned_incoming_indices"]) for index in range(n_rows)]
        predictions = [index in set(row["layer3_flagged_incoming_indices"]) for index in range(n_rows)]
        per_attack[row["attack_type"]] = _binary_metrics(predictions, truth)
    return per_attack


def evaluate() -> dict[str, Any]:
    manifest = load_manifest()
    rows = [evaluate_scenario(None, manifest)] + [evaluate_scenario(item["name"], manifest) for item in manifest["attacks"]]
    metrics = {layer: _metrics(rows, layer) for layer in LAYER_NAMES}
    operating_point = {
        **PRE_REGISTERED_OPERATING_POINT,
        "met": (metrics["combined"]["false_positive_rate"]
                <= PRE_REGISTERED_OPERATING_POINT["maximum_clean_batch_false_positive_rate"]),
    }
    return {"benchmark_id": manifest["benchmark_id"], "evaluation_revision": 3,
            "measurement_unit": "incident_batch", "threshold": THRESHOLD_SUSPICIOUS,
            "pre_registered_operating_point": operating_point, "metrics": metrics,
            "layer3_row_metrics": _layer3_row_metrics(rows), "scenarios": rows}


def write_report(
    report: dict[str, Any], output_dir: Path | None = None, label: str | None = None
) -> Path:
    """Write a versioned report without overwriting an earlier measurement."""
    output_dir = output_dir or ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = label or f"eval-r{report['evaluation_revision']}"
    path = output_dir / f"{date.today().isoformat()}-benchmark-v1-{suffix}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the immutable Veritas benchmark.")
    parser.add_argument("--label", help="Version label for this measurement report")
    args = parser.parse_args()
    print(write_report(evaluate(), label=args.label))
