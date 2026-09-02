"""Trace a benchmark scenario through the actual CSV upload pipeline.

Run ``PYTHONPATH=backend python -m research.diagnose_benchmark`` to compare
Layer 3 flags with benchmark-owned injected-row indices. This is a diagnostic
tool; it does not change thresholds or training data.
"""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import pandas as pd

from app.demo import data_generator as generator
from app.detection.pipeline import DetectionPipeline
from app.ingestion.csv_engine import CSVIngestionEngine
from research.evaluate import load_manifest


def trace_upload_scenario(attack_name: str = "boiling_frog") -> dict:
    manifest = load_manifest()
    np.random.seed(manifest["random_seed"])
    random.seed(manifest["random_seed"])
    samples = generator.generate_clean_dataset(manifest["clean_source"]["samples"])
    reference_n = manifest["clean_source"]["split"]["reference"]
    incoming = copy.deepcopy(samples[reference_n:])
    attack = next(item for item in manifest["attacks"] if item["name"] == attack_name)
    getattr(generator, attack["injector"])(incoming, n_poison=attack["poison_count"])
    poison_indices = {reference_n + index for index, sample in enumerate(incoming) if sample.get("attack_type") == attack_name}
    records = [{**{f"feature_{index}": value for index, value in enumerate(sample["feature_vector"])}, "label": sample["label"]}
               for sample in samples[:reference_n] + incoming]
    ingested = CSVIngestionEngine().ingest(pd.DataFrame(records).to_csv(index=False).encode(), f"benchmark_{attack_name}.csv")
    result = DetectionPipeline(random_state=manifest["random_seed"]).run_on_upload(ingested)
    layer3 = result["layer_results"]["layer3_ensemble"]
    flagged = set(layer3["flagged_indices"])
    return {"attack_type": attack_name, "reference_split": ingested["reference_split"],
            "poisoned_rows": len(poison_indices), "poisoned_rows_in_reference": len({index for index in poison_indices if index < ingested["reference_split"]}),
            "verdict": result["verdict"], "overall_suspicion": result["overall_suspicion_score"],
            "layer_scores": result["layer_scores"], "layer3_flagged_indices": sorted(flagged),
            "layer3_flagged_ratio": layer3["flagged_ratio"], "layer3_poison_overlap": len(flagged & poison_indices)}


if __name__ == "__main__":
    print(json.dumps(trace_upload_scenario(), indent=2))
