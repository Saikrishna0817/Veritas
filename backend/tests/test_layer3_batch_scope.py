import numpy as np
import json
from pathlib import Path

from app.detection.pipeline import DetectionPipeline, LAYER_WEIGHTS


class StubLayer3:
    """Returns incoming-local flags to verify pipeline index and ratio handling."""

    def __init__(self):
        self.analyzed_rows = None

    def analyze(self, matrix):
        self.analyzed_rows = len(matrix)
        return {
            "suspicion_score": 0.8,
            "flagged_ratio": 1.0,
            "flagged_count": len(matrix),
            "flagged_indices": list(range(len(matrix))),
            "expected_clean_flag_rate": 0.05,
        }

    def fit(self, matrix):
        return None


def test_layer3_flagged_ratio_computed_over_incoming_batch_only():
    pipeline = DetectionPipeline()
    pipeline.X_reference = np.zeros((100, 2))
    pipeline.l3 = StubLayer3()

    result = pipeline._analyze_l3_incoming(np.ones((4, 2)))

    assert pipeline.l3.analyzed_rows == 4
    assert result["flagged_ratio"] == 1.0
    assert result["flagged_indices_incoming"] == [0, 1, 2, 3]
    assert result["flagged_indices"] == [100, 101, 102, 103]


def test_layer3_flagged_ratio_not_diluted_by_reference_set_size():
    pipeline = DetectionPipeline()
    pipeline.X_reference = np.zeros((10_000, 2))
    pipeline.l3 = StubLayer3()

    result = pipeline._analyze_l3_incoming(np.ones((3, 2)))

    assert pipeline.l3.analyzed_rows == 3
    assert result["flagged_ratio"] == 1.0


def test_layer3_gate_does_not_suppress_high_confidence_batch_signal():
    raw_score = 0.8
    gated_score = DetectionPipeline._gate_l3_score(
        raw_score,
        {"flagged_ratio": 0.10, "expected_clean_flag_rate": 0.05},
    )

    assert gated_score == raw_score


def test_layer_weights_sum_to_one():
    assert sum(LAYER_WEIGHTS.values()) == 1.0


def test_updated_weights_are_documented_with_benchmark_citation():
    backend_root = Path(__file__).resolve().parents[1]
    pipeline_source = (backend_root / "app/detection/pipeline.py").read_text(encoding="utf-8")

    assert "2026-09-02-benchmark-v1-step2-l3-gate.json" in pipeline_source


def test_full_pipeline_meets_pre_registered_target_on_benchmark():
    backend_root = Path(__file__).resolve().parents[1]
    report_path = backend_root / "research/results/2026-09-02-benchmark-v1-eval-r3.json"
    report = json.loads(report_path.read_text())
    operating_point = report["pre_registered_operating_point"]

    assert operating_point["met"]
    assert (report["metrics"]["combined"]["false_positive_rate"]
            <= operating_point["maximum_clean_batch_false_positive_rate"])


def test_high_confidence_row_level_detection_produces_non_clean_batch_verdict():
    """A real strong incoming-only Layer-3 alarm must survive aggregation."""
    rng = np.random.default_rng(42)
    pipeline = DetectionPipeline(random_state=42)
    reference = rng.normal(0, 0.05, size=(100, 4))
    incoming = np.full((50, 4), 10.0)

    pipeline.fit_baseline(reference, np.zeros(len(reference), dtype=int))
    result = pipeline.analyze(incoming, np.zeros(len(incoming), dtype=int))

    assert result["details"]["layer3"]["flagged_ratio"] == 1.0
    assert result["verdict"] != "CLEAN"
