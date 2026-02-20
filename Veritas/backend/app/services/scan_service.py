"""Scanning services (thin wrappers around engines)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.detection.pipeline import DetectionPipeline
from app.ingestion.csv_engine import CSVIngestionEngine
from app.ingestion.model_engine import ModelScanEngine


def analyze_csv(csv_bytes: bytes, filename: str) -> Dict[str, Any]:
    engine = CSVIngestionEngine()
    ingested = engine.ingest(csv_bytes, filename=filename)
    pipeline = DetectionPipeline()
    return pipeline.run_on_upload(ingested)


def analyze_model(model_bytes: bytes, filename: str, dataset_bytes: Optional[bytes] = None, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    engine = ModelScanEngine()
    ingested = engine.ingest(model_bytes, filename, dataset_bytes, dataset_name)
    pipeline = DetectionPipeline()
    return pipeline.run_on_upload(ingested)

