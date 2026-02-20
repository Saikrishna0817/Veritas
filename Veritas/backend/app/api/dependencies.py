"""
Shared API dependencies/state.

This module centralizes singleton engines, caches, and persistence init so
route modules can stay thin and consistent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from app.models import database as db
from app.defense.engine import HumanInTheLoopQueue, RedTeamSimulator, StabilityAwareAutoDefense
from app.demo.data_generator import get_demo_data
from app.demo.real_datasets import DATASET_CATALOG, get_real_dataset
from app.detection.pipeline import DetectionPipeline
from app.forensics.engine import (
    AttackTypeClassifier,
    BlastRadiusMapper,
    CounterfactualSimulator,
    InjectionPatternReconstructor,
    SophisticationScorer,
)
from app.ingestion.csv_engine import CSVIngestionEngine
from app.ingestion.model_engine import ModelScanEngine
from app.utils.serialization import to_serializable


# Initialise SQLite on import (keeps startup simple)
db.init_db()


# ── Singletons ────────────────────────────────────────────────────────────────
pipeline = DetectionPipeline()
classifier = AttackTypeClassifier()
reconstructor = InjectionPatternReconstructor()
sophistication = SophisticationScorer()
blast_mapper = BlastRadiusMapper()
counterfactual = CounterfactualSimulator()
defense = StabilityAwareAutoDefense()
hitl = HumanInTheLoopQueue()
red_team = RedTeamSimulator()
model_engine = ModelScanEngine()


# ── In-memory caches (fast path) ─────────────────────────────────────────────
demo_result_cache: Dict[str, Any] = {}
upload_result_cache: Dict[str, Any] = {}  # keyed by dataset_id


async def broadcast_demo_events(manager, result: dict):
    """Broadcast detection events to WebSocket clients."""
    await asyncio.sleep(0.5)
    await manager.broadcast(
        "sample_analyzed",
        {
            "n_samples": result.get("n_samples"),
            "suspicion_score": result.get("overall_suspicion_score"),
            "layer_scores": result.get("layer_scores"),
        },
    )

    await asyncio.sleep(1.0)
    if result.get("verdict") and result.get("verdict") != "CLEAN":
        layer4 = (result.get("layer_results") or {}).get("layer4_causal") or {}
        blast = result.get("blast_radius") or {}
        attack_class = result.get("attack_classification") or {}
        pattern = result.get("injection_pattern") or {}

        await manager.broadcast(
            "attack_confirmed",
            {
                "attack_type": attack_class.get("attack_type"),
                "confidence": attack_class.get("confidence"),
                "causal_effect": layer4.get("causal_effect", 0),
                "narrative": (pattern.get("narrative") or "")[:200],
                "blast_radius": {
                    "n_batches": blast.get("n_batches_affected"),
                    "n_models": blast.get("n_models_affected"),
                    "impact_pct": blast.get("prediction_impact_pct"),
                },
            },
        )

    await asyncio.sleep(1.5)
    defense_action = result.get("defense_action") or {}
    if defense_action.get("action") in ("quarantine", "soft_quarantine"):
        await manager.broadcast(
            "defense_triggered",
            {
                "action": defense_action.get("action"),
                "samples_affected": defense_action.get("samples_affected"),
                "model_stable": defense_action.get("model_stable", True),
            },
        )

    if result.get("hitl_case"):
        await asyncio.sleep(0.5)
        await manager.broadcast(
            "human_review_required",
            {
                "case_id": result["hitl_case"].get("case_id"),
                "suspicion_score": result["hitl_case"].get("suspicion_score"),
                "deadline": result["hitl_case"].get("deadline"),
            },
        )

