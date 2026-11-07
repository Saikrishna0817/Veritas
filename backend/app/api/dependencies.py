"""
Shared API dependencies / state.

This module centralises singleton engines, caches, and persistence init so
route modules stay thin and consistent.

Changes from v2.2 review:
  - Shared ThreadPoolExecutor (M4) — no more per-request pool creation
  - Bounded LRU cache (M6) — prevent unbounded memory growth
  - Rate limiters exposed for all CPU-heavy endpoints (M1)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.models import database as db
from app.defense.engine import HumanInTheLoopQueue, StabilityAwareAutoDefense
from app.demo.data_generator import get_demo_data
from app.demo.real_datasets import DATASET_CATALOG, get_real_dataset
from app.detection.pipeline import DetectionPipeline
from app.forensics.engine import (
    AttackTypeClassifier,
    BlastRadiusMapper,
    InjectionPatternReconstructor,
)
from app.ingestion.csv_engine import CSVIngestionEngine
from app.ingestion.model_engine import ModelScanEngine
from app.utils.serialization import to_serializable
from app.core.rate_limit import SlidingWindowRateLimiter

# ── Database init ─────────────────────────────────────────────────────────────
db.init_db()


# ── Shared worker pool ────────────────────────────────────────────────────────
# CPU-bound detection work runs here (never on the event loop). Adjust the size
# via MAX_ANALYSIS_WORKERS. A thread pool is required: routes submit closures,
# which a process pool cannot pickle.
import os as _os

_MAX_WORKERS = int(_os.getenv("MAX_ANALYSIS_WORKERS", "4"))
_worker_pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="veritas-analysis")


def get_thread_pool() -> ThreadPoolExecutor:
    """Return the shared analysis worker pool."""
    return _worker_pool


# ── Singletons ────────────────────────────────────────────────────────────────
classifier    = AttackTypeClassifier()
reconstructor = InjectionPatternReconstructor()
blast_mapper  = BlastRadiusMapper()
defense       = StabilityAwareAutoDefense()
hitl          = HumanInTheLoopQueue()
model_engine  = ModelScanEngine()


def new_detection_pipeline(**kwargs) -> DetectionPipeline:
    """Create an isolated pipeline instance for concurrent analyses."""
    return DetectionPipeline(**kwargs)


# ── Bounded LRU result cache (M6) ─────────────────────────────────────────────

class _LRUCache:
    """Thread-safe bounded LRU cache backed by an OrderedDict."""

    def __init__(self, maxsize: int = 50) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)  # evict oldest

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            self._data.move_to_end(key)
            return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


demo_result_cache: _LRUCache = _LRUCache(maxsize=20)
upload_result_cache: _LRUCache = _LRUCache(maxsize=50)

def owner_scope(user: dict) -> str:
    """Whose rows a list/read endpoint may show: everyone's for an admin, otherwise only the caller's."""
    return db.ALL_OWNERS if user.get("role") == "admin" else user["id"]


def latest_key(owner_id: str) -> str:
    return f"latest:{owner_id}"


def resolve_latest_result(owner_id: str, prefer: str = "auto") -> dict | None:
    """The owner's best available analysis, without generating synthetic data. Never another user's.

    prefer:
      auto         → real first; demo only as last resort
      real         → upload / real_dataset only (no demo)
      upload       → upload only
      real_dataset → real_dataset only
      demo         → demo only
    """
    prefer = (prefer or "auto").lower()
    key = latest_key(owner_id)

    if prefer == "demo":
        return demo_result_cache.get(key) or db.get_latest(owner_id, source="demo")

    if prefer == "upload":
        return upload_result_cache.get(key) or db.get_latest(owner_id, source="upload")

    if prefer == "real_dataset":
        return db.get_latest(owner_id, source="real_dataset")

    upload_cached = upload_result_cache.get(key)
    if upload_cached:
        return upload_cached

    real = db.get_latest_real(owner_id)
    if real:
        return real

    if prefer == "real":
        return None

    return demo_result_cache.get(key) or db.get_latest(owner_id)


# ── Rate limiters ─────────────────────────────────────────────────────────────
# CPU-heavy analysis endpoints — shared limiter: 10 requests / 60 s per user+IP.
# Used by upload, model scan, real-dataset analysis, demo run, and red-team.
# Process-local by design for the single-instance prototype; production
# deployments should enforce the same policy at the gateway/shared store.
analysis_rate_limiter = SlidingWindowRateLimiter(limit=10, window_seconds=60)


# ── WebSocket broadcast helper ────────────────────────────────────────────────

async def broadcast_demo_events(manager, result: dict, owner_id: str) -> None:
    """Broadcast detection events to all connected WebSocket clients."""
    await manager.broadcast(
        "sample_analyzed",
        {
            "n_samples": result.get("n_samples"),
            "suspicion_score": result.get("overall_suspicion_score"),
            "layer_scores": result.get("layer_scores"),
        },
        owner_id=owner_id,
    )

    if result.get("verdict") and result.get("verdict") != "CLEAN":
        blast = result.get("blast_radius") or {}
        attack_class = result.get("attack_classification") or {}
        pattern = result.get("injection_pattern") or {}

        await manager.broadcast(
            "attack_confirmed",
            {
                "attack_type": attack_class.get("attack_type"),
                "confidence": attack_class.get("confidence"),
                "narrative": (pattern.get("narrative") or "")[:200],
                "blast_radius": {
                    "n_batches": blast.get("n_batches_affected"),
                    "n_models": blast.get("n_models_affected"),
                    "impact_pct": blast.get("prediction_impact_pct"),
                },
            },
            owner_id=owner_id,
        )

    defense_action = result.get("defense_action") or {}
    if defense_action.get("action") in ("quarantine", "soft_quarantine"):
        await manager.broadcast(
            "defense_triggered",
            {
                "action": defense_action.get("action"),
                "samples_affected": defense_action.get("samples_affected"),
                "model_stable": defense_action.get("model_stable", True),
            },
            owner_id=owner_id,
        )

    if result.get("hitl_case"):
        await manager.broadcast(
            "human_review_required",
            {
                "case_id": result["hitl_case"].get("case_id"),
                "suspicion_score": result["hitl_case"].get("suspicion_score"),
                "deadline": result["hitl_case"].get("deadline"),
            },
            owner_id=owner_id,
        )
