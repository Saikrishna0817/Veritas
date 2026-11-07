"""Tests for real-data-first result resolution."""

from __future__ import annotations

import asyncio
import uuid

import pytest


def _sample_result(source: str, score: float = 0.42) -> dict:
    return {
        "job_id": str(uuid.uuid4()),
        "source": source,
        "verdict": "SUSPICIOUS" if score > 0.3 else "CLEAN",
        "overall_suspicion_score": score,
        "n_samples": 100,
        "attack_classification": {"attack_type": "label_flip", "confidence": 0.5},
        "layer_results": {},
        "dataset_info": {"filename": f"{source}_file.csv"},
    }


@pytest.fixture()
def isolated_db(tmp_path):
    from app.models import database as db

    old_url = db.current_url()
    db.configure(f"sqlite:///{tmp_path / 'test_forensics.db'}")
    db.init_db()
    yield db
    db.configure(old_url)


def test_get_latest_real_empty(isolated_db):
    assert isolated_db.get_latest_real("u1") is None


def test_get_latest_real_prefers_upload_over_demo(isolated_db):
    isolated_db.save_result(_sample_result("demo", 0.9), "demo", "demo.csv", "u1")
    isolated_db.save_result(_sample_result("upload", 0.2), "upload", "real.csv", "u1")
    latest_real = isolated_db.get_latest_real("u1")
    assert latest_real is not None
    assert latest_real["source"] == "upload"


def test_resolve_prefer_real_skips_demo(isolated_db):
    from app.api import dependencies as deps

    with deps.upload_result_cache._lock:
        deps.upload_result_cache._data.clear()
    with deps.demo_result_cache._lock:
        deps.demo_result_cache._data.clear()
    isolated_db.save_result(_sample_result("demo", 0.8), "demo", "d.csv", "u1")
    assert deps.resolve_latest_result("u1", prefer="real") is None
    auto = deps.resolve_latest_result("u1", prefer="auto")
    assert auto is not None
    assert auto["source"] == "demo"


def test_trust_score_empty_is_not_grade_a(isolated_db):
    from app.api import dependencies as deps
    from app.api.routes.models import get_trust_score

    with deps.upload_result_cache._lock:
        deps.upload_result_cache._data.clear()
    with deps.demo_result_cache._lock:
        deps.demo_result_cache._data.clear()
    payload = asyncio.run(get_trust_score({"id": "u1", "name": "u1", "role": "user"}))
    assert payload["data_source"] == "none"
    assert payload["has_analysis"] is False
    assert payload["model_safety"]["grade"] is None
