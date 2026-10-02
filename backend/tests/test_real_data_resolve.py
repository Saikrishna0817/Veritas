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

    db_path = tmp_path / "test_forensics.db"
    old_path = db.DB_PATH
    db.DB_PATH = db_path
    if hasattr(db._local, "conn"):
        try:
            if db._local.conn is not None:
                db._local.conn.close()
        except Exception:
            pass
        db._local.conn = None
    db.init_db()
    yield db
    try:
        if hasattr(db._local, "conn") and db._local.conn is not None:
            db._local.conn.close()
    except Exception:
        pass
    db._local.conn = None
    db.DB_PATH = old_path


def test_get_latest_real_empty(isolated_db):
    assert isolated_db.get_latest_real() is None


def test_get_latest_real_prefers_upload_over_demo(isolated_db):
    isolated_db.save_result(_sample_result("demo", 0.9), "demo", "demo.csv")
    isolated_db.save_result(_sample_result("upload", 0.2), "upload", "real.csv")
    latest_real = isolated_db.get_latest_real()
    assert latest_real is not None
    assert latest_real["source"] == "upload"


def test_resolve_prefer_real_skips_demo(isolated_db):
    from app.api import dependencies as deps

    with deps.upload_result_cache._lock:
        deps.upload_result_cache._data.clear()
    with deps.demo_result_cache._lock:
        deps.demo_result_cache._data.clear()
    isolated_db.save_result(_sample_result("demo", 0.8), "demo", "d.csv")
    assert deps.resolve_latest_result(prefer="real") is None
    auto = deps.resolve_latest_result(prefer="auto")
    assert auto is not None
    assert auto["source"] == "demo"


def test_trust_score_empty_is_not_grade_a(isolated_db):
    from app.api import dependencies as deps
    from app.api.routes.models import get_trust_score

    with deps.upload_result_cache._lock:
        deps.upload_result_cache._data.clear()
    with deps.demo_result_cache._lock:
        deps.demo_result_cache._data.clear()
    payload = asyncio.run(get_trust_score())
    assert payload["data_source"] == "none"
    assert payload["has_analysis"] is False
    assert payload["model_safety"]["grade"] is None


def test_federated_clients_are_labelled_synthetic():
    from app.api.routes.models import get_federated_clients

    payload = asyncio.run(get_federated_clients())
    assert payload["synthetic"] is True
    assert payload["source"] == "demo"
