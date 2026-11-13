"""Regression checks for the 2026-11-14 code review fixes."""
import numpy as np
import pandas as pd

from app.defense.engine import StabilityAwareAutoDefense
from app.forensics.engine import AttackTypeClassifier
from app.ingestion.csv_engine import CSVIngestionEngine


def test_text_columns_become_features_and_many_class_labels_are_kept():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"age": rng.integers(18, 80, n), "color": rng.choice(["red", "blue", "green"], n),
                       "note": [f"free text {i}" for i in range(n)], "label": rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), n)})
    r = CSVIngestionEngine().ingest(df.to_csv(index=False).encode(), filename="t.csv")
    assert r["feature_names"] == ["age", "color"] and r["has_labels"] and len(np.unique(r["labels"])) == 26
    assert any("note" in w for w in r["warnings"])


def test_no_attack_is_named_without_an_attack_profile():
    """The old rules named 'label_flip' for any clean batch where one class dominated."""
    assert AttackTypeClassifier().classify({"attack_profile": None})["attack_type"] == "unknown"


def test_confirmed_batches_are_always_quarantined(monkeypatch):
    from app.models import database as db
    monkeypatch.setattr(db, "get_defense_actions", lambda owner, limit=100: [])
    monkeypatch.setattr(db, "save_defense_action", lambda action: None)
    samples = [{"id": f"row_{i}", "poison_status": "suspected", "row_score": float(i)} for i in range(40)]
    action = StabilityAwareAutoDefense().decide_action(samples, 0.66, "CONFIRMED_POISONED", scope="s", owner_id="u")
    assert action["action"] == "quarantine" and action["sample_ids"] == ["row_39", "row_38"]   # 5% cap, most suspicious first


def test_failed_upload_refunds_quota_and_stale_jobs_fail():   # the suite already runs on a scratch database
    from datetime import datetime, timedelta, timezone
    from app.core import quota
    from app.jobs.queue import job_store
    from app.models import database as db

    db.init_db()
    user = {"id": "q-user", "role": "user"}
    charged = quota.charge(user, "analyses", 1, 5)
    assert db.get_usage("q-user")["analyses"] == 1
    quota.refund("q-user", charged)
    assert db.get_usage("q-user")["analyses"] == 0

    db.create_job("stale-1", "csv_upload", "q-user", {"key": "k", "charged": quota.charge(user, "analyses", 1, 5)})
    db.update_job("stale-1", status="running", started_at=(datetime.now(timezone.utc) - timedelta(minutes=40)).isoformat())
    job = job_store.get("stale-1")
    assert job["status"] == "failed" and "worker stopped" in job["error"]
    assert db.get_usage("q-user")["analyses"] == 0


def test_upload_reports_suspicious_cells_with_file_lines_and_original_values():
    from app.api.analysis_helpers import _orchestrate_csv
    rng = np.random.default_rng(1)
    n = 900
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n)})
    df["label"] = (df["a"] + df["b"] > 0).astype(int)
    bad = np.arange(800, 830)                       # in the incoming 30%: out-of-range trigger plus a flipped label
    df.loc[bad, "b"] = 25.0
    df.loc[bad, "label"] = 1 - df.loc[bad, "label"]
    result, _ = _orchestrate_csv(df.to_csv(index=False).encode(), "t.csv", "cells-user")
    cells = result["suspicious_cells"]
    assert result["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED") and cells
    top = cells[0]
    assert top["csv_line"] - 2 in set(bad)                       # line in the uploaded file (header is line 1)
    assert {c["column"] for c in top["cells"]} & {"b", "label"}
    assert all(c["value"] in ("x", "y", "z") for row in cells for c in row["cells"] if c["column"] == "c")
    assert all(isinstance(c["value"], int) for row in cells for c in row["cells"] if c["is_label"])
