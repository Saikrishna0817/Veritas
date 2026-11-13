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
