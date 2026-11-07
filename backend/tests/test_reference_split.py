"""The trusted reference set can be a separate file, a column cutoff, or (default) file order."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.ingestion.csv_engine import CSVIngestionEngine
from app.main import app


def _frame(n=400, seed=0, flip=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[:flip] = 1 - y[:flip]
    return pd.DataFrame(X, columns=list("abcde")).assign(label=y)


def _bytes(df):
    return df.to_csv(index=False).encode()


def test_reference_file_is_the_reference_and_main_file_is_all_incoming():
    ref, main = _frame(300, 1), _frame(200, 2)
    out = CSVIngestionEngine().ingest(_bytes(main), reference_bytes=_bytes(ref))
    assert out["reference_split"] == 300 and out["n_rows"] == 500
    assert out["schema"]["split_method"] == "reference_file"
    # reference rows are scaled with themselves only, so reference min/max map to 0/1
    assert out["features"][:300].min() == 0 and out["features"][:300].max() == 1


def test_reference_file_must_have_the_same_columns():
    with pytest.raises(ValueError, match="same columns"):
        CSVIngestionEngine().ingest(_bytes(_frame()), reference_bytes=_bytes(_frame().drop(columns="e")))


def test_column_cutoff_puts_early_rows_in_reference_whatever_the_file_order():
    df = _frame(400).assign(week=np.random.RandomState(3).permutation(400))   # shuffled file
    out = CSVIngestionEngine().ingest(_bytes(df), split_column="week", split_cutoff="300")
    assert out["reference_split"] == 300 and "week" not in out["feature_names"]
    assert out["schema"]["split_method"] == "column_cutoff"


def test_column_cutoff_understands_dates():
    df = _frame(100).assign(day=pd.date_range("2026-01-01", periods=100).astype(str))
    out = CSVIngestionEngine().ingest(_bytes(df), split_column="day", split_cutoff="2026-03-01")
    assert out["reference_split"] == 59    # Jan 31 + Feb 28 days


def test_bad_split_requests_are_rejected():
    eng, df = CSVIngestionEngine(), _frame()
    with pytest.raises(ValueError, match="both"):
        eng.ingest(_bytes(df), split_column="a")
    with pytest.raises(ValueError, match="not both"):
        eng.ingest(_bytes(df), reference_bytes=_bytes(df), split_column="a", split_cutoff="0")
    with pytest.raises(ValueError, match="not in the CSV"):
        eng.ingest(_bytes(df), split_column="nope", split_cutoff="0")
    with pytest.raises(ValueError, match="at least"):
        eng.ingest(_bytes(df.assign(t=range(400))), split_column="t", split_cutoff="5")


def test_upload_endpoint_accepts_a_reference_file_and_finds_the_flipped_labels():
    headers = {"Authorization": f"Bearer {create_access_token({'id': 'u9', 'name': 'u9', 'role': 'user'})}"}
    ref, main = _frame(1500, 1), _frame(800, 2, flip=80)    # a shuffled main file the old file-order split would mishandle
    res = TestClient(app).post("/api/v1/analyze/upload", headers=headers, files={
        "file": ("main.csv", _bytes(main), "text/csv"), "reference_file": ("ref.csv", _bytes(ref), "text/csv")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["dataset_info"]["schema"]["split_method"] == "reference_file"
    assert body["verdict"] in ("SUSPICIOUS", "CONFIRMED_POISONED")
    flagged = set(body["layer_results"]["row_flags"]["flagged_indices_incoming"])
    assert len(flagged & set(range(80))) >= 50
