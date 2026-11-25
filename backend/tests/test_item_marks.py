"""Trial marks: per-owner upsert, clearing, isolation, admin CSV export, erasure."""
import csv
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.routes import marks
from app.main import app
from auth_helpers import token_for

client = TestClient(app)


def _h(uid, role="user"):
    return {"Authorization": f"Bearer {token_for({'id': uid, 'name': uid, 'role': role})}"}


def _csv():
    rng = np.random.RandomState(3)
    X = rng.randn(500, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[350:410] = 1 - y[350:410]
    return pd.DataFrame(X, columns=list("abcde")).assign(label=y).to_csv(index=False).encode()


@pytest.fixture(autouse=True)
def _fresh_limiter():
    marks.marks_rate_limiter._events.clear()


@pytest.fixture(scope="module")
def rid():
    r = client.post("/api/v1/analyze/upload", headers=_h("mk-alice"), files={"file": ("marks.csv", _csv(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_info"]["dataset_id"]


def _put(rid, item, verdict, rank=0, headers=None):
    return client.put(f"/api/v1/results/{rid}/marks", headers=headers or _h("mk-alice"), json={"item": item, "rank": rank, "verdict": verdict})


def _get(rid, headers=None):
    return client.get(f"/api/v1/results/{rid}/marks", headers=headers or _h("mk-alice"))


def test_upsert_overwrite_and_clear(rid):
    assert _put(rid, "line 5", "bad").status_code == 200
    assert _put(rid, "line 5", "fine").status_code == 200
    assert _put(rid, "line 9", "bad", rank=1).status_code == 200
    got = {m["item"]: m["verdict"] for m in _get(rid).json()["marks"]}
    assert got == {"line 5": "fine", "line 9": "bad"}
    assert _put(rid, "line 5", None).status_code == 200
    assert [m["item"] for m in _get(rid).json()["marks"]] == ["line 9"]
    assert _put(rid, "line 9", "maybe").status_code == 422
    assert _put(rid, "", "bad").status_code == 422
    assert _put(rid, "line 9", None).status_code == 200


def test_other_users_get_404(rid):
    assert _put(rid, "line 1", "bad", headers=_h("mk-bob")).status_code == 404
    assert _get(rid, headers=_h("mk-bob")).status_code == 404
    assert _put(rid, "line 1", "bad", headers=_h("mk-admin", "admin")).status_code == 404    # admins may not mark others' results
    assert _put("no-such-result", "line 1", "bad").status_code == 404


def test_export_is_admin_only_and_lists_marks(rid):
    _put(rid, "line 7", "bad", rank=2)
    assert client.get("/api/v1/marks/export", headers=_h("mk-alice")).status_code == 403
    r = client.get("/api/v1/marks/export", headers=_h("mk-admin", "admin"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    mine = [x for x in rows if x["result_id"] == rid and x["item"] == "line 7"]
    assert len(mine) == 1 and mine[0]["username"] == "mk-alice" and mine[0]["verdict"] == "bad"
    assert mine[0]["filename"] == "marks.csv" and mine[0]["rank"] == "2"
    _put(rid, "line 7", None)


def test_account_deletion_removes_marks():
    from app.models import database as db
    h = _h("mk-carol")
    r = client.post("/api/v1/analyze/upload", headers=h, files={"file": ("c.csv", _csv(), "text/csv")})
    crid = r.json()["dataset_info"]["dataset_id"]
    assert _put(crid, "line 3", "bad", headers=h).status_code == 200
    assert db.delete_owner_data("mk-carol")["item_marks"] == 1
    assert db.list_marks("mk-carol", crid) == []
