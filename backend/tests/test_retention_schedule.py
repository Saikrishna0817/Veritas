"""S2 (launch gap plan): retention runs on a schedule, not only when the process starts.

Before the fix `purge_older_than` ran once at import, so a server that stayed up kept results past
VERITAS_RETENTION_DAYS, which the privacy policy promises.
"""
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.api import dependencies as deps
from app.core.config import settings
from app.main import app
from app.models import database as db


def _aged_result(days_old):
    rid = "ret-" + uuid.uuid4().hex[:8]
    db.save_result({"dataset_id": rid, "verdict": "CLEAN"}, "upload", owner_id="ret-owner")
    old = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    conn = db._get_conn()
    conn.execute("UPDATE analysis_results SET created_at = ? WHERE id = ?", (old, rid))
    conn.commit()
    return rid


def test_purge_runs_again_once_the_interval_has_passed_without_a_restart():
    assert settings.retention_days > 0
    now = datetime.now(timezone.utc)
    deps.purge_if_due(now)                                       # whatever ran before, a purge has happened now
    rid = _aged_result(settings.retention_days + 5)
    assert deps.purge_if_due(now + timedelta(minutes=5)) is False    # not due yet: nothing runs
    assert db.get_result(rid, db.ALL_OWNERS) is not None
    deps.upload_result_cache[deps.latest_key("ret-owner")] = {"dataset_id": rid, "owner_id": "ret-owner"}
    assert deps.purge_if_due(now + deps.RETENTION_EVERY + timedelta(seconds=1)) is True
    assert db.get_result(rid, db.ALL_OWNERS) is None
    assert deps.upload_result_cache.get(deps.latest_key("ret-owner")) is None     # not served from memory either


def test_recent_results_are_kept():
    rid = _aged_result(1)
    deps.purge_if_due(datetime.now(timezone.utc) + 10 * deps.RETENTION_EVERY)
    assert db.get_result(rid, db.ALL_OWNERS) is not None


def test_the_app_schedules_the_purge_while_it_runs():
    with TestClient(app):
        task = app.state.retention_task
        assert not task.done()
    assert task.done()                                           # stopped with the app
