"""Durable background jobs. Status lives in the ``jobs`` table, so any API process can answer a poll and a restart
loses nothing. Execution is either in-process (a thread pool; development, tests, single box) or on a separate
worker through Redis (RQ) when VERITAS_REDIS_URL is set: run ``python -m app.jobs.worker`` next to the API."""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from app.core.config import settings
from app.models import database as db

logger = logging.getLogger(__name__)
QUEUE_NAME = "veritas"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# job_type -> function(payload, owner_id) -> JSON-serialisable result. Registered by app.jobs.tasks.
TASKS: dict[str, Callable[[dict, str], Any]] = {}


def execute(job_id: str) -> None:
    """Run one job to completion and record the outcome. Called by the worker or the in-process pool."""
    job = db.get_job(job_id)
    if job is None or job["status"] != "queued":
        return
    db.update_job(job_id, status="running", started_at=_now())
    try:
        result = TASKS[job["job_type"]](job["payload"], job["owner_id"])
        db.update_job(job_id, status="completed", result_json=json.dumps(result, default=str), completed_at=_now())
    except Exception as exc:  # noqa: BLE001 — job boundary must record failures
        logger.exception("job %s (%s) failed", job_id, job["job_type"])
        db.update_job(job_id, status="failed", error=str(exc)[:500], completed_at=_now())


class JobQueue:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-job")
        self._rq = None

    def _redis_queue(self):
        if self._rq is None:
            import redis
            from rq import Queue
            self._rq = Queue(QUEUE_NAME, connection=redis.Redis.from_url(settings.redis_url), default_timeout=1800)
        return self._rq

    def submit(self, job_type: str, payload: dict, owner_id: str) -> str:
        job_id = str(uuid.uuid4())
        db.create_job(job_id, job_type, owner_id, payload)
        if settings.redis_url:
            self._redis_queue().enqueue("app.jobs.queue.execute", job_id, job_timeout=1800, result_ttl=0)
        else:
            self._executor.submit(execute, job_id)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = db.get_job(job_id)
        if job:
            job.pop("payload", None)       # storage keys are internal
        return job


job_store = JobQueue()
