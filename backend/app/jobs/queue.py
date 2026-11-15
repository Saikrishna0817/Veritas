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
# Jobs that parse untrusted binary files (images, audio, video through ffmpeg, model files) go to their own queue, so
# only the isolated worker-media container (no internet, read-only filesystem, capped resources) ever runs them.
UNTRUSTED_QUEUE = "veritas-untrusted"
UNTRUSTED_JOBS = {"image_upload", "audio_upload", "video_upload", "model_scan"}
JOB_TIMEOUT = 1800                       # seconds a job may run (the RQ timeout)
STALE_AFTER = {"running": JOB_TIMEOUT + 300, "queued": 6 * 3600}   # then its worker is presumed gone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# job_type -> function(payload, owner_id) -> JSON-serialisable result. Registered by app.jobs.tasks.
TASKS: dict[str, Callable[[dict, str], Any]] = {}


def _count(job_type: str, status: str) -> None:
    from app.core.observability import JOBS
    JOBS.labels(job_type, status).inc()


def execute(job_id: str) -> None:
    """Run one job to completion and record the outcome. Called by the worker or the in-process pool."""
    job = db.get_job(job_id)
    if job is None or job["status"] != "queued":
        return
    db.update_job(job_id, status="running", started_at=_now())
    try:
        result = TASKS[job["job_type"]](job["payload"], job["owner_id"])
        _count(job["job_type"], "completed")
        db.update_job(job_id, status="completed", result_json=json.dumps(result, default=str), completed_at=_now())
    except Exception as exc:  # noqa: BLE001 — job boundary must record failures
        logger.exception("job %s (%s) failed", job_id, job["job_type"])
        from app.core import quota
        quota.refund(job["owner_id"], job["payload"].get("charged"))   # a failed job does not use up the quota
        _count(job["job_type"], "failed")
        db.update_job(job_id, status="failed", error=str(exc)[:500], completed_at=_now())


class JobQueue:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-job")
        self._rq: dict = {}

    def _redis_queue(self, job_type: str):
        name = UNTRUSTED_QUEUE if job_type in UNTRUSTED_JOBS else QUEUE_NAME
        if name not in self._rq:
            import redis
            from rq import Queue
            self._rq[name] = Queue(name, connection=redis.Redis.from_url(settings.redis_url), default_timeout=JOB_TIMEOUT)
        return self._rq[name]

    def submit(self, job_type: str, payload: dict, owner_id: str) -> str:
        job_id = str(uuid.uuid4())
        db.create_job(job_id, job_type, owner_id, payload)
        if settings.redis_url:
            self._redis_queue(job_type).enqueue("app.jobs.queue.execute", job_id, job_timeout=JOB_TIMEOUT, result_ttl=0)
        else:
            self._executor.submit(execute, job_id)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = db.get_job(job_id)
        if job and job["status"] in STALE_AFTER:
            since = job.get("started_at") if job["status"] == "running" else job.get("created_at")
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(since)).total_seconds() if since else 0
            if age > STALE_AFTER[job["status"]]:   # a worker that crashed or a process that restarted never reports back
                from app.core import quota
                db.update_job(job_id, status="failed", error=f"no progress for {age / 60:.0f} minutes: the worker stopped", completed_at=_now())
                quota.refund(job["owner_id"], job["payload"].get("charged"))
                job = db.get_job(job_id)
        if job:
            job.pop("payload", None)       # storage keys are internal
        return job


job_store = JobQueue()
