"""In-process job queue for CPU-heavy analysis tasks."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

MAX_JOBS = 50  # oldest finished jobs are evicted beyond this; results can be large


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Track background jobs and their results in memory (process-local)."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-job")

    def submit(self, job_type: str, fn: Callable[[], Any], owner_id: str) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "job_type": job_type,
                "owner_id": owner_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            }
            self._evict()

        def _update(**fields: Any) -> None:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].update(fields)

        def _run() -> None:
            _update(status="running", started_at=_now())
            try:
                _update(status="completed", result=fn(), completed_at=_now())
            except Exception as exc:  # noqa: BLE001 — job boundary must capture failures
                _update(status="failed", error=str(exc), completed_at=_now())

        self._executor.submit(_run)
        return job_id

    def _evict(self) -> None:
        """Drop oldest finished jobs while over the cap (caller holds the lock)."""
        for job_id in [j for j, v in self._jobs.items() if v["status"] in ("completed", "failed")]:
            if len(self._jobs) <= MAX_JOBS:
                break
            del self._jobs[job_id]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


job_store = JobStore()
