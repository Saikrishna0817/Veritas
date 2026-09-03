"""In-process job queue for CPU-heavy analysis tasks."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable


class JobStore:
    """Track background jobs and their results in memory."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-job")

    def submit(self, job_type: str, fn: Callable[[], Any]) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat() + "Z"
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "job_type": job_type,
                "status": "queued",
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
            }

        def _run() -> None:
            with self._lock:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat() + "Z"
            try:
                result = fn()
                with self._lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["result"] = result
                    self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat() + "Z"
            except Exception as exc:  # noqa: BLE001 — job boundary must capture failures
                with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(exc)
                    self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


job_store = JobStore()
