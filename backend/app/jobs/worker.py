"""Worker process:  python -m app.jobs.worker   (needs VERITAS_REDIS_URL, the same database and the same storage as the API).
VERITAS_WORKER_QUEUES picks the queues (comma-separated, default "veritas"); the isolated media worker runs
"veritas-untrusted" (see app.jobs.queue.UNTRUSTED_JOBS and docker-compose.yml)."""
import logging
import os

import redis
from rq import Queue, Worker

from app.core.config import settings
from app.jobs import tasks  # noqa: F401  (registers the job bodies)
from app.jobs.queue import QUEUE_NAME
from app.models import database as db


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not settings.redis_url:
        raise SystemExit("VERITAS_REDIS_URL is not set")
    db.init_db()
    conn = redis.Redis.from_url(settings.redis_url)
    names = [q.strip() for q in os.getenv("VERITAS_WORKER_QUEUES", QUEUE_NAME).split(",") if q.strip()]
    Worker([Queue(n, connection=conn) for n in names], connection=conn).work()


if __name__ == "__main__":
    main()
