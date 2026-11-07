"""Worker process:  python -m app.jobs.worker   (needs VERITAS_REDIS_URL, the same database and the same storage as the API)."""
import logging

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
    Worker([Queue(QUEUE_NAME, connection=conn)], connection=conn).work()


if __name__ == "__main__":
    main()
