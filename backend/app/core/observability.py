"""Error reporting (Sentry) and Prometheus metrics. Both are off unless configured."""
from __future__ import annotations

import hmac
import os
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("veritas_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("veritas_http_request_seconds", "HTTP request latency", ["method", "route"],
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
JOBS = Counter("veritas_jobs_total", "Background jobs finished", ["job_type", "status"])


def init_sentry() -> bool:
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        return False
    import sentry_sdk
    sentry_sdk.init(dsn=dsn, traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0")), send_default_pii=False,
                    environment=os.getenv("VERITAS_ENV", "production"))
    return True


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")       # template, never the raw URL (no ids, no cardinality blow-up)
        REQUESTS.labels(request.method, route, response.status_code).inc()
        LATENCY.labels(request.method, route).observe(time.perf_counter() - start)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request):
        token = os.getenv("VERITAS_METRICS_TOKEN", "")
        sent = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not token or not hmac.compare_digest(sent, token):     # disabled unless a scrape token is configured
            raise HTTPException(status_code=404)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
