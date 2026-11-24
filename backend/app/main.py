"""
AI Trust Forensics Platform v2.2 — FastAPI Main Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings, validate_security_settings

# Fail fast on missing/default secrets *before* importing routers: importing them
# initialises the database and seeds the admin account.
validate_security_settings()

from app.api.router import api_router, ws_router  # noqa: E402
from app.api.routes.websocket import ConnectionManager  # noqa: E402
from app.core import observability
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware, SecurityHeadersMiddleware

logger = get_logger(__name__)

# Global WebSocket manager
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise logging and shared state. Real-data primary — no silent demo preload."""
    configure_logging()
    logger.info("AI Trust Forensics Platform v2.2 starting (real-data primary mode)")
    from app.models import database as db
    try:
        stats = db.get_stats(db.ALL_OWNERS)
        logger.info("SQLite ready — total_analyses=%s by_source=%s",
                    stats.get("total_analyses", 0), stats.get("by_source", {}))
    except Exception as exc:
        logger.warning("SQLite stats unavailable at startup: %s", exc)
    yield
    logger.info("AI Trust Forensics Platform shutting down")


observability.init_sentry()

app = FastAPI(
    title="AI Trust Forensics Platform",
    description="Analyst-support platform for investigating risk signals consistent with adversarial data poisoning.",
    version="2.2.0",
    lifespan=lifespan,
)

# ── Middleware (outermost first — order matters) ──────────────────────────────

# 1. Request ID — must be first so all subsequent middleware/logs see the ID
app.add_middleware(RequestIDMiddleware)

# 2. Security headers — applied to every response
app.add_middleware(SecurityHeadersMiddleware)

# 3. CORS — restricted to configured origins, explicit methods and headers
#    allow_credentials=True is required for the Bearer-token WebSocket handshake.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-CSRF-Token"],
    expose_headers=["X-Request-ID", "X-Next-Before"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router)

observability.install(app)

from app.core import errors  # noqa: E402
errors.install(app)

# Store WebSocket manager on app state so routes can broadcast events
app.state.ws_manager = manager


@app.get("/health")
async def health():
    """Basic liveness probe. For a readiness probe use /ready."""
    from app.models import database as db
    try:
        stats = db.get_stats(db.ALL_OWNERS)
        db_ok = True
    except Exception:
        stats = {}
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "version": "2.2.0",
        "platform": "AI Trust Forensics",
        "db": "ok" if db_ok else "error",
        "total_analyses": stats.get("total_analyses", 0),
    }


@app.get("/ready")
async def ready():
    """Readiness probe: 200 only when the database answers (and Redis, when configured). 503 otherwise, so a load
    balancer stops sending traffic. Failure details go to the log, not the response."""
    from fastapi.responses import JSONResponse
    from app.models import database as db
    checks = {}
    try:
        db.connection().execute("SELECT 1").fetchone()
        checks["database"] = "ok"
    except Exception:
        logger.exception("Readiness: database check failed")
        checks["database"] = "error"
    if settings.redis_url:
        try:
            import redis
            redis.Redis.from_url(settings.redis_url, socket_timeout=1, socket_connect_timeout=1).ping()
            checks["redis"] = "ok"
        except Exception:
            logger.exception("Readiness: Redis check failed")
            checks["redis"] = "error"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@app.get("/")
async def root():
    return {
        "name": "AI Trust Forensics Platform",
        "version": "2.2.0",
        "docs": "/docs",
        "health": "/health",
    }
