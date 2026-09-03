"""
AI Trust Forensics Platform v2.2 — FastAPI Main Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router, ws_router
from app.api.routes.websocket import ConnectionManager
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware, SecurityHeadersMiddleware

logger = get_logger(__name__)

# Global WebSocket manager
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise logging, demo data, and shared state on startup."""
    configure_logging()
    logger.info("AI Trust Forensics Platform v2.2 starting")
    from app.demo.data_generator import get_demo_data
    data = get_demo_data()
    logger.info("Demo dataset ready: %s samples", data["total_samples"])
    yield
    logger.info("AI Trust Forensics Platform shutting down")


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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router)

# Store WebSocket manager on app state so routes can broadcast events
app.state.ws_manager = manager


@app.get("/health")
async def health():
    """Basic liveness probe.  For a readiness probe use /health/ready."""
    from app.models import database as db
    try:
        stats = db.get_stats()
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


@app.get("/")
async def root():
    return {
        "name": "AI Trust Forensics Platform",
        "version": "2.2.0",
        "docs": "/docs",
        "health": "/health",
    }
