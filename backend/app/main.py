"""
AI Trust Forensics Platform v2.2 — FastAPI Main Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router, ws_router
from app.api.routes.websocket import ConnectionManager
from app.core.config import settings
from app.core.logging import configure_logging
import logging

# Global state
manager = ConnectionManager()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize demo data on startup."""
    configure_logging()
    logger.info("AI Trust Forensics Platform v2.2 starting")
    from app.demo.data_generator import get_demo_data
    data = get_demo_data()
    logger.info("Demo dataset ready: %s samples", data["total_samples"])
    yield
    logger.info("AI Trust Forensics Platform shutting down")


app = FastAPI(
    title="AI Trust Forensics Platform",
    description="Causally Verifiable Poisoning Detection & Auto-Defense for AI Systems",
    version="2.2.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router)

# Store manager on app state
app.state.ws_manager = manager

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.2.0", "platform": "AI Trust Forensics"}


@app.get("/")
async def root():
    return {
        "name": "AI Trust Forensics Platform",
        "version": "2.2.0",
        "docs": "/docs",
        "health": "/health"
    }
