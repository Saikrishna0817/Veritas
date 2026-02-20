"""
AI Trust Forensics Platform v2.2 — FastAPI Main Application
"""
import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.websocket import ConnectionManager

# Global state
manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize demo data on startup."""
    print("🛡️  AI Trust Forensics Platform v2.2 starting...")
    from app.demo.data_generator import get_demo_data
    data = get_demo_data()
    print(f"✅ Demo dataset ready: {data['total_samples']} samples, {data['poisoned_samples']} poisoned")
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI Trust Forensics Platform",
    description="Causally Verifiable Poisoning Detection & Auto-Defense for AI Systems",
    version="2.2.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1")

# Store manager on app state
app.state.ws_manager = manager


@app.websocket("/ws/v1/detection-stream")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time detection event stream."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


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
