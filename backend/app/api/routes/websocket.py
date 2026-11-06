"""WebSocket routes + connection manager."""

from __future__ import annotations

import asyncio
import json
import time
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import decode_access_token, token_expiry

router = APIRouter()

AUTH_TIMEOUT_SECS = 5


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    def connect(self, websocket: WebSocket):
        """Register an already-accepted, authenticated socket."""
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: str, data: dict):
        message = json.dumps({"event": event, "data": data})
        dead = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)

    async def send_personal(self, websocket: WebSocket, event: str, data: dict):
        await websocket.send_json({"event": event, "data": data})


async def _authenticate(websocket: WebSocket) -> str | None:
    """Expect ``{"type": "auth", "token": ...}`` as the first frame; return the token.

    The token travels in a message, not the URL, so it never lands in proxy or
    access logs.
    """
    try:
        first = await asyncio.wait_for(websocket.receive_json(), AUTH_TIMEOUT_SECS)
        if first.get("type") != "auth":
            return None
        token = first["token"]
        decode_access_token(token)
        return token
    except Exception:
        return None


@router.websocket("/ws/v1/detection-stream")
async def detection_stream(websocket: WebSocket):
    """
    Real-time detection event stream.

    The ConnectionManager instance is stored on app.state.ws_manager by main.py.
    """
    await websocket.accept()
    token = await _authenticate(websocket)
    if token is None:
        await websocket.close(code=1008, reason="Authentication required")
        return

    expires_at = token_expiry(token)
    manager: ConnectionManager = websocket.app.state.ws_manager
    manager.connect(websocket)
    try:
        while True:
            remaining = expires_at - time.time()
            if remaining <= 0:
                await websocket.close(code=1008, reason="Token expired")
                break
            try:
                msg = json.loads(await asyncio.wait_for(websocket.receive_text(), remaining))
            except asyncio.TimeoutError:
                continue  # loop re-checks expiry and closes
            except ValueError:
                continue  # ignore malformed frames instead of dropping the socket
            if isinstance(msg, dict) and msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
