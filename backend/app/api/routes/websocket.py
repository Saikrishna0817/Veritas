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
        self._owners: dict[int, str] = {}

    def connect(self, websocket: WebSocket, owner_id: str):
        """Register an already-accepted, authenticated socket for one user."""
        self.active_connections.append(websocket)
        self._owners[id(websocket)] = owner_id

    def disconnect(self, websocket: WebSocket):
        self._owners.pop(id(websocket), None)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: str, data: dict, owner_id: str):
        """Send only to the sockets of ``owner_id``: events carry that user's own analysis results."""
        message = json.dumps({"event": event, "data": data})
        dead = []
        for connection in [c for c in list(self.active_connections) if self._owners.get(id(c)) == owner_id]:
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
    manager.connect(websocket, decode_access_token(token)["id"])
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
