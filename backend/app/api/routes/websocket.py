"""WebSocket routes + connection manager."""

from __future__ import annotations

import asyncio
import json
import time
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.security import SESSION_COOKIE, decode_access_token, token_expiry

router = APIRouter()

AUTH_TIMEOUT_SECS = 5
RECHECK_SECS = 30           # an open stream re-checks its session this often (and on every frame)


def _still_valid(token: str) -> bool:
    """False once the session is ended (sign out everywhere, password change), or the account is disabled or deleted."""
    try:
        decode_access_token(token)
        return True
    except Exception:
        return False


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._owners: dict[int, str] = {}
        self._tokens: dict[int, str] = {}

    def connect(self, websocket: WebSocket, owner_id: str, token: str | None = None):
        """Register an already-accepted, authenticated socket for one user."""
        self.active_connections.append(websocket)
        self._owners[id(websocket)] = owner_id
        if token:
            self._tokens[id(websocket)] = token

    def disconnect(self, websocket: WebSocket):
        self._owners.pop(id(websocket), None)
        self._tokens.pop(id(websocket), None)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: str, data: dict, owner_id: str):
        """Send only to the sockets of ``owner_id``: events carry that user's own analysis results."""
        message = json.dumps({"event": event, "data": data})
        dead = []
        for connection in [c for c in list(self.active_connections) if self._owners.get(id(c)) == owner_id]:
            token = self._tokens.get(id(connection))
            if token and not _still_valid(token):       # session ended since the socket opened: send nothing more
                dead.append(connection)
                try:
                    await connection.close(code=1008, reason="Session ended")
                except Exception:
                    pass
                continue
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)

    async def send_personal(self, websocket: WebSocket, event: str, data: dict):
        await websocket.send_json({"event": event, "data": data})


async def _authenticate(websocket: WebSocket) -> str | None:
    """Cookie session (browser) or an ``auth`` first frame (API clients). A cookie is an ambient credential, so the
    Origin header must be one of our own origins (blocks cross-site WebSocket hijacking)."""
    cookie = websocket.cookies.get(SESSION_COOKIE)
    if cookie:
        origin = websocket.headers.get("origin")
        if origin and origin not in settings.cors_allow_origins:
            return None
        try:
            decode_access_token(cookie)
            return cookie
        except Exception:
            return None
    return await _authenticate_frame(websocket)


async def _authenticate_frame(websocket: WebSocket) -> str | None:
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
    manager.connect(websocket, decode_access_token(token)["id"], token)
    try:
        while True:
            remaining = expires_at - time.time()
            if remaining <= 0:
                await websocket.close(code=1008, reason="Token expired")
                break
            if not _still_valid(token):
                await websocket.close(code=1008, reason="Session ended")
                break
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), min(remaining, RECHECK_SECS))
            except asyncio.TimeoutError:
                continue  # loop re-checks expiry and the session
            if not _still_valid(token):
                await websocket.close(code=1008, reason="Session ended")
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                continue  # ignore malformed frames instead of dropping the socket
            if isinstance(msg, dict) and msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
