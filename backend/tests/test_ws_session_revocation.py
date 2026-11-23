"""An open detection stream ends when its session does (sign out everywhere, disable, delete)."""
import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.routes.websocket import ConnectionManager
from app.core.security import create_access_token, hash_password
from app.main import app
from app.models import database as db


def _user():
    name = "ws_" + uuid.uuid4().hex[:8]
    uid = db.create_user(name, hash_password("ws-test-password-1"), role="user")
    return uid, create_access_token({"id": uid, "name": name, "role": "user"})


@pytest.mark.parametrize("end", [
    lambda uid: db.end_all_sessions(uid),
    lambda uid: db.set_user_disabled(uid, True),
    lambda uid: db.delete_account(uid),
], ids=["logout-all", "disabled", "deleted"])
def test_open_socket_closes_when_the_session_ends(end):
    uid, token = _user()
    client = TestClient(app, base_url="https://testserver")
    with client.websocket_connect("/ws/v1/detection-stream") as ws:
        ws.send_json({"type": "auth", "token": token})
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}
        end(uid)
        ws.send_json({"type": "ping"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008
    assert not app.state.ws_manager.active_connections


class _Sock:
    def __init__(self):
        self.sent, self.closed = [], None

    async def send_text(self, m):
        self.sent.append(m)

    async def close(self, code=1000, reason=""):
        self.closed = code


def test_broadcast_skips_a_socket_whose_session_ended():
    uid, token = _user()
    mgr, sock = ConnectionManager(), _Sock()
    mgr.connect(sock, uid, token)
    asyncio.run(mgr.broadcast("e", {"n": 1}, uid))
    assert len(sock.sent) == 1
    db.end_all_sessions(uid)
    asyncio.run(mgr.broadcast("e", {"n": 2}, uid))
    assert len(sock.sent) == 1 and sock.closed == 1008 and not mgr.active_connections
