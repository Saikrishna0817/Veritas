"""Session tokens for tests. Tokens are only accepted for accounts that exist, so the account is created first."""
from app.core.security import create_access_token, hash_password
from app.models import database as db

_HASH = None


def token_for(user: dict) -> str:
    """A token for {"id", "name", "role"}: creates (or updates the role of) the account with that id, then signs."""
    global _HASH
    stored = db.get_user_by_id(user["id"])
    if stored is None:
        _HASH = _HASH or hash_password("unused-test-password")
        name = user["name"] if db.get_user_by_username(user["name"]) is None else f"{user['name']}-{user['id']}"
        conn = db.connection()
        conn.execute("INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                     (user["id"], name, _HASH, user["role"], "2026-01-01T00:00:00+00:00"))
        conn.commit()
    elif stored["role"] != user["role"]:
        db.set_user_role(user["id"], user["role"])
    return create_access_token(user)
