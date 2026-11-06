from dataclasses import replace

import pytest

from app.core.config import settings, validate_security_settings

GOOD = replace(settings, jwt_secret="x" * 40, admin_username="ops", admin_password="a-strong-pass")


def test_valid_settings_pass():
    validate_security_settings(GOOD)


@pytest.mark.parametrize(
    "override",
    [
        {"jwt_secret": ""},
        {"jwt_secret": "short"},
        {"jwt_secret": "veritas-secret-key-change-me-in-production"},
        {"admin_username": ""},
        {"admin_password": ""},
        {"admin_password": "admin"},
        {"admin_password": "short"},
    ],
)
def test_insecure_settings_are_rejected(override):
    with pytest.raises(RuntimeError, match="Insecure configuration"):
        validate_security_settings(replace(GOOD, **override))


def test_legacy_default_user_is_purged_and_env_credentials_do_not_bypass_db():
    from app.core.security import authenticate_user, hash_password
    from app.models import database as db

    conn = db._get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO users (id, username, password_hash, role, created_at) VALUES (?,?,?,?,?)",
        ("user-regular-001", "user", hash_password("user123"), "user", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    db._ensure_bootstrap_admin(conn)
    assert authenticate_user("user", "user123") is None

    # Changing the stored admin password must invalidate the env password.
    conn.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password("rotated-password"), settings.admin_username))
    conn.commit()
    try:
        assert authenticate_user(settings.admin_username, settings.admin_password) is None
        assert authenticate_user(settings.admin_username, "rotated-password") is not None
    finally:
        conn.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(settings.admin_password), settings.admin_username))
        conn.commit()
