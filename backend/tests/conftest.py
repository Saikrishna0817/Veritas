import os
import tempfile

# Must be configured before importing the application modules.
os.environ.setdefault("VERITAS_JWT_SECRET", "test-only-secret-0123456789-abcdefghijklmnop")
os.environ.setdefault("VERITAS_ADMIN_USERNAME", "test-admin")
os.environ.setdefault("VERITAS_ADMIN_PASSWORD", "test-password")
os.environ.setdefault("CORS_ALLOW_ORIGINS", "http://localhost:5173")
# Isolated database so tests never touch (or depend on) a developer's real data.
os.environ.setdefault("FORENSICS_SQLITE_PATH", os.path.join(tempfile.mkdtemp(prefix="veritas-test-"), "test.db"))

# Optional: run the whole suite on PostgreSQL. VERITAS_TEST_PG_URL points at a server (any database); a fresh
# database is created for the session so nothing depends on, or leaks into, existing data.
_pg = os.environ.get("VERITAS_TEST_PG_URL")
if _pg:
    import uuid as _uuid

    import psycopg

    _name = "veritas_test_" + _uuid.uuid4().hex[:10]
    with psycopg.connect(_pg, autocommit=True) as _c:
        _c.execute(f"CREATE DATABASE {_name}")
    os.environ["VERITAS_DATABASE_URL"] = _pg.rsplit("/", 1)[0] + "/" + _name
