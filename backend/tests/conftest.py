import os
import tempfile

# Must be configured before importing the application modules.
os.environ.setdefault("VERITAS_JWT_SECRET", "test-only-secret-0123456789-abcdefghijklmnop")
os.environ.setdefault("VERITAS_ADMIN_USERNAME", "test-admin")
os.environ.setdefault("VERITAS_ADMIN_PASSWORD", "test-password")
os.environ.setdefault("CORS_ALLOW_ORIGINS", "http://localhost:5173")
# Isolated database so tests never touch (or depend on) a developer's real data.
os.environ.setdefault("FORENSICS_SQLITE_PATH", os.path.join(tempfile.mkdtemp(prefix="veritas-test-"), "test.db"))
