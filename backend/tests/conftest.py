import os

# Must be configured before importing the application modules.
os.environ.setdefault("VERITAS_JWT_SECRET", "test-only-secret")
os.environ.setdefault("VERITAS_ADMIN_USERNAME", "test-admin")
os.environ.setdefault("VERITAS_ADMIN_PASSWORD", "test-password")
os.environ.setdefault("CORS_ALLOW_ORIGINS", "http://localhost:5173")
