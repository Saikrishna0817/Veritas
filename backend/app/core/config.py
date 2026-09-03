"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# Load .env file if present in project root or backend dir
for parent in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
    env_file = parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Trust Forensics Platform"
    version: str = "2.2.0"
    cors_allow_origins: tuple[str, ...] = _csv_env("CORS_ALLOW_ORIGINS", "http://localhost:5173")
    api_prefix: str = "/api/v1"
    ws_path: str = "/ws/v1/detection-stream"
    sqlite_path: str = os.getenv("FORENSICS_SQLITE_PATH", "")
    jwt_secret: str = os.getenv("VERITAS_JWT_SECRET", "veritas-secret-key-change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = int(os.getenv("VERITAS_JWT_EXPIRY_MINUTES", "60"))
    admin_username: str = os.getenv("VERITAS_ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("VERITAS_ADMIN_PASSWORD", "admin")


settings = Settings()
