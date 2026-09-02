"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


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
    jwt_secret: str = os.getenv("VERITAS_JWT_SECRET", "")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = int(os.getenv("VERITAS_JWT_EXPIRY_MINUTES", "60"))
    admin_username: str = os.getenv("VERITAS_ADMIN_USERNAME", "")
    admin_password: str = os.getenv("VERITAS_ADMIN_PASSWORD", "")


settings = Settings()
