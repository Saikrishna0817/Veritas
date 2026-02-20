"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Trust Forensics Platform"
    version: str = "2.2.0"
    cors_allow_origins: str = "*"
    api_prefix: str = "/api/v1"
    ws_path: str = "/ws/v1/detection-stream"
    sqlite_path: str = os.getenv("FORENSICS_SQLITE_PATH", "")


settings = Settings()

