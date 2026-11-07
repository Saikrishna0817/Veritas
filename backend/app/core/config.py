"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend dir or project root; real environment variables win.
for parent in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]):
    load_dotenv(parent / ".env", override=False)


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
    # No defaults for secrets: unset means authentication fails closed and
    # validate_security_settings() stops the server at startup.
    jwt_secret: str = os.getenv("VERITAS_JWT_SECRET", "")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = int(os.getenv("VERITAS_JWT_EXPIRY_MINUTES", "60"))
    admin_username: str = os.getenv("VERITAS_ADMIN_USERNAME", "")
    admin_password: str = os.getenv("VERITAS_ADMIN_PASSWORD", "")
    # Stored results older than this many days are deleted at startup (0 keeps everything).
    retention_days: int = int(os.getenv("VERITAS_RETENTION_DAYS", "30"))
    # Per-user daily quotas (UTC day). Admins are exempt. 0 disables a limit.
    quota_analyses_per_day: int = int(os.getenv("VERITAS_QUOTA_ANALYSES_PER_DAY", "50"))
    quota_rows_per_day: int = int(os.getenv("VERITAS_QUOTA_ROWS_PER_DAY", "2000000"))
    quota_scan_mb_per_day: int = int(os.getenv("VERITAS_QUOTA_SCAN_MB_PER_DAY", "1000"))


settings = Settings()

MIN_JWT_SECRET_LEN = 32
MIN_ADMIN_PASSWORD_LEN = 8
_KNOWN_WEAK = {
    "admin", "password", "changeme", "change-me", "secret", "user123",
    "veritas-secret-key-change-me-in-production",
}


def validate_security_settings(s: Settings = settings) -> None:
    """Raise RuntimeError if secrets are missing, short, or well-known defaults."""
    problems: list[str] = []
    if len(s.jwt_secret) < MIN_JWT_SECRET_LEN or s.jwt_secret.lower() in _KNOWN_WEAK:
        problems.append(f"VERITAS_JWT_SECRET must be set to a random value of at least {MIN_JWT_SECRET_LEN} characters")
    if not s.admin_username:
        problems.append("VERITAS_ADMIN_USERNAME must be set")
    if len(s.admin_password) < MIN_ADMIN_PASSWORD_LEN or s.admin_password.lower() in _KNOWN_WEAK:
        problems.append(f"VERITAS_ADMIN_PASSWORD must be set to a non-default value of at least {MIN_ADMIN_PASSWORD_LEN} characters")
    if problems:
        raise RuntimeError("Insecure configuration: " + "; ".join(problems))
