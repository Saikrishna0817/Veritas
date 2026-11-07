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
    # postgresql://user:pass@host/db for production; empty means the SQLite file above.
    database_url: str = os.getenv("VERITAS_DATABASE_URL", "")
    # No defaults for secrets: unset means authentication fails closed and
    # validate_security_settings() stops the server at startup.
    jwt_secret: str = os.getenv("VERITAS_JWT_SECRET", "")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = int(os.getenv("VERITAS_JWT_EXPIRY_MINUTES", "60"))
    admin_username: str = os.getenv("VERITAS_ADMIN_USERNAME", "")
    admin_password: str = os.getenv("VERITAS_ADMIN_PASSWORD", "")
    # Shared state for rate limits and the job queue; empty keeps everything in-process (development, tests).
    redis_url: str = os.getenv("VERITAS_REDIS_URL", "")
    # Where uploads wait for the worker: file:///path (default, a local directory) or s3://bucket (S3, Cloudflare R2, MinIO).
    storage_url: str = os.getenv("VERITAS_STORAGE_URL", "")
    s3_endpoint: str = os.getenv("VERITAS_S3_ENDPOINT", "")
    # Public base URL used in emailed links, and outgoing mail (any SMTP provider: Resend, Brevo, ...). Without
    # smtp_host, messages are logged and kept in an in-memory outbox (development and tests only).
    public_url: str = os.getenv("VERITAS_PUBLIC_URL", "http://localhost:5173")
    smtp_host: str = os.getenv("VERITAS_SMTP_HOST", "")
    smtp_port: int = int(os.getenv("VERITAS_SMTP_PORT", "587"))
    smtp_user: str = os.getenv("VERITAS_SMTP_USER", "")
    smtp_password: str = os.getenv("VERITAS_SMTP_PASSWORD", "")
    mail_from: str = os.getenv("VERITAS_MAIL_FROM", "Veritas <no-reply@localhost>")
    # Self-serve sign-up. Verification can only be disabled for local development.
    signup_enabled: bool = os.getenv("VERITAS_SIGNUP_ENABLED", "true").lower() == "true"
    require_email_verification: bool = os.getenv("VERITAS_REQUIRE_EMAIL_VERIFICATION", "true").lower() == "true"
    turnstile_secret: str = os.getenv("VERITAS_TURNSTILE_SECRET", "")   # Cloudflare Turnstile; empty disables the check
    cookie_secure: bool = os.getenv("VERITAS_COOKIE_SECURE", "true").lower() == "true"   # false only for http://localhost
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
