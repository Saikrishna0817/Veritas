"""Structured logging configuration for the AI Trust Forensics Platform.

Provides:
- JSON-structured log output (controlled by LOG_FORMAT env var)
- Per-request correlation via a contextvars request_id
- Module-level logger factory
"""

from __future__ import annotations

import logging
import logging.config
import os
from contextvars import ContextVar

# Request-scoped correlation ID — set by RequestIDMiddleware
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIDFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure application-wide logging.

    Respects:
      LOG_LEVEL  env var (default INFO)
      LOG_FORMAT env var — set to "json" for JSON output (default: text)
    """
    effective_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, effective_level.upper(), logging.INFO)

    use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"

    if use_json:
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","request_id":"%(request_id)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.addFilter(_RequestIDFilter())

    root = logging.getLogger()
    # Clear existing handlers to avoid duplicate output on reload
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (convenience wrapper)."""
    return logging.getLogger(name)
