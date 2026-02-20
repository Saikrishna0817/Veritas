"""Security helpers (minimal stubs for this build)."""

from __future__ import annotations


def is_safe_filename(name: str) -> bool:
    # Basic guard: avoid path traversal
    return not any(x in name for x in ("..", "\\", "/"))

