"""User model (placeholder for future auth)."""

from __future__ import annotations

from pydantic import BaseModel


class User(BaseModel):
    id: str
    name: str
    role: str = "analyst"

