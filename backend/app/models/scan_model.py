"""Scan-related response models (optional typing layer)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ScanSummary(BaseModel):
    id: str
    verdict: Optional[str] = None
    score: Optional[float] = None
    attack_type: Optional[str] = None
    meta: Dict[str, Any] = {}

