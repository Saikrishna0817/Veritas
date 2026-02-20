"""Report models (optional typing layer)."""

from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel


class EvidenceReport(BaseModel):
    report_id: str
    title: str
    executive_summary: Dict[str, Any]
    evidence_bundle: Dict[str, Any]

