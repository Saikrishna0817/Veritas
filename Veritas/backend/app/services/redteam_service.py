"""Red-team helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from app.defense.engine import RedTeamSimulator


def run_attack(sim: RedTeamSimulator, attack_type: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sim.run_simulation(attack_type, samples)

