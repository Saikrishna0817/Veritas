"""Benchmark adapters around the shipped image detector (app.detection.veritas_mm), plus study-only variants.

The detector itself lives in the app so the benchmark scores the code users run.
"""
from __future__ import annotations

import numpy as np

from app.detection import veritas2 as v2
from app.detection.veritas_mm import (GATE, PENALTY, TAILS, components, full_reference as _full_ref,  # noqa: F401
                                      label_crossfit_component, label_evidence, pixel_component)
from app.detection.veritas_mm import report as _report
from research.mm import shared


def components_v6(d: dict, secs: dict | None = None) -> dict:
    """Study only (ref_study_dev.py): label (540 cal), label_cf, and every other check against all reference rows. ``secs`` collects seconds per check."""
    import time
    r = _full_ref(d)
    fs = {"label": lambda: v2.label_component(d), "label_cf": lambda: label_crossfit_component(d)[:2],
          "shift": lambda: v2.shift_component(r), "pixel": lambda: pixel_component(r),
          "dup": lambda: shared.duplicate_component(r), "cell": lambda: shared.shared_residual_cell_component(r),
          "resid": lambda: shared.shared_residual_component(r), "chroma": lambda: shared.chroma_floor_component(r)}
    out = {}
    for n, f in fs.items():
        t = time.perf_counter(); out[n] = f()
        if secs is not None: secs[n] = time.perf_counter() - t
    return out


_memo = [None, None]  # one slot: (batch, components), so several per-check detectors share one computation


def _components(d: dict) -> dict:
    if _memo[0] is not d: _memo[:] = [d, components(d)]  # holding d keeps its id from being reused
    return _memo[1]


def report(d: dict) -> dict:
    """veritas_mm.report, computing the components once per batch for the several per-check benchmark detectors."""
    return _report(d, _components(d))


def veritas_mm(d: dict):
    r = report(d)
    return np.max([c["rows"] for c in r.values() if c["admitted"]], axis=0), max(c["evidence"] for c in r.values())


def check_detector(name: str):
    """(rows, raw evidence) of one check."""
    return lambda d: (lambda c: (c["rows"], c["raw"]))(report(d)[name])


def mm_lead(d: dict):
    """(rows, penalised evidence) of the check with the highest penalised evidence."""
    c = max(report(d).values(), key=lambda c: c["evidence"])
    return c["rows"], c["evidence"]


def pixel_c2st(d: dict):
    """Baseline: the pixel check alone."""
    cal, inc = pixel_component(d)
    return inc, v2.exact_tail_test(cal, inc)
