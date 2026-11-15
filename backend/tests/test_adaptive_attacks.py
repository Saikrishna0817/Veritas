import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
from app.detection.veritas_mm import _residual  # noqa: E402
from research.adaptive.attacks import FAMILIES, make_context  # noqa: E402

GRID = {"lowfreq": 8, "warp": 1, "faint_patch": 0.4, "cl_lowfreq": 8, "boundary_flip": 1, "random_flip": 1}


def _setup():
    rng = np.random.default_rng(0)
    raw = rng.integers(20, 235, (60, 32, 32, 3), dtype=np.uint8)
    y = rng.integers(0, 10, 60)
    y[:12] = 3
    ctx = make_context(raw, 10, rng)
    ctx.update(source=1, target=3, target_score=rng.random(60))
    y[12:24] = 1
    return rng, raw, y, ctx


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_poison(name):
    rng, raw, y, ctx = _setup()
    before, y0 = raw.copy(), y.copy()
    idx = FAMILIES[name][0](raw, y, rng, 6, ctx, GRID[name])
    rest = np.setdiff1d(np.arange(60), idx)
    assert len(idx) == 6 and raw.dtype == np.uint8
    assert (raw[rest] == before[rest]).all() and (y[rest] == y0[rest]).all()
    if name in ("boundary_flip", "random_flip"):
        assert (raw == before).all() and (y[idx] == 3).all() and (y0[idx] == 1).all()
        return
    assert (raw[idx] != before[idx]).reshape(6, -1).any(axis=1).all()
    if name == "cl_lowfreq":
        assert (y == y0).all() and (y0[idx] == 3).all()
    else:
        assert (y[idx] == 3).all() and (y0[idx] != 3).all()


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_trigger_deterministic(name):
    _, raw, _, ctx = _setup()
    t = FAMILIES[name][1]
    a = t(raw[:5], ctx, GRID[name])
    assert a.dtype == np.uint8 and (a == t(raw[:5], ctx, GRID[name])).all()


def test_lowfreq_has_little_high_pass_energy():
    _, _, _, ctx = _setup()
    p = ctx["lowfreq"][None]
    assert (_residual(p) ** 2).sum() < 0.05 * (p ** 2).sum()
