import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torchvision")
from research.mm import ext_attacks  # noqa: E402

if not (ext_attacks._ROOT / "utils" / "bd_img_transform").is_dir():
    pytest.skip("BackdoorBench checkout missing (VERITAS_BACKDOORBENCH)", allow_module_level=True)


@pytest.mark.parametrize("name", sorted(ext_attacks.ATTACKS))
def test_attack(name):
    rng = np.random.default_rng(0)
    raw = rng.integers(20, 235, (40, 32, 32, 3), dtype=np.uint8)
    y = rng.integers(0, 10, 40)
    y[:10] = 3  # enough target-class images for the clean-label attack
    before, y0 = raw.copy(), y.copy()
    ctx = {"n_classes": 10, "source": 1, "target": 3}
    idx = ext_attacks.ATTACKS[name](raw, y, rng, 6, ctx)
    rest = np.setdiff1d(np.arange(40), idx)
    assert len(idx) == 6 and raw.dtype == np.uint8
    assert (raw[idx] != before[idx]).reshape(6, -1).any(axis=1).all()
    assert (raw[rest] == before[rest]).all() and (y[rest] == y0[rest]).all()
    if name == "bb_sig":
        assert (y == y0).all() and (y0[idx] == 3).all()
    else:
        assert (y[idx] == 3).all() and (y0[idx] != 3).all()
