"""The smooth within-class view removes the reference's strongest low-frequency directions and keeps the rest
(its detection numbers come from research/mm/weakness_study_dev.py on real dev images)."""
import numpy as np

from app.detection import veritas_mm as vmm


def test_smooth_view_removes_reference_directions():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, size=(300, 32, 32, 3)).astype(np.uint8)
    view = vmm.smooth_view(ref, k=20)
    low = ref.astype(np.float32).reshape(300, 8, 4, 8, 4, 3).mean(axis=(2, 4)).reshape(300, -1)
    V = np.linalg.svd(low - low.mean(0), full_matrices=False)[2]
    out = view(ref)
    assert out.shape == (300, 192)
    assert np.abs(out @ V[:20].T).max() < 1e-2                         # the 20 strongest directions are gone
    assert np.allclose(out @ V[20:].T, (low - low.mean(0)) @ V[20:].T, atol=1e-2)   # the rest is untouched


def test_component_scores_every_incoming_item():
    rng = np.random.default_rng(1)
    d = {"R_cal": rng.integers(0, 256, size=(400, 32, 32, 3)).astype(np.uint8), "y_cal": rng.integers(0, 4, 400),
         "R_inc": rng.integers(0, 256, size=(200, 32, 32, 3)).astype(np.uint8), "y_inc": rng.integers(0, 4, 200)}
    cal, inc = vmm.class_signature_component(d, view=vmm.smooth_view(d["R_cal"]))
    assert inc.shape == (200,) and np.isfinite(cal).all() and (inc >= 0).all()
