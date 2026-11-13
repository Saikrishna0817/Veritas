"""Low-level pixel statistics, for attacks a semantic embedding cannot see.

A ResNet embedding is built to ignore corner patches and faint noise; these features are built to keep them:
per-cell brightness and high-frequency energy on a 4x4 grid (a stamped patch lights up one cell), a radial
power spectrum (blended and additive noise raise the high bands), per-channel means and the exact corner pixels.
"""
from app.detection.veritas_mm import GRID, RADIAL_BINS, pixel_features  # noqa: F401  (moved into the app)
