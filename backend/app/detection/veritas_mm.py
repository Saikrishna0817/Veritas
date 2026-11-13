"""Image detector (veritas_mm): veritas2's label check plus checks on other views of the images.

Frozen after dev-only tuning; measured in ``backend/research`` (docs/image-benchmark-findings.md, v8). The benchmark
imports this module, so it scores the code users run.

* label  - logistic regression on the semantic (ResNet-18) embedding, cross-fitted; always ranks rows.
* shift  - classifier two-sample test on the embedding (duplicates, large distribution changes).
* pixel  - classifier two-sample test on low-level pixel statistics (stamped patches, noise, blending).
* dup / cell / resid - what poisoned images share: near-copies, a common local patch, a common faint pattern.
* chroma - colour noise in the quietest block of an image (per-channel noise patterns, blends).

Every check except label compares the incoming images with all reference images; each becomes an exact tail-test
evidence over several tail sizes. A check joins the row ranking only when its own evidence reaches its ``GATE``;
its batch evidence is lowered by ``PENALTY`` (weighted Bonferroni).

Inputs (``d``): X_fit/X_cal/X_inc embeddings (PCA), y_*, n_classes, P_* pixel features (standardised on the fit
rows), R_* raw images (N, 32, 32, 3) uint8.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.detection import veritas2 as v2

GATE = {"shift": 3.0, "pixel": 1.0, "dup": 1.0, "cell": 1.0, "resid": 1.0, "chroma": 1.0}
PENALTY = {"label": 0.1, "shift": 1.0, "pixel": 1.0, "dup": 1.0, "cell": 1.0, "resid": 1.0, "chroma": 1.0}  # -log10 share of the false-alarm budget
TAILS = v2.TAILS
CHECKS = {"label": "label check (labels that do not fit the image)", "shift": "distribution shift of the image content",
          "pixel": "low-level pixel statistics (patches, noise, blending)", "dup": "near-duplicate images",
          "cell": "a patch shared by several images", "resid": "a faint pattern shared by several images",
          "chroma": "colour noise in flat regions"}

# ── Pixel features ──────────────────────────────────────────────────────────────────────────────────────────────────
GRID, RADIAL_BINS = 4, 8


def pixel_features(raw: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, D) float32: per-cell brightness and high-frequency energy on a 4x4 grid (a stamped
    patch lights up one cell), a radial power spectrum (blended and additive noise raise the high bands), per-channel
    means and spreads, and the exact corner pixels. A ResNet embedding is built to ignore exactly these."""
    x = raw.astype(np.float32) / 255.0
    gray = x.mean(axis=-1)
    n, h, w = gray.shape
    pad = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="edge")   # high-pass: pixel minus the mean of its 4 neighbours
    hf = np.abs(gray - 0.25 * (pad[:, :-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, :-2] + pad[:, 1:-1, 2:]))
    cells = lambda a: a[:, :h // GRID * GRID, :w // GRID * GRID].reshape(n, GRID, h // GRID, GRID, w // GRID)  # noqa: E731
    cell_mean = cells(gray).mean(axis=(2, 4)).reshape(n, -1)
    cell_hf = cells(hf).mean(axis=(2, 4)).reshape(n, -1)
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    yy, xx = np.indices((h, w))
    radius = np.hypot(yy - h / 2, xx - w / 2)
    which = np.digitize(radius, np.linspace(0, radius.max() + 1e-6, RADIAL_BINS + 1)) - 1
    spectrum = np.stack([np.log1p(power[:, which == b].mean(axis=1)) for b in range(RADIAL_BINS)], axis=1)
    channels = np.concatenate([x.mean(axis=(1, 2)), x.std(axis=(1, 2))], axis=1)
    k = max(3, h // 8)
    corners = np.concatenate([x[:, :k, :k].reshape(n, -1), x[:, -k:, -k:].reshape(n, -1)], axis=1)
    return np.concatenate([cell_mean, cell_hf, spectrum, channels, corners], axis=1).astype(np.float32)


# ── Shared-pattern checks: each image ranked against all OTHER images, reference and incoming pooled ────────────────
def _split(d, s):
    n = len(d["R_cal"])
    return s[:n], s[n:]


def _pool(d):
    return np.concatenate([d["R_cal"], d["R_inc"]]).astype(np.float32)


def _unit(a):
    a = a.reshape(len(a), -1)
    a = a - a.mean(axis=1, keepdims=True)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.where(n > 1e-6, a / np.maximum(n, 1e-6), 0).astype(np.float32)


def _kth_other(U, k):
    """k-th largest cosine similarity of each row to the other rows (U unit rows). Builds the full n x n matrix
    (n = 6,000 images: 144 MB); ponytail: compute it in row blocks if larger batches are allowed (last-bit rounding
    then differs from the benchmarked version)."""
    S = U @ U.T
    np.fill_diagonal(S, -1)
    return np.partition(S, -k, axis=1)[:, -k]


def _down8(x):
    n, h, w, c = x.shape
    H, W = -(-h // 8) * 8, -(-w // 8) * 8
    x = np.pad(x, ((0, 0), (0, H - h), (0, W - w), (0, 0)), mode="edge")
    return x.reshape(n, 8, H // 8, 8, W // 8, c).mean(axis=(2, 4))


def _residual(x):
    """pixel minus 3x3 local mean, per channel (edge padded)."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = x.shape[1:3]
    return x - sum(p[:, i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9


def duplicate_component(d):
    return _split(d, _kth_other(_unit(_down8(_pool(d))), 1))


def shared_residual_component(d, k=3):
    return _split(d, _kth_other(_unit(_residual(_pool(d))), k))


def shared_residual_cell_component(d, k=3, cell=8):
    """As above, correlation per 8x8 cell, max over cells, so a small corner patch is not drowned by the rest."""
    r = _residual(_pool(d))
    n, h, w, c = r.shape
    H, W = -(-h // cell) * cell, -(-w // cell) * cell
    r = np.pad(r, ((0, 0), (0, H - h), (0, W - w), (0, 0)))
    best = np.full(n, -1.0, dtype=np.float32)
    for i in range(0, H, cell):
        for j in range(0, W, cell):
            best = np.maximum(best, _kth_other(_unit(r[:, i:i + cell, j:j + cell]), k))
    return _split(d, best)


def chroma_floor_component(d, block=4):
    """Quietest 4x4 block of the high-pass residual of the colour differences R-G and G-B (independent per-channel noise lifts it)."""
    r = _residual(_pool(d))
    n, h, w, _ = r.shape
    H, W = h // block * block, w // block * block
    blocks = lambda a: (a[:, :H, :W] ** 2).reshape(n, H // block, block, W // block, block).mean(axis=(2, 4)).reshape(n, -1)  # noqa: E731
    return _split(d, ((blocks(r[..., 0] - r[..., 1]) + blocks(r[..., 1] - r[..., 2])) / 2).min(1))


# ── Model-based checks ──────────────────────────────────────────────────────────────────────────────────────────────
def pixel_component(d: dict):
    """(cal, inc) out-of-fold 'looks like the incoming batch' probabilities on pixel features."""
    Z = np.vstack([d["P_cal"], d["P_inc"]])
    t = np.r_[np.zeros(len(d["P_cal"])), np.ones(len(d["P_inc"]))]
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Z, t,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return p[: len(d["P_cal"])], p[len(d["P_cal"]):]


def full_reference(d: dict) -> dict:
    """d with all reference rows (fit + cal) as the clean comparison set."""
    return {**d, "X_cal": np.vstack([d["X_fit"], d["X_cal"]]), "y_cal": np.concatenate([d["y_fit"], d["y_cal"]]),
            "P_cal": np.vstack([d["P_fit"], d["P_cal"]]), "R_cal": np.concatenate([d["R_fit"], d["R_cal"]])}


def label_crossfit_component(d: dict):
    """(ref, inc, one) label suspicion over all reference rows: out-of-fold for reference rows; for incoming rows the
    mean of the fold models (row ranking) and one fold model per row (``one``: same footing as the out-of-fold
    reference rows, so it is used for the batch evidence). Logistic regression: 100x faster than GBM and ranks
    flipped labels better on dev."""
    X, y = np.vstack([d["X_fit"], d["X_cal"]]), np.concatenate([d["y_fit"], d["y_cal"]])
    cv = (StratifiedKFold if np.bincount(y)[np.unique(y)].min() >= 5 else KFold)(5, shuffle=True, random_state=0)
    ref, inc, one = np.zeros(len(y)), np.zeros(len(d["y_inc"])), np.zeros(len(d["y_inc"]))
    fold = np.arange(len(d["y_inc"])) % 5
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)).fit(X[tr], y[tr])
        score = lambda X_, y_: 1.0 - v2._probs(model, X_, d["n_classes"])[np.arange(len(y_)), y_]  # noqa: E731
        ref[te] = score(X[te], y[te])
        s = score(d["X_inc"], d["y_inc"])
        inc += s / 5
        one[fold == k] = s[fold == k]
    return ref, inc, one


def label_evidence(d: dict, ref, one) -> float:
    """Best of the pooled tail test and the best per-claimed-class one (Bonferroni over classes; a targeted flip puts
    all its rows in one claimed class), corrected for trying both."""
    y_ref, y_inc = np.concatenate([d["y_fit"], d["y_cal"]]), d["y_inc"]
    per_class = max([v2.multi_tail_test(ref[y_ref == c], one[y_inc == c], TAILS) for c in np.unique(y_inc)
                     if (y_ref == c).any()] or [0.0]) - np.log10(d["n_classes"])
    return max(max(v2.multi_tail_test(ref, one, TAILS), per_class) - np.log10(2), 0.0)


def components(d: dict) -> dict:
    """name -> (clean, inc[, one]) scores. Only the label check needs held-out rows (cross-fitted); every other check
    compares the incoming rows with all reference rows, so a handful of poisoned rows can still give strong evidence."""
    r = full_reference(d)
    return {"label": label_crossfit_component(d), "shift": v2.shift_component(r), "pixel": pixel_component(r),
            "dup": duplicate_component(r), "cell": shared_residual_cell_component(r),
            "resid": shared_residual_component(r), "chroma": chroma_floor_component(r)}


def report(d: dict, comps: dict | None = None) -> dict:
    """Per check: penalised ``evidence``, ``raw`` evidence, whether it is ``admitted`` to the ranking, its row scores."""
    out = {}
    for name, (cal, inc, *one) in (comps or components(d)).items():
        e = label_evidence(d, cal, one[0]) if name == "label" else v2.multi_tail_test(cal, inc, TAILS)
        out[name] = {"evidence": e - PENALTY[name], "raw": e, "admitted": name == "label" or e >= GATE[name],
                     "rows": v2.extrapolated_neg_log_p(cal, inc)}  # rows beyond every clean row keep their margin
    return out


def detect(d: dict) -> dict:
    """Row scores (max over admitted checks), the check that gave each row its score, batch evidence and the report."""
    r = report(d)
    admitted = [n for n, c in r.items() if c["admitted"]]
    rows = np.stack([r[n]["rows"] for n in admitted])
    return {"row_scores": rows.max(0), "row_check": [admitted[i] for i in rows.argmax(0)],
            "batch_evidence": max(max(c["evidence"] for c in r.values()), 0.0), "report": r}
