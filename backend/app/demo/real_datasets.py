"""
Real Dataset Library — AI Trust Forensics Platform v2.2

Ships 4 real public datasets (from sklearn.datasets) with controlled
poison injection so the platform can demonstrate detection on real data.

Datasets:
  - iris_poisoned       : UCI Iris (150 samples, 4 features, 3 classes)
  - wine_poisoned       : UCI Wine (178 samples, 13 features, 3 classes)
  - breast_cancer       : Wisconsin Breast Cancer (569 samples, 30 features)
  - digits_subset       : MNIST-like Digits (500 samples, 64 features)

Each is returned as a ready-to-download CSV with poison injected.
"""
import io
import numpy as np
import pandas as pd
from typing import Dict, Any


# Cache so we don't regenerate on every request
_cache: Dict[str, bytes] = {}


def _inject_label_flip(df: pd.DataFrame, label_col: str, rate: float = 0.12,
                        seed: int = 42) -> pd.DataFrame:
    """Flip `rate` fraction of labels to a random wrong class."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    classes = df[label_col].unique()
    n_flip = max(1, int(len(df) * rate))
    idx = rng.choice(len(df), n_flip, replace=False)
    for i in idx:
        current = df.iloc[i][label_col]
        wrong = rng.choice([c for c in classes if c != current])
        df.at[df.index[i], label_col] = wrong
    return df


def _inject_feature_noise(df: pd.DataFrame, label_col: str, rate: float = 0.08,
                           seed: int = 99) -> pd.DataFrame:
    """Add extreme outlier values to a fraction of feature rows (clean-label style)."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    feat_cols = [c for c in df.columns if c != label_col]
    n_poison = max(1, int(len(df) * rate))
    idx = rng.choice(len(df), n_poison, replace=False)
    for i in idx:
        col = rng.choice(feat_cols)
        df.at[df.index[i], col] = float(df[col].mean() + rng.uniform(8, 15) * df[col].std())
    return df


def _make_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def get_real_dataset(name: str) -> Dict[str, Any]:
    """
    Return a real dataset as CSV bytes with metadata.
    name: one of 'iris', 'wine', 'breast_cancer', 'digits'
    """
    if name in _cache:
        data = _cache[name]
        return _build_response(name, data)

    from sklearn import datasets

    if name == "iris":
        raw = datasets.load_iris(as_frame=True)
        df = raw.frame.rename(columns={"target": "label"})
        df = _inject_label_flip(df, "label", rate=0.13)
        df = _inject_feature_noise(df, "label", rate=0.07)
        desc = "UCI Iris — 150 samples, 4 features (sepal/petal length & width), 3 flower species"
        poison_note = "13% label flips + 7% feature outliers injected"

    elif name == "wine":
        raw = datasets.load_wine(as_frame=True)
        df = raw.frame.rename(columns={"target": "label"})
        df = _inject_label_flip(df, "label", rate=0.15)
        df = _inject_feature_noise(df, "label", rate=0.10)
        desc = "UCI Wine — 178 samples, 13 chemical features, 3 wine cultivars"
        poison_note = "15% label flips + 10% feature outliers injected"

    elif name == "breast_cancer":
        raw = datasets.load_breast_cancer(as_frame=True)
        df = raw.frame.rename(columns={"target": "label"})
        df = _inject_label_flip(df, "label", rate=0.10)
        df = _inject_feature_noise(df, "label", rate=0.06)
        desc = "Wisconsin Breast Cancer — 569 samples, 30 features, binary (malignant/benign)"
        poison_note = "10% label flips + 6% feature outliers injected"

    elif name == "digits":
        raw = datasets.load_digits(as_frame=True)
        # Use first 500 samples for speed
        df = raw.frame.iloc[:500].rename(columns={"target": "label"})
        df = _inject_label_flip(df, "label", rate=0.12)
        desc = "MNIST-like Digits — 500 samples, 64 pixel features, 10 digit classes"
        poison_note = "12% label flips injected"

    else:
        raise ValueError(f"Unknown dataset: '{name}'. Choose from: iris, wine, breast_cancer, digits")

    csv_bytes = _make_csv(df)
    _cache[name] = csv_bytes
    return _build_response(name, csv_bytes, desc, poison_note, len(df))


def _build_response(name: str, csv_bytes: bytes,
                    description: str = "", poison_note: str = "",
                    n_rows: int = 0) -> Dict[str, Any]:
    return {
        "name": name,
        "filename": f"{name}_poisoned.csv",
        "csv_bytes": csv_bytes,
        "description": description,
        "poison_note": poison_note,
        "n_rows": n_rows,
        "size_kb": round(len(csv_bytes) / 1024, 1),
    }


DATASET_CATALOG = [
    {
        "id": "iris",
        "name": "UCI Iris (Poisoned)",
        "description": "150 samples · 4 features · 3 classes · 13% label flips + 7% outliers",
        "domain": "Biology",
        "n_rows": 150,
        "n_features": 4,
        "attack_types": ["label_flip", "clean_label"],
    },
    {
        "id": "wine",
        "name": "UCI Wine (Poisoned)",
        "description": "178 samples · 13 features · 3 classes · 15% label flips + 10% outliers",
        "domain": "Chemistry",
        "n_rows": 178,
        "n_features": 13,
        "attack_types": ["label_flip", "clean_label"],
    },
    {
        "id": "breast_cancer",
        "name": "Breast Cancer Wisconsin (Poisoned)",
        "description": "569 samples · 30 features · binary · 10% label flips + 6% outliers",
        "domain": "Healthcare",
        "n_rows": 569,
        "n_features": 30,
        "attack_types": ["label_flip", "clean_label"],
    },
    {
        "id": "digits",
        "name": "MNIST Digits Subset (Poisoned)",
        "description": "500 samples · 64 pixel features · 10 classes · 12% label flips",
        "domain": "Computer Vision",
        "n_rows": 500,
        "n_features": 64,
        "attack_types": ["label_flip"],
    },
]
