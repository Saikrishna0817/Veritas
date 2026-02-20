"""Quick smoke test for DB + real datasets + model scan engine."""

import sys

sys.path.insert(0, ".")

from app.models.database import get_stats, init_db  # noqa: E402
from app.demo.real_datasets import DATASET_CATALOG, get_real_dataset  # noqa: E402
from app.ingestion.model_engine import ModelScanEngine  # noqa: E402

init_db()
print("DB init OK:", get_stats())

for ds in DATASET_CATALOG:
    d = get_real_dataset(ds["id"])
    print(f"Dataset {ds['id']}: {d['n_rows']} rows, {d['size_kb']}KB")

import pickle  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)
model = LogisticRegression().fit(X, y)
model_bytes = pickle.dumps(model)

engine = ModelScanEngine()
ingested = engine.ingest(model_bytes, "test_model.pkl")
print(f"Model scan: {ingested['n_rows']} param units, {ingested['n_features']} dims")
print(f"Model type: {ingested['model_type']}")
print(f"Param stats: {ingested['extraction_info']['param_stats']}")
print("All checks passed!")

