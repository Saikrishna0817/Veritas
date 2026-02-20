"""Quick smoke test for the three new features."""
import sys
sys.path.insert(0, '.')

from app.db.store import init_db, get_stats
from app.demo.real_datasets import get_real_dataset, DATASET_CATALOG
from app.ingestion.model_engine import ModelScanEngine

# Test DB
init_db()
print("DB init OK:", get_stats())

# Test real datasets
for ds in DATASET_CATALOG:
    d = get_real_dataset(ds['id'])
    print(f"Dataset {ds['id']}: {d['n_rows']} rows, {d['size_kb']}KB")

# Test model engine
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.random.randn(100, 4)
y = (X[:, 0] > 0).astype(int)
model = LogisticRegression().fit(X, y)
model_bytes = pickle.dumps(model)

engine = ModelScanEngine()
ingested = engine.ingest(model_bytes, 'test_model.pkl')
print(f"Model scan: {ingested['n_rows']} param units, {ingested['n_features']} dims")
print(f"Model type: {ingested['model_type']}")
print(f"Param stats: {ingested['extraction_info']['param_stats']}")
print("All checks passed!")
