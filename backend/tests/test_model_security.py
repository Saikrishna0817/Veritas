import pickle

import pytest
from sklearn.linear_model import LogisticRegression

from app.ingestion.model_engine import ModelScanEngine


class MaliciousPayload:
    def __reduce__(self):
        return (eval, ("__import__('os').environ.__setitem__('VERITAS_PICKLE_RCE', '1')",))


def test_malicious_pickle_is_rejected_without_execution(monkeypatch):
    monkeypatch.delenv("VERITAS_PICKLE_RCE", raising=False)
    with pytest.raises(ValueError, match="Cannot unpickle"):
        ModelScanEngine().load_and_validate(pickle.dumps(MaliciousPayload()), "malicious.pkl")
    assert "VERITAS_PICKLE_RCE" not in __import__("os").environ


def test_supported_sklearn_model_loads_through_restricted_unpickler():
    model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    loaded, metadata = ModelScanEngine().load_and_validate(pickle.dumps(model), "model.pkl")
    assert type(loaded).__name__ == "LogisticRegression"
    assert metadata["model_type"] == "LogisticRegression"
