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


def _supported_models():
    import numpy as np
    from sklearn import ensemble as en, linear_model as lm, naive_bayes as nbay
    from sklearn import neighbors as nb, neural_network as nn, svm, tree

    rng = np.random.RandomState(0)
    X = rng.rand(120, 4)
    y_cls = (X[:, 0] > 0.5).astype(int)
    y_reg = X[:, 0]
    cases = [
        (lm.LogisticRegression(), y_cls), (lm.Ridge(), y_reg), (lm.Lasso(), y_reg),
        (lm.ElasticNet(), y_reg), (lm.SGDClassifier(), y_cls), (lm.SGDRegressor(), y_reg),
        (svm.SVC(), y_cls), (svm.SVR(), y_reg), (svm.LinearSVC(), y_cls), (svm.LinearSVR(), y_reg),
        (tree.DecisionTreeClassifier(), y_cls), (tree.DecisionTreeRegressor(), y_reg),
        (en.RandomForestClassifier(n_estimators=3), y_cls), (en.RandomForestRegressor(n_estimators=3), y_reg),
        (en.ExtraTreesClassifier(n_estimators=3), y_cls), (en.ExtraTreesRegressor(n_estimators=3), y_reg),
        (en.GradientBoostingClassifier(n_estimators=3), y_cls), (en.GradientBoostingRegressor(n_estimators=3), y_reg),
        (en.AdaBoostClassifier(n_estimators=3), y_cls), (en.AdaBoostRegressor(n_estimators=3), y_reg),
        (en.BaggingClassifier(n_estimators=3), y_cls), (en.BaggingRegressor(n_estimators=3), y_reg),
        (nn.MLPClassifier(max_iter=30), y_cls), (nn.MLPRegressor(max_iter=30), y_reg),
        (nb.KNeighborsClassifier(), y_cls), (nb.KNeighborsClassifier(algorithm="ball_tree"), y_cls),
        (nb.KNeighborsClassifier(algorithm="kd_tree"), y_cls), (nb.KNeighborsRegressor(), y_reg),
        (nbay.GaussianNB(), y_cls), (nbay.BernoulliNB(), y_cls), (nbay.MultinomialNB(), y_cls),
    ]
    return [(m.fit(X, y), m.__class__.__name__) for m, y in cases]


@pytest.mark.parametrize("model,name", _supported_models(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_supported_model_type_loads_and_extracts(model, name):
    engine = ModelScanEngine()
    loaded, meta = engine.load_and_validate(pickle.dumps(model), "m.pkl")
    assert meta["model_type"] == name
    matrix, _ = engine.extract_features(loaded)
    assert matrix.ndim == 2 and len(matrix) > 0


def test_supported_model_set_is_fully_covered():
    from app.ingestion.model_engine import ALLOWED_CLASSES

    assert {name for _, name in _supported_models()} == set(ALLOWED_CLASSES)


def test_model_scan_is_deterministic():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(1)
    X = rng.rand(30, 6000)  # > 5000 parameter rows would trigger sampling in extract_features
    model = LogisticRegression(max_iter=5).fit(X.T[:100], (np.arange(100) % 2))
    engine = ModelScanEngine()
    assert (engine.extract_features(model)[0] == engine.extract_features(model)[0]).all()
