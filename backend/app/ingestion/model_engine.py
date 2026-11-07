"""Restricted loader for scikit-learn pickles, used only to read metadata (model type, size, hash).

It runs AFTER the static scanner (app/scanner/model_scan.py) has found no code execution; the scanner, not this
allow-list, decides whether a file is safe. Parameter-level "poisoning" analysis was removed: it had no benchmark.
"""
import io
import pickle
import hashlib
from typing import Dict, Any, Tuple


# ── Allowed model types (security: whitelist only) ────────────────────────────
ALLOWED_CLASSES = {
    "LogisticRegression", "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "SVC", "SVR", "LinearSVC", "LinearSVR",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "MLPClassifier", "MLPRegressor",
    "SGDClassifier", "SGDRegressor",
    "KNeighborsClassifier", "KNeighborsRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier", "AdaBoostRegressor",
    "BaggingClassifier", "BaggingRegressor",
    "Ridge", "Lasso", "ElasticNet",
    "GaussianNB", "BernoulliNB", "MultinomialNB",
}

MAX_MODEL_SIZE = 50 * 1024 * 1024  # 50 MB

# Pickles are executable programs, not a safe interchange format.  This allow-list
# permits the data containers needed by current scikit-learn estimators while
# refusing arbitrary globals (for example ``os.system``) *before* the object is
# constructed. Keep this small and extend it only with a regression fixture.
SAFE_GLOBALS = {
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"),
    ("collections", "OrderedDict"), ("copyreg", "_reconstructor"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer"),
    ("scipy.sparse._csr", "csr_matrix"), ("scipy.sparse._csc", "csc_matrix"),
    # Estimator internals required by the supported models (each is exercised by
    # tests/test_model_security.py::test_every_supported_model_type_loads_and_extracts).
    ("sklearn.tree._tree", "Tree"),
    ("sklearn.dummy", "DummyClassifier"), ("sklearn.dummy", "DummyRegressor"),
    ("sklearn.tree._classes", "ExtraTreeClassifier"), ("sklearn.tree._classes", "ExtraTreeRegressor"),
    ("sklearn.preprocessing._label", "LabelBinarizer"),
    ("sklearn._loss.loss", "HalfBinomialLoss"), ("sklearn._loss.loss", "HalfSquaredError"),
    ("sklearn._loss.link", "IdentityLink"), ("sklearn._loss.link", "LogitLink"),
    ("sklearn._loss.link", "Interval"),
    ("_loss", "CyHalfBinomialLoss"), ("_loss", "CyHalfSquaredError"),
    ("sklearn.linear_model._sgd_fast", "Hinge"),
    ("sklearn.metrics._dist_metrics", "newObj"),
    ("sklearn.metrics._dist_metrics", "EuclideanDistance64"), ("sklearn.metrics._dist_metrics", "EuclideanDistance32"),
    ("sklearn.neural_network._stochastic_optimizers", "AdamOptimizer"),
    ("sklearn.neural_network._stochastic_optimizers", "SGDOptimizer"),
    ("sklearn.neighbors._kd_tree", "KDTree"), ("sklearn.neighbors._kd_tree", "newObj"),
    ("sklearn.neighbors._ball_tree", "BallTree"), ("sklearn.neighbors._ball_tree", "newObj"),
    ("numpy.random._pickle", "__randomstate_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._mt19937", "MT19937"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"), ("numpy.random.bit_generator", "SeedSequence"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that rejects every global outside the reviewed allow-list.

    This reduces the legacy-pickle risk but is not a reason to accept arbitrary
    pickle formats indefinitely. Production deployments should prefer a
    non-executable model format or execute this legacy path in an isolated worker.
    """

    def find_class(self, module: str, name: str) -> Any:
        if name in ALLOWED_CLASSES and module.startswith("sklearn."):
            return super().find_class(module, name)
        if (module, name) in SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")


def restricted_loads(model_bytes: bytes) -> Any:
    """Deserialize legacy pickle bytes through the restrictive global resolver."""
    return RestrictedUnpickler(io.BytesIO(model_bytes)).load()


class ModelScanEngine:
    """Loads an allow-listed sklearn pickle and reports its metadata."""

    def load_and_validate(self, model_bytes: bytes, filename: str) -> Tuple[Any, Dict]:
        """
        Safely load a pickle, validate it's an allowed sklearn model.
        Returns (model_object, metadata_dict).
        Raises ValueError on any security or format issue.
        """
        if len(model_bytes) > MAX_MODEL_SIZE:
            raise ValueError(f"Model file too large (max {MAX_MODEL_SIZE // 1024 // 1024}MB)")

        # Compute hash for audit trail
        sha256 = hashlib.sha256(model_bytes).hexdigest()

        # Validate every class global before it can be instantiated. Never use
        # pickle.loads here: it resolves arbitrary globals during construction.
        try:
            model = restricted_loads(model_bytes)
        except Exception as e:
            raise ValueError(f"Cannot unpickle model: {e}")

        # Whitelist check
        class_name = type(model).__name__
        module_name = type(model).__module__

        if not module_name.startswith("sklearn"):
            raise ValueError(
                f"Only scikit-learn models are supported. "
                f"Got: {module_name}.{class_name}"
            )
        if class_name not in ALLOWED_CLASSES:
            raise ValueError(
                f"Model type '{class_name}' is not in the supported list. "
                f"Supported: {sorted(ALLOWED_CLASSES)}"
            )

        metadata = {
            "model_type": class_name,
            "module": module_name,
            "filename": filename,
            "file_size_kb": round(len(model_bytes) / 1024, 1),
            "sha256": sha256,
        }
        return model, metadata
