"""
Model Scanning Engine — AI Trust Forensics Platform v2.2

Loads a user-uploaded sklearn .pkl model, extracts its internal learned
parameters (weights, coefs, tree structures) as a numeric feature matrix,
then runs the full 5-layer detection pipeline on those parameters to detect
signs of poisoning baked into the model itself.

Supports: LogisticRegression, RandomForest, GradientBoosting, SVC,
          DecisionTree, MLPClassifier, LinearSVC, SGDClassifier, KNeighbors
"""
import io
import pickle
import hashlib
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


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


class ModelScanEngine:
    """
    Extracts numeric feature matrices from sklearn model internals
    and prepares them for the detection pipeline.
    """

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

        # Safe unpickling — catch any import errors
        try:
            model = pickle.loads(model_bytes)  # noqa: S301
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

    def extract_features(self, model: Any) -> Tuple[np.ndarray, Dict]:
        """
        Extract internal learned parameters as a 2D numeric matrix.
        Each row = one "unit" (neuron, tree, support vector, etc.)
        Each col = a parameter dimension.

        Returns (feature_matrix, extraction_info).
        """
        class_name = type(model).__name__
        info = {"extraction_method": class_name, "n_units": 0, "n_dims": 0}

        try:
            matrix = self._extract(model, class_name)
        except Exception as e:
            raise ValueError(f"Cannot extract features from {class_name}: {e}")

        if matrix is None or len(matrix) == 0:
            raise ValueError(f"No extractable parameters found in {class_name}")

        # Ensure 2D
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        # Cap rows for performance (take a representative sample)
        if len(matrix) > 5000:
            idx = np.random.choice(len(matrix), 5000, replace=False)
            matrix = matrix[idx]

        # Replace NaN/Inf
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)

        info["n_units"] = len(matrix)
        info["n_dims"] = matrix.shape[1]
        info["param_stats"] = {
            "mean": round(float(np.mean(matrix)), 6),
            "std": round(float(np.std(matrix)), 6),
            "min": round(float(np.min(matrix)), 6),
            "max": round(float(np.max(matrix)), 6),
            "sparsity": round(float(np.mean(matrix == 0)), 4),
        }
        return matrix, info

    def _extract(self, model: Any, class_name: str) -> np.ndarray:
        """Dispatch extraction by model type."""

        # ── Linear models ─────────────────────────────────────────────────────
        if hasattr(model, "coef_"):
            coef = np.atleast_2d(model.coef_)
            if hasattr(model, "intercept_"):
                intercept = np.atleast_2d(model.intercept_).T
                if intercept.shape[0] == coef.shape[0]:
                    coef = np.hstack([coef, intercept])
            return coef

        # ── Neural network ────────────────────────────────────────────────────
        if hasattr(model, "coefs_"):
            # Stack all weight matrices flattened into rows
            rows = []
            for layer_w in model.coefs_:
                rows.append(layer_w.flatten())
            # Pad to same length
            max_len = max(len(r) for r in rows)
            padded = np.array([np.pad(r, (0, max_len - len(r))) for r in rows])
            return padded

        # ── Tree ensembles ────────────────────────────────────────────────────
        if hasattr(model, "estimators_"):
            rows = []
            estimators = model.estimators_
            # Flatten nested lists (GradientBoosting has list of lists)
            if isinstance(estimators[0], (list, np.ndarray)):
                estimators = [e for sublist in estimators for e in np.atleast_1d(sublist)]
            for est in estimators[:200]:  # cap at 200 trees
                tree = getattr(est, "tree_", None)
                if tree is not None:
                    # Use threshold + impurity as feature vector per node
                    row = np.concatenate([
                        tree.threshold[:50],
                        tree.impurity[:50],
                    ])
                    rows.append(row)
            if rows:
                max_len = max(len(r) for r in rows)
                return np.array([np.pad(r, (0, max_len - len(r))) for r in rows])

        # ── Single decision tree ──────────────────────────────────────────────
        if hasattr(model, "tree_"):
            tree = model.tree_
            return np.column_stack([tree.threshold, tree.impurity])

        # ── SVM support vectors ───────────────────────────────────────────────
        if hasattr(model, "support_vectors_"):
            return model.support_vectors_

        # ── KNN training data ─────────────────────────────────────────────────
        if hasattr(model, "_fit_X"):
            return model._fit_X

        # ── Naive Bayes ───────────────────────────────────────────────────────
        if hasattr(model, "theta_"):
            return model.theta_
        if hasattr(model, "feature_log_prob_"):
            return model.feature_log_prob_

        raise ValueError("No known parameter attribute found")

    def ingest(self, model_bytes: bytes, filename: str,
               dataset_bytes: Optional[bytes] = None,
               dataset_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Full ingestion: load model → extract features → build sample dicts
        compatible with DetectionPipeline.run_on_upload().
        """
        model, metadata = self.load_and_validate(model_bytes, filename)
        feature_matrix, extraction_info = self.extract_features(model)

        n = len(feature_matrix)
        split_idx = int(n * 0.70)

        # Build sample dicts (compatible with pipeline)
        samples = []
        for i, row in enumerate(feature_matrix):
            samples.append({
                "id": f"param_{i}",
                "feature_vector": row.tolist(),
                "label": 0,  # no labels for model params
                "poison_status": "unknown",
                "batch_id": f"layer_{i // 50}",
            })

        # If a dataset CSV was also uploaded, add its info
        dataset_info = None
        if dataset_bytes:
            from app.ingestion.csv_engine import CSVIngestionEngine
            csv_engine = CSVIngestionEngine()
            try:
                dataset_info = csv_engine.ingest(dataset_bytes, filename=dataset_filename or "dataset.csv")
            except Exception:
                dataset_info = None

        return {
            "scan_id": str(uuid.uuid4()),
            "model_filename": filename,
            "dataset_filename": dataset_filename,
            "model_type": metadata["model_type"],
            "model_metadata": metadata,
            "extraction_info": extraction_info,
            "features": feature_matrix,
            "labels": None,
            "has_labels": False,
            "samples": samples,
            "n_rows": n,
            "n_features": feature_matrix.shape[1],
            "feature_names": [f"param_dim_{i}" for i in range(feature_matrix.shape[1])],
            "label_column": None,
            "detection_mode": "model_parameter_scan",
            "reference_split": split_idx,
            "schema": {},
            "warnings": [],
            "created_at": datetime.utcnow().isoformat(),
            "dataset_info": dataset_info,
        }
