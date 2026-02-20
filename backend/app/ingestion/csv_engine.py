"""
CSV Ingestion Engine — Auto-detects schema, cleans data, prepares for detection pipeline.
Handles up to 50,000 rows. No external baseline required.
"""
import io
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid


MAX_ROWS = 50_000
MAX_FILE_SIZE_MB = 10


class CSVIngestionEngine:
    """
    Ingests a CSV file and prepares it for the detection pipeline.
    
    Key design: completely self-contained — no external baseline.
    The engine splits the data internally (70% reference / 30% incoming)
    to establish what "normal" looks like within the dataset itself.
    """

    def __init__(self):
        self.schema: Dict[str, Any] = {}
        self.label_column: Optional[str] = None
        self.feature_columns: List[str] = []
        self.has_labels: bool = False
        self.detection_mode: str = "supervised"  # or "unsupervised"

    # ── Schema Detection ──────────────────────────────────────────────────────

    def _detect_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect the label/target column using heuristics:
        1. Column named 'label', 'target', 'class', 'y', 'output', 'poisoned', etc.
        2. Binary or low-cardinality integer column
        3. Last column if it's binary
        """
        # Priority 1: name-based detection
        label_names = {
            'label', 'labels', 'target', 'targets', 'class', 'classes',
            'y', 'output', 'result', 'category', 'poisoned', 'is_poisoned',
            'attack', 'anomaly', 'fraud', 'diagnosis', 'prediction', 'ground_truth'
        }
        for col in df.columns:
            if col.lower().strip() in label_names:
                return col

        # Priority 2: binary integer column with low cardinality
        for col in df.columns:
            if df[col].dtype in [np.int64, np.int32, np.float64]:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, -1, True, False}):
                    return col

        # Priority 3: last column if low cardinality
        last_col = df.columns[-1]
        if df[last_col].nunique() <= 10 and df[last_col].dtype in [np.int64, np.int32, object]:
            return last_col

        return None

    def _detect_feature_columns(self, df: pd.DataFrame, label_col: Optional[str]) -> List[str]:
        """Select numeric feature columns, excluding label and ID-like columns."""
        exclude = set()
        if label_col:
            exclude.add(label_col)

        # Exclude ID-like columns
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['id', 'uuid', 'index', 'timestamp', 'date', 'time', 'name']):
                if df[col].nunique() > len(df) * 0.9:  # near-unique = ID
                    exclude.add(col)

        feature_cols = [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        return feature_cols

    def _encode_label_column(self, series: pd.Series) -> np.ndarray:
        """Encode label column to binary 0/1."""
        if series.dtype in [np.int64, np.int32, np.float64]:
            vals = series.values
            unique = np.unique(vals[~np.isnan(vals)])
            if len(unique) == 2:
                return (vals == unique[1]).astype(int)
            return (vals > vals.mean()).astype(int)
        else:
            # String labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            return le.fit_transform(series.fillna('unknown').astype(str))

    # ── Data Cleaning ─────────────────────────────────────────────────────────

    def _clean_dataframe(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Clean feature columns: fill NaN, clip outliers, normalize."""
        df = df.copy()

        for col in feature_cols:
            # Fill NaN with median
            median = df[col].median()
            df[col] = df[col].fillna(median)

            # Clip extreme outliers (beyond 5 IQR)
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[col] = df[col].clip(q1 - 5 * iqr, q3 + 5 * iqr)

        return df

    # ── Main Ingestion ─────────────────────────────────────────────────────────

    def ingest(self, csv_bytes: bytes, filename: str = "upload.csv") -> Dict[str, Any]:
        """
        Main ingestion method. Returns a structured dataset ready for detection.
        
        Returns:
            {
                "dataset_id": str,
                "samples": List[Dict],          # for detection pipeline
                "features": np.ndarray,          # raw feature matrix
                "labels": np.ndarray | None,     # label array or None
                "feature_names": List[str],
                "label_column": str | None,
                "has_labels": bool,
                "detection_mode": str,           # supervised | unsupervised
                "n_rows": int,
                "n_features": int,
                "schema": Dict,
                "warnings": List[str],
                "reference_split": int,          # index where reference ends
                "original_df": pd.DataFrame,
            }
        """
        warnings = []

        # ── Parse CSV ──────────────────────────────────────────────────────
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

        # Row limit
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            warnings.append(f"Dataset truncated to {MAX_ROWS} rows (limit).")

        if len(df) < 20:
            raise ValueError("Dataset too small (minimum 20 rows required).")

        # ── Schema Detection ───────────────────────────────────────────────
        label_col = self._detect_label_column(df)
        feature_cols = self._detect_feature_columns(df, label_col)

        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found in the CSV.")

        if len(feature_cols) > 100:
            # Keep top 100 by variance
            variances = df[feature_cols].var().sort_values(ascending=False)
            feature_cols = list(variances.head(100).index)
            warnings.append("More than 100 features detected. Keeping top 100 by variance.")

        # ── Clean Data ─────────────────────────────────────────────────────
        df_clean = self._clean_dataframe(df, feature_cols)

        # ── Prepare Labels ─────────────────────────────────────────────────
        has_labels = label_col is not None
        labels = None
        if has_labels:
            try:
                labels = self._encode_label_column(df_clean[label_col])
                n_classes = len(np.unique(labels))
                if n_classes > 20:
                    warnings.append(f"Label column has {n_classes} unique values — treating as regression target, running unsupervised mode.")
                    has_labels = False
                    labels = None
                    label_col = None
            except Exception as e:
                warnings.append(f"Could not encode label column: {e}. Running unsupervised.")
                has_labels = False
                labels = None
                label_col = None

        detection_mode = "supervised" if has_labels else "unsupervised"

        # ── Feature Matrix ─────────────────────────────────────────────────
        features = df_clean[feature_cols].values.astype(np.float64)

        # Normalize features to [0, 1] range per column
        col_min = features.min(axis=0)
        col_max = features.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1  # avoid division by zero
        features_norm = (features - col_min) / col_range

        # ── Self-contained Split ───────────────────────────────────────────
        # 70% reference (what "normal" looks like), 30% incoming (what we analyze)
        # This is purely internal — no external baseline needed
        n = len(features_norm)
        split_idx = int(n * 0.70)

        # ── Build Samples List ─────────────────────────────────────────────
        samples = []
        for i in range(n):
            sample = {
                "id": f"row_{i}",
                "row_index": i,
                "feature_vector": features_norm[i].tolist(),
                "label": int(labels[i]) if has_labels and labels is not None else -1,
                "poison_status": "unknown",  # we don't know — that's what we're detecting
                "ingested_at": datetime.utcnow().isoformat(),
                "batch_id": f"batch_{i // 100}",
                "source_id": filename,
                "client_id": "upload_client",
                "split": "reference" if i < split_idx else "incoming"
            }
            samples.append(sample)

        # ── Schema Summary ─────────────────────────────────────────────────
        schema = {
            "total_columns": len(df.columns),
            "feature_columns": feature_cols,
            "label_column": label_col,
            "n_rows": n,
            "n_features": len(feature_cols),
            "dtypes": {col: str(df[col].dtype) for col in feature_cols},
            "missing_pct": {
                col: round(df[col].isna().mean() * 100, 2)
                for col in feature_cols
            },
            "feature_stats": {
                col: {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                }
                for col in feature_cols[:10]  # first 10 for summary
            }
        }

        if has_labels and labels is not None:
            label_counts = np.bincount(labels)
            schema["label_distribution"] = {
                str(i): int(c) for i, c in enumerate(label_counts)
            }
            schema["class_balance"] = round(float(label_counts.min() / label_counts.max()), 4)

        self.schema = schema
        self.label_column = label_col
        self.feature_columns = feature_cols
        self.has_labels = has_labels
        self.detection_mode = detection_mode

        return {
            "dataset_id": str(uuid.uuid4()),
            "filename": filename,
            "samples": samples,
            "features": features_norm,
            "labels": labels,
            "feature_names": feature_cols,
            "label_column": label_col,
            "has_labels": has_labels,
            "detection_mode": detection_mode,
            "n_rows": n,
            "n_features": len(feature_cols),
            "schema": schema,
            "warnings": warnings,
            "reference_split": split_idx,
            "original_df": df_clean,
            "created_at": datetime.utcnow().isoformat()
        }
