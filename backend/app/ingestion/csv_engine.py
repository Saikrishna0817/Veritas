"""
CSV Ingestion Engine — Auto-detects schema, cleans data, prepares for detection pipeline.
Handles up to 200,000 rows. No external baseline required.
"""
import io
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import uuid


MAX_ROWS = 200_000

# Columns that hold the answer key (or a model's output), not training data. They are
# never used as features or as the training label; the schema reports them instead.
GROUND_TRUTH_NAMES = {
    'poisoned', 'is_poisoned', 'is_poison', 'poison', 'poison_status',
    'attack', 'is_attack', 'ground_truth', 'prediction', 'predicted',
}
# Name tokens that mark a row identifier / timestamp (matched as whole tokens).
ID_TOKENS = {'id', 'uuid', 'guid', 'index', 'idx', 'timestamp', 'date', 'time', 'name'}


def _name_tokens(col: str) -> set:
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(col))  # userId -> user_Id
    return {t for t in re.split(r'[^a-z0-9]+', spaced.lower()) if t}


_REF_COL = "__veritas_is_reference"
_ROW_COL = "__veritas_source_row"      # 0-based data row in the file it came from; travels through cleaning
MIN_REFERENCE_ROWS, MIN_INCOMING_ROWS = 30, 10
MAX_CATEGORIES = 50     # text columns with up to this many distinct values become category codes (as in the benchmark)
MAX_CLASSES = 100       # more label values than this is treated as a regression target


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
        self.categories: Dict[str, List[str]] = {}   # text column -> its values, in code order

    # ── Schema Detection ──────────────────────────────────────────────────────

    def _detect_label_column(self, df: pd.DataFrame, exclude: set = frozenset()) -> Optional[str]:
        """
        Auto-detect the label/target column using heuristics:
        1. Column named 'label', 'target', 'class', 'y', 'output', etc.
        2. Binary or low-cardinality integer column
        3. Last column if it's binary
        """
        # Priority 1: name-based detection
        label_names = {
            'label', 'labels', 'target', 'targets', 'class', 'classes',
            'y', 'output', 'result', 'category', 'anomaly', 'fraud', 'diagnosis',
        }
        candidates = [c for c in df.columns if c not in exclude]
        for col in candidates:
            if col.lower().strip() in label_names:
                return col

        # Priority 2: binary integer column with low cardinality
        for col in candidates:
            if df[col].dtype in [np.int64, np.int32, np.float64]:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, -1, True, False}):
                    return col

        # Priority 3: last column if low cardinality
        last_col = candidates[-1] if candidates else None
        if last_col is not None and df[last_col].nunique() <= 10 and df[last_col].dtype in [np.int64, np.int32, object]:
            return last_col

        return None

    def _detect_feature_columns(self, df: pd.DataFrame, label_col: Optional[str],
                                exclude: set = frozenset()) -> List[str]:
        """Select numeric feature columns, excluding label and ID-like columns."""
        exclude = set(exclude)
        if label_col:
            exclude.add(label_col)

        # Exclude ID-like columns: a whole name token says id/time/etc. AND the column is
        # near-unique AND (integer-valued or monotonic). Continuous float measurements that
        # happen to be named e.g. 'time' or 'valid_score' are real features and are kept.
        for col in df.columns:
            if _name_tokens(col) & ID_TOKENS and df[col].nunique() > len(df) * 0.9:
                if not pd.api.types.is_float_dtype(df[col]) or df[col].is_monotonic_increasing:
                    exclude.add(col)

        feature_cols = [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        return feature_cols

    def _encode_text_columns(self, df: pd.DataFrame, exclude: set, warnings: List[str]) -> pd.DataFrame:
        """Low-cardinality text columns become integer category codes (missing = its own code) so poisoned
        categorical values are checked too; free-text and ID-like text columns are dropped, with a warning."""
        df, dropped = df.copy(), []
        for col in df.columns:
            if col in exclude or pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                continue
            if df[col].nunique() <= MAX_CATEGORIES:
                codes, uniques = pd.factorize(df[col], sort=True)
                df[col] = codes.astype(float)                                 # NaN -> -1, a category of its own
                self.categories[str(col)] = [str(u) for u in uniques]
            else:
                dropped.append(str(col))
        if dropped:
            warnings.append(f"Ignored {len(dropped)} text column(s) with more than {MAX_CATEGORIES} distinct values: "
                            + ", ".join(dropped[:5]) + (f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""))
        return df

    def _encode_label_column(self, series: pd.Series) -> np.ndarray:
        """Encode labels as integer classes 0..k-1 (missing values become their own class)."""
        codes, uniques = pd.factorize(series, sort=True)
        codes = np.where(codes < 0, len(uniques), codes)  # NaN -> extra class
        return codes.astype(int)

    # ── Data Cleaning Pipeline ─────────────────────────────────────────────────
    #
    # Real-world CSV uploads are messy. This pipeline runs BEFORE data reaches
    # the detector, ensuring it receives a clean numeric matrix.
    #
    # Steps:
    #   1. Force numeric coercion (catch stray strings in numeric columns)
    #   2. Replace ±inf with NaN
    #   3. Drop rows that are >50% NaN (corrupted / mostly-empty)
    #   4. Report exact duplicate rows (kept: duplicates can be a poisoning signal)
    #   5. Drop zero-variance / constant columns (cause div-by-zero in layers)
    #   6. Impute remaining NaN (median); outliers are NOT clipped
    #
    # Every action is logged in the warnings list so the user knows what changed.
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str] = None,
        warnings: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Comprehensive data cleaning pipeline.

        Returns:
            (cleaned_df, surviving_feature_cols)
            Feature cols list may shrink if constant columns are dropped.
        """
        if warnings is None:
            warnings = []

        df = df.copy()
        n_original = len(df)

        # ── Step 1: Force numeric coercion ─────────────────────────────────
        # Some "numeric" columns may contain stray strings like "N/A", "?", "--"
        coercion_issues = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                before_nan = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                after_nan = df[col].isna().sum()
                new_nan = after_nan - before_nan
                if new_nan > 0:
                    coercion_issues.append(f"{col} ({new_nan} non-numeric values)")
        if coercion_issues:
            warnings.append(
                f"Coerced non-numeric values to NaN in: {', '.join(coercion_issues[:5])}"
                + (f" (+{len(coercion_issues)-5} more)" if len(coercion_issues) > 5 else "")
            )

        # ── Step 2: Replace ±inf with NaN ──────────────────────────────────
        # Infinite values crash numpy linalg, KL divergence, spectral analysis
        inf_counts = {}
        for col in feature_cols:
            n_inf = np.isinf(df[col].values.astype(float)).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
            if n_inf > 0:
                inf_counts[col] = int(n_inf)
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if inf_counts:
            total_inf = sum(inf_counts.values())
            warnings.append(
                f"Replaced {total_inf} infinite value(s) with NaN "
                f"in {len(inf_counts)} column(s)."
            )

        # ── Step 3: Drop rows that are >50% NaN ───────────────────────────
        # Rows where most feature values are missing are likely corrupted
        nan_per_row = df[feature_cols].isna().sum(axis=1)
        threshold = len(feature_cols) * 0.5
        corrupted_mask = nan_per_row > threshold
        n_corrupted = corrupted_mask.sum()
        if n_corrupted > 0:
            df = df[~corrupted_mask].reset_index(drop=True)
            warnings.append(
                f"Dropped {n_corrupted} corrupted row(s) "
                f"(>{int(threshold)} of {len(feature_cols)} features missing)."
            )

        # ── Step 4: Report (do not drop) exact duplicate rows ───────────────
        # Duplicated records are a classic poisoning pattern, and Layer 3 scores
        # near-duplicate rate, so removing them here would hide the evidence.
        cols_to_check = feature_cols + ([label_col] if label_col else [])
        n_dupes = int(df.duplicated(subset=cols_to_check).sum())
        if n_dupes > 0:
            warnings.append(
                f"{n_dupes} exact duplicate row(s) "
                f"({round(n_dupes / len(df) * 100, 1)}% of data) kept: duplicates can be a poisoning signal."
            )

        # ── Step 5: Drop zero-variance / constant columns ─────────────────
        # Constant features cause division-by-zero in normalization, KL
        # divergence, spectral gap, and DBSCAN eps estimation.
        surviving_cols = []
        dropped_const = []
        for col in feature_cols:
            col_std = df[col].std()
            if col_std is None or col_std == 0 or (pd.api.types.is_numeric_dtype(df[col]) and np.isnan(col_std)):
                dropped_const.append(col)
            else:
                surviving_cols.append(col)
        if dropped_const:
            warnings.append(
                f"Dropped {len(dropped_const)} constant-value column(s): "
                f"{', '.join(dropped_const[:5])}"
                + (f" (+{len(dropped_const)-5} more)" if len(dropped_const) > 5 else "")
            )
        feature_cols = surviving_cols

        # ── Step 6: Impute remaining NaN (outliers are kept: they are signal) ──
        nan_imputed = 0
        for col in feature_cols:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                median = df[col].median()
                # If entire column is NaN after cleaning, fill with 0
                if pd.isna(median):
                    median = 0.0
                df[col] = df[col].fillna(median)
                nan_imputed += n_nan

        if nan_imputed > 0:
            warnings.append(
                f"Imputed {nan_imputed} missing value(s) with column median."
            )

        # ── Summary ────────────────────────────────────────────────────────
        n_final = len(df)
        if n_final < n_original:
            warnings.append(
                f"Data cleaning: {n_original} → {n_final} rows "
                f"({n_original - n_final} removed)."
            )

        return df, feature_cols

    # ── Main Ingestion ─────────────────────────────────────────────────────────

    def _parse(self, csv_bytes: bytes, what: str = "CSV") -> pd.DataFrame:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse {what}: {e}")
        return df

    @staticmethod
    def _cutoff_mask(df: pd.DataFrame, column: str, cutoff: str) -> np.ndarray:
        """True for rows strictly before ``cutoff`` in ``column`` (numbers or dates)."""
        if column not in df.columns:
            raise ValueError(f"Split column {column!r} is not in the CSV.")
        series = df[column]
        try:
            numeric = pd.to_numeric(series, errors="raise")
            values, edge = numeric, float(cutoff)
        except (ValueError, TypeError):
            try:
                values, edge = pd.to_datetime(series, errors="raise"), pd.to_datetime(cutoff)
            except (ValueError, TypeError):
                raise ValueError(f"Column {column!r} must hold numbers or dates, and the cutoff {cutoff!r} must match.")
        if values.isna().any():
            raise ValueError(f"Split column {column!r} has empty values; every row needs a value to be placed.")
        return (values < edge).to_numpy()

    def ingest(self, csv_bytes: bytes, filename: str = "upload.csv", *, reference_bytes: bytes | None = None,
               split_column: str | None = None, split_cutoff: str | None = None) -> Dict[str, Any]:
        """
        Main ingestion method. Returns a structured dataset ready for detection.

        The trusted reference rows (what "normal" looks like) come from one of:
          * ``reference_bytes``: a separate CSV with the same columns (it is the reference, the main file is incoming)
          * ``split_column`` + ``split_cutoff``: rows of the main file strictly before the cutoff (number or date)
          * neither: the first 70% of the main file in file order
        
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
        if reference_bytes is not None and (split_column or split_cutoff):
            raise ValueError("Give either a reference file or a split column and cutoff, not both.")
        if bool(split_column) != bool(split_cutoff):
            raise ValueError("A split needs both a column and a cutoff.")

        df = self._parse(csv_bytes)
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            warnings.append(f"Dataset truncated to {MAX_ROWS} rows (limit).")
        is_reference = None          # None = file order
        split_method = "file_order"
        extra_excluded: set = set()
        if reference_bytes is not None:
            ref = self._parse(reference_bytes, "reference CSV")
            if len(ref) > MAX_ROWS:
                ref = ref.head(MAX_ROWS)
                warnings.append(f"Reference file truncated to {MAX_ROWS} rows (limit).")
            if set(ref.columns) != set(df.columns):
                only_ref, only_main = sorted(set(ref.columns) - set(df.columns)), sorted(set(df.columns) - set(ref.columns))
                raise ValueError(f"Reference and main CSV must have the same columns (only in reference: {only_ref}; only in main: {only_main}).")
            df = pd.concat([ref[df.columns], df], ignore_index=True)
            is_reference = np.r_[np.ones(len(ref), bool), np.zeros(len(df) - len(ref), bool)]
            split_method = "reference_file"
        elif split_column:
            is_reference = self._cutoff_mask(df, split_column, split_cutoff)
            extra_excluded = {split_column}
            split_method = "column_cutoff"

        if len(df) < 20:
            raise ValueError("Dataset too small (minimum 20 rows required).")

        # ── Schema Detection ───────────────────────────────────────────────
        ground_truth_cols = [c for c in df.columns if str(c).lower().strip() in GROUND_TRUTH_NAMES and c not in extra_excluded]
        if ground_truth_cols:
            warnings.append(
                f"Column(s) {', '.join(map(str, ground_truth_cols))} look like an answer key or model output; "
                "excluded from features and labels."
            )
        label_col = self._detect_label_column(df, exclude=set(ground_truth_cols) | extra_excluded)
        df = self._encode_text_columns(df, exclude=set(ground_truth_cols) | extra_excluded | {label_col}, warnings=warnings)
        feature_cols = self._detect_feature_columns(df, label_col, exclude=set(ground_truth_cols) | extra_excluded)

        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found in the CSV.")

        if len(feature_cols) > 100:
            # Keep top 100 by variance
            variances = df[feature_cols].var().sort_values(ascending=False)
            feature_cols = list(variances.head(100).index)
            warnings.append("More than 100 features detected. Keeping top 100 by variance.")

        # ── Clean Data ─────────────────────────────────────────────────────
        n_ref_file = int(is_reference.sum()) if split_method == "reference_file" else 0
        df = df.assign(**{_ROW_COL: np.r_[np.arange(n_ref_file), np.arange(len(df) - n_ref_file)]})
        if is_reference is not None:
            df = df.assign(**{_REF_COL: is_reference})   # cleaning drops rows; these tags travel with them
        df_clean, feature_cols = self._clean_dataframe(
            df, feature_cols, label_col=label_col, warnings=warnings
        )

        if len(feature_cols) == 0:
            raise ValueError("No usable feature columns remain after data cleaning.")

        if len(df_clean) < 20:
            raise ValueError(
                f"Only {len(df_clean)} rows remain after cleaning "
                f"(minimum 20 required). Input data may be too noisy."
            )

        # ── Prepare Labels ─────────────────────────────────────────────────
        has_labels = label_col is not None
        labels = None
        if has_labels:
            try:
                labels = self._encode_label_column(df_clean[label_col])
                n_classes = len(np.unique(labels))
                raw = df_clean[label_col]
                continuous = pd.api.types.is_float_dtype(raw) and not np.allclose(raw.dropna(), raw.dropna().round())
                if n_classes > MAX_CLASSES or (continuous and n_classes > 20):
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

        # ── Reference / incoming split ─────────────────────────────────────
        n = len(features)
        if is_reference is None:
            split_idx = int(n * 0.70)
            warnings.append(
                f"Reference = first {split_idx} rows in file order; incoming = remaining {n - split_idx}. "
                "If rows are not ordered by time, or the file may contain bad rows anywhere, give a trusted "
                "reference file or a split column and cutoff instead."
            )
        else:
            tag = df_clean[_REF_COL].to_numpy(bool)
            order = np.argsort(~tag, kind="stable")      # reference rows first, order kept within each side
            features, df_clean = features[order], df_clean.iloc[order].reset_index(drop=True)
            if labels is not None:
                labels = labels[order]
            df_clean = df_clean.drop(columns=_REF_COL)
            split_idx = int(tag.sum())
            if split_idx < MIN_REFERENCE_ROWS or n - split_idx < MIN_INCOMING_ROWS:
                raise ValueError(
                    f"The split leaves {split_idx} reference and {n - split_idx} incoming rows after cleaning; "
                    f"need at least {MIN_REFERENCE_ROWS} and {MIN_INCOMING_ROWS}."
                )
            warnings.append(
                f"Reference = {split_idx} rows from {'the reference file' if split_method == 'reference_file' else f'{split_column} < {split_cutoff}'}; "
                f"incoming = {n - split_idx} rows."
            )

        source_rows = df_clean.pop(_ROW_COL).to_numpy(int)
        from_ref_file = np.arange(n) < (split_idx if split_method == "reference_file" else 0)

        # Scale to [0, 1] using the REFERENCE rows only, so incoming rows cannot shape
        # the baseline. Incoming values are not clipped: a value outside the reference
        # range (>1 or <0) is exactly the kind of shift the detector should see.
        ref = features[:split_idx]
        col_min = ref.min(axis=0)
        col_range = ref.max(axis=0) - col_min
        col_range[col_range == 0] = 1  # avoid division by zero
        features_norm = (features - col_min) / col_range

        # ── Build Samples List ─────────────────────────────────────────────
        samples = []
        for i in range(n):
            sample = {
                "id": f"row_{i}",
                "row_index": i,
                "feature_vector": features_norm[i].tolist(),
                "label": int(labels[i]) if has_labels and labels is not None else -1,
                "poison_status": "unknown",  # we don't know — that's what we're detecting
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "batch_id": f"batch_{i // 100}",
                "source_id": filename,
                "client_id": "upload_client",
                "split": "reference" if i < split_idx else "incoming",
                # where the row is in the user's file: line 1 is the header
                "source_file": "reference" if from_ref_file[i] else "main",
                "csv_line": int(source_rows[i]) + 2,
            }
            samples.append(sample)

        # ── Schema Summary ─────────────────────────────────────────────────
        schema = {
            "total_columns": len(df.columns),
            "feature_columns": feature_cols,
            "label_column": label_col,
            "n_rows": n,
            "n_features": len(feature_cols),
            "ground_truth_columns": [str(c) for c in ground_truth_cols],
            "split_method": split_method,
            "dtypes": {col: str(df_clean[col].dtype) for col in feature_cols},
            "missing_pct": {
                col: round(df_clean[col].isna().mean() * 100, 2)
                for col in feature_cols
            },
            "feature_stats": {
                col: {
                    "mean": round(float(df_clean[col].mean()), 4),
                    "std": round(float(df_clean[col].std()), 4),
                    "min": round(float(df_clean[col].min()), 4),
                    "max": round(float(df_clean[col].max()), 4),
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
            "categories": self.categories,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
