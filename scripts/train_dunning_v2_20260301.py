"""
Train dunning recovery model v2 (2026-03-01): strict temporal split for March 2026 refresh.

- Data: 2025-01-02 through 2026-03-01 (inclusive).
- Training: 2025-01-02 to 2026-01-31 (inclusive).
- Validation: 2026-02-01 to 2026-02-23 (inclusive). Calibration fit on validation set.
- Raw data is cached to data/raw/dunning_raw.parquet; use FORCE_QUERY=1 to re-query.
- Saves models to models/; outputs eval plot to reports/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    PrecisionRecallDisplay,
)

# Project root and paths (script lives in scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import run_pipeline
from src.features import (
    engineer_dunning_features,
    KEEP_COLS,
    TARGET,
    GROUP_COL,
    DROP_COLS,
    CAT_FEATURES,
    sanitize_for_catboost,
)
from src.model import IsotonicCalibratedClassifier

# Register for joblib unpickle (models reference __main__.IsotonicCalibratedClassifier)
setattr(sys.modules["__main__"], "IsotonicCalibratedClassifier", IsotonicCalibratedClassifier)

# ---------------------------------------------------------------------------
# Temporal windows (March 2026 refresh)
# ---------------------------------------------------------------------------
GLOBAL_START = "2025-01-02"
TRAIN_END = "2026-01-31"
VAL_START = "2026-02-01"
VAL_END = "2026-02-23"
HOLDOUT_START = "2026-02-24"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_CACHE_PATH = DATA_RAW_DIR / "dunning_raw.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=50,
)
SUFFIX = "20260301"


def load_raw_data(force_query: bool = False) -> pd.DataFrame:
    """Load raw pipeline data: from local cache if present, else query BigQuery and save cache."""
    force_query = force_query or (os.environ.get("FORCE_QUERY", "").lower() in ("1", "true", "yes"))
    if not force_query and DATA_CACHE_PATH.exists():
        print(f"Loading cached data from {DATA_CACHE_PATH}...")
        df = pd.read_parquet(DATA_CACHE_PATH)
        print("Cached data loaded successfully")
        return df
    print("Querying data from BigQuery...")
    df = run_pipeline()
    print("Data loaded successfully")
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_CACHE_PATH, index=False)
    print(f"Saved raw data to {DATA_CACHE_PATH} for future runs")
    return df


def load_and_prepare_data(force_query: bool = False):
    """Load raw data, engineer features, filter by temporal windows; return X_train, X_val, y_train, y_val."""
    df = load_raw_data(force_query=force_query)
    processed_df = engineer_dunning_features(df)
    print("Features engineered successfully")

    if "updated_at" not in processed_df.columns:
        raise ValueError("processed_df must have 'updated_at'; ensure run_pipeline() returns it.")

    updated_at = pd.to_datetime(processed_df["updated_at"])
    in_range = (updated_at >= GLOBAL_START) & (updated_at < HOLDOUT_START)
    processed_df = processed_df.loc[in_range].copy()
    updated_at = pd.to_datetime(processed_df["updated_at"])

    missing = [c for c in KEEP_COLS if c not in processed_df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    df_dunning = processed_df[KEEP_COLS].copy()
    X = df_dunning.drop(columns=DROP_COLS)
    y = df_dunning[TARGET]

    train_mask = (updated_at >= GLOBAL_START) & (updated_at <= TRAIN_END)
    val_mask = (updated_at >= VAL_START) & (updated_at <= VAL_END)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    return X_train, X_val, y_train, y_val


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    force_query = os.environ.get("FORCE_QUERY", "").lower() in ("1", "true", "yes")

    print("Loading data and applying temporal windows...")
    X_train, X_val, y_train, y_val = load_and_prepare_data(force_query=force_query)

    print(f"  Train: {len(X_train)} rows ({GLOBAL_START} to {TRAIN_END})")
    print(f"  Val:   {len(X_val)} rows ({VAL_START} to {VAL_END})")
    print(f"  (No data from {HOLDOUT_START} onwards.)")

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train or validation set is empty; check date ranges and data.")

    X_train = sanitize_for_catboost(X_train)
    X_val = sanitize_for_catboost(X_val)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    calibrated = IsotonicCalibratedClassifier(model, method="isotonic")
    calibrated.fit(X_val, y_val)

    raw_path = MODELS_DIR / f"catboost_dunning_{SUFFIX}.joblib"
    cal_path = MODELS_DIR / f"catboost_dunning_calibrated_{SUFFIX}.joblib"
    joblib.dump(model, raw_path)
    joblib.dump(calibrated, cal_path)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {cal_path}")

    preds_val = calibrated.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, preds_val)
    pr_auc_val = average_precision_score(y_val, preds_val)

    print("\n--- Validation set evaluation ---")
    print(f"  AUC:    {auc_val:.4f}")
    print(f"  PR-AUC: {pr_auc_val:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    display = PrecisionRecallDisplay.from_predictions(y_val, preds_val, ax=ax)
    ax.set_title(f"Dunning recovery model {SUFFIX} — Validation set PR curve")
    ax.legend(loc="lower left")
    plot_path = REPORTS_DIR / f"eval_plots_{SUFFIX}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")


if __name__ == "__main__":
    main()
