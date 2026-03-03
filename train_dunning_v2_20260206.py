"""
Train dunning recovery model v2 (2026-02-24): strict temporal split for February 2026 production refresh.

- Training: 2025-01-02 to 2026-02-07 (inclusive).
- Validation: 2026-02-08 to 2026-02-14 (inclusive).
- No data from 2026-02-15 onwards (reserved for shadow phase and buffer).
- CatBoost without auto_class_weights; calibration via isotonic regression on validation set.
- Raw data is cached to data/dunning_raw.parquet after first BigQuery run; later runs use cache
  unless FORCE_QUERY=1. Set FORCE_QUERY=1 to re-query and refresh the cache.
- Saves raw and calibrated models to models/ with suffix _20260224; outputs AUC/PR-AUC and PR curve plot.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    PrecisionRecallDisplay,
)

# Project root: allow importing txn_pipeline when run from repo root or from this dir.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Temporal windows (February 2026 production refresh)
# ---------------------------------------------------------------------------
GLOBAL_START = "2025-01-02"
TRAIN_END = "2026-02-07"   # inclusive
VAL_START = "2026-02-08"   # inclusive
VAL_END = "2026-02-14"     # inclusive
HOLDOUT_START = "2026-02-15"  # do not use any data from this date onwards

# ---------------------------------------------------------------------------
# Feature engineering (aligned with dunning_modeling.ipynb)
# ---------------------------------------------------------------------------

def engineer_dunning_features(df: pd.DataFrame) -> pd.DataFrame:
    """Soft-decline dunning attempts with temporal and transactional features."""
    df = df.sort_values(by=["linked_invoice_id", "updated_at"]).copy()
    df["prev_decline_code"] = df.groupby("linked_invoice_id")["Decline_code_norm"].shift(1)
    df["prev_card_status"] = df.groupby("linked_invoice_id")["card_status"].shift(1)
    df["prev_decline_type"] = df.groupby("linked_invoice_id")["Decline_type_for_retry"].shift(1)
    df["prev_attempt_time"] = df.groupby("linked_invoice_id")["updated_at"].shift(1)
    df["time_since_prev_attempt"] = (df["updated_at"] - df["prev_attempt_time"]).dt.total_seconds() / 3600
    df["first_attempt_at"] = df.groupby("linked_invoice_id")["updated_at"].transform("min")
    df["cumulative_delay_hours"] = (df["updated_at"] - df["first_attempt_at"]).dt.total_seconds() / 3600

    df = df[
        (df["is_attached_invoice_1st_attempt"] == "Dunning attempt")
        & (df["prev_decline_type"] == "Soft decline")
    ].copy()

    df["local_day_of_week"] = df["localized_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["local_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["local_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["local_day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["local_day_of_week"] / 7)
    df["max_days"] = df["localized_time"].dt.days_in_month
    df["day_sin"] = np.sin(2 * np.pi * (df["local_day_of_month"] - 1) / df["max_days"])
    df["day_cos"] = np.cos(2 * np.pi * (df["local_day_of_month"] - 1) / df["max_days"])
    df["dist_to_payday"] = df["local_day_of_month"].apply(
        lambda x: min(abs(x - 1), abs(x - 15), abs(x - 30))
    )
    df["log_charge_amount"] = np.log1p(df["amount"])
    df["is_debit"] = (df["funding_type_norm"].str.lower() == "debit").astype(int)
    df["amt_per_attempt"] = df["amount"] / (df["invoice_attempt_no"] + 1)
    df["is_success"] = (df["status"] == "success").astype(int)
    df["billing_country"] = df["billing_country"].fillna("UNKNOWN").astype(str)
    df["prev_decline_code"] = df["prev_decline_code"].fillna("UNKNOWN").astype(str)
    return df


# Feature set (must match notebook keep_cols minus target/group)
KEEP_COLS = [
    "linked_invoice_id", "prev_decline_code", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "day_sin", "day_cos", "dist_to_payday", "log_charge_amount", "is_debit", "amt_per_attempt",
    "time_since_prev_attempt", "cumulative_delay_hours",
    "billing_country", "gateway", "funding_type_norm", "card_brand", "prev_card_status",
    "Domain_category", "invoice_attempt_no", "is_success",
]
TARGET = "is_success"
GROUP_COL = "linked_invoice_id"
DROP_COLS = [TARGET, GROUP_COL]

# Categorical features for CatBoost (same as notebook; notebook uses billing_country, not cus_country)
CAT_FEATURES = [
    "prev_decline_code", "billing_country", "gateway",
    "funding_type_norm", "card_brand", "Domain_category", "prev_card_status"
]


def _sanitize_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN/None in categorical and numeric columns so CatBoost does not raise TypeError."""
    X = X.copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN").astype(str).replace("nan", "UNKNOWN")
    # Numeric columns: no NaN/inf (CatBoost requires real numbers)
    for col in X.columns:
        if col not in CAT_FEATURES:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# ---------------------------------------------------------------------------
# Model config (no auto_class_weights — calibration handles probability scale)
# ---------------------------------------------------------------------------
CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=50,
)
SUFFIX = "20260224"
MODELS_DIR = _SCRIPT_DIR / "models"
DATA_CACHE_DIR = _SCRIPT_DIR / "data"
DATA_CACHE_PATH = DATA_CACHE_DIR / "dunning_raw.parquet"


class IsotonicCalibratedClassifier:
    """Wraps a fitted classifier and calibrates its probabilities using IsotonicRegression on (X_cal, y_cal)."""

    def __init__(self, estimator, method="isotonic"):
        self.estimator = estimator
        self.method = method
        self.calibrator_ = None

    def fit(self, X_cal, y_cal):
        p = self.estimator.predict_proba(X_cal)[:, 1]
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(p, y_cal)
        return self

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)[:, 1]
        p_cal = self.calibrator_.predict(p).reshape(-1, 1)
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.hstack([1 - p_cal, p_cal])

    @property
    def feature_names_(self):
        return getattr(self.estimator, "feature_names_", None)


def load_raw_data(force_query: bool = False) -> pd.DataFrame:
    """Load raw pipeline data: from local cache if present, else query BigQuery and save cache."""
    force_query = force_query or (os.environ.get("FORCE_QUERY", "").lower() in ("1", "true", "yes"))
    if not force_query and DATA_CACHE_PATH.exists():
        print(f"Loading cached data from {DATA_CACHE_PATH}...")
        df = pd.read_parquet(DATA_CACHE_PATH)
        print("Cached data loaded successfully")
        return df
    from txn_pipeline import run_pipeline

    print("Querying data from BigQuery...")
    df = run_pipeline()
    print("Data loaded successfully")
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_CACHE_PATH, index=False)
    print(f"Saved raw data to {DATA_CACHE_PATH} for future runs")
    return df


def load_and_prepare_data(force_query: bool = False):
    """Load raw data (from cache or BigQuery), engineer features, filter by temporal windows; return X_train, X_val, y_train, y_val."""
    df = load_raw_data(force_query=force_query)
    processed_df = engineer_dunning_features(df)
    print("Features engineered successfully")

    if "updated_at" not in processed_df.columns:
        raise ValueError("processed_df must have 'updated_at'; ensure run_pipeline() returns it.")

    updated_at = pd.to_datetime(processed_df["updated_at"])
    # Only use data from global start and strictly before holdout (no 2026-02-01+)
    in_range = (updated_at >= GLOBAL_START) & (updated_at < HOLDOUT_START)
    processed_df = processed_df.loc[in_range].copy()
    updated_at = pd.to_datetime(processed_df["updated_at"])

    # Ensure we have all required columns
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
    force_query = os.environ.get("FORCE_QUERY", "").lower() in ("1", "true", "yes")

    print("Loading data and applying temporal windows...")
    X_train, X_val, y_train, y_val = load_and_prepare_data(force_query=force_query)

    print(f"  Train: {len(X_train)} rows ({GLOBAL_START} to {TRAIN_END})")
    print(f"  Val:   {len(X_val)} rows ({VAL_START} to {VAL_END})")
    print(f"  (No data from {HOLDOUT_START} onwards.)")

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train or validation set is empty; check date ranges and data.")

    # Sanitize: CatBoost raises TypeError on NaN/None in categoricals or NaN/inf in numerics
    X_train = _sanitize_for_catboost(X_train)
    X_val = _sanitize_for_catboost(X_val)

    # 1) Train raw CatBoost (no auto_class_weights)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    # 2) Calibrate on validation set (isotonic); sklearn 1.2+ removed cv="prefit", so we use a small wrapper
    calibrated = IsotonicCalibratedClassifier(model, method="isotonic")
    calibrated.fit(X_val, y_val)

    # 3) Save both models
    raw_path = MODELS_DIR / f"catboost_dunning_{SUFFIX}.joblib"
    cal_path = MODELS_DIR / f"catboost_dunning_calibrated_{SUFFIX}.joblib"
    joblib.dump(model, raw_path)
    joblib.dump(calibrated, cal_path)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {cal_path}")

    # 4) Evaluation on validation set (use calibrated probabilities for reporting)
    preds_val = calibrated.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, preds_val)
    pr_auc_val = average_precision_score(y_val, preds_val)

    print("\n--- Validation set evaluation ---")
    print(f"  AUC:    {auc_val:.4f}")
    print(f"  PR-AUC: {pr_auc_val:.4f}")

    # 5) Precision-Recall curve and save
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    display = PrecisionRecallDisplay.from_predictions(y_val, preds_val, ax=ax)
    ax.set_title(f"Dunning recovery model {SUFFIX} — Validation set PR curve")
    ax.legend(loc="lower left")
    plot_path = _SCRIPT_DIR / f"eval_plots_{SUFFIX}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")


if __name__ == "__main__":
    main()
