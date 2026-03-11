"""
Train dunning recovery model v2 (2026-03-11): strict temporal split after shadowing period.

- Data: 2025-01-02 through 2026-03-10 (inclusive) for train/cal/val.
- Training: 2025-01-02 to 2026-02-14 (inclusive) — CatBoost fit.
- Calibration: 2026-02-15 to 2026-02-28 (inclusive) — isotonic (and optional temperature) fit on held-out data to reduce overfitting and optimism.
- Validation: 2026-03-01 to 2026-03-10 (inclusive) — evaluation only (no calibration fit).
- Holdout: 2026-03-11 onwards (reserved for future shadow/eval).
- Raw data is cached to data/raw/dunning_raw.parquet; use FORCE_QUERY=1 to re-query.
- Saves models to models/; outputs eval plot to reports/.
- Calibration tuning (reduce optimism): see docs/CALIBRATION_TUNING.md and CALIBRATION_TEMPERATURE env.
"""
from __future__ import annotations

import json
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
    brier_score_loss,
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
# Temporal windows (post-shadowing refresh: train / cal / val)
# Override via env for cron/Airflow: TRAIN_END, CAL_START, CAL_END, VAL_START, VAL_END, HOLDOUT_START, SUFFIX
# ---------------------------------------------------------------------------
GLOBAL_START = os.environ.get("GLOBAL_START", "2025-01-02")
TRAIN_END = os.environ.get("TRAIN_END", "2026-02-14")       # CatBoost training end (inclusive)
CAL_START = os.environ.get("CAL_START", "2026-02-15")       # Calibration set start (inclusive)
CAL_END = os.environ.get("CAL_END", "2026-02-28")        # Calibration set end (inclusive)
VAL_START = os.environ.get("VAL_START", "2026-03-01")      # Validation (eval only) start (inclusive)
VAL_END = os.environ.get("VAL_END", "2026-03-10")        # Validation end (inclusive)
HOLDOUT_START = os.environ.get("HOLDOUT_START", "2026-03-11")  # No train/cal/val data from here
SUFFIX = os.environ.get("SUFFIX", "20260311")

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

# Calibration: temperature > 1 pulls probabilities toward 0.5 (reduces optimism)
CALIBRATION_TEMPERATURE = float(os.environ.get("CALIBRATION_TEMPERATURE", "1.25"))
ECE_N_BINS = 10


def _ece(y_true, y_pred, n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error: weighted mean of |mean_pred - actual_rate| over bins."""
    if len(y_true) == 0 or n_bins < 1:
        return 0.0
    try:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return float(np.abs(y_pred.mean() - y_true.mean()))
        bin_ids = np.digitize(y_pred, bins[1:-1])
        ece = 0.0
        for i in range(len(bins)):
            mask = bin_ids == i
            if mask.sum() == 0:
                continue
            mean_pred = y_pred[mask].mean()
            actual_rate = y_true[mask].mean()
            weight = mask.sum() / len(y_true)
            ece += weight * abs(mean_pred - actual_rate)
        return float(ece)
    except Exception:
        return float(np.nan)


def _mce(y_true, y_pred, n_bins: int = ECE_N_BINS) -> float:
    """Maximum Calibration Error: max over bins of |mean_pred - actual_rate|."""
    if len(y_true) == 0 or n_bins < 1:
        return 0.0
    try:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return float(np.abs(y_pred.mean() - y_true.mean()))
        bin_ids = np.digitize(y_pred, bins[1:-1])
        mce = 0.0
        for i in range(len(bins)):
            mask = bin_ids == i
            if mask.sum() == 0:
                continue
            mean_pred = y_pred[mask].mean()
            actual_rate = y_true[mask].mean()
            mce = max(mce, abs(mean_pred - actual_rate))
        return float(mce)
    except Exception:
        return float(np.nan)


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
    """Load raw data, engineer features, filter by temporal windows; return X_train, X_cal, X_val, y_train, y_cal, y_val."""
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
    cal_mask = (updated_at >= CAL_START) & (updated_at <= CAL_END)
    val_mask = (updated_at >= VAL_START) & (updated_at <= VAL_END)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_cal = X.loc[cal_mask]
    y_cal = y.loc[cal_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    return X_train, X_cal, X_val, y_train, y_cal, y_val


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    force_query = os.environ.get("FORCE_QUERY", "").lower() in ("1", "true", "yes")

    print("Loading data and applying temporal windows...")
    X_train, X_cal, X_val, y_train, y_cal, y_val = load_and_prepare_data(force_query=force_query)

    print(f"  Train: {len(X_train)} rows ({GLOBAL_START} to {TRAIN_END})")
    print(f"  Cal:   {len(X_cal)} rows ({CAL_START} to {CAL_END})")
    print(f"  Val:   {len(X_val)} rows ({VAL_START} to {VAL_END})")
    print(f"  (No data from {HOLDOUT_START} onwards.)")

    if len(X_train) == 0:
        raise ValueError("Training set is empty; check date ranges and data.")
    if len(X_cal) == 0:
        raise ValueError("Calibration set is empty; need cal data to fit isotonic. Check CAL_START/CAL_END.")
    if len(X_val) == 0:
        raise ValueError("Validation set is empty; check date ranges and data.")

    X_train = sanitize_for_catboost(X_train)
    X_cal = sanitize_for_catboost(X_cal)
    X_val = sanitize_for_catboost(X_val)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    # Fit calibration on held-out cal set (not val) to reduce overfitting and optimism
    calibrated = IsotonicCalibratedClassifier(
        model, method="isotonic", temperature=CALIBRATION_TEMPERATURE
    )
    calibrated.fit(X_cal, y_cal)

    raw_path = MODELS_DIR / f"catboost_dunning_{SUFFIX}.joblib"
    cal_path = MODELS_DIR / f"catboost_dunning_calibrated_{SUFFIX}.joblib"
    joblib.dump(model, raw_path)
    joblib.dump(calibrated, cal_path)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {cal_path}")
    print(f"  Calibration temperature: {CALIBRATION_TEMPERATURE} (set CALIBRATION_TEMPERATURE env to tune)")

    preds_val = calibrated.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, preds_val)
    pr_auc_val = average_precision_score(y_val, preds_val)
    brier_val = brier_score_loss(y_val, preds_val)
    ece_val = _ece(y_val.values, preds_val)
    mce_val = _mce(y_val.values, preds_val)

    print("\n--- Validation set evaluation ---")
    print("  Discriminative:")
    print(f"    AUC:    {auc_val:.4f}")
    print(f"    PR-AUC: {pr_auc_val:.4f}")
    print("  Calibration (lower is better):")
    print(f"    Brier:  {brier_val:.4f}")
    print(f"    ECE:    {ece_val:.4f}")
    print(f"    MCE:    {mce_val:.4f}")

    # PR curve
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    display = PrecisionRecallDisplay.from_predictions(y_val, preds_val, ax=ax)
    ax.set_title(f"Dunning recovery model {SUFFIX} — Validation set PR curve")
    ax.legend(loc="lower left")
    plot_path = REPORTS_DIR / f"eval_plots_{SUFFIX}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")

    # Reliability diagram (deciles) on validation
    deciles = np.percentile(preds_val, np.linspace(0, 100, 11))
    deciles = np.unique(deciles)
    if len(deciles) >= 2:
        bin_ids = np.digitize(preds_val, deciles[1:-1])
        bin_means = []
        bin_actuals = []
        for i in range(10):
            mask = bin_ids == i
            if mask.sum() > 0:
                bin_means.append(np.mean(preds_val[mask]))
                bin_actuals.append(np.mean(y_val.values[mask]))
            else:
                bin_means.append(np.nan)
                bin_actuals.append(np.nan)
        bin_means = np.array(bin_means)
        bin_actuals = np.array(bin_actuals)
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        x = np.arange(10)
        ax2.bar(x - 0.2, np.nan_to_num(bin_means, nan=0), 0.4, label="Mean predicted", color="steelblue", alpha=0.8)
        ax2.bar(x + 0.2, np.nan_to_num(bin_actuals, nan=0), 0.4, label="Actual rate", color="coral", alpha=0.8)
        ax2.plot([0, 9], [y_val.mean(), y_val.mean()], "k--", alpha=0.7, label="Global rate")
        ax2.set_xlabel("Decile (0=lowest P, 9=highest P)")
        ax2.set_ylabel("Rate")
        ax2.set_title(f"Validation calibration (reliability diagram) — ECE={ece_val:.3f}, MCE={mce_val:.3f}")
        ax2.set_xticks(x)
        ax2.legend()
        ax2.set_ylim(0, None)
        fig2.tight_layout()
        rel_path = REPORTS_DIR / f"reliability_{SUFFIX}.png"
        fig2.savefig(rel_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {rel_path}")

    # Export training run log for retrain job (when EXPORT_TRAINING_LOG=1)
    if os.environ.get("EXPORT_TRAINING_LOG", "").lower() in ("1", "true", "yes"):
        from datetime import datetime, timezone
        log_path = MODELS_DIR / f"training_run_{SUFFIX}.json"
        log_data = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "suffix": SUFFIX,
            "global_start": GLOBAL_START,
            "train_end": TRAIN_END,
            "cal_start": CAL_START,
            "cal_end": CAL_END,
            "val_start": VAL_START,
            "val_end": VAL_END,
            "holdout_start": HOLDOUT_START,
            "n_train": int(len(X_train)),
            "n_cal": int(len(X_cal)),
            "n_val": int(len(X_val)),
            "auc_val": float(auc_val),
            "pr_auc_val": float(pr_auc_val),
            "brier_val": float(brier_val),
            "ece_val": float(ece_val),
            "mce_val": float(mce_val),
            "calibration_temperature": float(CALIBRATION_TEMPERATURE),
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
