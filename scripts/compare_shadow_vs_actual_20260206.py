"""
Compare shadow monitoring output with actual Chargebee outcomes.

- Loads shadow log (artifacts/shadow_log.csv), fetches actual outcomes from BigQuery
  (or optional CSV), merges on invoice_id, computes calibration/discriminative metrics,
  bootstrap CIs, decile lift, TTR, EV lift, decline-code matrix; writes comparison CSV,
  text report, calibration plot, and cumulative gains plot.
- Paths: if not absolute, resolved relative to PROJECT_ROOT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)

# Project root (script lives in scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bq_client import get_bq_client, load_bigquery_table

# ---------------------------------------------------------------------------
# Directories and default paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

SHADOW_LOG_PATH = ARTIFACTS_DIR / "shadow_log.csv"
COMPARISON_OUTPUT_PATH = ARTIFACTS_DIR / "shadow_vs_actual_comparison.csv"
REPORT_OUTPUT_PATH = ARTIFACTS_DIR / "shadow_vs_actual_report.txt"
CALIBRATION_PLOT_PATH = ARTIFACTS_DIR / "calibration_plot.png"
GAINS_PLOT_PATH = ARTIFACTS_DIR / "cumulative_gains_plot.png"

# Config constants
OUTCOMES_START_DATE = "2026-02-15"
BQ_TABLE = "aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback"
BQ_PROJECT = "aa-datamart"
BQ_LOCATION = "europe-west1"
TEMPORAL_WINDOWS_H = [6, 12, 24]
ECE_N_BINS = 10
BOOTSTRAP_N = int(os.environ.get("SHADOW_BOOTSTRAP_N", "500"))
BOOTSTRAP_ALPHA = 0.05
BASELINE_LIFT_FACTOR = float(os.environ.get("BASELINE_LIFT_FACTOR", "3.0"))

INVOICE_ID_COL = "invoice_id"
SHADOW_ID_COL = "invoice_id"  # shadow log uses invoice_id (= linked_invoice_id)


def _resolve_path(path: Path, base: Path) -> Path:
    """Return path unchanged if absolute, else base / path."""
    if path.is_absolute():
        return path
    return base / path


def fetch_actual_outcomes(
    start_date: str | None = None,
    bq_table: str | None = None,
) -> pd.DataFrame:
    """
    Load actual outcomes from BigQuery: recovered, recovered_at, last_attempt_at,
    prev_decline_code, prev_card_status per linked_invoice_id (attempts with
    updated_at >= start_date). prev_* from "previous" attempt (last failure before
    success for recovered, latest for unrecovered).
    """
    start_date = start_date or OUTCOMES_START_DATE
    bq_table = bq_table or BQ_TABLE
    full_table = f"`{bq_table}`"

    query = f"""
    WITH base AS (
      SELECT linked_invoice_id, updated_at, status,
        COALESCE(CAST(Decline_code_norm AS STRING), 'unknown') AS prev_decline_code,
        COALESCE(CAST(card_status AS STRING), 'unknown') AS prev_card_status
      FROM {full_table}
      WHERE updated_at >= TIMESTAMP('{start_date}')
    ),
    recovered_at_per_inv AS (
      SELECT linked_invoice_id,
        MAX(CASE WHEN LOWER(TRIM(COALESCE(status, ''))) = 'success' THEN 1 ELSE 0 END) AS recovered,
        MIN(CASE WHEN LOWER(TRIM(COALESCE(status, ''))) = 'success' THEN updated_at END) AS recovered_at,
        MAX(updated_at) AS last_attempt_at
      FROM base
      GROUP BY linked_invoice_id
    ),
    ranked AS (
      SELECT b.linked_invoice_id, b.updated_at, b.prev_decline_code, b.prev_card_status,
        r.recovered, r.recovered_at, r.last_attempt_at,
        ROW_NUMBER() OVER (
          PARTITION BY b.linked_invoice_id
          ORDER BY
            CASE
              WHEN r.recovered = 1 AND b.updated_at < r.recovered_at THEN b.updated_at
              WHEN r.recovered = 1 THEN TIMESTAMP('1970-01-01')
              ELSE b.updated_at
            END DESC
        ) AS rn_prev
      FROM base b
      JOIN recovered_at_per_inv r ON b.linked_invoice_id = r.linked_invoice_id
      WHERE (r.recovered = 1 AND b.updated_at < r.recovered_at) OR (r.recovered = 0)
    ),
    prev_attempt AS (
      SELECT linked_invoice_id, prev_decline_code, prev_card_status
      FROM ranked
      WHERE rn_prev = 1
    )
    SELECT r.linked_invoice_id AS invoice_id, r.recovered, r.recovered_at, r.last_attempt_at,
      COALESCE(p.prev_decline_code, 'unknown') AS prev_decline_code,
      COALESCE(p.prev_card_status, 'unknown') AS prev_card_status
    FROM recovered_at_per_inv r
    LEFT JOIN prev_attempt p ON r.linked_invoice_id = p.linked_invoice_id
    ORDER BY r.linked_invoice_id
    """
    client = get_bq_client(project=BQ_PROJECT, location=BQ_LOCATION)
    df = load_bigquery_table(client, query)
    # Normalize to UTC then tz-naive
    for col in ("recovered_at", "last_attempt_at"):
        if col in df.columns and df[col].notna().any():
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    return df


def load_shadow_log(path: Path) -> pd.DataFrame:
    """Load shadow log CSV; path resolved with PROJECT_ROOT if not absolute."""
    resolved = _resolve_path(Path(path) if not isinstance(path, Path) else path, PROJECT_ROOT)
    return pd.read_csv(resolved)


def _build_merged(
    shadow: pd.DataFrame,
    actuals: pd.DataFrame,
    invoice_id_col: str = INVOICE_ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two merged DataFrames: initial (oldest record per invoice) and latest
    (latest record per invoice). Both left-join actuals on invoice_id.
    """
    shadow = shadow.copy()
    shadow["inference_run_at"] = pd.to_datetime(shadow["inference_run_at"], utc=True).dt.tz_localize(None)

    actuals = actuals.copy()
    actual_cols = [c for c in actuals.columns if c != invoice_id_col]
    rename_actual = {c: c for c in actual_cols}

    # Initial: oldest per invoice (by inference_run_at)
    initial = shadow.sort_values("inference_run_at").groupby(invoice_id_col, as_index=False).first()
    initial = initial.merge(
        actuals[[invoice_id_col] + actual_cols],
        on=invoice_id_col,
        how="left",
        suffixes=("", "_actual"),
    )
    # Drop any duplicate column names from merge (keep left)
    for c in list(initial.columns):
        if c.endswith("_actual"):
            base = c.replace("_actual", "")
            if base in initial.columns and base in actual_cols:
                initial = initial.drop(columns=[c])

    # Latest: latest per invoice
    latest = shadow.sort_values("inference_run_at").groupby(invoice_id_col, as_index=False).last()
    latest = latest.merge(
        actuals[[invoice_id_col] + actual_cols],
        on=invoice_id_col,
        how="left",
    )
    for c in list(latest.columns):
        if c.endswith("_actual"):
            base = c.replace("_actual", "")
            if base in latest.columns and base in actual_cols:
                latest = latest.drop(columns=[c])

    return initial, latest


def _parse_amount_from_snapshot(raw: str) -> float | None:
    """From raw_features_snapshot JSON get expm1(log_charge_amount) or None."""
    if pd.isna(raw):
        return None
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        log_amt = d.get("log_charge_amount")
        if log_amt is None:
            return None
        return float(np.expm1(float(log_amt)))
    except (TypeError, ValueError, KeyError):
        return None


def _enrich_merged(
    merged: pd.DataFrame,
    temporal_windows_h: list[int] | None = None,
    baseline_lift_factor: float = BASELINE_LIFT_FACTOR,
) -> pd.DataFrame:
    """
    Add: hours_suggested_to_recovered (valid only when recovered_at >= inference_run_at),
    high_prob_missed_by_chargebee, high_prob_threshold_used, decile, temporal window
    flags, invoice_amount, ev_lift. Fill prev_decline_code/prev_card_status from
    raw_features_snapshot when missing from actuals.
    """
    temporal_windows_h = temporal_windows_h or TEMPORAL_WINDOWS_H
    df = merged.copy()

    # UTC normalize
    for col in ("inference_run_at", "suggested_optimal_retry_at", "recovered_at", "last_attempt_at"):
        if col in df.columns and df[col].notna().any():
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)

    # TTR: only when recovered and recovered_at >= inference_run_at
    df["hours_suggested_to_recovered"] = np.nan
    if "recovered_at" in df.columns and "inference_run_at" in df.columns and "suggested_optimal_retry_at" in df.columns:
        valid = df["recovered"].fillna(0).astype(bool) & (df["recovered_at"] >= df["inference_run_at"])
        delta = df.loc[valid, "recovered_at"] - df.loc[valid, "suggested_optimal_retry_at"]
        df.loc[valid, "hours_suggested_to_recovered"] = delta.dt.total_seconds() / 3600

    # Global recovery rate and high-value threshold
    global_rate = df["recovered"].fillna(0).mean()
    high_threshold = baseline_lift_factor * global_rate
    df["high_prob_threshold_used"] = high_threshold
    has_suggestion = df["suggested_max_prob"].notna()
    df["high_prob_missed_by_chargebee"] = (
        (df["recovered"].fillna(0) == 0)
        & has_suggestion
        & (df["suggested_max_prob"] > high_threshold)
    )

    # Deciles (rows with suggestion)
    df["decile"] = np.nan
    if has_suggestion.any() and "suggested_max_prob" in df.columns:
        try:
            deciles = pd.qcut(df.loc[has_suggestion, "suggested_max_prob"], 10, labels=False, duplicates="drop")
            df.loc[has_suggestion, "decile"] = deciles.values
        except Exception:
            pass

    # Temporal windows
    df["recovered_within_top1_window"] = False
    if "hours_suggested_to_recovered" in df.columns:
        df.loc[df["recovered"].fillna(0).astype(bool), "recovered_within_top1_window"] = (
            df.loc[df["recovered"].fillna(0).astype(bool), "hours_suggested_to_recovered"].abs() <= 6
        )
    for h in temporal_windows_h:
        col = f"recovered_within_±{h}h"
        df[col] = False
        if "hours_suggested_to_recovered" in df.columns:
            df.loc[df["recovered"].fillna(0).astype(bool), col] = (
                df.loc[df["recovered"].fillna(0).astype(bool), "hours_suggested_to_recovered"].abs() <= h
            )

    # prev_decline_code / prev_card_status from snapshot when missing
    if "raw_features_snapshot" in df.columns:
        for key, col in (("prev_decline_code", "prev_decline_code"), ("prev_card_status", "prev_card_status")):
            if col not in df.columns:
                df[col] = None
            missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
            if missing.any():
                def _get(s):
                    try:
                        d = json.loads(s) if isinstance(s, str) else s
                        return d.get(key)
                    except Exception:
                        return None
                df.loc[missing, col] = df.loc[missing, "raw_features_snapshot"].map(_get)
    for col in ("prev_decline_code", "prev_card_status"):
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().replace("", "unknown")

    # Invoice amount: from actuals 'amount' if present, else from raw_features_snapshot
    df["invoice_amount"] = np.nan
    if "amount" in df.columns:
        df["invoice_amount"] = df["amount"]
    if "raw_features_snapshot" in df.columns:
        amts = df["raw_features_snapshot"].map(_parse_amount_from_snapshot)
        df.loc[df["invoice_amount"].isna(), "invoice_amount"] = amts[df["invoice_amount"].isna()]
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce")

    # EV lift: amount * (suggested_max_prob - baseline)
    df["ev_lift"] = np.nan
    if "invoice_amount" in df.columns and "suggested_max_prob" in df.columns:
        df["ev_lift"] = df["invoice_amount"].fillna(0) * (df["suggested_max_prob"].fillna(0) - global_rate)

    return df


def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error: weighted mean of |mean_pred - actual_rate| over bins."""
    if len(y_true) == 0 or n_bins < 1:
        return 0.0
    try:
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


def _mce(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """Maximum Calibration Error: max over bins of |mean_pred - actual_rate|."""
    if len(y_true) == 0 or n_bins < 1:
        return 0.0
    try:
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


def _ttr_mean(df: pd.DataFrame, valid_only: bool = True) -> float:
    """Mean TTR (hours_suggested_to_recovered) for recovered rows; valid_only = recovered_at >= inference_run_at."""
    col = "hours_suggested_to_recovered"
    if col not in df.columns:
        return float(np.nan)
    if valid_only:
        vals = df[df["recovered"].fillna(0).astype(bool)][col].dropna()
    else:
        vals = df[col].dropna()
    return float(vals.mean()) if len(vals) else float(np.nan)


def _decile_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Decile table: count, recovered, mean_predicted, actual_rate, calibration_error."""
    has_dec = "decile" in df.columns and df["decile"].notna().any()
    has_prob = "suggested_max_prob" in df.columns
    if not has_dec or not has_prob:
        return pd.DataFrame()
    sub = df[df["decile"].notna()].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    sub["recovered"] = sub["recovered"].fillna(0).astype(int)
    agg = sub.groupby("decile", as_index=True).agg(
        count=("suggested_max_prob", "count"),
        recovered=("recovered", "sum"),
        mean_predicted=("suggested_max_prob", "mean"),
    )
    agg["actual_rate"] = agg["recovered"] / agg["count"].replace(0, np.nan)
    agg["calibration_error"] = (agg["mean_predicted"] - agg["actual_rate"]).abs()
    return agg


def _decline_code_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Recovery rate and mean TTR per prev_decline_code."""
    if "prev_decline_code" not in df.columns:
        return pd.DataFrame()
    sub = df.copy()
    sub["recovered"] = sub["recovered"].fillna(0).astype(int)
    agg = sub.groupby("prev_decline_code", as_index=True).agg(
        n=("recovered", "count"),
        recovered=("recovered", "sum"),
    )
    agg["recovery_rate"] = agg["recovered"] / agg["n"].replace(0, np.nan)
    ttr = sub[sub["recovered"] == 1].groupby("prev_decline_code")["hours_suggested_to_recovered"].mean()
    agg["mean_ttr_h"] = ttr
    return agg


def _bootstrap_ci(
    df: pd.DataFrame,
    metric_fn,
    stratify_col: str = "recovered",
    n: int = BOOTSTRAP_N,
    alpha: float = BOOTSTRAP_ALPHA,
) -> tuple[float, float, float]:
    """Bootstrap (stratified by stratify_col) and return (point_estimate, lo, hi)."""
    if n <= 0 or len(df) == 0:
        point = metric_fn(df)
        return point, point, point
    rng = np.random.default_rng(42)
    if stratify_col not in df.columns:
        indices = np.arange(len(df))
        boots = []
        for _ in range(n):
            idx = rng.choice(indices, size=len(indices), replace=True)
            boots.append(metric_fn(df.iloc[idx]))
        boots = np.array(boots)
    else:
        boots = []
        for _ in range(n):
            resampled = df.groupby(stratify_col, group_keys=False).apply(
                lambda g: g.sample(n=len(g), replace=True, random_state=rng.integers(0, 2**31))
            )
            boots.append(metric_fn(resampled))
        boots = np.array(boots)
    point = metric_fn(df)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


def run_comparison(
    shadow_path: Path,
    actuals_df: pd.DataFrame | None,
    out_path: Path,
    rpath: Path,
    cal_plot_path: Path | None = None,
    gains_plot_path: Path | None = None,
    actual_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load shadow log, merge with actuals (or load from actual_csv_path), enrich,
    compute metrics, write comparison CSV, report, and optional plots.
    out_path, rpath, cal_plot_path, gains_plot_path: if not absolute, resolved with PROJECT_ROOT.
    """
    out_path = _resolve_path(Path(out_path), PROJECT_ROOT)
    rpath = _resolve_path(Path(rpath), PROJECT_ROOT)
    if cal_plot_path is not None:
        cal_plot_path = _resolve_path(Path(cal_plot_path), PROJECT_ROOT)
    if gains_plot_path is not None:
        gains_plot_path = _resolve_path(Path(gains_plot_path), PROJECT_ROOT)

    shadow = load_shadow_log(shadow_path)
    if actuals_df is None and actual_csv_path is not None:
        actuals_df = pd.read_csv(_resolve_path(Path(actual_csv_path), PROJECT_ROOT))
    if actuals_df is None:
        actuals_df = fetch_actual_outcomes()

    initial_merged, latest_merged = _build_merged(shadow, actuals_df)
    initial_enriched = _enrich_merged(initial_merged)
    latest_enriched = _enrich_merged(latest_merged)

    # Primary output: last-mile (latest)
    df = latest_enriched
    with_suggestion = df["suggested_max_prob"].notna()
    n_with_suggestion = with_suggestion.sum()
    global_rate = df["recovered"].fillna(0).mean()
    n_recovered = int(df["recovered"].fillna(0).sum())
    n_valid_ttr = df.loc[df["recovered"].fillna(0).astype(bool), "hours_suggested_to_recovered"].notna().sum()

    # Calibration / discriminative (rows with suggestion)
    y_true = df.loc[with_suggestion, "recovered"].fillna(0).astype(int).values
    y_pred = df.loc[with_suggestion, "suggested_max_prob"].values
    brier = brier_score_loss(y_true, y_pred) if len(y_true) else float(np.nan)
    ece = _ece(y_true, y_pred) if len(y_true) else float(np.nan)
    mce = _mce(y_true, y_pred) if len(y_true) else float(np.nan)
    auc_roc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float(np.nan)
    auc_pr = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else float(np.nan)

    # Dual-path
    initial_rate = initial_enriched["recovered"].fillna(0).mean()
    initial_ttr = _ttr_mean(initial_enriched, valid_only=True)
    latest_rate = latest_enriched["recovered"].fillna(0).mean()
    latest_ttr = _ttr_mean(latest_enriched, valid_only=True)

    # Bootstrap CIs
    def _recovery_rate(d):
        return d["recovered"].fillna(0).mean()

    def _ttr_mean_fn(d):
        return _ttr_mean(d, valid_only=True)

    def _top_decile_lift(d):
        dec = _decile_stats(d)
        if dec.empty or 9 not in dec.index:
            return np.nan
        top_rate = dec.loc[9, "actual_rate"] if 9 in dec.index else np.nan
        return top_rate / global_rate if global_rate and not np.isnan(top_rate) else np.nan

    def _ev_lift_sum(d):
        return d["ev_lift"].fillna(0).sum()

    rr_point, rr_lo, rr_hi = _bootstrap_ci(df, _recovery_rate, n=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA)
    ttr_point, ttr_lo, ttr_hi = _bootstrap_ci(
        df[df["recovered"].fillna(0).astype(bool)].dropna(subset=["hours_suggested_to_recovered"]),
        _ttr_mean_fn,
        stratify_col="recovered",
        n=BOOTSTRAP_N,
        alpha=BOOTSTRAP_ALPHA,
    )
    lift_point, lift_lo, lift_hi = _bootstrap_ci(df[with_suggestion], _top_decile_lift, n=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA)
    ev_point, ev_lo, ev_hi = _bootstrap_ci(df[with_suggestion], _ev_lift_sum, n=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA)

    decile_df = _decile_stats(df)
    top_decile_lift = np.nan
    if not decile_df.empty and 9 in decile_df.index:
        top_decile_lift = decile_df.loc[9, "actual_rate"] / global_rate if global_rate else np.nan

    high_threshold = df["high_prob_threshold_used"].iloc[0] if "high_prob_threshold_used" in df.columns else (BASELINE_LIFT_FACTOR * global_rate)
    n_high_missed = int(df["high_prob_missed_by_chargebee"].fillna(False).sum())

    # Temporal proximity
    recovered = df[df["recovered"].fillna(0).astype(bool)]
    pct_6h = (recovered["recovered_within_±6h"].fillna(False).sum() / len(recovered) * 100) if len(recovered) else 0
    pct_12h = (recovered["recovered_within_±12h"].fillna(False).sum() / len(recovered) * 100) if len(recovered) else 0
    pct_24h = (recovered["recovered_within_±24h"].fillna(False).sum() / len(recovered) * 100) if len(recovered) else 0
    n_6h = int(recovered["recovered_within_±6h"].fillna(False).sum())
    n_12h = int(recovered["recovered_within_±12h"].fillna(False).sum())
    n_24h = int(recovered["recovered_within_±24h"].fillna(False).sum())

    total_ev = df["ev_lift"].fillna(0).sum()
    inf_range = f"{df['inference_run_at'].min()} to {df['inference_run_at'].max()}" if "inference_run_at" in df.columns else ""

    # Write comparison CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Report
    rpath.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 70,
        "Shadow vs Actual (Chargebee) — Statistical Evaluation Report",
        "=" * 70,
        "",
        "0. EXECUTIVE SUMMARY",
        "-" * 50,
        f"Shadow compared {df[INVOICE_ID_COL].nunique()} unique invoices (last-mile merge) to Chargebee outcomes.",
        f"Overall recovery rate: {latest_rate:.2%} ({n_recovered} recoveries).",
        f"Model ranking: Top decile lift = {top_decile_lift:.2f}x global rate; AUC-ROC = {auc_roc:.3f}, AUC-PR = {auc_pr:.3f}.",
        f"Calibration: Brier = {brier:.4f}, ECE = {ece:.4f}, MCE = {mce:.4f} (lower is better).",
        f"TTR (recovered, valid): mean = {latest_ttr:.2f}h (negative = recovery before model slot).",
        f"High-value missed (Chargebee failed, model P > {high_threshold:.2%}): {n_high_missed:,}.",
        "",
        "1. DATA",
        "-" * 50,
        f"Total Invoices Shadowed (unique):         {df[INVOICE_ID_COL].nunique():,}",
        f"Total shadow rows:                         {len(shadow):,}",
        f"Rows with model suggestion (this merge):   {n_with_suggestion:,}",
        f"Actual outcomes matched:                  {len(df):,}",
        f"Shadow inference date range (last-mile):  {inf_range}",
        f"Recovered rows with valid TTR:            {n_valid_ttr} (recovered_at >= inference_run_at)",
        "",
        "2. CALIBRATION METRICS (rows with suggestion)",
        "-" * 50,
        f"Brier score (lower better):     {brier:.4f}",
        f"Expected Calibration Error:     {ece:.4f}",
        f"Max Calibration Error:         {mce:.4f}",
        "",
        "3. DISCRIMINATIVE METRICS (rows with suggestion)",
        "-" * 50,
        f"AUC-ROC:                        {auc_roc:.3f}",
        f"AUC-PR (average precision):    {auc_pr:.3f}",
        "",
        "4. DUAL-PATH EVALUATION (Initial Intent vs Last-Mile Accuracy)",
        "-" * 50,
        "Initial Intent (oldest record per invoice):",
        f"  Overall Recovery Rate:    {initial_rate:.2%}",
        f"  TTR Lift (recovered, valid only): {initial_ttr:.2f}h",
        "",
        "Last-Mile Accuracy (latest record per invoice):",
        f"  Overall Recovery Rate:    {latest_rate:.2%}",
        f"  TTR Lift (recovered, valid only): {latest_ttr:.2f}h",
        "",
        "5. RECOVERIES (Last-Mile)",
        "-" * 50,
        f"Total Recoveries:              {n_recovered}",
        f"Overall recovery rate:         {latest_rate:.2%}",
    ]
    if BOOTSTRAP_N > 0:
        lines.append(f"  95% CI (bootstrap n={BOOTSTRAP_N}):        [{rr_lo:.2%}, {rr_hi:.2%}]")
    lines.extend([
        "",
        "6. RELATIVE OPPORTUNITY GAP",
        "-" * 50,
        f"High-Value threshold (= {BASELINE_LIFT_FACTOR} x global rate): {high_threshold:.4f}",
        f"Chargebee FAILED but model P > threshold (high-value missed): {n_high_missed:,}",
        "",
        "7. TEMPORAL PROXIMITY (recovered within ±Nh of model #1 slot)",
        "-" * 50,
        f"  ±6h:  {n_6h} ({pct_6h:.1f}%)",
        f"  ±12h: {n_12h} ({pct_12h:.1f}%)",
        f"  ±24h: {n_24h} ({pct_24h:.1f}%)",
        "",
        "8. TTR LIFT (Recovered, valid TTR only: recovered_at >= inference_run_at)",
        "-" * 50,
        f"Mean (recovered_at - suggested_optimal_retry_at): {latest_ttr:.2f}h",
    ])
    if BOOTSTRAP_N > 0:
        lines.append(f"  95% CI (bootstrap n={BOOTSTRAP_N}): [{ttr_lo:.2f}h, {ttr_hi:.2f}h]")
    lines.extend([
        "",
        "9. DECILE CALIBRATION (Lift Curve)",
        "-" * 50,
    ])
    if not decile_df.empty:
        lines.append(decile_df.to_string())
        lines.append("")
        lines.append(f"Top Decile Lift Factor (Top Decile Success Rate / Global Rate): {top_decile_lift:.2f}")
        if BOOTSTRAP_N > 0:
            lines.append(f"  95% CI (bootstrap n={BOOTSTRAP_N}): [{lift_lo:.2f}, {lift_hi:.2f}]")
    lines.extend([
        "",
        "10. EXPECTED VALUE (EV) LIFT",
        "-" * 50,
        f"Total Potential Revenue Lift (sum of amount * (P - baseline)): {total_ev:,.2f}",
    ])
    if BOOTSTRAP_N > 0:
        lines.append(f"  95% CI (bootstrap n={BOOTSTRAP_N}): [{ev_lo:,.2f}, {ev_hi:,.2f}]")
    lines.extend([
        "",
        "11. TRANSITION MATRIX BY DECLINE CODE",
        "-" * 50,
    ])
    decline_df = _decline_code_matrix(df)
    if not decline_df.empty:
        lines.append(decline_df.to_string())
    lines.append("")
    lines.append("=" * 70)

    with open(rpath, "w") as f:
        f.write("\n".join(lines))

    if cal_plot_path:
        _save_calibration_plot(df, cal_plot_path)
    if gains_plot_path:
        _save_gains_plot(df, gains_plot_path)

    return df


def _save_calibration_plot(merged_df: pd.DataFrame, plot_path: Path) -> None:
    """Save reliability diagram (deciles): mean predicted vs actual rate. plot_path resolved with PROJECT_ROOT if not absolute."""
    plot_path = _resolve_path(Path(plot_path), PROJECT_ROOT)
    dec = _decile_stats(merged_df)
    if dec.empty:
        return
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = np.arange(len(dec))
    width = 0.35
    ax.bar(x - width / 2, dec["mean_predicted"], width, label="Mean predicted", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, dec["actual_rate"], width, label="Actual rate", color="coral", alpha=0.8)
    ax.plot([0, len(dec) - 1], [merged_df["recovered"].fillna(0).mean()] * 2, "k--", alpha=0.7, label="Global rate")
    ax.set_xlabel("Decile (0=lowest P, 9=highest P)")
    ax.set_ylabel("Rate")
    ax.set_title("Calibration: Predicted vs Actual Recovery Rate by Decile")
    ax.set_xticks(x)
    ax.set_xticklabels(dec.index.astype(int))
    ax.legend()
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_gains_plot(merged_df: pd.DataFrame, plot_path: Path) -> None:
    """Cumulative gains: x = % of invoices (sorted by prob high to low), y = % of total recoveries captured. plot_path resolved with PROJECT_ROOT if not absolute."""
    plot_path = _resolve_path(Path(plot_path), PROJECT_ROOT)
    sub = merged_df[merged_df["suggested_max_prob"].notna()].copy()
    if len(sub) == 0:
        return
    sub = sub.sort_values("suggested_max_prob", ascending=False).reset_index(drop=True)
    sub["pct_invoices"] = (np.arange(len(sub)) + 1) / len(sub) * 100
    sub["recovered"] = sub["recovered"].fillna(0).astype(int)
    total_recovered = sub["recovered"].sum()
    if total_recovered == 0:
        return
    sub["cumul_recovered"] = sub["recovered"].cumsum()
    sub["pct_recoveries"] = sub["cumul_recovered"] / total_recovered * 100
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(sub["pct_invoices"], sub["pct_recoveries"], "b-", label="Model")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.6, label="Random")
    ax.set_xlabel("% of Invoices (sorted by model probability, high to low)")
    ax.set_ylabel("% of Total Recoveries Captured")
    ax.set_title("Cumulative Gains")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shadow monitoring output with actual Chargebee outcomes.",
    )
    parser.add_argument(
        "--shadow",
        type=str,
        default=os.environ.get("SHADOW_LOG_PATH", str(SHADOW_LOG_PATH)),
        help="Path to shadow log CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("COMPARISON_OUTPUT_PATH", str(COMPARISON_OUTPUT_PATH)),
        help="Path to comparison output CSV",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=os.environ.get("REPORT_OUTPUT_PATH", str(REPORT_OUTPUT_PATH)),
        help="Path to text report",
    )
    parser.add_argument(
        "--cal-plot",
        type=str,
        default=os.environ.get("CALIBRATION_PLOT_PATH", str(CALIBRATION_PLOT_PATH)),
        help="Path to calibration (reliability) plot PNG",
    )
    parser.add_argument(
        "--gains-plot",
        type=str,
        default=os.environ.get("GAINS_PLOT_PATH", str(GAINS_PLOT_PATH)),
        help="Path to cumulative gains plot PNG",
    )
    parser.add_argument(
        "--actual-csv",
        type=str,
        default=None,
        help="If set, load actual outcomes from this CSV instead of BigQuery",
    )
    args = parser.parse_args()

    shadow_path = Path(args.shadow)
    out_path = Path(args.output)
    rpath = Path(args.report)
    cal_plot_path = Path(args.cal_plot)
    gains_plot_path = Path(args.gains_plot)
    actual_csv = Path(args.actual_csv) if args.actual_csv else None

    actuals_df = None
    if actual_csv is not None:
        actuals_df = pd.read_csv(_resolve_path(actual_csv, PROJECT_ROOT))

    run_comparison(
        shadow_path=shadow_path,
        actuals_df=actuals_df,
        out_path=out_path,
        rpath=rpath,
        cal_plot_path=cal_plot_path,
        gains_plot_path=gains_plot_path,
        actual_csv_path=actual_csv,
    )
    print(f"Comparison CSV: {_resolve_path(out_path, PROJECT_ROOT)}")
    print(f"Report: {_resolve_path(rpath, PROJECT_ROOT)}")
    print(f"Calibration plot: {_resolve_path(cal_plot_path, PROJECT_ROOT)}")
    print(f"Gains plot: {_resolve_path(gains_plot_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
