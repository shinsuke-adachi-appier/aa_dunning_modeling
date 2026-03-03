"""
Compare shadow monitoring output (model's suggested optimal retry) with actual Chargebee outcomes.

Implements a robust statistical evaluation:
- Dual-path evaluation: Initial Intent (oldest record per invoice) vs Last-Mile Accuracy (latest record).
- Quantile-based calibration (deciles) and Top Decile Lift Factor.
- Relative opportunity gap: HIGH_PROB = BASELINE_LIFT_FACTOR * global_recovery_rate.
- Expected Value (EV) Lift: sum of invoice_amount * (suggested_max_prob - baseline_rate).
- Transition matrix by decline code: prev_decline_code (and prev_card_status) from actuals when available
  (BQ "previous" attempt: last failure before success for recovered, latest attempt for unrecovered), else from shadow raw_features_snapshot.
- Reliability: all timestamps normalized to UTC; TTR excludes rows where recovered_at < inference_run_at.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTCOMES_START_DATE = "2026-02-15"  # Timeline of attempts starting from this date (UTC)
TOP1_WINDOW_HOURS = 6  # +/- hours for "Top-1 Performance"
NUM_DECILES = 10
BASELINE_LIFT_FACTOR = 3.0  # High-value opportunity: predicted P >= this factor * global_recovery_rate

SHADOW_LOG_PATH = os.environ.get("SHADOW_LOG_PATH", _SCRIPT_DIR / "artifacts" / "shadow_log.csv")
COMPARISON_OUTPUT_PATH = os.environ.get("COMPARISON_OUTPUT_PATH", _SCRIPT_DIR / "artifacts" / "shadow_vs_actual_comparison.csv")
REPORT_OUTPUT_PATH = os.environ.get("REPORT_OUTPUT_PATH", _SCRIPT_DIR / "artifacts" / "shadow_vs_actual_report.txt")
CALIBRATION_PLOT_PATH = os.environ.get("CALIBRATION_PLOT_PATH", _SCRIPT_DIR / "artifacts" / "calibration_plot.png")
GAINS_PLOT_PATH = os.environ.get("GAINS_PLOT_PATH", _SCRIPT_DIR / "artifacts" / "cumulative_gains_plot.png")

BQ_PROJECT = "aa-datamart"
BQ_LOCATION = "europe-west1"
BQ_TABLE = "aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback"


def load_shadow_log(path: Optional[str] = None) -> pd.DataFrame:
    """Load shadow log CSV (artifacts/shadow_log.csv). Uses invoice_id (same as linked_invoice_id).
    Expects columns: invoice_id, inference_run_at, suggested_optimal_retry_at, suggested_max_prob (and optional metadata).
    Coerces inference_run_at and suggested_optimal_retry_at to datetime, suggested_max_prob to numeric.
    """
    p = Path(path or SHADOW_LOG_PATH)
    if not p.is_absolute():
        p = _SCRIPT_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Shadow log not found: {p}. Run shadow_monitoring_20260206.py first.")
    df = pd.read_csv(p)
    for col in ("inference_run_at", "suggested_optimal_retry_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "suggested_max_prob" in df.columns:
        df["suggested_max_prob"] = pd.to_numeric(df["suggested_max_prob"], errors="coerce")
    return df


def fetch_actual_outcomes() -> pd.DataFrame:
    """
    Pull from BigQuery: per linked_invoice_id, recovered, recovered_at, last_attempt_at,
    and the "previous" attempt's Decline_code_norm / card_status as prev_decline_code / prev_card_status.
    The "previous" attempt is the one that is the context for the next prediction:
    - Recovered: the last attempt before the first success (the failure that preceded recovery).
    - Unrecovered: the latest attempt (current failure).
    So recovered invoices get the decline code of the failure before the success, not UNKNOWN.
    All timestamps in UTC for consistency.
    """
    from txn_pipeline import get_bq_client, load_bigquery_table

    client = get_bq_client(project=BQ_PROJECT, location=BQ_LOCATION)

    # "Previous" attempt = last failure before first success (recovered) or latest attempt (unrecovered).
    # Table must have Decline_code_norm and card_status (or adjust the base CTE to match your schema).
    query = f"""
    WITH base AS (
      SELECT
        linked_invoice_id,
        updated_at,
        status,
        COALESCE(CAST(Decline_code_norm AS STRING), 'UNKNOWN') AS Decline_code_norm,
        COALESCE(CAST(card_status AS STRING), 'UNKNOWN') AS card_status
      FROM `{BQ_TABLE}`
      WHERE updated_at >= TIMESTAMP('{OUTCOMES_START_DATE}')
    ),
    agg AS (
      SELECT
        linked_invoice_id AS invoice_id,
        MAX(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS recovered,
        MIN(CASE WHEN status = 'success' THEN updated_at END) AS recovered_at,
        MAX(updated_at) AS last_attempt_at
      FROM base
      GROUP BY linked_invoice_id
    ),
    prev_attempt AS (
      SELECT
        b.linked_invoice_id,
        b.Decline_code_norm AS prev_decline_code,
        b.card_status AS prev_card_status,
        ROW_NUMBER() OVER (
          PARTITION BY b.linked_invoice_id
          ORDER BY
            CASE
              WHEN a.recovered_at IS NOT NULL AND b.updated_at < a.recovered_at THEN b.updated_at
              WHEN a.recovered_at IS NULL THEN b.updated_at
              ELSE TIMESTAMP('1970-01-01')
            END DESC
        ) AS rn
      FROM base b
      JOIN agg a ON b.linked_invoice_id = a.invoice_id
      WHERE (a.recovered_at IS NULL) OR (b.updated_at < a.recovered_at)
    )
    SELECT
      a.invoice_id,
      a.recovered,
      a.recovered_at,
      a.last_attempt_at,
      p.prev_decline_code,
      p.prev_card_status
    FROM agg a
    LEFT JOIN (SELECT linked_invoice_id, prev_decline_code, prev_card_status FROM prev_attempt WHERE rn = 1) p
      ON a.invoice_id = p.linked_invoice_id
    """
    df = load_bigquery_table(client, query)
    if "recovered_at" in df.columns:
        df["recovered_at"] = pd.to_datetime(df["recovered_at"], utc=True).dt.tz_localize(None)
    if "last_attempt_at" in df.columns:
        df["last_attempt_at"] = pd.to_datetime(df["last_attempt_at"], utc=True).dt.tz_localize(None)
    return df


def _to_utc_naive(s: pd.Series) -> pd.Series:
    """Force series to UTC then strip tz for consistent subtraction."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        return s
    if s.dt.tz is not None:
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def _parse_prev_decline_code(raw_features_snapshot: pd.Series) -> pd.Series:
    """Extract prev_decline_code from raw_features_snapshot JSON. Returns 'UNKNOWN' on parse error."""
    return _parse_json_feature(raw_features_snapshot, "prev_decline_code")


def _parse_json_feature(raw_features_snapshot: pd.Series, key: str) -> pd.Series:
    """Extract a string feature from raw_features_snapshot JSON. Returns 'UNKNOWN' on parse error."""
    def get_val(x):
        if pd.isna(x) or not isinstance(x, str):
            return "UNKNOWN"
        try:
            d = json.loads(x)
            return str(d.get(key, "UNKNOWN"))
        except (json.JSONDecodeError, TypeError):
            return "UNKNOWN"
    return raw_features_snapshot.map(get_val)


def _ensure_invoice_amount(merged: pd.DataFrame) -> pd.DataFrame:
    """Add invoice_amount from actuals or derive from raw_features_snapshot (log_charge_amount -> expm1)."""
    if "invoice_amount" in merged.columns:
        return merged
    if "amount" in merged.columns:
        merged["invoice_amount"] = pd.to_numeric(merged["amount"], errors="coerce")
        return merged
    # Derive from log_charge_amount in raw_features_snapshot
    def get_log_amount(x):
        if pd.isna(x) or not isinstance(x, str):
            return np.nan
        try:
            d = json.loads(x)
            return float(d.get("log_charge_amount", np.nan))
        except (json.JSONDecodeError, TypeError, ValueError):
            return np.nan
    log_amt = merged["raw_features_snapshot"].map(get_log_amount)
    merged["invoice_amount"] = np.expm1(log_amt)
    return merged


def _build_merged(
    shadow: pd.DataFrame,
    actual_df: pd.DataFrame,
    keep: str,
) -> pd.DataFrame:
    """Build merged DF: shadow deduplicated by invoice_id (keep='first' or 'last'), merged with actuals."""
    sh = shadow.sort_values("inference_run_at")
    if keep == "first":
        sh = sh.drop_duplicates(subset=["invoice_id"], keep="first").reset_index(drop=True)
    else:
        sh = sh.drop_duplicates(subset=["invoice_id"], keep="last").reset_index(drop=True)
    merged = sh.merge(actual_df, on="invoice_id", how="left")
    return merged


def _enrich_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """Add recovered, has_suggestion, UTC timestamps, valid TTR mask, high_prob threshold, deciles, EV, decline code."""
    if "recovered" in merged.columns:
        merged["recovered"] = merged["recovered"].fillna(0).astype(int)
    has_suggestion = merged["suggested_optimal_retry_at"].notna()

    for col in ("inference_run_at", "recovered_at", "suggested_optimal_retry_at", "last_attempt_at"):
        if col in merged.columns and pd.api.types.is_datetime64_any_dtype(merged[col]):
            merged[col] = _to_utc_naive(merged[col])

    # TTR: exclude "future peeking" — only where recovered_at >= inference_run_at
    if "recovered_at" in merged.columns and "suggested_optimal_retry_at" in merged.columns and "inference_run_at" in merged.columns:
        merged["hours_suggested_to_recovered"] = (
            merged["recovered_at"] - merged["suggested_optimal_retry_at"]
        ).dt.total_seconds() / 3600.0
        valid_ttr = merged["recovered_at"] >= merged["inference_run_at"]
        merged.loc[~valid_ttr, "hours_suggested_to_recovered"] = np.nan
    else:
        merged["hours_suggested_to_recovered"] = np.nan

    global_rate = merged["recovered"].mean() if "recovered" in merged.columns else 0.0
    high_prob_threshold = min(1.0, BASELINE_LIFT_FACTOR * global_rate) if global_rate > 0 else 0.0
    chargebee_failed = merged["recovered"] == 0 if "recovered" in merged.columns else pd.Series(False, index=merged.index)
    merged["high_prob_missed_by_chargebee"] = (
        chargebee_failed & has_suggestion & (merged["suggested_max_prob"] > high_prob_threshold)
    )
    merged["high_prob_threshold_used"] = high_prob_threshold

    # Deciles (equal-sized bins by suggested_max_prob)
    prob_valid = merged["suggested_max_prob"].notna() & has_suggestion
    merged["decile"] = pd.NA
    if prob_valid.any():
        try:
            merged.loc[prob_valid, "decile"] = pd.qcut(
                merged.loc[prob_valid, "suggested_max_prob"],
                q=NUM_DECILES,
                duplicates="drop",
                labels=False,
            )
        except Exception:
            pass

    # Top-1 window
    hrs = merged["hours_suggested_to_recovered"]
    merged["recovered_within_top1_window"] = (
        merged["recovered"].eq(1) & hrs.notna() & (hrs.abs() <= TOP1_WINDOW_HOURS)
    ) if "recovered" in merged.columns else False

    merged = _ensure_invoice_amount(merged)
    # prev_decline_code / prev_card_status: prefer actuals (BQ "previous" attempt) when available, else shadow raw_features_snapshot
    if "raw_features_snapshot" in merged.columns:
        from_shadow_decline = _parse_prev_decline_code(merged["raw_features_snapshot"])
        from_shadow_card = _parse_json_feature(merged["raw_features_snapshot"], "prev_card_status")
    else:
        from_shadow_decline = pd.Series("UNKNOWN", index=merged.index)
        from_shadow_card = pd.Series("UNKNOWN", index=merged.index)
    if "prev_decline_code" in merged.columns:
        actuals_decline = merged["prev_decline_code"].astype(str).str.strip().replace("", np.nan)
        merged["prev_decline_code"] = actuals_decline.fillna(from_shadow_decline).fillna("UNKNOWN")
    else:
        merged["prev_decline_code"] = from_shadow_decline.fillna("UNKNOWN")
    if "prev_card_status" in merged.columns:
        actuals_card = merged["prev_card_status"].astype(str).str.strip().replace("", np.nan)
        merged["prev_card_status"] = actuals_card.fillna(from_shadow_card).fillna("UNKNOWN")
    else:
        merged["prev_card_status"] = from_shadow_card.fillna("UNKNOWN")

    baseline = merged["recovered"].mean() if "recovered" in merged.columns else 0.0
    merged["ev_lift"] = np.nan
    if "invoice_amount" in merged.columns and "suggested_max_prob" in merged.columns:
        amt = merged["invoice_amount"].fillna(0)
        prob = merged["suggested_max_prob"].fillna(0)
        merged["ev_lift"] = np.where(has_suggestion, amt * (prob - baseline), np.nan)

    return merged


def run_comparison(
    shadow_path: Optional[str] = None,
    actual_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    calibration_plot_path: Optional[str] = None,
    gains_plot_path: Optional[str] = None,
    dedupe_shadow_latest_per_invoice: bool = True,
) -> pd.DataFrame:
    """
    Dual-path evaluation: build Initial Intent (oldest per invoice) and Last-Mile (latest per invoice),
    compare recovery rate and TTR for both, then report decile calibration, EV lift, and decline-code matrix.
    Returns the Last-Mile merged DF (or Initial if --no-dedupe is used elsewhere).
    """
    shadow = load_shadow_log(shadow_path)
    if actual_df is None:
        print("Fetching actual outcomes from BigQuery...")
        actual_df = fetch_actual_outcomes()
    else:
        actual_df = actual_df.copy()
    if "linked_invoice_id" in actual_df.columns and "invoice_id" not in actual_df.columns:
        actual_df = actual_df.rename(columns={"linked_invoice_id": "invoice_id"})
    if "invoice_id" not in actual_df.columns:
        raise ValueError("Actual outcomes must have invoice_id (or linked_invoice_id).")
    actual_df = actual_df.drop_duplicates(subset=["invoice_id"], keep="last")
    actual_df["invoice_id"] = actual_df["invoice_id"].astype(str)
    shadow["invoice_id"] = shadow["invoice_id"].astype(str)

    # Dual path: Initial (oldest) and Latest (newest) per invoice
    merged_initial = _build_merged(shadow, actual_df, keep="first")
    merged_latest = _build_merged(shadow, actual_df, keep="last")
    merged_initial = _enrich_merged(merged_initial)
    merged_latest = _enrich_merged(merged_latest)

    # Use Latest for primary output and reporting (configurable behavior is preserved via which merge we report)
    merged = merged_latest
    has_suggestion = merged["suggested_optimal_retry_at"].notna()
    global_rate = merged["recovered"].mean()
    high_prob_threshold = merged["high_prob_threshold_used"].iloc[0] if len(merged) else 0.0

    # ---- Report ----
    report_lines = [
        "=" * 70,
        "Shadow vs Actual (Chargebee) — Statistical Evaluation Report",
        "=" * 70,
        "",
        "1. DATA",
        "-" * 50,
        f"Total Invoices Shadowed (unique):         {shadow['invoice_id'].nunique():,}",
        f"Total shadow rows:                         {len(shadow):,}",
        f"Rows with model suggestion (this merge):   {has_suggestion.sum():,}",
        f"Actual outcomes matched:                  {merged['invoice_id'].isin(actual_df['invoice_id']).sum():,}",
        "",
        "2. DUAL-PATH EVALUATION (Initial Intent vs Last-Mile Accuracy)",
        "-" * 50,
        f"Initial Intent (oldest record per invoice):",
        f"  Overall Recovery Rate:    {merged_initial['recovered'].mean() * 100:.2f}%",
        f"  TTR Lift (recovered, valid only): {_ttr_mean(merged_initial):.2f}h",
        "",
        f"Last-Mile Accuracy (latest record per invoice):",
        f"  Overall Recovery Rate:    {merged_latest['recovered'].mean() * 100:.2f}%",
        f"  TTR Lift (recovered, valid only): {_ttr_mean(merged_latest):.2f}h",
        "",
    ]

    # Recoveries
    total_recovered = merged["recovered"].sum()
    report_lines.extend([
        "3. RECOVERIES (Last-Mile)",
        "-" * 50,
        f"Total Recoveries:              {int(total_recovered):,}",
        f"Overall recovery rate:         {global_rate * 100:.2f}%",
        "",
    ])

    # Relative opportunity gap
    high_prob_missed = merged["high_prob_missed_by_chargebee"].sum()
    report_lines.extend([
        "4. RELATIVE OPPORTUNITY GAP",
        "-" * 50,
        f"High-Value threshold (= {BASELINE_LIFT_FACTOR} x global rate): {high_prob_threshold:.4f}",
        f"Chargebee FAILED but model P > threshold (high-value missed): {int(high_prob_missed):,}",
        "",
    ])

    # Top-1
    if merged["recovered"].eq(1).any():
        within_top1 = merged["recovered_within_top1_window"].sum()
        report_lines.extend([
            "5. TOP-1 PERFORMANCE",
            "-" * 50,
            f"Recoveries within +/- {TOP1_WINDOW_HOURS}h of model #1 slot: {int(within_top1):,} ({100.0 * within_top1 / total_recovered:.1f}%)",
            "",
        ])

    # TTR (valid only)
    ttr_mean = _ttr_mean(merged)
    report_lines.extend([
        "6. TTR LIFT (Recovered, valid TTR only: recovered_at >= inference_run_at)",
        "-" * 50,
        f"Mean (recovered_at - suggested_optimal_retry_at): {ttr_mean:.2f}h",
        "",
    ])

    # Decile calibration & Top Decile Lift
    decile_stats, top_decile_lift = _decile_stats(merged)
    if decile_stats is not None:
        report_lines.extend([
            "7. DECILE CALIBRATION (Lift Curve)",
            "-" * 50,
            decile_stats.to_string(),
            "",
        ])
        if top_decile_lift is not None:
            report_lines.append(f"Top Decile Lift Factor (Top Decile Success Rate / Global Rate): {top_decile_lift:.2f}")
            report_lines.append("")

    # EV Lift
    total_ev = merged["ev_lift"].sum()
    if pd.notna(total_ev):
        report_lines.extend([
            "8. EXPECTED VALUE (EV) LIFT",
            "-" * 50,
            f"Total Potential Revenue Lift (sum of amount * (P - baseline)): {total_ev:,.2f}",
            "",
        ])

    # Transition matrix by decline code
    decline_matrix = _decline_code_matrix(merged)
    if decline_matrix is not None:
        report_lines.extend([
            "9. TRANSITION MATRIX BY DECLINE CODE",
            "-" * 50,
            decline_matrix,
            "",
        ])

    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save outputs
    out_path = Path(output_path or COMPARISON_OUTPUT_PATH)
    if not out_path.is_absolute():
        out_path = _SCRIPT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Comparison table saved: {out_path}")

    rpath = Path(report_path or REPORT_OUTPUT_PATH)
    if not rpath.is_absolute():
        rpath = _SCRIPT_DIR / rpath
    rpath.write_text(report_text, encoding="utf-8")
    print(f"Report saved: {rpath}")

    # Plots: Reliability diagram (deciles) + Cumulative gains
    if has_suggestion.any() and "recovered" in merged.columns:
        _save_calibration_plot(merged, calibration_plot_path)
        _save_gains_plot(merged, gains_plot_path)

    return merged


def _ttr_mean(merged: pd.DataFrame) -> float:
    """Mean TTR in hours for recovered rows with valid TTR (recovered_at >= inference_run_at)."""
    if "recovered" not in merged.columns or "hours_suggested_to_recovered" not in merged.columns:
        return float("nan")
    rec = merged[merged["recovered"] == 1]["hours_suggested_to_recovered"].dropna()
    return float(rec.mean()) if len(rec) > 0 else float("nan")


def _decile_stats(merged: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Actual recovery rate per decile; Top Decile Lift Factor."""
    if "decile" not in merged.columns or merged["decile"].isna().all():
        return None, None
    valid = merged["decile"].notna()
    if not valid.any():
        return None, None
    cal = merged[valid].groupby("decile", observed=True).agg(
        count=("recovered", "count"),
        recovered=("recovered", "sum"),
        mean_predicted=("suggested_max_prob", "mean"),
    )
    cal["actual_rate"] = cal["recovered"] / cal["count"]
    global_rate = merged["recovered"].mean()
    top_decile_lift = None
    if len(cal) > 0 and global_rate > 0:
        top_decile = cal.index.max()
        top_rate = cal.loc[top_decile, "actual_rate"]
        top_decile_lift = top_rate / global_rate
    return cal, top_decile_lift


def _decline_code_matrix(merged: pd.DataFrame) -> Optional[str]:
    """TTR and recovery rate by prev_decline_code."""
    if "prev_decline_code" not in merged.columns:
        return None
    agg = merged.groupby("prev_decline_code", observed=True).agg(
        n=("recovered", "count"),
        recovered=("recovered", "sum"),
    )
    agg["recovery_rate"] = agg["recovered"] / agg["n"]
    if "hours_suggested_to_recovered" in merged.columns:
        ttr = merged[merged["recovered"] == 1].groupby("prev_decline_code", observed=True)["hours_suggested_to_recovered"].mean()
        agg["mean_ttr_h"] = agg.index.map(ttr)
    return agg.to_string()


def _save_calibration_plot(merged: pd.DataFrame, path: Optional[str] = None) -> None:
    """Reliability diagram: deciles, mean predicted vs actual rate."""
    valid = merged["decile"].notna()
    if not valid.any():
        return
    cal = merged[valid].groupby("decile", observed=True).agg(
        mean_predicted=("suggested_max_prob", "mean"),
        actual_rate=("recovered", "mean"),
        count=("recovered", "count"),
    ).reset_index()
    if len(cal) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.bar(cal["decile"].astype(float) - 0.4, cal["mean_predicted"], width=0.4, label="Mean predicted", alpha=0.7)
    ax.bar(cal["decile"].astype(float), cal["actual_rate"], width=0.4, label="Actual rate", alpha=0.7)
    ax.plot([0, NUM_DECILES - 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Decile (0=lowest P, 9=highest P)")
    ax.set_ylabel("Rate")
    ax.set_title("Calibration (Reliability Diagram by Decile)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plot_path = Path(path or CALIBRATION_PLOT_PATH)
    if not plot_path.is_absolute():
        plot_path = _SCRIPT_DIR / plot_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved: {plot_path}")


def _save_gains_plot(merged: pd.DataFrame, path: Optional[str] = None) -> None:
    """Cumulative gains: X = % of invoices (sorted by model prob), Y = % of total recoveries captured."""
    with_suggestion = merged[merged["suggested_optimal_retry_at"].notna()].copy()
    if with_suggestion.empty or "recovered" not in with_suggestion.columns:
        return
    with_suggestion = with_suggestion.sort_values("suggested_max_prob", ascending=False).reset_index(drop=True)
    n = len(with_suggestion)
    total_recoveries = with_suggestion["recovered"].sum()
    if total_recoveries == 0:
        return
    pct_invoices = np.arange(1, n + 1) / n * 100
    cum_recoveries = with_suggestion["recovered"].cumsum()
    pct_recoveries = cum_recoveries / total_recoveries * 100
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(pct_invoices, pct_recoveries, "b-", label="Model (by P)")
    ax.plot([0, 100], [0, 100], "k--", label="Random")
    ax.set_xlabel("% of Invoices (sorted by predicted P, high to low)")
    ax.set_ylabel("% of Total Recoveries Captured")
    ax.set_title("Cumulative Gains Chart")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plot_path = Path(path or GAINS_PLOT_PATH)
    if not plot_path.is_absolute():
        plot_path = _SCRIPT_DIR / plot_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cumulative gains plot saved: {plot_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare shadow log with actual Chargebee outcomes.")
    parser.add_argument("--shadow", default=None, help="Path to shadow log CSV")
    parser.add_argument("--actual-csv", default=None, help="Path to actual outcomes CSV (skip BigQuery)")
    parser.add_argument("--output", default=None, help="Output comparison CSV path")
    parser.add_argument("--report", default=None, help="Output report path")
    parser.add_argument("--cal-plot", default=None, help="Output calibration (reliability) plot path")
    parser.add_argument("--gains-plot", default=None, help="Output cumulative gains plot path")
    parser.add_argument("--no-dedupe", action="store_true", help="Unused; dual-path (initial + latest) is always run")
    parser.add_argument("--no-query", action="store_true", help="Skip BigQuery; load only shadow log")
    args = parser.parse_args()

    if args.no_query:
        shadow = load_shadow_log(args.shadow)
        print(f"Loaded shadow log: {len(shadow)} rows.")
        return

    actual_df = None
    if args.actual_csv:
        actual_df = pd.read_csv(args.actual_csv)
        for col in ("recovered_at", "last_attempt_at"):
            if col in actual_df.columns:
                actual_df[col] = pd.to_datetime(actual_df[col], errors="coerce")
        print(f"Loaded actual outcomes from CSV: {len(actual_df)} rows")

    run_comparison(
        shadow_path=args.shadow,
        actual_df=actual_df,
        output_path=args.output,
        report_path=args.report,
        calibration_plot_path=args.cal_plot,
        gains_plot_path=args.gains_plot,
    )


if __name__ == "__main__":
    main()
