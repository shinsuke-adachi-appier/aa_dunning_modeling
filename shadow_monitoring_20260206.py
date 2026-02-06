"""
Shadow monitoring: log the model's "best choice" (optimal retry time) for every invoice
currently in a dunning cycle for future comparison against Chargebee's actual performance.

Uses the same inference logic as dunning_modeling.ipynb:
- Calibrated CatBoost model (model_temporal_calibrated)
- generate_candidate_slots / optimal_slot_for_invoice from ranking_backtest.py
- Point-in-time: base_timestamp = most recent payment_failed; slots 24h–120h after base.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow importing ranking_backtest when run from repo root (e.g. python modeling/shadow_monitoring_20260206.py).
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

# ---------------------------------------------------------------------------
# 1. Data connection & filter (framework — fill BigQuery later)
# ---------------------------------------------------------------------------

# Training window cutoff: exclude data after this date to prevent leakage.
TRAINING_CUTOFF_DATE = "2026-01-31"

def fetch_active_dunning_invoices() -> pd.DataFrame:
    """
    Connect to BigQuery and pull active dunning invoices.

    Filter:
    - Dunning status = 'active' OR last failure within the last 7 days.
    - Do not use data after TRAINING_CUTOFF_DATE for training; this script
      only runs inference on current state.

    Returns:
        DataFrame with one row per invoice (or per last attempt). Required columns
        (adjust to match your BigQuery schema):
        - linked_invoice_id (invoice_id)
        - customer_id
        - updated_at (or last_failure_at): timestamp of most recent payment_failed
        - first_attempt_at
        - prev_decline_code, billing_country, gateway, funding_type_norm, card_brand,
          prev_card_status, Domain_category, invoice_attempt_no, amount
        - localized_time or timezone + updated_at for local hour/day (see build_invoice_row)
    """
    # TODO: Implement BigQuery connection and query.
    # Example placeholder:
    # from google.cloud import bigquery
    # client = bigquery.Client()
    # query = """
    #   SELECT ...
    #   FROM ...
    #   WHERE (dunning_status = 'active' OR last_failure_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY))
    #   AND last_failure_at < TIMESTAMP('{cutoff}')
    # """.format(cutoff=TRAINING_CUTOFF_DATE)
    # return client.query(query).to_dataframe()

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Feature row builder (point-in-time, aligned with notebook)
# ---------------------------------------------------------------------------

# Model feature set (must match dunning_modeling.ipynb keep_cols minus target/group).
MODEL_FEATURE_NAMES = [
    "prev_decline_code", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_sin", "day_cos",
    "dist_to_payday", "log_charge_amount", "is_debit", "amt_per_attempt",
    "time_since_prev_attempt", "cumulative_delay_hours",
    "billing_country", "gateway", "funding_type_norm", "card_brand", "prev_card_status",
    "Domain_category", "invoice_attempt_no",
]

CAT_FEATURES = [
    "prev_decline_code", "billing_country", "gateway",
    "funding_type_norm", "card_brand", "Domain_category",
]


def _safe_str(s: Any) -> str:
    """Coerce to str for categoricals; avoid high-cardinality leakage."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "UNKNOWN"
    return str(s).strip() or "UNKNOWN"


def build_invoice_row(
    row: pd.Series,
    base_timestamp: pd.Timestamp,
    first_attempt_at: pd.Timestamp,
) -> pd.Series:
    """
    Build one feature row for inference from a raw invoice/attempt row.
    Temporal features are set at base_timestamp; generate_candidate_slots will
    overwrite them per slot (time_since_prev_attempt, cumulative_delay_hours, hour_*, dow_*, day_*, dist_to_payday).
    """
    base = pd.Timestamp(base_timestamp)
    first = pd.Timestamp(first_attempt_at)

    # Temporal at base (will be overwritten per slot; needed for column presence)
    hour = base.hour + base.minute / 60.0 + base.second / 3600.0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow = base.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    day_of_month = base.day
    max_days = base.days_in_month
    day_sin = np.sin(2 * np.pi * (day_of_month - 1) / max_days)
    day_cos = np.cos(2 * np.pi * (day_of_month - 1) / max_days)
    dist_to_payday = min(abs(day_of_month - 1), abs(day_of_month - 15), abs(day_of_month - 30))
    time_since_prev = 0.0  # at base
    cumulative_delay = (base - first).total_seconds() / 3600.0

    amount = float(row.get("amount", 0) or 0)
    attempt_no = int(row.get("invoice_attempt_no", 0) or 0)
    log_charge_amount = np.log1p(amount)
    is_debit = 1 if (str(row.get("funding_type_norm", "") or "").lower() == "debit") else 0
    amt_per_attempt = amount / (attempt_no + 1) if (attempt_no + 1) != 0 else 0.0

    out = pd.Series({
        "prev_decline_code": _safe_str(row.get("prev_decline_code")),
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "day_sin": day_sin, "day_cos": day_cos,
        "dist_to_payday": dist_to_payday,
        "log_charge_amount": log_charge_amount,
        "is_debit": is_debit,
        "amt_per_attempt": amt_per_attempt,
        "time_since_prev_attempt": time_since_prev,
        "cumulative_delay_hours": cumulative_delay,
        "billing_country": _safe_str(row.get("billing_country")),
        "gateway": _safe_str(row.get("gateway")),
        "funding_type_norm": _safe_str(row.get("funding_type_norm")),
        "card_brand": _safe_str(row.get("card_brand")),
        "prev_card_status": _safe_str(row.get("prev_card_status")),
        "Domain_category": _safe_str(row.get("Domain_category")),
        "invoice_attempt_no": attempt_no,
    })
    return out.reindex(MODEL_FEATURE_NAMES).fillna(0)


# ---------------------------------------------------------------------------
# 3. Inference: load model, score slots, get optimal retry
# ---------------------------------------------------------------------------

def load_calibrated_model(model_path: str) -> Any:
    """Load the calibrated CatBoost model (saved from notebook)."""
    import joblib
    path = Path(model_path)
    if not path.is_absolute():
        path = _MODULE_DIR / path
    return joblib.load(path)


def run_inference_for_invoice(
    invoice_row: pd.Series,
    base_timestamp: pd.Timestamp,
    first_attempt_at: pd.Timestamp,
    model: Any,
) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[dict]]:
    """
    Score candidate slots and return (optimal_retry_at, max_prob, raw_features_snapshot).
    Uses optimal_slot_for_invoice from ranking_backtest (same as dunning_modeling.ipynb).
    """
    from ranking_backtest import (
        DEFAULT_DELAY_HOURS,
        optimal_slot_for_invoice,
    )

    slot_num, max_prob, slots_df = optimal_slot_for_invoice(
        invoice_row,
        base_timestamp,
        model,
        CAT_FEATURES,
        first_attempt_timestamp=first_attempt_at,
        feature_cols=[c for c in MODEL_FEATURE_NAMES if c in invoice_row.index],
    )
    delay_h = DEFAULT_DELAY_HOURS[slot_num - 1]
    optimal_retry_at = pd.Timestamp(base_timestamp) + pd.Timedelta(hours=delay_h)

    raw_snapshot = slots_df.iloc[slot_num - 1].drop("prob", errors="ignore").to_dict()
    for k, v in raw_snapshot.items():
        if isinstance(v, (np.floating, np.integer)):
            raw_snapshot[k] = float(v)

    return optimal_retry_at, max_prob, raw_snapshot


# ---------------------------------------------------------------------------
# 4. Shadow logging: main loop and output
# ---------------------------------------------------------------------------

def run_shadow_monitoring(
    active_df: pd.DataFrame,
    model_path: str,
    model_version_id: Optional[str] = None,
    output_path: Optional[str] = None,
    max_hours_since_base: int = 120,
) -> pd.DataFrame:
    """
    For each invoice in active_df, compute the model's best retry time and log to a results DataFrame.

    - base_timestamp = most recent payment_failed (e.g. updated_at or last_failure_at).
    - If (now - base_timestamp) > max_hours_since_base (120h), log as EXPIRED_DUNNING.
    - Otherwise run inference and record suggested_optimal_retry_at, suggested_max_prob, raw_features_snapshot.
    """
    from ranking_backtest import DEFAULT_DELAY_HOURS

    model = load_calibrated_model(model_path)
    inference_run_at = pd.Timestamp.utcnow()
    version_id = model_version_id or os.path.basename(model_path).replace(".joblib", "").replace(".pkl", "")

    # Expect one row per invoice (last attempt) with updated_at = last failure, first_attempt_at
    time_col = "updated_at" if "updated_at" in active_df.columns else "last_failure_at"
    if time_col not in active_df.columns:
        raise ValueError(f"Active dataframe must have '{time_col}' (or 'updated_at' / 'last_failure_at') for base timestamp.")
    first_col = "first_attempt_at"

    results: List[dict] = []
    invoices = active_df.drop_duplicates(subset=["linked_invoice_id"], keep="last")
    if "linked_invoice_id" not in invoices.columns and "invoice_id" in invoices.columns:
        invoices = invoices.rename(columns={"invoice_id": "linked_invoice_id"})

    for _, raw in tqdm(invoices.iterrows(), total=len(invoices), desc="Shadow inference"):
        invoice_id = raw.get("linked_invoice_id") or raw.get("invoice_id")
        customer_id = raw.get("customer_id")
        base_ts = pd.Timestamp(raw[time_col])
        first_ts = raw.get(first_col)
        if first_ts is None or pd.isna(first_ts):
            first_ts = base_ts
        else:
            first_ts = pd.Timestamp(first_ts)

        hours_since_base = (inference_run_at - base_ts).total_seconds() / 3600.0
        current_attempt_no = int(raw.get("invoice_attempt_no", 0) or 0)

        if hours_since_base > max_hours_since_base:
            results.append({
                "invoice_id": invoice_id,
                "customer_id": customer_id,
                "inference_run_at": inference_run_at,
                "model_version_id": version_id,
                "suggested_optimal_retry_at": pd.NaT,
                "suggested_max_prob": None,
                "current_attempt_no": current_attempt_no,
                "raw_features_snapshot": json.dumps({"status": "EXPIRED_DUNNING", "hours_since_base": round(hours_since_base, 2)}),
            })
            continue

        try:
            invoice_row = build_invoice_row(raw, base_ts, first_ts)
        except Exception as e:
            results.append({
                "invoice_id": invoice_id,
                "customer_id": customer_id,
                "inference_run_at": inference_run_at,
                "model_version_id": version_id,
                "suggested_optimal_retry_at": pd.NaT,
                "suggested_max_prob": None,
                "current_attempt_no": current_attempt_no,
                "raw_features_snapshot": json.dumps({"error": "build_row", "message": str(e)}),
            })
            continue

        try:
            optimal_retry_at, max_prob, raw_snapshot = run_inference_for_invoice(
                invoice_row, base_ts, first_ts, model,
            )
        except Exception as e:
            results.append({
                "invoice_id": invoice_id,
                "customer_id": customer_id,
                "inference_run_at": inference_run_at,
                "model_version_id": version_id,
                "suggested_optimal_retry_at": pd.NaT,
                "suggested_max_prob": None,
                "current_attempt_no": current_attempt_no,
                "raw_features_snapshot": json.dumps({"error": "inference", "message": str(e)}),
            })
            continue

        results.append({
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "inference_run_at": inference_run_at,
            "model_version_id": version_id,
            "suggested_optimal_retry_at": optimal_retry_at,
            "suggested_max_prob": round(max_prob, 6),
            "current_attempt_no": current_attempt_no,
            "raw_features_snapshot": json.dumps(raw_snapshot),
        })

    out = pd.DataFrame(results)

    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = _MODULE_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"Shadow log saved: {path}")

    return out


# ---------------------------------------------------------------------------
# 5. Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # 1) Fetch active dunning invoices (implement BigQuery in fetch_active_dunning_invoices)
    active_df = fetch_active_dunning_invoices()
    if active_df.empty:
        print("No active dunning invoices returned. Fill fetch_active_dunning_invoices() with your BigQuery logic.")
        return

    # Optional: filter to training window (inference uses current state; cutoff is for training only)
    # if "updated_at" in active_df.columns:
    #     active_df = active_df[pd.to_datetime(active_df["updated_at"]).dt.date <= pd.to_datetime(TRAINING_CUTOFF_DATE).date()]

    # 2) Model path: export from notebook with joblib.dump(model_temporal_calibrated, path)
    model_path = os.environ.get("DUNNING_MODEL_PATH", _MODULE_DIR / "artifacts" / "model_temporal_calibrated.joblib")
    model_version_id = os.environ.get("DUNNING_MODEL_VERSION", None)
    output_path = os.environ.get("SHADOW_LOG_PATH", _MODULE_DIR / "artifacts" / "shadow_log.csv")

    # 3) Run shadow inference and save
    result_df = run_shadow_monitoring(
        active_df,
        str(model_path),
        model_version_id=model_version_id,
        output_path=str(output_path),
        max_hours_since_base=120,
    )
    print(f"Logged {len(result_df)} invoices.")


if __name__ == "__main__":
    main()
