"""
Shadow monitoring: log the model's "best choice" (optimal retry time) for every invoice
currently in a dunning cycle for future comparison and probability drift over a 14-day period.

- Append-mode CSV: each cron run appends rows (same invoice can appear multiple times).
- Time-aware features: time_since_prev_attempt and cumulative_delay_hours at current run time.
- Slots: 4-hour resolution, 24–120h from current run time (5-day window).
- Metadata: inference_run_id, snapshot_hour, days_into_dunning for evaluation.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 4-hour resolution, 120h (5 days) from run time
SHADOW_DELAY_HOURS = list(range(24, 121, 4))  # 24, 28, ..., 120

# Allow importing ranking_backtest when run from repo root (e.g. python modeling/shadow_monitoring_20260206.py).
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

# ---------------------------------------------------------------------------
# 1. Data connection & filter (framework — fill BigQuery later)
# ---------------------------------------------------------------------------

# Training window cutoff metadata: shadow period starts on/after this date.
TRAINING_CUTOFF_DATE = "2026-02-14"
LAST_DUNNING_AT_OR_AFTER = "2026-02-15"

def fetch_active_dunning_invoices() -> pd.DataFrame:
    """
    Connect to BigQuery and pull active dunning invoices (one row per invoice = latest attempt).
    The model predicts the *next* attempt, so the latest attempt's outcome is the "previous" state:
    we use its Decline_code_norm and card_status as prev_decline_code and prev_card_status.
    No LAG: we simply take the latest row and rename those columns.
    """
    try:
        from txn_pipeline import get_bq_client, load_bigquery_table, add_timezone_features
    except Exception as e:
        print(f"Error importing txn_pipeline: {e}", file=sys.stderr)
        raise
    try:
        client = get_bq_client(project="aa-datamart", location="europe-west1")
        query = """
            WITH LatestState AS (
                SELECT *,
                    ROW_NUMBER() OVER(PARTITION BY linked_invoice_id ORDER BY updated_at DESC) AS latest_row,
                    MIN(updated_at) OVER(PARTITION BY linked_invoice_id) AS first_attempt_at_calc
                FROM `aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback`
                WHERE invoice_success_attempt_no IS NULL
                  AND invoice_attempt_count < 12
                  AND updated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 DAY)
            )
            SELECT * EXCEPT(latest_row)
            FROM LatestState
            WHERE latest_row = 1
            ORDER BY updated_at ASC;
        """
        active_df = load_bigquery_table(client, query)
        # Rename latest attempt's outcome columns → "previous" for the next prediction
        rename_map = {}
        if "Decline_code_norm" in active_df.columns:
            rename_map["Decline_code_norm"] = "prev_decline_code"
        if "card_status" in active_df.columns:
            rename_map["card_status"] = "prev_card_status"
        if "first_attempt_at" not in active_df.columns and "first_attempt_at_calc" in active_df.columns:
            rename_map["first_attempt_at_calc"] = "first_attempt_at"
        if rename_map:
            active_df = active_df.rename(columns=rename_map)
        active_df = add_timezone_features(active_df)
        return active_df
    except Exception as e:
        print(f"BigQuery error in fetch_active_dunning_invoices: {e}", file=sys.stderr)
        raise


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
    as_of_timestamp: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Build one feature row for inference.

    - base_timestamp: latest attempt's updated_at (current failure time). Used as "previous" for the next prediction.
    - time_since_prev_attempt = (as_of_timestamp - base_timestamp) = (inference_run_at - latest_failure_updated_at).
    - cumulative_delay_hours = (as_of_timestamp - first_attempt_at) = (inference_run_at - first_attempt_at).
    - invoice_attempt_no: from the latest record (row), i.e. the attempt number of the current failure.
    - prev_decline_code / prev_card_status: from the latest attempt (renamed from Decline_code_norm / card_status in fetch).
    generate_candidate_slots overwrites temporal features per slot.
    """
    base = pd.Timestamp(base_timestamp)   # latest_failure_updated_at
    first = pd.Timestamp(first_attempt_at)
    as_of = pd.Timestamp(as_of_timestamp) if as_of_timestamp is not None else base

    # time_since_prev_attempt = hours since latest (current) failure
    time_since_prev = (as_of - base).total_seconds() / 3600.0
    # cumulative_delay_hours = hours since first attempt
    cumulative_delay = (as_of - first).total_seconds() / 3600.0

    # Cyclic features at as_of (placeholder; generate_candidate_slots overwrites per slot)
    hour = as_of.hour + as_of.minute / 60.0 + as_of.second / 3600.0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow = as_of.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    day_of_month = as_of.day
    max_days = as_of.days_in_month
    day_sin = np.sin(2 * np.pi * (day_of_month - 1) / max_days)
    day_cos = np.cos(2 * np.pi * (day_of_month - 1) / max_days)
    dist_to_payday = min(abs(day_of_month - 1), abs(day_of_month - 15), abs(day_of_month - 30))

    amount = float(row.get("amount", 0) or 0)
    # invoice_attempt_no from latest record (current failure attempt number)
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
    """Load the calibrated model. Exits with code 1 on failure so cron can log."""
    import joblib

    try:
        from train_dunning_v2_20260206 import IsotonicCalibratedClassifier
        setattr(sys.modules["__main__"], "IsotonicCalibratedClassifier", IsotonicCalibratedClassifier)
    except ImportError:
        pass

    path = Path(model_path)
    if not path.is_absolute():
        path = _MODULE_DIR / path
    if not path.exists():
        print(f"Model file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load model from {path}: {e}", file=sys.stderr)
        sys.exit(1)


def run_inference_for_invoice(
    invoice_row: pd.Series,
    base_timestamp_for_slots: pd.Timestamp,
    first_attempt_at: pd.Timestamp,
    model: Any,
) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[dict], Dict[int, float]]:
    """
    Score candidate slots 24–120h (4-hour resolution) from base_timestamp_for_slots (current run time).
    Returns (optimal_retry_at, max_prob, raw_features_snapshot, probs_by_delay) where probs_by_delay
    maps delay_hours (e.g. 24, 28, ..., 120) to probability.
    """
    from ranking_backtest import optimal_slot_for_invoice

    slot_num, max_prob, slots_df = optimal_slot_for_invoice(
        invoice_row,
        base_timestamp_for_slots,
        model,
        CAT_FEATURES,
        first_attempt_timestamp=first_attempt_at,
        feature_cols=[c for c in MODEL_FEATURE_NAMES if c in invoice_row.index],
        delay_hours=SHADOW_DELAY_HOURS,
    )
    base_ts = pd.Timestamp(base_timestamp_for_slots)
    delay_h = SHADOW_DELAY_HOURS[slot_num - 1]
    optimal_retry_at = base_ts + pd.Timedelta(hours=delay_h)
    optimal_retry_at = optimal_retry_at.round("h")

    raw_snapshot = slots_df.iloc[slot_num - 1].drop("prob", errors="ignore").to_dict()
    for k, v in raw_snapshot.items():
        if isinstance(v, (np.floating, np.integer)):
            raw_snapshot[k] = float(v)

    probs_by_delay: Dict[int, float] = {
        d: round(float(slots_df.iloc[i]["prob"]), 6)
        for i, d in enumerate(SHADOW_DELAY_HOURS)
    }

    return optimal_retry_at, max_prob, raw_snapshot, probs_by_delay


# ---------------------------------------------------------------------------
# 4. Shadow logging: main loop and output
# ---------------------------------------------------------------------------

def _default_slot_log_path(output_path: Optional[str]) -> Optional[str]:
    """Derive slot log path from main shadow log path: same dir, file shadow_slot_log.csv."""
    if not output_path:
        return None
    p = Path(output_path)
    if not p.is_absolute():
        p = _MODULE_DIR / p
    return str(p.parent / "shadow_slot_log.csv")


def run_shadow_monitoring(
    active_df: pd.DataFrame,
    model_path: str,
    model_version_id: Optional[str] = None,
    output_path: Optional[str] = None,
    slot_log_path: Optional[str] = None,
    max_hours_since_base: int = 120,
) -> pd.DataFrame:
    """
    For each invoice, compute best retry in the next 5 days (from current run time). Append rows to CSV.
    Also appends one row per invoice (wide format: prob_24h, ..., prob_120h) to a separate slot log CSV.

    Feature alignment:
    - Each row from BQ is the latest attempt (current failure). Its outcome is the "previous" for the next prediction.
    - time_since_prev_attempt = (inference_run_at - latest_failure_updated_at), i.e. inference_run_at - row.updated_at.
    - cumulative_delay_hours = (inference_run_at - first_attempt_at).
    - invoice_attempt_no and prev_decline_code / prev_card_status come from the latest record.
    """
    model = load_calibrated_model(model_path)
    inference_run_at = pd.Timestamp.utcnow()
    inference_run_id = str(uuid.uuid4())
    snapshot_hour = int(inference_run_at.hour)

    version_id = model_version_id or os.path.basename(model_path).replace(".joblib", "").replace(".pkl", "")

    time_col = "updated_at"
    if time_col not in active_df.columns:
        raise ValueError(f"Active dataframe must have '{time_col}'.")
    first_col = "first_attempt_at"

    slot_path = slot_log_path if slot_log_path is not None else _default_slot_log_path(output_path)

    results: List[dict] = []
    slot_rows_all: List[dict] = []
    # One row per invoice = latest attempt (current failure)
    invoices = active_df.drop_duplicates(subset=["linked_invoice_id"], keep="last")
    if "linked_invoice_id" not in invoices.columns and "invoice_id" in invoices.columns:
        invoices = invoices.rename(columns={"invoice_id": "linked_invoice_id"})

    for _, raw in tqdm(invoices.iterrows(), total=len(invoices), desc="Shadow inference"):
        invoice_id = raw.get("linked_invoice_id") or raw.get("invoice_id")
        customer_id = raw.get("customer_id")
        # base_ts = latest attempt's updated_at (current failure) → used for time_since_prev_attempt
        base_ts = pd.Timestamp(raw[time_col])
        first_ts = raw.get(first_col)
        if first_ts is None or pd.isna(first_ts):
            first_ts = base_ts
        else:
            first_ts = pd.Timestamp(first_ts)

        hours_since_base = (inference_run_at - base_ts).total_seconds() / 3600.0
        cumulative_delay_hours = (inference_run_at - first_ts).total_seconds() / 3600.0
        days_into_dunning = cumulative_delay_hours / 24.0
        current_attempt_no = int(raw.get("invoice_attempt_no", 0) or 0)

        meta = {
            "inference_run_id": inference_run_id,
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "inference_run_at": inference_run_at,
            "model_version_id": version_id,
            "current_attempt_no": current_attempt_no,
            "snapshot_hour": snapshot_hour,
            "days_into_dunning": round(days_into_dunning, 4),
        }

        if hours_since_base > max_hours_since_base:
            results.append({
                **meta,
                "suggested_optimal_retry_at": pd.NaT,
                "suggested_max_prob": None,
                "raw_features_snapshot": json.dumps({"status": "EXPIRED_DUNNING", "hours_since_base": round(hours_since_base, 2)}),
            })
            continue

        try:
            invoice_row = build_invoice_row(
                raw,
                base_ts,   # latest_failure_updated_at
                first_ts,  # first_attempt_at
                as_of_timestamp=inference_run_at,
            )
        except Exception as e:
            results.append({
                **meta,
                "suggested_optimal_retry_at": pd.NaT,
                "suggested_max_prob": None,
                "raw_features_snapshot": json.dumps({"error": "build_row", "message": str(e)}),
            })
            continue

        try:
            optimal_retry_at, max_prob, raw_snapshot, probs_by_delay = run_inference_for_invoice(
                invoice_row,
                base_timestamp_for_slots=inference_run_at,
                first_attempt_at=first_ts,
                model=model,
            )
        except Exception as e:
            results.append({
                **meta,
                "suggested_optimal_retry_at": pd.NaT,
                "suggested_max_prob": None,
                "raw_features_snapshot": json.dumps({"error": "inference", "message": str(e)}),
            })
            continue

        results.append({
            **meta,
            "suggested_optimal_retry_at": optimal_retry_at,
            "suggested_max_prob": round(max_prob, 6),
            "raw_features_snapshot": json.dumps(raw_snapshot),
        })

        wide_row = {
            "inference_run_id": inference_run_id,
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "inference_run_at": inference_run_at,
            "snapshot_hour_jst": snapshot_hour,
            "days_into_dunning": round(days_into_dunning, 4),
        }
        for d in SHADOW_DELAY_HOURS:
            wide_row[f"prob_{d}h"] = probs_by_delay.get(d)
        slot_rows_all.append(wide_row)

    out = pd.DataFrame(results)

    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = _MODULE_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        out.to_csv(path, mode="a", header=write_header, index=False, date_format="%Y-%m-%d %H:%M:%S")

    if slot_path and slot_rows_all:
        slot_path_p = Path(slot_path)
        if not slot_path_p.is_absolute():
            slot_path_p = _MODULE_DIR / slot_path_p
        slot_path_p.parent.mkdir(parents=True, exist_ok=True)
        slot_df = pd.DataFrame(slot_rows_all)
        write_header_slot = not slot_path_p.exists()
        slot_df.to_csv(slot_path_p, mode="a", header=write_header_slot, index=False, date_format="%Y-%m-%d %H:%M:%S")

    out.attrs["slot_rows_appended"] = len(slot_rows_all)
    return out


# ---------------------------------------------------------------------------
# 5. Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        active_df = fetch_active_dunning_invoices()
    except Exception as e:
        print(f"fetch_active_dunning_invoices failed: {e}", file=sys.stderr)
        sys.exit(1)

    if active_df.empty:
        print("No active dunning invoices returned; nothing to append.")
        return

    model_path = os.environ.get("DUNNING_MODEL_PATH", _MODULE_DIR / "models" / "catboost_dunning_calibrated_20260224.joblib")
    model_version_id = os.environ.get("DUNNING_MODEL_VERSION", None)
    output_path = os.environ.get("SHADOW_LOG_PATH", str(_MODULE_DIR / "artifacts" / "shadow_log.csv"))
    slot_log_path = os.environ.get("SHADOW_SLOT_LOG_PATH") or _default_slot_log_path(output_path)

    result_df = run_shadow_monitoring(
        active_df,
        str(model_path),
        model_version_id=model_version_id,
        output_path=output_path,
        slot_log_path=slot_log_path,
        max_hours_since_base=120,
    )
    n = len(result_df)
    slot_count = result_df.attrs.get("slot_rows_appended", 0)
    print(f"Appended {n} new predictions to {output_path}.")
    if slot_log_path and slot_count:
        print(f"Appended {slot_count} slot log rows (one per invoice, prob_24h–prob_120h) to {slot_log_path}.")


if __name__ == "__main__":
    main()
