"""
Feature building for inference: one row per invoice at as_of_timestamp.
Aligned with shadow_monitoring and dunning_modeling notebook.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# 4-hour resolution, 24–120h from run time (25 slots)
DELAY_HOURS = list(range(24, 121, 4))

MODEL_FEATURE_NAMES = [
    "prev_decline_code", "prev_advice_code_group", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_sin", "day_cos",
    "dist_to_payday", "log_charge_amount", "is_debit", "amt_per_attempt",
    "time_since_prev_attempt", "cumulative_delay_hours",
    "billing_country", "gateway", "funding_type_norm", "card_brand", "prev_card_status",
    "Domain_category", "invoice_attempt_no",
]

CAT_FEATURES = [
    "prev_decline_code", "prev_advice_code_group", "billing_country", "gateway",
    "funding_type_norm", "card_brand", "Domain_category", "prev_card_status",
]

# Feature log table schema (optional drift table). Single source of truth; table can be created from this.
# Order: metadata first, then MODEL_FEATURE_NAMES, then outputs.
FEATURE_LOG_METADATA_COLUMNS = ["inference_run_id", "created_at", "invoice_id", "model_version_id"]
FEATURE_LOG_OUTPUT_COLUMNS = ["max_prob", "optimal_retry_at_utc"]
# BQ type per column: STRING, FLOAT64, INT64, TIMESTAMP
FEATURE_LOG_SCHEMA = (
    [("inference_run_id", "STRING"), ("created_at", "TIMESTAMP"), ("invoice_id", "STRING"), ("model_version_id", "STRING")]
    + [
        ("prev_decline_code", "STRING"),
        ("prev_advice_code_group", "STRING"),
        ("hour_sin", "FLOAT64"), ("hour_cos", "FLOAT64"),
        ("dow_sin", "FLOAT64"), ("dow_cos", "FLOAT64"),
        ("day_sin", "FLOAT64"), ("day_cos", "FLOAT64"),
        ("dist_to_payday", "FLOAT64"), ("log_charge_amount", "FLOAT64"),
        ("is_debit", "INT64"), ("amt_per_attempt", "FLOAT64"),
        ("time_since_prev_attempt", "FLOAT64"), ("cumulative_delay_hours", "FLOAT64"),
        ("billing_country", "STRING"), ("gateway", "STRING"),
        ("funding_type_norm", "STRING"), ("card_brand", "STRING"),
        ("prev_card_status", "STRING"), ("Domain_category", "STRING"),
        ("invoice_attempt_no", "INT64"),
    ]
    + [("max_prob", "FLOAT64"), ("optimal_retry_at_utc", "TIMESTAMP")]
)


def _safe_str(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "UNKNOWN"
    return str(s).strip() or "UNKNOWN"


def build_invoice_row(
    row: pd.Series,
    base_timestamp: pd.Timestamp,
    first_attempt_at: pd.Timestamp,
    as_of_timestamp: Optional[pd.Timestamp] = None,
    timezone: Optional[str] = None,
    as_of_localized: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Build one feature row for inference.
    base_timestamp = latest attempt's updated_at; time_since_prev_attempt = as_of - base;
    cumulative_delay_hours = as_of - first_attempt_at.
    Cyclic features (hour/dow/day) use the localized timestamp when as_of_localized is
    provided (e.g. row's localized_time column); otherwise as_of is converted to local
    using timezone (aligns with training).
    """
    base = pd.Timestamp(base_timestamp)
    first = pd.Timestamp(first_attempt_at)
    as_of = pd.Timestamp(as_of_timestamp) if as_of_timestamp is not None else base

    time_since_prev = (as_of - base).total_seconds() / 3600.0
    cumulative_delay = (as_of - first).total_seconds() / 3600.0

    # Time features: use localized timestamp column when provided; else convert as_of to local
    use_local = pd.Timestamp(as_of_localized) if as_of_localized is not None and not pd.isna(as_of_localized) else None
    if use_local is not None:
        local_dt = use_local
    elif timezone and str(timezone).strip() and str(timezone) != "UTC":
        try:
            if as_of.tzinfo is None:
                as_of = as_of.tz_localize("UTC")
            local_dt = as_of.tz_convert(str(timezone).strip())
        except Exception:
            local_dt = as_of
    else:
        local_dt = as_of

    hour = local_dt.hour + local_dt.minute / 60.0 + local_dt.second / 3600.0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow = local_dt.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    day_of_month = local_dt.day
    max_days = local_dt.days_in_month
    day_sin = np.sin(2 * np.pi * (day_of_month - 1) / max_days)
    day_cos = np.cos(2 * np.pi * (day_of_month - 1) / max_days)
    dist_to_payday = min(abs(day_of_month - 1), abs(day_of_month - 15), abs(day_of_month - 30))

    amount = float(row.get("amount", 0) or 0)
    attempt_no = int(row.get("invoice_attempt_no", 0) or 0)
    log_charge_amount = np.log1p(amount)
    is_debit = 1 if (str(row.get("funding_type_norm", "") or "").lower() == "debit") else 0
    amt_per_attempt = amount / (attempt_no + 1) if (attempt_no + 1) != 0 else 0.0

    out = pd.Series({
        "prev_decline_code": _safe_str(row.get("prev_decline_code")),
        "prev_advice_code_group": _safe_str(row.get("prev_advice_code_group")),
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
