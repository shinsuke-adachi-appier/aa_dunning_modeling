"""Shared feature engineering and model feature names (training + inference)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


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

# Categorical features for CatBoost
CAT_FEATURES = [
    "prev_decline_code", "billing_country", "gateway",
    "funding_type_norm", "card_brand", "Domain_category", "prev_card_status"
]

# Model feature names for inference (no target/group)
MODEL_FEATURE_NAMES = [
    "prev_decline_code", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_sin", "day_cos",
    "dist_to_payday", "log_charge_amount", "is_debit", "amt_per_attempt",
    "time_since_prev_attempt", "cumulative_delay_hours",
    "billing_country", "gateway", "funding_type_norm", "card_brand", "prev_card_status",
    "Domain_category", "invoice_attempt_no",
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
    base_timestamp: latest attempt's updated_at; time_since_prev_attempt = as_of - base;
    cumulative_delay_hours = as_of - first_attempt_at.
    """
    base = pd.Timestamp(base_timestamp)
    first = pd.Timestamp(first_attempt_at)
    as_of = pd.Timestamp(as_of_timestamp) if as_of_timestamp is not None else base

    time_since_prev = (as_of - base).total_seconds() / 3600.0
    cumulative_delay = (as_of - first).total_seconds() / 3600.0

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


def sanitize_for_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN/None in categorical and numeric columns so CatBoost does not raise TypeError."""
    X = X.copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN").astype(str).replace("nan", "UNKNOWN")
    for col in X.columns:
        if col not in CAT_FEATURES:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X
