"""
Candidate slot generation and optimal-slot scoring (24–120h, 4-hour resolution).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import features

TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_sin", "day_cos",
    "dist_to_payday", "time_since_prev_attempt", "cumulative_delay_hours",
]


def generate_candidate_slots(
    invoice_row: pd.Series,
    base_timestamp: pd.Timestamp,
    first_attempt_timestamp: Optional[pd.Timestamp] = None,
    delay_hours: Optional[List[int]] = None,
    feature_cols: Optional[List[str]] = None,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """One row per slot; temporal features recomputed per slot in **local** time when timezone is set."""
    if first_attempt_timestamp is None:
        first_attempt_timestamp = base_timestamp
    base = pd.Timestamp(base_timestamp)
    first = pd.Timestamp(first_attempt_timestamp)
    if delay_hours is None:
        delay_hours = features.DELAY_HOURS

    all_cols = [c for c in invoice_row.index if c in invoice_row]
    static_cols = [c for c in all_cols if c not in TEMPORAL_FEATURES]
    row_base = invoice_row.copy()

    if base.tzinfo is None:
        base = base.tz_localize("UTC")
    if first.tzinfo is None and base.tzinfo is not None:
        first = first.tz_localize("UTC")

    rows = []
    for hours_after in delay_hours:
        slot_dt = base + pd.Timedelta(hours=hours_after)
        if base.tz is not None and slot_dt.tz is None:
            slot_dt = slot_dt.tz_localize(base.tz)

        if timezone and str(timezone).strip() and str(timezone) != "UTC":
            try:
                slot_local = slot_dt.tz_convert(str(timezone).strip())
            except Exception:
                slot_local = slot_dt
        else:
            slot_local = slot_dt

        hour = slot_local.hour + slot_local.minute / 60.0 + slot_local.second / 3600.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow = slot_local.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        day_of_month = slot_local.day
        max_days = slot_local.days_in_month
        day_sin = np.sin(2 * np.pi * (day_of_month - 1) / max_days)
        day_cos = np.cos(2 * np.pi * (day_of_month - 1) / max_days)
        dist_to_payday = min(abs(day_of_month - 1), abs(day_of_month - 15), abs(day_of_month - 30))
        time_since_prev = float(hours_after)
        cumulative_delay = (slot_dt - first).total_seconds() / 3600

        r = {c: row_base[c] for c in static_cols}
        r.update({
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "dow_sin": dow_sin, "dow_cos": dow_cos,
            "day_sin": day_sin, "day_cos": day_cos,
            "dist_to_payday": dist_to_payday,
            "time_since_prev_attempt": time_since_prev,
            "cumulative_delay_hours": cumulative_delay,
        })
        rows.append(r)
    out = pd.DataFrame(rows)
    if feature_cols is not None:
        out = out[[c for c in feature_cols if c in out.columns]]
    return out


def optimal_slot_for_invoice(
    invoice_row: pd.Series,
    base_timestamp: pd.Timestamp,
    model,
    cat_features: List[str],
    first_attempt_timestamp: Optional[pd.Timestamp] = None,
    feature_cols: Optional[List[str]] = None,
    delay_hours: Optional[List[int]] = None,
    timezone: Optional[str] = None,
) -> Tuple[int, float, pd.DataFrame]:
    """Returns (1-based slot index of max P, max P, slots DataFrame with prob column)."""
    model_feature_names = getattr(model, "feature_names_", None) or list(invoice_row.index)
    if feature_cols is None:
        feature_cols = [c for c in model_feature_names if c in invoice_row.index]
    slots = generate_candidate_slots(
        invoice_row, base_timestamp,
        first_attempt_timestamp=first_attempt_timestamp,
        delay_hours=delay_hours,
        feature_cols=feature_cols,
        timezone=timezone,
    )
    order = [c for c in model_feature_names if c in slots.columns]
    if order:
        slots = slots[order]
    probs = model.predict_proba(slots)[:, 1]
    best_idx = int(np.argmax(probs))
    return best_idx + 1, float(probs[best_idx]), slots.assign(prob=probs)


def run_inference_for_invoice(
    invoice_row: pd.Series,
    base_timestamp_for_slots: pd.Timestamp,
    first_attempt_at: pd.Timestamp,
    model,
    timezone: Optional[str] = None,
) -> Tuple[pd.Timestamp, float, dict, dict]:
    """Returns (optimal_retry_at, max_prob, raw_snapshot_dict, probs_by_delay)."""
    slot_num, max_prob, slots_df = optimal_slot_for_invoice(
        invoice_row,
        base_timestamp_for_slots,
        model,
        features.CAT_FEATURES,
        first_attempt_timestamp=first_attempt_at,
        feature_cols=[c for c in features.MODEL_FEATURE_NAMES if c in invoice_row.index],
        delay_hours=features.DELAY_HOURS,
        timezone=timezone,
    )
    base_ts = pd.Timestamp(base_timestamp_for_slots)
    delay_h = features.DELAY_HOURS[slot_num - 1]
    optimal_retry_at = (base_ts + pd.Timedelta(hours=delay_h)).round("h")

    raw_snapshot = slots_df.iloc[slot_num - 1].drop("prob", errors="ignore").to_dict()
    for k, v in raw_snapshot.items():
        if isinstance(v, (np.floating, np.integer)):
            raw_snapshot[k] = float(v)

    probs_by_delay = {
        d: round(float(slots_df.iloc[i]["prob"]), 6)
        for i, d in enumerate(features.DELAY_HOURS)
    }
    return optimal_retry_at, max_prob, raw_snapshot, probs_by_delay
