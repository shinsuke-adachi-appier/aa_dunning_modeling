"""
Ranking Backtest & Optimal Slot Simulation for Dunning Success Model.
Evaluates model's ability to RANK candidate slots (maximize recall via best slot choice).
Uses: X_hold, y_hold, model_temporal, invoice_ids, optional timestamps for TTR.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# 1. Invoice grouping & ranking backtest
# ---------------------------------------------------------------------------

def run_ranking_backtest(
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    invoice_ids: pd.Series,
    model,
    cat_features: List[str],
) -> pd.DataFrame:
    """
    Group holdout by linked_invoice_id; for recovered invoices, rank attempts by P(success).
    Returns a DataFrame with one row per recovered invoice: invoice_id, rank_of_success, prob_success, n_attempts.
    """
    # Align: same index as X_hold
    invoice_ids = invoice_ids.reindex(X_hold.index).dropna()
    preds = model.predict_proba(X_hold.loc[invoice_ids.index])[:, 1]
    y = y_hold.reindex(invoice_ids.index).fillna(0).astype(int)

    df = pd.DataFrame({
        "invoice_id": invoice_ids.values,
        "y": y.values,
        "prob": preds,
        "idx": invoice_ids.index,
    })

    # Invoices that had at least one success
    recovered_invoices = df.groupby("invoice_id")["y"].max()
    recovered_invoices = recovered_invoices[recovered_invoices >= 1].index.tolist()

    rows = []
    for inv in recovered_invoices:
        sub = df[df["invoice_id"] == inv].copy()
        sub["rank"] = sub["prob"].rank(method="first", ascending=False).astype(int)
        success_row = sub[sub["y"] == 1].iloc[0]
        rank_of_success = int(success_row["rank"])
        prob_success = success_row["prob"]
        n_attempts = len(sub)
        rows.append({
            "invoice_id": inv,
            "rank_of_success": rank_of_success,
            "prob_success": prob_success,
            "n_attempts": n_attempts,
        })
    return pd.DataFrame(rows)


def top1_accuracy(backtest_df: pd.DataFrame) -> float:
    """Fraction of recovered invoices where the actual success was ranked #1 by the model."""
    if len(backtest_df) == 0:
        return 0.0
    return (backtest_df["rank_of_success"] == 1).mean()


def rank_distribution(backtest_df: pd.DataFrame, max_rank: int = 10) -> pd.Series:
    """Count of recovered invoices by rank of the successful attempt (1 = model's top pick)."""
    return backtest_df["rank_of_success"].clip(upper=max_rank).value_counts().sort_index()


# ---------------------------------------------------------------------------
# 2. Time-to-Recovery (TTR) analysis
# ---------------------------------------------------------------------------

def ttr_analysis(
    backtest_df: pd.DataFrame,
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    invoice_ids: pd.Series,
    holdout_timestamps: pd.DataFrame,
    model,
    cat_features: List[str],
) -> Tuple[float, float, float]:
    """
    holdout_timestamps: DataFrame with same index as X_hold, columns [updated_at] and optionally [first_attempt_at].
    Returns: (baseline_ttr_hours, model_top1_ttr_hours, pct_recovered_at_top1).
    """
    common_idx = X_hold.index.intersection(invoice_ids.index).intersection(holdout_timestamps.index)
    if "updated_at" not in holdout_timestamps.columns or len(common_idx) == 0:
        return np.nan, np.nan, 0.0

    inv = invoice_ids.reindex(common_idx).dropna()
    preds = model.predict_proba(X_hold.loc[inv.index])[:, 1]
    y = y_hold.reindex(inv.index).fillna(0).astype(int)
    ts = holdout_timestamps.loc[inv.index].copy()
    ts["updated_at"] = pd.to_datetime(ts["updated_at"])

    df = pd.DataFrame({"invoice_id": inv.values, "y": y.values, "prob": preds}, index=inv.index)
    df["updated_at"] = ts["updated_at"].values
    if "first_attempt_at" in ts.columns:
        df["first_attempt_at"] = pd.to_datetime(ts["first_attempt_at"]).values
    else:
        first_per_inv = df.groupby("invoice_id")["updated_at"].min()
        df = df.join(first_per_inv.rename("first_attempt_at"), on="invoice_id")

    recovered = df[df["y"] == 1].copy()
    if len(recovered) == 0:
        return np.nan, np.nan, 0.0

    # Baseline TTR: hours from first_attempt_at to success
    u = pd.to_datetime(recovered["updated_at"])
    r = pd.to_datetime(recovered["first_attempt_at"])
    baseline_ttr = (u - r).dt.total_seconds() / 3600
    baseline_ttr = baseline_ttr.replace([np.inf, -np.inf], np.nan).dropna()
    baseline_mean = float(baseline_ttr.mean()) if len(baseline_ttr) else np.nan

    # Model Top-1 TTR: for invoices where model ranked success as #1, hours from ref to success
    df["rank"] = df.groupby("invoice_id")["prob"].rank(method="first", ascending=False)
    success_rank1 = df[(df["y"] == 1) & (df["rank"] == 1)]
    if len(success_rank1) == 0:
        model_mean, pct_top1 = np.nan, 0.0
    else:
        u1 = pd.to_datetime(success_rank1["updated_at"])
        r1 = pd.to_datetime(success_rank1["first_attempt_at"])
        model_ttr = (u1 - r1).dt.total_seconds() / 3600
        model_ttr = model_ttr.replace([np.inf, -np.inf], np.nan).dropna()
        model_mean = float(model_ttr.mean()) if len(model_ttr) else np.nan
        pct_top1 = len(success_rank1) / len(backtest_df) if len(backtest_df) else 0.0
    return baseline_mean, model_mean, pct_top1


# ---------------------------------------------------------------------------
# 3. Rank distribution visualization
# ---------------------------------------------------------------------------

def plot_rank_distribution(
    backtest_df: pd.DataFrame,
    max_rank: int = 10,
    title: str = "Rank of Successful Attempt (by model P(success))",
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
):
    """Histogram of which rank (1st through max_rank) the successful attempt fell into."""
    dist = rank_distribution(backtest_df, max_rank=max_rank)
    # Fill missing ranks with 0
    full = pd.Series(0, index=range(1, max_rank + 1))
    full.update(dist)
    full = full.sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(full.index, full.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Rank (1 = model's top pick)")
    ax.set_ylabel("Number of recovered invoices")
    ax.set_title(title)
    ax.set_xticks(full.index)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Generate candidate slots by hours after previous attempt (24, 36, ..., 120h)
# ---------------------------------------------------------------------------

# Slots = fixed hours after base_timestamp (previous attempt)
# 4-hour resolution to break 12h "frequency trap" (24, 28, ..., 120 → 25 slots)
DEFAULT_DELAY_HOURS = list(range(24, 121, 6))  # 25 slots

# Temporal features we recompute per slot; the rest are copied from invoice_row
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
) -> pd.DataFrame:
    """
    Create hypothetical feature rows: one per slot at 24, 36, 48, ..., 120 hours after base (previous attempt).
    invoice_row: one row from X (same columns as model). Static features copied; temporal features recomputed.
    base_timestamp: reference time (e.g. last attempt). Each slot_dt = base + delay_hours[i].
    Returns: DataFrame with len(delay_hours) rows, same columns as model input.
    """
    if first_attempt_timestamp is None:
        first_attempt_timestamp = base_timestamp
    base = pd.Timestamp(base_timestamp)
    first = pd.Timestamp(first_attempt_timestamp)
    if delay_hours is None:
        delay_hours = list(DEFAULT_DELAY_HOURS)

    # Static = all features except temporal (copy from invoice_row)
    all_cols = [c for c in invoice_row.index if c in invoice_row]
    static_cols = [c for c in all_cols if c not in TEMPORAL_FEATURES]
    row_base = invoice_row.copy()

    rows = []
    for hours_after in delay_hours:
        slot_dt = base + pd.Timedelta(hours=hours_after)
        if base.tz is not None and slot_dt.tz is None:
            slot_dt = slot_dt.tz_localize(base.tz)

        hour = slot_dt.hour + slot_dt.minute / 60.0 + slot_dt.second / 3600.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow = slot_dt.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        day_of_month = slot_dt.day
        max_days = slot_dt.days_in_month
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
) -> Tuple[int, float, pd.DataFrame]:
    """
    Generate candidate slots (24, 36, ..., 120h after base), score with model; return (1-based slot index of max P, max P, candidate DataFrame).
    """
    try:
        model_feature_names = getattr(model, "feature_names_", None)
    except Exception:
        model_feature_names = None
    if model_feature_names is None:
        model_feature_names = list(invoice_row.index)
    if feature_cols is None:
        feature_cols = [c for c in model_feature_names if c in invoice_row.index]
    slots = generate_candidate_slots(
        invoice_row, base_timestamp,
        first_attempt_timestamp=first_attempt_timestamp,
        feature_cols=feature_cols,
    )
    # Ensure column order matches model
    order = [c for c in model_feature_names if c in slots.columns]
    if order:
        slots = slots[order]
    probs = model.predict_proba(slots)[:, 1]
    best_idx = int(np.argmax(probs))
    return best_idx + 1, float(probs[best_idx]), slots.assign(prob=probs)


def rank1_slot_per_invoice(
    backtest_df: pd.DataFrame,
    X_hold: pd.DataFrame,
    invoice_ids: pd.Series,
    holdout_timestamps: Optional[pd.DataFrame],
    model,
    cat_features: List[str],
) -> pd.Series:
    """
    For each recovered invoice, which candidate slot (0-based index) is ranked #1 by the model.
    Uses generate_candidate_slots (24, 36, ..., 120h after previous attempt).
    Returns Series: index=invoice_id, value=slot_index (0..n_slots-1).
    """
    invoice_ids_aligned = invoice_ids.reindex(X_hold.index).dropna()
    if holdout_timestamps is None or "updated_at" not in holdout_timestamps.columns:
        return pd.Series(dtype=int)

    slot_indices = []
    invoice_ids_list = []
    for inv in backtest_df["invoice_id"]:
        inv_idx = invoice_ids_aligned[invoice_ids_aligned == inv].index
        if len(inv_idx) == 0:
            continue
        idx = inv_idx[0]
        row = X_hold.loc[idx]
        base_ts = pd.Timestamp(holdout_timestamps.loc[idx, "updated_at"])
        first_ts = None
        if "first_attempt_at" in holdout_timestamps.columns:
            first_ts = pd.Timestamp(holdout_timestamps.loc[idx, "first_attempt_at"])
        try:
            slot_num, _, _ = optimal_slot_for_invoice(
                row, base_ts, model, cat_features,
                first_attempt_timestamp=first_ts,
            )
            slot_indices.append(slot_num - 1)
            invoice_ids_list.append(inv)
        except Exception:
            continue
    return pd.Series(slot_indices, index=invoice_ids_list)


def rank1_slot_labels(delay_hours: Optional[List[int]] = None) -> List[str]:
    """Labels for slot indices by hours after previous attempt: e.g. ['24h', '36h', ..., '120h']."""
    if delay_hours is None:
        delay_hours = DEFAULT_DELAY_HOURS
    return [f"{h}h" for h in delay_hours]


# ---------------------------------------------------------------------------
# 5. Main: run backtest and print metrics + plot
# ---------------------------------------------------------------------------

def run_full_backtest(
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    invoice_ids: pd.Series,
    model,
    cat_features: List[str],
    holdout_timestamps: Optional[pd.DataFrame] = None,
    max_rank: int = 10,
    plot: bool = True,
):
    """
    Run ranking backtest, compute Top-1 accuracy, rank distribution, optional TTR; plot rank distribution.
    """
    backtest_df = run_ranking_backtest(X_hold, y_hold, invoice_ids, model, cat_features)
    if len(backtest_df) == 0:
        print("No recovered invoices in holdout; cannot compute ranking metrics.")
        return backtest_df

    top1 = top1_accuracy(backtest_df)
    print("=== Ranking Backtest (holdout) ===\n")
    print(f"Recovered invoices (with ≥1 success): {len(backtest_df)}")
    print(f"Top-1 Accuracy: {top1:.2%} (success was model's #1 pick)")
    print(f"Rank distribution (1 = model's top):\n{rank_distribution(backtest_df, max_rank=max_rank)}")

    if holdout_timestamps is not None and len(holdout_timestamps.columns):
        baseline_ttr, model_ttr, pct_top1 = ttr_analysis(
            backtest_df, X_hold, y_hold, invoice_ids, holdout_timestamps,
            model, cat_features,
        )
        print(f"\n=== Time-to-Recovery (TTR) ===")
        print(f"Baseline (avg hours to recovery): {baseline_ttr:.1f}")
        print(f"Model Top-1 (avg hours when success was rank #1): {model_ttr:.1f}")
        print(f"% recovered where model ranked success #1: {pct_top1:.1%}")

    if plot:
        plot_rank_distribution(backtest_df, max_rank=max_rank)
    return backtest_df
