"""
Circadian Bias Diagnostic for Payment Intent (Dunning) Model.
Compares localized hour of initial failure vs model's predicted optimal slot;
computes clock-time shift and runs a shuffle test to separate time-of-day vs delay bias.
Run from project root (aa_dunning_modeling) so that src is importable.
"""

import sys
from pathlib import Path

# Project root so that "from src.*" works (notebooks/ is under project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

# Import from src package
def _get_rank1_slot_with_shuffle(
    invoice_row: pd.Series,
    base_ts: pd.Timestamp,
    model,
    cat_features: List[str],
    first_attempt_timestamp: Optional[pd.Timestamp],
    delay_hours: List[int],
    shuffle_hour: bool,
    feature_cols: Optional[List[str]] = None,
):
    """Get 0-based rank-1 slot index, optionally with hour_sin/hour_cos shuffled across slot rows."""
    from src.evaluation.ranking_backtest import generate_candidate_slots

    try:
        model_feature_names = getattr(model, "feature_names_", None)
    except Exception:
        model_feature_names = None
    if model_feature_names is None:
        model_feature_names = list(invoice_row.index)
    if feature_cols is None:
        feature_cols = [c for c in model_feature_names if c in invoice_row.index]

    slots = generate_candidate_slots(
        invoice_row, base_ts,
        first_attempt_timestamp=first_attempt_timestamp,
        delay_hours=delay_hours,
        feature_cols=feature_cols,
    )
    order = [c for c in model_feature_names if c in slots.columns]
    if order:
        slots = slots[order]

    if shuffle_hour and "hour_sin" in slots.columns and "hour_cos" in slots.columns:
        perm = np.random.permutation(len(slots))
        slots = slots.copy()
        slots["hour_sin"] = slots["hour_sin"].values[perm]
        slots["hour_cos"] = slots["hour_cos"].values[perm]

    probs = model.predict_proba(slots)[:, 1]
    best_idx = int(np.argmax(probs))
    return best_idx


def run_circadian_diagnostic(
    backtest_df: pd.DataFrame,
    X_hold: pd.DataFrame,
    invoice_ids_hold: pd.Series,
    holdout_ts: pd.DataFrame,
    processed_df: pd.DataFrame,
    model,
    cat_features: List[str],
    delay_hours: Optional[List[int]] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    For each recovered invoice: initial failure localized hour, optimal-slot localized hour,
    and (optionally) rank-1 slot with shuffled hour for bias test.

    processed_df must have same index as X_hold (or alignable) and columns:
      updated_at, timezone, local_hour (or localized_time for .dt.hour).
    holdout_ts: index = X_hold.index, columns updated_at, first_attempt_at.

    Returns:
      diag_df: columns [invoice_id, initial_failure_hour, optimal_slot_hour, clock_shift_hours, rank1_slot_idx, rank1_slot_idx_shuffled]
      results: dict with shift stats and slot counts for shuffle test.
    """
    from src.evaluation.ranking_backtest import (
        DEFAULT_DELAY_HOURS,
        rank1_slot_per_invoice,
        generate_candidate_slots,
    )
    if delay_hours is None:
        delay_hours = list(DEFAULT_DELAY_HOURS)

    np.random.seed(seed)
    invoice_ids_aligned = invoice_ids_hold.reindex(X_hold.index).dropna()
    if holdout_ts is None or "updated_at" not in holdout_ts.columns:
        return pd.DataFrame(), {}

    # First attempt index per invoice (chronological): row with min(updated_at) per invoice
    ts = holdout_ts.loc[invoice_ids_aligned.index].copy()
    ts["updated_at"] = pd.to_datetime(ts["updated_at"])
    ts["invoice_id"] = invoice_ids_aligned.values
    first_attempt_idx = ts.groupby("invoice_id")["updated_at"].idxmin()  # Series: invoice_id -> index of first attempt

    # Align processed_df with X_hold index for timezone and local_hour
    common_idx = processed_df.index.intersection(X_hold.index)
    if "local_hour" not in processed_df.columns and "localized_time" in processed_df.columns:
        local_hour_series = processed_df.loc[common_idx, "localized_time"].dt.hour
    else:
        local_hour_series = processed_df.loc[common_idx, "local_hour"] if "local_hour" in processed_df.columns else None
    if local_hour_series is None:
        return pd.DataFrame(), {"error": "processed_df needs local_hour or localized_time"}

    timezone_series = processed_df.loc[common_idx, "timezone"] if "timezone" in processed_df.columns else None
    if timezone_series is None:
        return pd.DataFrame(), {"error": "processed_df needs timezone"}

    rank1_series = rank1_slot_per_invoice(
        backtest_df, X_hold, invoice_ids_hold, holdout_ts, model, cat_features
    )

    rows = []
    for inv in backtest_df["invoice_id"]:
        inv_idx = invoice_ids_aligned[invoice_ids_aligned == inv].index
        if len(inv_idx) == 0:
            continue
        idx = inv_idx[0]  # row used for base_ts and slot generation in rank1_slot_per_invoice
        fa_idx = first_attempt_idx.get(inv)
        if fa_idx is None or pd.isna(fa_idx):
            continue
        if fa_idx not in local_hour_series.index or fa_idx not in timezone_series.index:
            continue

        initial_hour = float(local_hour_series.loc[fa_idx])
        tz = timezone_series.loc[fa_idx]
        base_ts = pd.Timestamp(holdout_ts.loc[idx, "updated_at"])
        first_ts = None
        if "first_attempt_at" in holdout_ts.columns:
            first_ts = pd.Timestamp(holdout_ts.loc[idx, "first_attempt_at"])

        rank1_idx = rank1_series.get(inv)
        if rank1_idx is None:
            continue
        slot_delay = delay_hours[int(rank1_idx)]
        slot_utc = base_ts + pd.Timedelta(hours=slot_delay)
        if slot_utc.tzinfo is None:
            slot_utc = slot_utc.tz_localize("UTC")
        try:
            slot_local = slot_utc.tz_convert(tz)
        except Exception:
            slot_local = slot_utc
        optimal_hour = slot_local.hour + slot_local.minute / 60.0 + slot_local.second / 3600.0

        clock_shift = (optimal_hour - initial_hour) % 24
        if clock_shift > 12:
            clock_shift -= 24

        # Shuffle test: get rank-1 slot with shuffled hour_sin/hour_cos
        row_series = X_hold.loc[idx]
        rank1_shuffled = _get_rank1_slot_with_shuffle(
            row_series, base_ts, model, cat_features, first_ts, delay_hours, shuffle_hour=True
        )

        rows.append({
            "invoice_id": inv,
            "initial_failure_hour": initial_hour,
            "optimal_slot_hour": optimal_hour,
            "clock_shift_hours": clock_shift,
            "rank1_slot_idx": int(rank1_idx),
            "rank1_slot_label": f"{delay_hours[int(rank1_idx)]}h",
            "rank1_slot_idx_shuffled": rank1_shuffled,
            "rank1_slot_label_shuffled": f"{delay_hours[rank1_shuffled]}h",
        })

    diag_df = pd.DataFrame(rows)
    if len(diag_df) == 0:
        return diag_df, {}

    # Clock-time shift stats
    shift_mean = diag_df["clock_shift_hours"].mean()
    shift_median = diag_df["clock_shift_hours"].median()
    day_start, day_end = 8, 20
    initial_day = (diag_df["initial_failure_hour"] >= day_start) & (diag_df["initial_failure_hour"] < day_end)
    optimal_night = (diag_df["optimal_slot_hour"] < day_start) | (diag_df["optimal_slot_hour"] >= day_end)
    pct_day_to_night = (initial_day & optimal_night).sum() / max(initial_day.sum(), 1) * 100

    results = {
        "clock_shift_mean_hours": shift_mean,
        "clock_shift_median_hours": shift_median,
        "pct_initial_day_that_optimal_night": pct_day_to_night,
        "rank1_slot_counts": diag_df["rank1_slot_label"].value_counts().to_dict(),
        "rank1_slot_counts_shuffled": diag_df["rank1_slot_label_shuffled"].value_counts().to_dict(),
    }
    return diag_df, results


def plot_circadian_diagnostic(
    diag_df: pd.DataFrame,
    results: dict,
    save_path: Optional[str] = None,
):
    """Plot initial vs optimal localized hour, clock-time shift, and shuffle comparison."""
    if len(diag_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Initial failure localized hour
    ax = axes[0, 0]
    ax.hist(diag_df["initial_failure_hour"], bins=24, range=(0, 24), color="steelblue", edgecolor="white")
    ax.axvspan(0, 8, alpha=0.2, color="gray", label="Night")
    ax.axvspan(8, 20, alpha=0.2, color="yellow", label="Day")
    ax.set_xlabel("Localized hour of initial failure")
    ax.set_ylabel("Count")
    ax.set_title("Initial failure: time of day (local)")
    ax.legend(loc="upper right")

    # 2) Optimal slot localized hour
    ax = axes[0, 1]
    ax.hist(diag_df["optimal_slot_hour"], bins=24, range=(0, 24), color="coral", edgecolor="white")
    ax.axvspan(0, 8, alpha=0.2, color="gray")
    ax.axvspan(8, 20, alpha=0.2, color="yellow")
    ax.set_xlabel("Localized hour of model's predicted optimal slot")
    ax.set_ylabel("Count")
    ax.set_title("Optimal slot: time of day (local)")

    # 3) Clock-time shift (optimal - initial, wrapped)
    ax = axes[1, 0]
    ax.hist(diag_df["clock_shift_hours"], bins=24, color="green", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Clock-time shift (optimal − initial, hours)")
    ax.set_ylabel("Count")
    shift_mean = results.get("clock_shift_mean_hours", 0)
    shift_med = results.get("clock_shift_median_hours", 0)
    ax.set_title(f"Clock-time shift (mean={shift_mean:.1f}h, median={shift_med:.1f}h)")

    # 4) Shuffle test: rank-1 slot distribution normal vs shuffled
    ax = axes[1, 1]
    counts = diag_df["rank1_slot_label"].value_counts()
    counts_sh = diag_df["rank1_slot_label_shuffled"].value_counts()
    all_labels = sorted(set(counts.index) | set(counts_sh.index), key=lambda x: int(x.replace("h", "")))
    n = len(all_labels)
    x = np.arange(n)
    w = 0.35
    v1 = counts.reindex(all_labels, fill_value=0).values
    v2 = counts_sh.reindex(all_labels, fill_value=0).values
    ax.bar(x - w / 2, v1, width=w, label="Normal", color="steelblue")
    ax.bar(x + w / 2, v2, width=w, label="Hour shuffled", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Rank-1 slot: normal vs shuffle hour (if similar → bias in delay, not clock)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def run_shuffle_test_summary(diag_df: pd.DataFrame, results: dict) -> None:
    """Print summary: if rank-1 distribution is similar with shuffled hour, bias is in time_since_prev_attempt."""
    if len(diag_df) == 0:
        print("No diagnostic data.")
        return
    print("--- Circadian bias diagnostic ---")
    print(f"Clock-time shift (hours): mean = {results.get('clock_shift_mean_hours', np.nan):.2f}, median = {results.get('clock_shift_median_hours', np.nan):.2f}")
    print(f"% of initial failures in Day (8–20h) for which optimal slot is Night: {results.get('pct_initial_day_that_optimal_night', np.nan):.1f}%")
    print("\nRank-1 slot distribution (normal):", results.get("rank1_slot_counts", {}))
    print("Rank-1 slot distribution (hour_sin/hour_cos shuffled):", results.get("rank1_slot_counts_shuffled", {}))
    print("\nShuffle test: If the two distributions are very similar, the model is largely using delay (time_since_prev_attempt), not clock time.")
