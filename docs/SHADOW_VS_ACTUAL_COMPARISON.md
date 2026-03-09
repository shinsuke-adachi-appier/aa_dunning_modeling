# Shadow vs Actual (Chargebee) Comparison

This document describes the **statistical evaluation pipeline** that compares the model's shadow predictions (optimal retry time per invoice) with actual Chargebee outcomes from BigQuery. The script implements dual-path evaluation, **calibration and discriminative metrics**, **bootstrap confidence intervals**, quantile-based calibration (deciles), relative opportunity gap, Expected Value (EV) lift, **multiple temporal windows**, and decline-code segmentation.

---

## 1. Data Loading

### Shadow log

- **Source:** `artifacts/shadow_log.csv` (produced by `shadow_monitoring_20260206.py`). Only `shadow_log.csv` is used; `shadow_slot_log.csv` is not used by this comparison.
- **Key columns:** `invoice_id` (= `linked_invoice_id`), `suggested_optimal_retry_at`, `suggested_max_prob`, `inference_run_at`, `current_attempt_no`, `raw_features_snapshot`.
- **Dual-path:** The same log is evaluated in two ways: **Initial Intent** (oldest record per `invoice_id`) and **Last-Mile Accuracy** (latest record per `invoice_id`).

### Actual outcomes (BigQuery)

- **Source:** BigQuery table/view (default: `aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback`).
- **Query logic:** For each `linked_invoice_id`, attempts with `updated_at >= OUTCOMES_START_DATE` (default `2026-02-15`) are aggregated to derive:
  - **recovered:** 1 if any attempt had `status = 'success'`, else 0.
  - **recovered_at:** timestamp of the first successful attempt (if any).
  - **last_attempt_at:** timestamp of the last attempt.
  - **prev_decline_code** and **prev_card_status:** from the **"previous" attempt** (the attempt that is the context for the next prediction). For **recovered** invoices this is the **last attempt before the first success** (the failure that preceded recovery), so recovered invoices get the real decline code of that failure, not UNKNOWN. For **unrecovered** invoices this is the **latest** attempt (current failure). The query uses the table’s `Decline_code_norm` and `card_status` columns; the table must have these for the full query to run.
- **Timestamps:** All timestamps are normalized to **UTC** (then stored tz-naive) for consistent subtraction and comparison.
- **Merge:** Left-join of actuals onto each shadow view (Initial and Latest) on **`invoice_id`**. The primary output CSV and most report metrics use the **Last-Mile** (latest per invoice) merge.

---

## 2. Dual-Path Evaluation (Initial vs. Latest)

- **Initial Intent:** Keeps the **oldest** shadow record per `invoice_id` (by `inference_run_at`). This evaluates the model’s **original plan** at the start of the dunning cycle.
- **Last-Mile Accuracy:** Keeps the **latest** shadow record per `invoice_id`. This evaluates how well the model’s most recent suggestion aligned with outcomes.

The report compares **Overall Recovery Rate** and **TTR Lift** for both modes so you can see whether the initial plan vs. the final suggestion drives the metrics.

---

## 3. Reliability and TTR Validity

- **UTC:** All datetime columns (`inference_run_at`, `recovered_at`, `suggested_optimal_retry_at`, `last_attempt_at`) are forced to UTC (then tz-naive) before any subtraction.
- **Valid TTR only:** Rows where **`recovered_at` is before `inference_run_at`** are excluded from TTR calculations (no “future peeking” or use of stale recovery data). For those rows, `hours_suggested_to_recovered` is set to missing.

---

## 4. Core Metrics

### Calibration metrics (rows with suggestion)

- **Brier score:** Mean squared error between predicted probability and binary outcome. Lower is better; measures both calibration and discriminative ability.
- **Expected Calibration Error (ECE):** Weighted mean over probability bins of |mean predicted − actual rate|. Lower is better; reflects how well predicted probabilities match observed frequencies.
- **Maximum Calibration Error (MCE):** Max over bins of |mean predicted − actual rate|. Highlights worst-case miscalibration.

### Discriminative metrics (rows with suggestion)

- **AUC-ROC:** Area under the ROC curve; ranking quality (0.5 = random, 1 = perfect).
- **AUC-PR (average precision):** Area under the precision–recall curve; appropriate for imbalanced outcomes.

### Bootstrap confidence intervals

- **Stratified bootstrap** (by `recovered`) is run for: **recovery rate**, **TTR mean**, **Top Decile Lift Factor**, and **EV lift**.
- **BOOTSTRAP_N** (default 500, env `SHADOW_BOOTSTRAP_N`) replicates; **BOOTSTRAP_ALPHA** (default 0.05) for 95% percentile CIs. Set `SHADOW_BOOTSTRAP_N=0` to disable.

### Relative opportunity gap (high-value threshold)

- **Definition:** Instead of a fixed probability threshold, the script uses a **dynamic** threshold: **HIGH_PROB_THRESHOLD = BASELINE_LIFT_FACTOR × global_recovery_rate** (default `BASELINE_LIFT_FACTOR = 3.0`).
- **Logic:** If an invoice has predicted success probability **at least 3×** the global recovery rate, it is a “High-Value Opportunity,” whether that probability is 2% or 20%.
- **Flag:** `high_prob_missed_by_chargebee` = (recovered = 0) and (model has a suggestion) and (`suggested_max_prob` > HIGH_PROB_THRESHOLD).

### Temporal proximity (TTR)

- **Definition:** `hours_suggested_to_recovered` = `recovered_at` − `suggested_optimal_retry_at` (in hours), **only when** `recovered_at >= inference_run_at`.
- **Use:** Positive = Chargebee recovered *after* the model’s slot (model could have “saved” time); negative = recovered *before* the model’s slot.

### Quantile-based calibration (deciles)

- **Definition:** Rows with a model suggestion are binned into **10 equal-sized groups** by `suggested_max_prob` using `pd.qcut(..., 10, duplicates='drop')`.
- **Metrics:** For each decile: **count**, **recovered**, **mean_predicted**, **actual_rate**, and **calibration_error** = |mean_predicted − actual_rate|.
- **Top Decile Lift Factor** = (Top decile success rate) / (Global success rate). A lift > 1 indicates the highest-probability segment contains more successes than average (lift curve).

### Expected Value (EV) lift

- **Definition:** For every row where the model suggested a retry, **EV_lift = invoice_amount × (suggested_max_prob − baseline_recovery_rate)**.
- **Invoice amount:** Taken from actuals (e.g. BigQuery `amount` if present) or derived from `raw_features_snapshot` as **expm1(log_charge_amount)**.
- **KPI:** **Total Potential Revenue Lift** = sum of EV_lift over the shadow period. This is the core business-case metric.

### Transition matrix by decline code

- **Source of prev_decline_code / prev_card_status:** When available from **actuals** (BigQuery or CSV), the script uses those values—they come from the "previous" attempt as above (last failure before success for recovered, latest attempt for unrecovered). Where actuals are missing or null, the script falls back to parsing **`raw_features_snapshot`** JSON from the shadow log. This way recovered invoices show the decline code of the failure that preceded recovery, not UNKNOWN.
- **Metrics:** For each decline category (e.g. Insufficient Funds, Technical, Expired Card, UNKNOWN): **recovery rate** and **mean TTR** (hours) among recovered invoices. Use this to see if the model performs better on specific decline types.

### Top-1 and temporal proximity

- **Top-1 performance:** Among **recovered** invoices, the share where |`hours_suggested_to_recovered`| ≤ 6 (recovery within ±6 h of the model’s #1 slot).
- **Multiple temporal windows:** The report also reports the share of recoveries within ±12 h and ±24 h of the model’s suggested slot (config: **TEMPORAL_WINDOWS_H** = [6, 12, 24] in the script).

---

## 5. Reporting Outputs

### Text report: `artifacts/shadow_vs_actual_report.txt`

Contains:

0. **Executive summary:** High-level summary (recovery rate, ranking, calibration, TTR, high-value missed).
1. **Data:** Unique invoices shadowed; total shadow rows; rows with suggestion; actual outcomes matched; shadow inference date range; recovered rows with valid TTR.
2. **Calibration metrics:** Brier score, ECE, MCE (rows with suggestion).
3. **Discriminative metrics:** AUC-ROC, AUC-PR (rows with suggestion).
4. **Dual-path evaluation:** Overall recovery rate and TTR lift for **Initial Intent** and **Last-Mile Accuracy**.
5. **Recoveries (Last-Mile):** Total recoveries, overall recovery rate, and optional 95% bootstrap CI.
6. **Relative opportunity gap:** Dynamic threshold (BASELINE_LIFT_FACTOR × global rate); count of high-value opportunities missed by Chargebee.
7. **Temporal proximity:** % of recoveries within ±6 h, ±12 h, ±24 h of the model’s #1 slot.
8. **TTR lift:** Mean (recovered_at − suggested_optimal_retry_at) in hours, **valid TTR only**, and optional 95% bootstrap CI.
9. **Decile calibration:** Table with count, recovered, mean_predicted, actual_rate, calibration_error per decile; **Top Decile Lift Factor** and optional 95% bootstrap CI.
10. **EV lift:** Total potential revenue lift and optional 95% bootstrap CI.
11. **Transition matrix by decline code:** Recovery rate and mean TTR per `prev_decline_code`.

### Comparison CSV: `artifacts/shadow_vs_actual_comparison.csv`

- Based on the **Last-Mile** merge (latest record per invoice). Columns include: shadow fields, `recovered`, `recovered_at`, `last_attempt_at`, `hours_suggested_to_recovered`, `high_prob_missed_by_chargebee`, `high_prob_threshold_used`, `decile`, `recovered_within_top1_window`, **`recovered_within_±6h`**, **`recovered_within_±12h`**, **`recovered_within_±24h`** (when TEMPORAL_WINDOWS_H = [6, 12, 24]), `invoice_amount`, `ev_lift`, `prev_decline_code`, `prev_card_status`. The latter two come from actuals (BQ "previous" attempt) when available, else from the shadow log’s `raw_features_snapshot`.

### Calibration plot: `artifacts/calibration_plot.png`

- **Reliability diagram (deciles):** Bar chart of **mean predicted P(success)** vs **actual recovery rate** by decile (0 = lowest P, 9 = highest P). Reference line for perfect calibration.

### Cumulative gains plot: `artifacts/cumulative_gains_plot.png`

- **X-axis:** % of total invoices (sorted by model probability, high to low).
- **Y-axis:** % of total actual recoveries captured.
- **Use:** Shows how much of total recovery is captured by the top X% of invoices when ordered by model score; curve above the diagonal indicates lift over random.

---

## 6. How to Run

```bash
cd aa_dunning_modeling

# Use BigQuery for actual outcomes (default)
python compare_shadow_vs_actual_20260206.py

# Use a CSV of actual outcomes instead of BigQuery
python compare_shadow_vs_actual_20260206.py --actual-csv path/to/actuals.csv

# Custom paths
python compare_shadow_vs_actual_20260206.py \
  --shadow artifacts/shadow_log.csv \
  --output artifacts/shadow_vs_actual_comparison.csv \
  --report artifacts/shadow_vs_actual_report.txt \
  --cal-plot artifacts/calibration_plot.png \
  --gains-plot artifacts/cumulative_gains_plot.png
```

**Environment variables (optional):**

- `SHADOW_LOG_PATH` — shadow log CSV
- `COMPARISON_OUTPUT_PATH` — comparison CSV
- `REPORT_OUTPUT_PATH` — report text file
- `CALIBRATION_PLOT_PATH` — calibration (reliability) plot PNG
- `GAINS_PLOT_PATH` — cumulative gains plot PNG
- `SHADOW_BOOTSTRAP_N` — number of bootstrap replicates for CIs (default 500; set to 0 to disable)

**Script config constants** (in `compare_shadow_vs_actual_20260206.py`): `TEMPORAL_WINDOWS_H`, `ECE_N_BINS`, `BOOTSTRAP_N`, `BOOTSTRAP_ALPHA`.

---

## 7. BigQuery Table

The script uses `BQ_TABLE` (default: `aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback`). The table must have:

- **Required:** `linked_invoice_id`, `updated_at`, `status` — used to compute `recovered`, `recovered_at`, and `last_attempt_at` per invoice.
- **Required for prev_decline_code / prev_card_status:** `Decline_code_norm` and `card_status` — the query uses them to populate the "previous" attempt’s decline code and card status (so recovered invoices get the decline code of the failure before the success, not UNKNOWN). If your schema uses different column names (e.g. `decline_code_norm`), adjust the `base` CTE in `fetch_actual_outcomes()`.

The query filters `updated_at >= OUTCOMES_START_DATE` (default `2026-02-15`) and returns one row per `linked_invoice_id` with the aggregates and the previous attempt’s `prev_decline_code` and `prev_card_status`. For **invoice_amount** (EV lift), the script derives it from the shadow log’s `raw_features_snapshot` (expm1(log_charge_amount)) or from an `amount` column if you provide actuals via CSV; you can extend the BQ query to include `MAX(CAST(amount AS FLOAT64)) AS amount` if the table has it.
