# Runbook: Retrain Model & Two-Week Automated Shadow Run

Use this to retrain the dunning model and run automated shadow monitoring for **two weeks**, then compare against Chargebee outcomes.

---

## Prerequisites

- Python env and dependencies (see [SHADOW_MONITORING_SETUP.md](SHADOW_MONITORING_SETUP.md)).
- BigQuery access for training data and shadow active-invoices.
- Cron (or another scheduler) available for daily runs.

---

## Step 1: Retrain the model (today)

1. **Optional: refresh training data from BigQuery**  
   To use the latest data, re-query and overwrite the cache:
   ```bash
   cd aa_dunning_modeling
   source .venv/bin/activate
   FORCE_QUERY=1 python train_dunning_v2_20260206.py
   ```
   To use existing cached data (faster):
   ```bash
   python train_dunning_v2_20260206.py
   ```

2. **Check outputs**  
   - Model: `models/catboost_dunning_calibrated_20260206.joblib`
   - Report: validation AUC and PR-AUC in the console; `eval_plots_20260206.png` in the project root.

3. **Confirm cron uses this model**  
   Default in `run_shadow_cron.sh` is:
   `DUNNING_MODEL_PATH=$AA_DUNNING_ROOT/models/catboost_dunning_calibrated_20260206.joblib`  
   If you use a different path, set `DUNNING_MODEL_PATH` in `.env`.

---

## Step 2: Enable daily shadow run for two weeks (start tomorrow)

1. **Cron schedule**  
   Run the shadow script **once per day** (e.g. 6:00 AM):
   ```bash
   crontab -e
   ```
   Add (replace with your actual path):
   ```cron
   0 6 * * * /path/to/aa_dunning_modeling/run_shadow_cron.sh
   ```

2. **Date-stamped logs**  
   The cron wrapper now writes a **new file each day**:
   - `artifacts/shadow_log_YYYYMMDD.csv` (e.g. `shadow_log_20260211.csv`, `shadow_log_20260212.csv`, …)
   - No overwrite between days; after 14 days you have 14 files.

3. **Duration**  
   Leave the cron entry in place for **14 days** (e.g. from tomorrow until 14 days later). No code change needed to “stop” after 2 weeks; you simply stop editing crontab or remove the line after the run period.

4. **Optional: single fixed path**  
   If you prefer one file overwritten daily, set in `.env`:
   ```bash
   SHADOW_LOG_PATH=/path/to/aa_dunning_modeling/artifacts/shadow_log.csv
   ```

---

## Step 3: After the two-week window — run comparison

1. **Combine shadow logs (optional)**  
   If you want one CSV for the whole period:
   ```bash
   cd aa_dunning_modeling/artifacts
   # Example: concatenate all shadow_log_*.csv, keep header once
   head -1 shadow_log_20260211.csv > shadow_log_2weeks.csv
   tail -n +2 -q shadow_log_202602*.csv >> shadow_log_2weeks.csv
   ```
   Or point the comparison script at a single day’s file (see below).

2. **Run comparison vs actuals**  
   Uses the latest shadow log by default (or a specific file). Actuals come from BigQuery (or `--actual-csv`):
   ```bash
   cd aa_dunning_modeling
   source .venv/bin/activate
   python compare_shadow_vs_actual_20260206.py
   ```
   To use a specific shadow file (e.g. one day or the combined 2-week file):
   ```bash
   python compare_shadow_vs_actual_20260206.py --shadow artifacts/shadow_log_2weeks.csv
   ```

3. **Check outputs**  
   - `artifacts/shadow_vs_actual_comparison.csv`
   - `artifacts/shadow_vs_actual_report.txt`
   - `artifacts/calibration_plot.png`

---

## Summary checklist

| When            | Action |
|-----------------|--------|
| **Today**       | 1. Run `FORCE_QUERY=1 python train_dunning_v2_20260206.py` (or without `FORCE_QUERY` to use cache). 2. Confirm `models/catboost_dunning_calibrated_20260206.joblib` exists. |
| **Tomorrow**    | Cron starts; first `artifacts/shadow_log_YYYYMMDD.csv` is written. |
| **Days 2–14**   | Cron runs daily; one new `shadow_log_YYYYMMDD.csv` per day. |
| **After day 14**| Run `python compare_shadow_vs_actual_20260206.py` (optionally with `--shadow artifacts/shadow_log_2weeks.csv` if you combined files). Review report and calibration plot. |

---

## Optimal retry time: rounded to nearest hour

The model’s **suggested optimal retry time** (`suggested_optimal_retry_at`) is **rounded to the nearest hour** before being written to the shadow log (e.g. 13:53 → 14:00, 13:22 → 13:00). This is done in `shadow_monitoring_20260206.py` so dunning send time is always at the hour.
