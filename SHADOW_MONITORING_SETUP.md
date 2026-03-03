# Shadow Monitoring Setup (Local Cron)

Step-by-step instructions to run the dunning shadow monitoring script on a schedule using a local cron job. The script fetches active dunning invoices from BigQuery, runs the calibrated model to get the "model's best choice" retry time per invoice, and writes a CSV log for later comparison with Chargebee.

---

## 1. Prerequisites

- **Python 3.8+** (3.9+ recommended; BigQuery client may warn on 3.8).
- **Project directory:** All commands assume you are in or reference the `aa_dunning_modeling` folder (e.g. `~/Documents/local/AA/aa_dunning_modeling` or `/path/to/aa_dunning_modeling`).
### 1.1 Dependencies

From the project root that contains `aa_dunning_modeling`:

```bash
cd aa_dunning_modeling
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn catboost joblib tqdm \
  google-cloud-bigquery pydata-google-auth timezonefinder pgeocode db-dtypes
```

If `txn_pipeline.py` imports `country_timezones`, ensure that module is on `PYTHONPATH` or in the same directory (e.g. copy `country_timezones.py` into `aa_dunning_modeling` if it lives elsewhere).

### 1.2 BigQuery access

- The script uses **pydata_google_auth** (interactive or cached credentials). For a **cron job**, you must have credentials that work non-interactively:
  - **Option A:** Run the script once manually from the same user that will run cron; pydata_google_auth may cache tokens (depends on config).
  - **Option B:** Use a **service account**: set `GOOGLE_APPLICATION_CREDENTIALS` to a JSON key path and ensure `txn_pipeline.get_bq_client` uses it (you may need to avoid clearing `GOOGLE_APPLICATION_CREDENTIALS` in that case).
- The script queries `aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback` in project `aa-datamart`, location `europe-west1`. Your account or service account must have read access to that dataset.

### 1.3 Trained model artifact

The shadow script needs the **calibrated** model file. Either:

- **From training script:** After running `train_dunning_v2_20260206.py`, the calibrated model is saved as  
  `aa_dunning_modeling/models/catboost_dunning_calibrated_20260224.joblib`.
- **From notebook:** Export the calibrated model from `dunning_modeling.ipynb`:
  ```python
  import joblib
  joblib.dump(model_temporal_calibrated, "models/catboost_dunning_calibrated_20260224.joblib")
  ```

Ensure the path you will use in Step 3 exists (e.g. create `models/` or `artifacts/` as needed).

---

## 2. Directory layout

After setup, you should have something like:

```
aa_dunning_modeling/
├── .env                            # optional; copy from .env.example (gitignore)
├── .env.example                    # template for .env
├── .venv/                          # virtualenv (optional but recommended)
├── notebook/
│   └── dunning_modeling.ipynb
├── models/                         # or artifacts/
│   └── catboost_dunning_calibrated_20260224.joblib
├── ranking_backtest.py
├── shadow_monitoring_20260206.py
├── train_dunning_v2_20260206.py
├── txn_pipeline.py
├── country_timezones.py            # if required by txn_pipeline
├── SHADOW_MONITORING_SETUP.md      # this file
├── run_shadow_cron.sh              # optional wrapper (Step 4)
└── logs/                           # optional; for cron stdout/stderr
```

---

## 3. Environment variables and .env (optional)

You can override defaults by setting environment variables. The **recommended** way is to use a `.env` file so paths and version stay in one place and are loaded automatically by `run_shadow_cron.sh`.

### 3.1 Create .env from the example

```bash
cd aa_dunning_modeling
cp .env.example .env
# Edit .env and set paths for your machine (see below).
```

Add `.env` to `.gitignore` so you don’t commit machine-specific paths or secrets:

```bash
echo ".env" >> .gitignore
```

### 3.2 Variables

| Variable | Default | Description |
|---------|--------|-------------|
| `DUNNING_MODEL_PATH` | `models/catboost_dunning_calibrated_20260224.joblib` (under script dir) | Full path to the calibrated joblib model file. |
| `DUNNING_MODEL_VERSION` | `20260224` | String written as `model_version_id` in the CSV. |
| `SHADOW_LOG_PATH` | `artifacts/shadow_log.csv` | Full path for the main shadow output CSV (append mode). |
| `SHADOW_SLOT_LOG_PATH` | `artifacts/shadow_slot_log.csv` | Full path for per-invoice wide slot probabilities CSV (append mode). |
| `GOOGLE_APPLICATION_CREDENTIALS` | (none) | For cron: path to service account JSON key for non-interactive BigQuery. |

**Recommendation:** In `.env`, set `DUNNING_MODEL_PATH` to the full path of your trained model so the script doesn’t rely on the default.

Example `.env` (uncomment and set as needed):

```bash
# DUNNING_MODEL_PATH=/Users/you/path/to/aa_dunning_modeling/models/catboost_dunning_calibrated_20260224.joblib
# SHADOW_LOG_PATH=/Users/you/path/to/aa_dunning_modeling/artifacts/shadow_log.csv
# SHADOW_SLOT_LOG_PATH=/Users/you/path/to/aa_dunning_modeling/artifacts/shadow_slot_log.csv
# DUNNING_MODEL_VERSION=20260224
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

Alternatively, set variables in the shell before running (or in your crontab); the cron wrapper uses defaults when variables are unset.

---

## 4. Run once manually

Before scheduling, run the script once from the terminal to confirm BigQuery, model load, and CSV output.

```bash
cd /path/to/aa_dunning_modeling
source .venv/bin/activate
# Optional: load .env then run
# [ -f .env ] && set -a && . .env && set +a
python shadow_monitoring_20260206.py
```

**Two-week shadow run:** See [RUNBOOK_TWO_WEEK_SHADOW.md](RUNBOOK_TWO_WEEK_SHADOW.md) for retrain → daily cron for 14 days → comparison.

- If BigQuery auth is interactive, complete the browser/login step.
- Check that the script prints "Appended N new predictions to ..." and (if enabled) "Appended N slot log rows ...".
- Open `artifacts/shadow_log.csv` (or `SHADOW_LOG_PATH`) and confirm columns:
  `inference_run_id,invoice_id,customer_id,inference_run_at,model_version_id,current_attempt_no,snapshot_hour,days_into_dunning,suggested_optimal_retry_at,suggested_max_prob,raw_features_snapshot`.
- Open `artifacts/shadow_slot_log.csv` (or `SHADOW_SLOT_LOG_PATH`) and confirm columns:
  `inference_run_id,invoice_id,customer_id,inference_run_at,snapshot_hour_jst,days_into_dunning,prob_24h,prob_28h,...,prob_120h`.

---

## 5. Optional: wrapper script for cron

Cron runs with a minimal environment. A small wrapper script sets the working directory, virtualenv, and env vars, and logs stdout/stderr.

Create `aa_dunning_modeling/run_shadow_cron.sh`:

```bash
#!/usr/bin/env bash
set -e
AA_DUNNING_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$AA_DUNNING_ROOT"
export DUNNING_MODEL_PATH="$AA_DUNNING_ROOT/models/catboost_dunning_calibrated_20260224.joblib"
export SHADOW_LOG_PATH="$AA_DUNNING_ROOT/artifacts/shadow_log.csv"
export SHADOW_SLOT_LOG_PATH="$AA_DUNNING_ROOT/artifacts/shadow_slot_log.csv"
export DUNNING_MODEL_VERSION="20260224"

LOG_DIR="$AA_DUNNING_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/shadow_$(date +%Y%m%d_%H%M%S).log"

if [ -d "$AA_DUNNING_ROOT/.venv" ]; then
  source "$AA_DUNNING_ROOT/.venv/bin/activate"
fi
python shadow_monitoring_20260206.py >> "$LOG" 2>&1
```

Make it executable:

```bash
chmod +x aa_dunning_modeling/run_shadow_cron.sh
```

Run it once to confirm:

```bash
/path/to/aa_dunning_modeling/run_shadow_cron.sh
cat aa_dunning_modeling/logs/shadow_*.log
```

---

## 6. Install the cron job

### 6.1 Open crontab

```bash
crontab -e
```

### 6.2 Add a line for the shadow script

Run every day at 6:00 AM (adjust time and path to your setup):

```cron
0 6 * * * /path/to/aa_dunning_modeling/run_shadow_cron.sh
```

Or run the Python script directly (replace with your paths and venv):

```cron
0 6 * * * cd /path/to/aa_dunning_modeling && .venv/bin/python shadow_monitoring_20260206.py >> /path/to/aa_dunning_modeling/logs/shadow_cron.log 2>&1
```

Use **absolute paths** in crontab. Example with wrapper:

```cron
0 6 * * * /Users/you/Documents/local/AA/aa_dunning_modeling/run_shadow_cron.sh
```

### 6.3 Save and exit

Save the crontab file and exit the editor (e.g. in vim: `:wq`).

### 6.4 Verify

```bash
crontab -l
```

You should see the line you added.

---

## 7. Logs and output

- **CSV output (append mode):**
  - `SHADOW_LOG_PATH` defaults to `artifacts/shadow_log.csv`
  - `SHADOW_SLOT_LOG_PATH` defaults to `artifacts/shadow_slot_log.csv`
  - Both files append one batch per run (headers written only when the file does not exist).
  - Keep these as canonical files; optional archival (weekly/monthly copies) can be done separately.

- **Cron logs:** If you use `run_shadow_cron.sh`, each run creates `logs/shadow_YYYYMMDD_HHMMSS.log`. Optionally prune old logs (e.g. keep last 30 days):

  ```bash
  find /path/to/aa_dunning_modeling/logs -name 'shadow_*.log' -mtime +30 -delete
  ```

  You can add this to a weekly cron job if you like.

---

## 8. Troubleshooting

| Issue | What to check |
|-------|----------------|
| **"No module named 'txn_pipeline'"** | Run from `aa_dunning_modeling` or set `PYTHONPATH` to `aa_dunning_modeling` in the cron wrapper. |
| **"No module named 'country_timezones'"** | Copy `country_timezones.py` into `aa_dunning_modeling` or add its parent to `PYTHONPATH`. |
| **"No active dunning invoices"** | BigQuery query returns empty (e.g. no data in last 5 days). Check query and filters in `fetch_active_dunning_invoices()`. |
| **BigQuery auth error in cron** | Use a service account and `GOOGLE_APPLICATION_CREDENTIALS`, or ensure cached user credentials are available for the cron user. |
| **Model file not found** | Set `DUNNING_MODEL_PATH` to the full path of `catboost_dunning_calibrated_20260224.joblib` (e.g. in the wrapper script). |
| **Permission denied** | `chmod +x run_shadow_cron.sh` and ensure cron user can read the repo and write to `artifacts/` and `logs/`. |

---

## 9. Quick reference

| Step | Command / action |
|------|------------------|
| One-time setup | `cd aa_dunning_modeling && python -m venv .venv && source .venv/bin/activate && pip install ...` |
| Train model | `python train_dunning_v2_20260206.py` → writes `models/catboost_dunning_calibrated_20260224.joblib` |
| Run shadow once | `python shadow_monitoring_20260206.py` |
| Cron (daily 6 AM) | `0 6 * * * /absolute/path/to/aa_dunning_modeling/run_shadow_cron.sh` |
| View last log | `cat aa_dunning_modeling/logs/shadow_*.log \| tail -100` |
| Output CSVs | Defaults: `aa_dunning_modeling/artifacts/shadow_log.csv` and `aa_dunning_modeling/artifacts/shadow_slot_log.csv` |
