# Dunning Model: Production Deployment on Apache Airflow

This guide describes how to run the dunning retry-time model in **production on Apache Airflow**: predict the hour with highest success probability per invoice and **send an API request (e.g. Chargebee retry) at that hour**. Deployment uses the **standalone `deploy/` folder**: two **Airflow DAGs** (inference + trigger) with the same code, plus **Airflow Variables/Connections** for config and secrets. Guardrails: **rate limiting**, **idempotency**, **velocity cap**, **feature drift logging**, **model fallback**, and **shadow-to-live** rollout.

---

## Part 1 — Architecture on Airflow

### 1.1 Components

| Component | Implementation on Airflow |
|-----------|---------------------------|
| **Inference** | One **Airflow DAG**. Runs every 4–6 h (`schedule_interval`). Fetches active dunning from BigQuery, loads model from GCS, scores 25 slots per invoice, writes to `production_dunning_schedule` (+ optional feature log). Fallback: 24h if model fails. Task runs `python inference_job/main.py` (or equivalent container) with env from Airflow Variables/Connections. |
| **Trigger** | One **Airflow DAG**. Runs **hourly** (`schedule_interval`). Reads schedule for current hour (UTC), applies jitter + rate limit + velocity cap, calls Retry API with `invoice_id` and Idempotency-Key, writes to `dunning_retry_trigger_log`. If an invoice is already paid, the Retry API returns an error and we log it. Task runs `python trigger_job/main.py` with env from Variables/Connections. |
| **Scheduling** | **Airflow**: `schedule_interval` on each DAG (e.g. inference `0 */6 * * *`, trigger `0 * * * *`). No separate scheduler service. |
| **Secrets / Config** | **Airflow Variables** for non-secret config (e.g. `BQ_PROJECT`, `BQ_DATASET`, `GCS_MODEL_URI`). **Airflow Connections** or **Variables** (encrypted) for Billing/Chargebee API keys. BigQuery uses the Airflow worker’s credentials (e.g. GCP connection or default service account). |

### 1.2 Data Flow

```
Airflow (every 6h) → DAG: dunning_inference
    → Task runs inference_job/main.py
    → BQ (active dunning) → Model (GCS) → production_dunning_schedule [+ optional feature_log]

Airflow (hourly) → DAG: dunning_trigger
    → Task runs trigger_job/main.py
    → BQ (schedule, current hour) → jitter + rate limit + velocity cap
    → jitter + rate limit → POST /retry (invoice_id + Idempotency-Key) → dunning_retry_trigger_log
```

### 1.3 Source Data and Refresh

- **Data source:** The inference job does **not** read a single pre-built view. It runs an **embedded pipeline query** in `deploy/lib/bq_fetch.py` (`ACTIVE_DUNNING_QUERY`) that reads from:
  - `billing_dm.v_subscription_customer_flat` (one row per customer for subscription context),
  - `billing_stg.stg_cb_transactions_enriched_tbl` (transaction data with `raw_json` for linked invoices).
  **Active dunning** is defined in the query as: `invoice_success_attempt_no IS NULL`, `invoice_attempt_count < 12`, `updated_at` within the last **5 days**, **Soft decline only** (`Decline_type_for_retry = 'Soft decline'`; hard-decline types are excluded), and the **latest row per `linked_invoice_id`**.
- **Refresh:** The pipeline uses the tables above; refresh them (or their upstream sources) at least every 6–12 h (ideally 4–6 h) so inference sees new failures sooner. If they are daily, run inference shortly after refresh and document in the runbook.
- **Pre-Flight:** Not used. The Retry API is called for all schedule rows for the current hour; if an invoice is already paid, the API returns an error and we log it.

---

## Part 2 — Critical Guardrails

### 2.1 Jitter and Rate Limiting (Mandatory)

The trigger job shuffles the list of due invoices and applies a random delay per invoice (`JITTER_MAX_SECONDS`, default 1800) and a cap on requests per minute (`RATE_LIMIT_PER_MIN`, default 50). Configure via env (Airflow Variables) on the trigger task.

### 2.2 Idempotency Key (Mandatory)

Use a deterministic key: `invoice_id|attempt_number|scheduled_hour_iso` (e.g. `inv_xyz|3|2026-02-28T10:00:00Z`). Send as header `Idempotency-Key` to the Retry API so duplicate runs (e.g. Airflow retry, double trigger) do not cause double charges. **A Chargebee implementation is provided** in `deploy/trigger_job/main.py`: `call_retry_api(invoice_id, idempotency_key)` calls POST `/invoices/{id}/collect_payment`. For a different Retry API, implement your own using `RETRY_API_URL` and auth.

### 2.3 Velocity cap (rolling window)

To avoid being aggressive and being flagged by the payment provider, the trigger job enforces **no more than N retries per invoice in any rolling N-day window**. Before calling the Retry API for an invoice, it queries `dunning_retry_trigger_log` for that invoice’s count of **real** Retry API attempts (excluding `DRY_RUN`, `VELOCITY_CAP_7D`, and `call_retry_api not implemented`) in the last N days. If the count is already ≥ the cap, the job logs a row with `error_message='VELOCITY_CAP_7D'` and does **not** call the Retry API. Configure via env: `VELOCITY_CAP_DAYS` (default 7), `VELOCITY_CAP_MAX_RETRIES_7D` (default 3).

### 2.4 Model Fallback

If the model fails to load or inference fails for an invoice, the inference job sets `optimal_retry_at_utc = now + 24h`. The schedule is still written so dunning continues.

---

## Part 3 — Deploy Folder and Code

All production code lives in **`deploy/`** (standalone; no parent repo required). Airflow tasks invoke this code (e.g. from a checked-out repo or mounted path).

| Path | Purpose |
|------|--------|
| **deploy/lib/** | Model load (GCS/local), features, slot generation, BQ fetch. |
| **deploy/inference_job/main.py** | Entrypoint for the inference task. |
| **deploy/trigger_job/main.py** | Entrypoint for the trigger task. **Chargebee implementation included:** `call_retry_api` (POST collect_payment). Replace with your own if using a different Retry API. No pre-flight status check; the Retry API returns an error for already-paid invoices. |
| **deploy/bq_schema/*.sql** | DDL for `production_dunning_schedule` and `dunning_retry_trigger_log`. Feature log table is optional and created from Python when first writing. |
| **deploy/requirements-deploy.txt** | Pinned dependencies: **scikit-learn>=1.8,<1.9** (match training for model unpickling), **pandas-gbq>=0.26.1** (for BigQuery DataFrame load), catboost, google-cloud-bigquery, google-cloud-storage, python-dotenv (optional, for local .env loading), etc. |
| **deploy/scripts/insert_test_schedule_rows.py** | Optional: one-off script to insert test rows into the schedule table for the current hour (UTC) for local trigger-job testing. See **docs/LOCAL_TESTING.md**. |
| **deploy/Dockerfile** | Optional: build image for DockerOperator/KubernetesPodOperator. |
| **deploy/README.md** | Build/run instructions and env reference. |
| **docs/INFERENCE_JOB_FLOW.md** | Step-by-step flow of the inference job (in repo docs). |

---

## Part 4 — BigQuery

### 4.1 Tables to Create

1. **production_dunning_schedule**  
   Columns: `invoice_id`, `optimal_retry_at_utc`, `attempt_number`, `max_prob`, `inference_run_id`, `created_at`, `status`, `model_version_id`. Partition by `DATE(optimal_retry_at_utc)`. Use `deploy/bq_schema/production_dunning_schedule.sql` (replace `PROJECT_ID`, `DATASET`).  
   **If the table already exists** without `model_version_id`, add it:  
   `ALTER TABLE \`project.dataset.production_dunning_schedule\` ADD COLUMN IF NOT EXISTS model_version_id STRING;`

2. **dunning_retry_trigger_log**  
   Columns: `invoice_id`, `optimal_retry_at_utc`, `triggered_at`, `idempotency_key`, `api_response_status`, `error_message`, `status`, `message`, `txn_id`, `attempt_number`, `comment_status`. Use `deploy/bq_schema/dunning_retry_trigger_log.sql`.

3. **dunning_inference_feature_log** (optional)  
   No SQL needed. If you set `FEATURE_LOG_TABLE`, the inference job creates the table on first write using the schema in `lib/features.py` (`FEATURE_LOG_SCHEMA`), which includes `model_version_id`.

### 4.2 Permissions

The identity used by Airflow to run the tasks (e.g. worker service account, or the account from a GCP connection) needs:  
- BigQuery Data Viewer (or read access) on the **tables used by the inference query**: `billing_dm.v_subscription_customer_flat` and `billing_stg.stg_cb_transactions_enriched_tbl` (replace project/dataset if your query uses different names; the query is in `deploy/lib/bq_fetch.py`).  
- BigQuery Data Editor (or write access) on the **dataset** where `production_dunning_schedule`, `dunning_retry_trigger_log`, and optionally `dunning_inference_feature_log` live.  
- If using GCS for the model: Storage Object Viewer on the model bucket.

---

## Part 4.5 — After shadow dunning: steps to deploy on Airflow

Once **shadow dunning** is finished and you are satisfied with the model (e.g. shadow vs. reality metrics, recovery rate), follow these steps in order to go live on Airflow.

### Prerequisite

- Shadow dunning has been run (e.g. via `shadow_monitoring_*.py` or notebook). You have a calibrated model artifact (e.g. `.joblib`) and have validated schedule vs. actual outcomes.
- Airflow is available (self-hosted, Cloud Composer, or other). Workers have access to BigQuery and GCS (or you use Docker/Kubernetes and provide credentials via env).

### Step-by-step

| # | Step | What to do | See |
|---|------|------------|-----|
| 1 | **Create BigQuery tables** | Run the DDL in `deploy/bq_schema/production_dunning_schedule.sql` and `dunning_retry_trigger_log.sql`. Replace `PROJECT_ID` and `DATASET` with your project and dataset. If the schedule table already exists without `model_version_id`, run `ALTER TABLE ... ADD COLUMN IF NOT EXISTS model_version_id STRING`. Optionally create the feature log table later (inference job can create it from Python schema when `FEATURE_LOG_TABLE` is set). | Part 4.1 |
| 2 | **Grant BQ (and GCS) permissions** | Ensure the identity used by Airflow tasks has **BigQuery Data Viewer** on the source table/view and **BigQuery Data Editor** on the dataset for schedule and trigger log. If using GCS model, **Storage Object Viewer** on the model bucket. | Part 4.2 |
| 3 | **Trigger implementation** | In `deploy/trigger_job/main.py`: **Chargebee is already implemented**. Set `CHARGEBEE_SITE` and `CHARGEBEE_API_KEY` via Airflow Variables (or Connection). For a different Retry API, implement `call_retry_api` and use your env vars. No pre-flight; the Retry API returns an error if the invoice is already paid. | Part 2.2, Part 8 |
| 4 | **Configure Airflow Variables / Connections** | Create Airflow Variables for: `BQ_PROJECT`, `BQ_DATASET`, `SCHEDULE_TABLE`, `TRIGGER_LOG_TABLE`, `GCS_MODEL_URI` (or path), optional `FEATURE_LOG_TABLE`, `FEATURE_LOG_SAMPLE_PCT`, `DUNNING_MODEL_VERSION`, `BQ_LOCATION`. For trigger: `CHARGEBEE_SITE`, `CHARGEBEE_API_KEY` (or store in Connection / secret backend). Optional: `RATE_LIMIT_PER_MIN`, `JITTER_MAX_SECONDS`, `DRY_RUN`, `TRAFFIC_SPLIT_MODEL_PCT`, `VELOCITY_CAP_DAYS`, `VELOCITY_CAP_MAX_RETRIES_7D`. (Inference reads from the embedded pipeline query in `lib/bq_fetch.py`; no `BQ_SOURCE_TABLE`.) | Part 6 |
| 5 | **Upload model to GCS** | Upload your calibrated model (e.g. `catboost_dunning_calibrated_YYYYMMDD.joblib`) to a GCS bucket, e.g. `gs://YOUR_BUCKET/dunning/models/`. Set `GCS_MODEL_URI` in Airflow Variables. | Part 9 #4 |
| 6 | **Make deploy code available to Airflow** | Ensure the `deploy/` folder (or repo containing it) is on the worker path: e.g. clone repo in DAGs folder, mount a volume, or use a Docker image that includes `deploy/` and run with DockerOperator. Install dependencies (`pip install -r deploy/requirements-deploy.txt`) in the worker environment or inside the image. | Part 6 |
| 7 | **Create the inference DAG** | DAG with `schedule_interval` e.g. `0 */6 * * *` (every 6 hours). One task: run `python inference_job/main.py` from `deploy/` with env populated from Airflow Variables (and Connections for secrets if used). Set task timeout (e.g. 3600 s) if you have many invoices. | Part 6.1, Part 7 |
| 8 | **Create the trigger DAG** | DAG with `schedule_interval` `0 * * * *` (hourly). One task: run `python trigger_job/main.py` with env from Variables/Connections. Set `DRY_RUN=1` initially for testing. | Part 6.2, Part 7 |
| 9 | **Test with DRY_RUN** | Run the **trigger** DAG manually with `DRY_RUN=1`. It should write to `dunning_retry_trigger_log` but **not** call the Retry API. Check logs and BQ. Run the **inference** DAG once and confirm rows in `production_dunning_schedule`. | Part 9 #9 |
| 10 | **Go live** | Set `DRY_RUN=0` (or remove it) for the trigger. Optionally start with `TRAFFIC_SPLIT_MODEL_PCT=10` and increase after validating recovery. Monitor trigger log and Retry API outcomes; roll back by updating the model URI Variable or setting traffic split to 0. | Part 9 #10, #11 |

### Summary flow

```
Shadow dunning done → BQ tables + permissions → Configure Variables/Connections
  → Upload model to GCS → Make deploy/ available to workers → Create inference DAG → Create trigger DAG
  → Test with DRY_RUN=1 → Go live (optionally with traffic split)
```

---

## Part 5 — Docker Image (Optional)

If you run tasks via **DockerOperator** or **KubernetesPodOperator**, build an image from the deploy folder:

1. **Build from the deploy folder:**

   ```bash
   cd aa_dunning_modeling/deploy
   docker build -t dunning-prod .
   ```

2. **Tag and push** to your container registry (e.g. Artifact Registry, ECR). Ensure Airflow workers (or K8s) can pull the image.

3. In the DAG, use DockerOperator/KubernetesPodOperator with this image, command `python`, args `inference_job/main.py` or `trigger_job/main.py`, and pass env from Airflow Variables (e.g. via `environment` or `env_vars`).

If you run tasks with **BashOperator** or **PythonOperator** on workers that have the repo and dependencies installed, you do not need a Docker image.

---

## Part 6 — Airflow DAGs and Tasks

### 6.1 Inference DAG

- **Schedule:** `schedule_interval='0 */6 * * *'` (every 6 hours) or a cron that matches your source refresh.
- **Task:** Execute `python inference_job/main.py` from the `deploy/` directory (or equivalent in-container command). Working directory must be the deploy folder so `inference_job/main.py` and `lib` resolve correctly.
- **Env (required):**  
  `BQ_PROJECT`, `BQ_DATASET`,  
  `GCS_MODEL_URI` (e.g. `gs://your-bucket/dunning/models/catboost_dunning_calibrated_20260224.joblib`) or `DUNNING_MODEL_PATH`.  
  **Production (Airflow):** Set `GCS_MODEL_URI`. The inference job prefers `DUNNING_MODEL_PATH` when that path exists (for local dev); on Airflow workers you typically only set `GCS_MODEL_URI`. The job loads `deploy/.env` when present (local runs); in production, pass all env from Airflow Variables.
- **Env (optional):**  
  `SCHEDULE_TABLE`, `FEATURE_LOG_TABLE`, `FEATURE_LOG_SAMPLE_PCT`, `DUNNING_MODEL_VERSION`, `BQ_LOCATION`.
- **Passing env:** Read Airflow Variables in the DAG and pass them as env to the task (e.g. `env={**os.environ, 'BQ_PROJECT': Variable.get('BQ_PROJECT'), ...}` for BashOperator, or equivalent for DockerOperator).
- **Timeout:** Set task execution timeout (e.g. 3600 s) for large runs.

**Example (pseudo-code; adapt to your Airflow version):**

```python
# In inference DAG
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable

with DAG(
    'dunning_inference',
    schedule_interval='0 */6 * * *',
    ...
) as dag:
    BashOperator(
        task_id='run_inference',
        bash_command='cd /path/to/deploy && python inference_job/main.py',
        env={
            'BQ_PROJECT': Variable.get('BQ_PROJECT'),
            'BQ_DATASET': Variable.get('BQ_DATASET'),
            'GCS_MODEL_URI': Variable.get('GCS_MODEL_URI'),
            # ... other vars
        },
        execution_timeout=timedelta(seconds=3600),
    )
```

### 6.2 Trigger DAG

- **Schedule:** `schedule_interval='0 * * * *'` (hourly).
- **Task:** Execute `python trigger_job/main.py` from the `deploy/` directory.
- **Env (required):**  
  `BQ_PROJECT`, `BQ_DATASET`, `SCHEDULE_TABLE`, `TRIGGER_LOG_TABLE`.
- **Env (optional):**  
  `RATE_LIMIT_PER_MIN`, `JITTER_MAX_SECONDS`, `DRY_RUN`, `TRAFFIC_SPLIT_MODEL_PCT`, `VELOCITY_CAP_DAYS`, `VELOCITY_CAP_MAX_RETRIES_7D`.  
  **For Chargebee:** `CHARGEBEE_SITE`, `CHARGEBEE_API_KEY` (from Variable or Connection).
- **Logging:** The trigger job prints progress to stderr: schedule fetch count, traffic/velocity summary, per-invoice DRY_RUN or API call result, and a final summary (total, dry_run, velocity_cap, success, failed). Use these logs for monitoring and debugging.
- **Secrets:** Prefer Airflow Connection (e.g. type HTTP with password for API key) or your secret backend; pass as env to the task.

### 6.3 Retries and Timeouts

- Configure task **retries** and **timeout** per DAG. For trigger, avoid aggressive retries that could double-invoke the Retry API (idempotency key protects, but rate limiting still applies).
- Ensure the worker (or container) uses the **same** dependency versions as in `deploy/requirements-deploy.txt`: in particular **scikit-learn>=1.8,<1.9** (model was pickled with 1.8.x) and **pandas-gbq>=0.26.1** (required for loading DataFrames into BigQuery). Install with `pip install -r deploy/requirements-deploy.txt`.

---

## Part 7 — Scheduling (Airflow)

- **Inference DAG:** Set `schedule_interval='0 */6 * * *'` (or `"0 */6 * * *"`) for every 6 hours at minute 0. Use a timezone-aware DAG if your Airflow is configured with a timezone.
- **Trigger DAG:** Set `schedule_interval='0 * * * *'` for hourly at minute 0.
- Trigger runs on **UTC** hour boundaries if the worker clock is UTC; the trigger code queries the schedule for the current UTC hour. Align DAG schedule with that (e.g. run at :00 so “current hour” is correct).

---

## Part 8 — APIs and Contracts

### 8.1 Retry API

- **Request:** `POST` with body `{ "invoice_id": "<linked_invoice_id>" }` and header `Idempotency-Key: <key>`.
- **Key format:** `{invoice_id}|{attempt_number}|{scheduled_hour_iso}` (e.g. `inv_xyz|3|2026-02-28T10:00:00Z`).
- **Response:** 200 success; 409 if key already used; 4xx/5xx with clear error.
- **Chargebee implementation:** `call_retry_api(invoice_id, idempotency_key)` in `deploy/trigger_job/main.py` calls Chargebee POST `/invoices/{id}/collect_payment` using `CHARGEBEE_SITE` and `CHARGEBEE_API_KEY`. For another Retry API, implement your own using `RETRY_API_URL` and auth from env/Variables. If an invoice is already paid, the Retry API returns an error and we log it.

---

## Part 9 — Step-by-Step Checklist (Airflow)

| # | Step | Notes |
|---|------|--------|
| 1 | Create BQ tables | Run `deploy/bq_schema/production_dunning_schedule.sql` and `dunning_retry_trigger_log.sql` (replace PROJECT_ID, DATASET). Add `model_version_id` to the schedule table if using older DDL: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS model_version_id STRING`. |
| 2 | **Retry API client** | **Chargebee:** Already implemented; set `CHARGEBEE_SITE` and `CHARGEBEE_API_KEY` in Variables/Connection. **Other:** Implement `call_retry_api()` in `deploy/trigger_job/main.py`; add Retry API URL and auth. No pre-flight status check; the Retry API returns an error for already-paid invoices. |
| 3 | Upload model to GCS | e.g. `gs://your-bucket/dunning/models/catboost_dunning_calibrated_20260224.joblib`; set `GCS_MODEL_URI` in Airflow Variables. |
| 4 | Configure Airflow Variables/Connections | All env vars used by inference and trigger (see deploy/env.example). Pass to tasks as env. |
| 5 | Make deploy code available | Repo on worker path or use Docker image; install requirements. |
| 6 | Create inference DAG | One task: run `python inference_job/main.py` with env from Variables; schedule every 6 h. |
| 7 | Create trigger DAG | One task: run `python trigger_job/main.py` with env from Variables/Connections; schedule hourly. |
| 8 | Test with DRY_RUN | Set `DRY_RUN=1` for trigger and run DAG manually; confirm logs and no real Retry API calls. For local testing, you can insert test rows for the current hour with `python deploy/scripts/insert_test_schedule_rows.py` (see **docs/LOCAL_TESTING.md**), then run the trigger job so it has rows to process. |
| 9 | Traffic split (optional) | Set `TRAFFIC_SPLIT_MODEL_PCT=10`; increase after validating recovery rate. |
| 10 | Monitoring and rollback | Model vs. Reality dashboard; rollback by updating `GCS_MODEL_URI` Variable to a previous model or setting `TRAFFIC_SPLIT_MODEL_PCT=0`. |

---

## Part 10 — Timezone and Idempotency

- **Inference and schedule:** All times in **UTC** (`optimal_retry_at_utc`, `inference_run_at`).
- **Trigger:** Query schedule for the **current hour UTC** (e.g. 08:00–08:59).
- **Idempotency key:** Use the same hour truncation (e.g. `2026-02-28T08:00:00Z`) so the key is deterministic for that hour.

---

## Part 11 — Feature Consistency and Dependency Pinning

- The environment that runs the inference task must use the **same** `scikit-learn` and `catboost` versions as the training environment so the calibrated model loads and behaves correctly. In `deploy/requirements-deploy.txt`, **scikit-learn** is pinned to `>=1.8,<1.9` (model is pickled with 1.8.x). **pandas-gbq>=0.26.1** is required for writing DataFrames to BigQuery. Use that environment (or image) in Airflow.
- **python-dotenv** is listed for optional local development (inference job loads `deploy/.env` when the file exists); in production, env is provided by Airflow Variables.
- The deploy code includes the `IsotonicCalibratedClassifier` class in `lib/model.py` so joblib can unpickle models saved at training time.

---

This guide is tailored for **Apache Airflow**: two DAGs (inference + trigger), schedule via `schedule_interval`, BigQuery, GCS for the model, and Airflow Variables/Connections for config and secrets. The **`deploy/`** folder is self-contained; make it available to Airflow workers (or run it inside a container) and invoke `inference_job/main.py` and `trigger_job/main.py` from your DAGs.
