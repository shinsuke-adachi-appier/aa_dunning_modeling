# Dunning model production deployment (standalone)

This **deploy** folder is self-contained. Upload only this folder to Google Cloud Run (or Cloud Build). No parent repo or other files are required.

---

## Layout

| Path | Purpose |
|------|--------|
| **lib/** | Model load (GCS/local), features, slot generation, BQ fetch. No dependency on parent repo. |
| **inference_job/main.py** | Fetch active dunning from BQ → run model → write `production_dunning_schedule` + optional feature log. |
| **trigger_job/main.py** | Query schedule for current hour → Pre-Flight → jitter + rate limit → Retry API → log. **You implement** `resolve_invoice_status` and `call_retry_api`. |
| **bq_schema/*.sql** | DDL for **schedule** and **trigger_log** only. Feature log table is optional and created from Python (lib/features.py) when first writing. |
| **requirements-deploy.txt** | Pinned production deps. |
| **env.example** | Env template. Copy to `.env` or set in Cloud Run. |
| **Dockerfile** | Build from **this directory**: `docker build -t dunning-deploy .` |
| **LOCAL_TESTING.md** | How to dry run and test inference + trigger locally before pushing to Google Cloud. |

---

## Build and run (standalone)

```bash
# From the deploy folder (no parent repo needed)
cd deploy
docker build -t dunning-deploy .

# Inference (set env or use .env)
docker run --rm -e BQ_PROJECT=... -e BQ_DATASET=... -e BQ_SOURCE_TABLE=billing_dm.MISc_vw_txn_enriched_subID_fallback -e GCS_MODEL_URI=gs://... dunning-deploy python inference_job/main.py

# Trigger
docker run --rm -e BQ_PROJECT=... -e BQ_DATASET=... dunning-deploy python trigger_job/main.py
```

---

## Required env (inference)

| Variable | Description |
|----------|-------------|
| **BQ_PROJECT** | BigQuery project. |
| **BQ_DATASET** | BigQuery dataset for schedule/feature_log tables. |
| **BQ_SOURCE_TABLE** | Source view/table for active dunning (e.g. `billing_dm.MISc_vw_txn_enriched_subID_fallback` or `project.dataset.table`). |
| **GCS_MODEL_URI** or **DUNNING_MODEL_PATH** | Model artifact (GCS URI or local path). |
| **SCHEDULE_TABLE** | Table name for schedule (default `production_dunning_schedule`). |
| **FEATURE_LOG_TABLE** | Optional; if set, a sample of features is written for drift. Table is created automatically from `lib.features.FEATURE_LOG_SCHEMA` on first write. |
| **BQ_LOCATION** | Optional; BQ location (default `europe-west1`). |

---

## Required env (trigger)

| Variable | Description |
|----------|-------------|
| **BQ_PROJECT**, **BQ_DATASET**, **SCHEDULE_TABLE**, **TRIGGER_LOG_TABLE** | BigQuery tables. |
| **resolve_invoice_status** / **call_retry_api** | Implement in `trigger_job/main.py` (use your Resolver URL, Retry API URL, auth from env or Secret Manager). |
| **DRY_RUN**, **RATE_LIMIT_PER_MIN**, **JITTER_MAX_SECONDS**, **TRAFFIC_SPLIT_MODEL_PCT** | See env.example. |

---

## What you must implement

1. **BigQuery**  
   Run the SQL in `bq_schema/` (replace `PROJECT_ID` and `DATASET`). Create the source view/table if needed.

2. **Trigger: `resolve_invoice_status(invoice_ids)`**  
   In `trigger_job/main.py`, implement the call to your Invoice State Resolver (e.g. `POST /invoices/status`). Return `dict[invoice_id, status]`. Use `INVOICE_STATE_RESOLVER_URL` and auth from env or Secret Manager.

3. **Trigger: `call_retry_api(invoice_id, idempotency_key)`**  
   In `trigger_job/main.py`, implement `POST` to your Retry API with body `{"invoice_id": "<id>"}` and header `Idempotency-Key: <key>`. Return `(http_status_code, error_message)`. Use `RETRY_API_URL` and auth.

4. **Secrets**  
   Store Billing/Chargebee API keys in Secret Manager; mount as env in the Cloud Run Job.

---

## Cloud Run Jobs

- Build image from **deploy** (this folder) and push to Artifact Registry.
- **Inference job:** Command `python inference_job/main.py`; schedule every 4–6 h (e.g. Cloud Scheduler).
- **Trigger job:** Command `python trigger_job/main.py`; schedule hourly.
- Set env vars (or use Secret Manager) as in env.example.

See PRODUCTION_DEPLOYMENT_GUIDE.md in the parent repo for full architecture and guardrails.
