# Dunning model production deployment (standalone)

This **deploy** folder is self-contained. It is run by **Apache Airflow** (or locally for testing). No parent repo is required at runtime; make this folder available to Airflow workers or run it inside a container.

---

## Layout

| Path | Purpose |
|------|--------|
| **lib/** | Model load (GCS/local), features, slot generation, BQ fetch. No dependency on parent repo. |
| **inference_job/main.py** | Fetch active dunning from BQ → run model → write `production_dunning_schedule` + optional feature log. |
| **trigger_job/main.py** | Query schedule for current hour → Pre-Flight → jitter + rate limit → Retry API → log. **Chargebee implementation included;** replace if using a different billing/retry API. |
| **bq_schema/*.sql** | DDL for **schedule** and **trigger_log** only. Feature log table is optional and created from Python (lib/features.py) when first writing. |
| **requirements-deploy.txt** | Pinned production deps. |
| **env.example** | Env template. Copy to `.env` for local runs, or set via Airflow Variables/Connections in production. |
| **Dockerfile** | Build from **this directory**: `docker build -t dunning-deploy .` (optional for DockerOperator). |
| **LOCAL_TESTING.md** | In repo **docs**: `../docs/LOCAL_TESTING.md`. How to dry run and test inference + trigger locally before deploying to Airflow. |

---

## Build and run (standalone)

```bash
# From the deploy folder (no parent repo needed)
cd deploy
docker build -t dunning-deploy .

# Inference (set env or use .env)
docker run --rm -e BQ_PROJECT=... -e BQ_DATASET=... -e GCS_MODEL_URI=gs://... dunning-deploy python inference_job/main.py

# Trigger
docker run --rm -e BQ_PROJECT=... -e BQ_DATASET=... dunning-deploy python trigger_job/main.py
```

---

## Required env (inference)

| Variable | Description |
|----------|-------------|
| **BQ_PROJECT** | BigQuery project. |
| **BQ_DATASET** | BigQuery dataset for schedule/feature_log tables. |
| **GCS_MODEL_URI** or **DUNNING_MODEL_PATH** | Model artifact (GCS URI or local path). |
| **SCHEDULE_TABLE** | Table name for schedule (default `production_dunning_schedule`). |
| **FEATURE_LOG_TABLE** | Optional; if set, a sample of features is written for drift. Table is created automatically from `lib.features.FEATURE_LOG_SCHEMA` on first write. |
| **BQ_LOCATION** | Optional; BQ location (default `europe-west1`). |

---

## Required env (trigger)

| Variable | Description |
|----------|-------------|
| **BQ_PROJECT**, **BQ_DATASET**, **SCHEDULE_TABLE**, **TRIGGER_LOG_TABLE** | BigQuery tables. |
| **resolve_invoice_status** / **call_retry_api** | Chargebee is implemented in `trigger_job/main.py`; set `CHARGEBEE_SITE` and `CHARGEBEE_API_KEY`. For another system, implement in code and use env or Airflow Variables. |
| **DRY_RUN**, **RATE_LIMIT_PER_MIN**, **JITTER_MAX_SECONDS**, **TRAFFIC_SPLIT_MODEL_PCT** | See env.example. |

---

## What you must implement

1. **BigQuery**  
   Run the SQL in `bq_schema/` (replace `PROJECT_ID` and `DATASET`). Create the source view/table if needed.

2. **Trigger: `resolve_invoice_status(invoice_ids)`**  
   Chargebee is already implemented (GET `/invoices/{id}`). For another system, implement in `trigger_job/main.py`; use env or Airflow Variables.

3. **Trigger: `call_retry_api(invoice_id, idempotency_key)`**  
   Chargebee is already implemented (POST collect_payment). For another Retry API, implement in `trigger_job/main.py`; use env or Airflow Variables.

4. **Secrets**  
   Store Billing/Chargebee API keys in Airflow Variables (encrypted) or Connections; pass as env to the trigger task.

---

## Airflow DAGs

- Make **deploy** available to workers (clone repo or use Docker image).
- **Inference DAG:** Run `python inference_job/main.py`; schedule every 4–6 h (e.g. `schedule_interval='0 */6 * * *'`).
- **Trigger DAG:** Run `python trigger_job/main.py`; schedule hourly (`schedule_interval='0 * * * *'`).
- Set env from Airflow Variables/Connections as in env.example.

See **../docs/PRODUCTION_DEPLOYMENT_GUIDE.md** for full architecture, guardrails, and Airflow setup.
