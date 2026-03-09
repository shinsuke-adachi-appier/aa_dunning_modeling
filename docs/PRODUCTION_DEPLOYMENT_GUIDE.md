# Dunning Model: Production Deployment on Google Cloud Run

This guide describes how to run the dunning retry-time model in **production on Google Cloud Run**: predict the hour with highest success probability per invoice and **send an API request (e.g. Chargebee retry) at that hour**. Deployment uses the **standalone `deploy/` folder**: two **Cloud Run Jobs** (inference + trigger) and **Cloud Scheduler** for cron. Guardrails: **pre-flight checks**, **rate limiting**, **idempotency**, **feature drift logging**, **model fallback**, and **shadow-to-live** rollout.

---

## Part 1 — Architecture on Cloud Run

### 1.1 Components

| Component | Implementation on Cloud Run |
|-----------|----------------------------|
| **Inference** | One **Cloud Run Job**. Runs every 4–6 h (Cloud Scheduler). Fetches active dunning from BigQuery, loads model from GCS, scores 25 slots per invoice, writes to `production_dunning_schedule` (+ optional feature log). Fallback: 24h if model fails. |
| **Trigger** | One **Cloud Run Job**. Runs **hourly** (Cloud Scheduler). Reads schedule for current hour (UTC), calls Invoice State Resolver (Pre-Flight), applies jitter + rate limit, calls Retry API with `invoice_id` and Idempotency-Key, writes to `dunning_retry_trigger_log`. |
| **Scheduling** | **Cloud Scheduler**: one job for inference (e.g. `0 */6 * * *`), one for trigger (`0 * * * *`). Each invokes the corresponding Cloud Run Job. |
| **Secrets** | **Secret Manager** for Billing/Chargebee API keys; mount as env vars on the trigger Job. BigQuery uses the Job's service account (no secret needed if it has BQ access). |

### 1.2 Data Flow

```
Cloud Scheduler (every 6h) → Cloud Run Job (inference)
    → BQ (active dunning) → Model (GCS) → production_dunning_schedule [+ optional feature_log]

Cloud Scheduler (hourly) → Cloud Run Job (trigger)
    → BQ (schedule, current hour) → Pre-Flight (Invoice State Resolver) → filter unpaid
    → jitter + rate limit → POST /retry (invoice_id + Idempotency-Key) → dunning_retry_trigger_log
```

### 1.3 Source Data and Refresh

- **Data source:** BigQuery view/table (e.g. `MISc_vw_txn_enriched_subID_fallback`). Active dunning: `invoice_success_attempt_no IS NULL`, `invoice_attempt_count < 12`, last 5 days; latest row per `linked_invoice_id`.
- **Refresh:** Prefer refreshing the source at least every 6–12 h (ideally 4–6 h) so inference sees new failures sooner. If it stays daily, run inference shortly after the refresh and document in the runbook.
- **Pre-Flight:** At trigger time, the **source of truth** for "should we retry?" is the **Billing API / Invoice State Resolver**, not the schedule table. The trigger job must never call the Retry API without confirming the invoice is still unpaid.

---

## Part 2 — Critical Guardrails

### 2.1 Pre-Flight Check (Mandatory)

Before calling the Retry API, the **trigger job** must call your **Invoice State Resolver** (Billing API or internal service) with the list of `invoice_id` for the current hour. Filter to `status in ('unpaid', 'open')` only; do **not** retry paid/cancelled invoices. Implement `resolve_invoice_status(invoice_ids)` in `deploy/trigger_job/main.py`; use env (e.g. `INVOICE_STATE_RESOLVER_URL`) and auth from Secret Manager.

### 2.2 Jitter and Rate Limiting (Mandatory)

The trigger job shuffles the list of due invoices and applies a random delay per invoice (`JITTER_MAX_SECONDS`, default 1800) and a cap on requests per minute (`RATE_LIMIT_PER_MIN`, default 50). Configure via env on the Cloud Run Job.

### 2.3 Idempotency Key (Mandatory)

Use a deterministic key: `invoice_id|attempt_number|scheduled_hour_iso` (e.g. `inv_xyz|3|2026-02-28T10:00:00Z`). Send as header `Idempotency-Key` to the Retry API so duplicate runs (e.g. Cloud Run retry, double cron) do not cause double charges. Implement `call_retry_api(invoice_id, idempotency_key)` in `deploy/trigger_job/main.py`.

### 2.4 Model Fallback

If the model fails to load or inference fails for an invoice, the inference job sets `optimal_retry_at_utc = now + 24h` and `model_version_id = 'fallback_24h'`. The schedule is still written so dunning continues.

---

## Part 3 — Deploy Folder and Code

All production code lives in **`deploy/`** (standalone; no parent repo required).

| Path | Purpose |
|------|--------|
| **deploy/lib/** | Model load (GCS/local), features, slot generation, BQ fetch. |
| **deploy/inference_job/main.py** | Entrypoint for the inference Cloud Run Job. |
| **deploy/trigger_job/main.py** | Entrypoint for the trigger Cloud Run Job. You implement `resolve_invoice_status` and `call_retry_api`. |
| **deploy/bq_schema/*.sql** | DDL for `production_dunning_schedule` and `dunning_retry_trigger_log`. Feature log table is optional and created from Python when first writing. |
| **deploy/requirements-deploy.txt** | Pinned dependencies (scikit-learn, catboost, google-cloud-bigquery, google-cloud-storage, etc.). |
| **deploy/Dockerfile** | Build from the **deploy** directory only. |
| **deploy/README.md** | Build/run instructions and env reference. |
| **deploy/INFERENCE_JOB_FLOW.md** | Step-by-step flow of the inference job. |

---

## Part 4 — BigQuery

### 4.1 Tables to Create

1. **production_dunning_schedule**  
   Columns: `invoice_id`, `optimal_retry_at_utc`, `attempt_number`, `model_version_id`, `max_prob`, `inference_run_id`, `created_at`, `status`. Partition by `DATE(optimal_retry_at_utc)`. Use `deploy/bq_schema/production_dunning_schedule.sql` (replace `PROJECT_ID`, `DATASET`).

2. **dunning_retry_trigger_log**  
   Columns: `invoice_id`, `optimal_retry_at_utc`, `triggered_at`, `idempotency_key`, `api_response_status`, `model_version_id`, `error_message`. Use `deploy/bq_schema/dunning_retry_trigger_log.sql`.

3. **dunning_inference_feature_log** (optional)  
   No SQL needed. If you set `FEATURE_LOG_TABLE`, the inference job creates the table on first write using the schema in `lib/features.py` (`FEATURE_LOG_SCHEMA`).

### 4.2 Permissions

The Cloud Run Job's service account needs:  
- BigQuery Data Viewer (or read access) on the **source** table/view (e.g. `billing_dm.MISc_vw_txn_enriched_subID_fallback`).  
- BigQuery Data Editor (or write access) on the **dataset** where `production_dunning_schedule`, `dunning_retry_trigger_log`, and optionally `dunning_inference_feature_log` live.

---

## Part 4.5 — After shadow dunning: steps to deploy on Cloud Run + Scheduler

Once **shadow dunning** is finished and you are satisfied with the model (e.g. shadow vs. reality metrics, recovery rate), follow these steps in order to go live on Google Cloud Run and Cloud Scheduler.

### Prerequisite

- Shadow dunning has been run (e.g. via `shadow_monitoring_*.py` or notebook). You have a calibrated model artifact (e.g. `.joblib`) and have validated schedule vs. actual outcomes.
- You have a GCP project with Cloud Run, Cloud Scheduler, BigQuery, Artifact Registry, and Secret Manager enabled. You have a service account for the Jobs (BQ + GCS read, and optionally Secret Manager for API keys).

### Step-by-step

| # | Step | What to do | See |
|---|------|------------|-----|
| 1 | **Create BigQuery tables** | Run the DDL in `deploy/bq_schema/production_dunning_schedule.sql` and `dunning_retry_trigger_log.sql`. Replace `PROJECT_ID` and `DATASET` with your project and dataset. Optionally create the feature log table later (inference job can create it from Python schema when `FEATURE_LOG_TABLE` is set). | Part 4.1 |
| 2 | **Grant BQ permissions** | Ensure the service account you will use for Cloud Run Jobs has **BigQuery Data Viewer** on the source table/view and **BigQuery Data Editor** on the dataset for schedule and trigger log. | Part 4.2 |
| 3 | **Implement trigger stubs** | In `deploy/trigger_job/main.py`: implement `resolve_invoice_status(invoice_ids)` (call your Invoice State Resolver/Billing API) and `call_retry_api(invoice_id, idempotency_key)` (call your Retry API with Idempotency-Key header). Use env vars (e.g. `INVOICE_STATE_RESOLVER_URL`, `RETRY_API_URL`) and auth from Secret Manager if needed. | Part 2.1, 2.3, Part 8 |
| 4 | **Create secrets (if needed)** | In Secret Manager, create secrets for Billing/Chargebee API keys (or other auth). You will mount these as env vars on the **trigger** Cloud Run Job. | Part 1.1, Part 6.2 |
| 5 | **Upload model to GCS** | Upload your calibrated model (e.g. `catboost_dunning_calibrated_YYYYMMDD.joblib`) to a GCS bucket, e.g. `gs://YOUR_BUCKET/dunning/models/`. Ensure the Jobs’ service account has **Storage Object Viewer** on that bucket. | Part 9 #4 |
| 6 | **Build and push Docker image** | From the **deploy** folder: `docker build -t dunning-prod .`. Tag and push to Artifact Registry (e.g. `REGION-docker.pkg.dev/PROJECT/REPO/dunning-prod:latest`). Ensure the Cloud Run Jobs’ service account can pull from the repo. | Part 5 |
| 7 | **Create the inference Cloud Run Job** | Create a Cloud Run Job with the image from step 6. Command: `python`, Args: `inference_job/main.py`. Set env: `BQ_PROJECT`, `BQ_DATASET`, `BQ_SOURCE_TABLE`, `GCS_MODEL_URI` (or `DUNNING_MODEL_PATH`). Optionally: `SCHEDULE_TABLE`, `FEATURE_LOG_TABLE`, `FEATURE_LOG_SAMPLE_PCT`, `DUNNING_MODEL_VERSION`, `BQ_LOCATION`. Use a region that matches your BQ dataset (e.g. `europe-west1`). Set timeout (e.g. 3600 s) if you have many invoices. | Part 6.1 |
| 8 | **Create the trigger Cloud Run Job** | Create a second Cloud Run Job with the **same** image. Command: `python`, Args: `trigger_job/main.py`. Set env: `BQ_PROJECT`, `BQ_DATASET`, `SCHEDULE_TABLE`, `TRIGGER_LOG_TABLE`, and (when implemented) Resolver/Retry API URLs. Mount secrets for API keys. Set `DRY_RUN=1` initially for testing. Optionally: `RATE_LIMIT_PER_MIN`, `JITTER_MAX_SECONDS`, `TRAFFIC_SPLIT_MODEL_PCT`. | Part 6.2 |
| 9 | **Create Cloud Scheduler jobs** | Create two HTTP targets that POST to the Cloud Run Jobs run endpoint. (1) **Inference:** schedule `0 */6 * * *` (every 6 hours), URI `https://run.googleapis.com/v2/projects/PROJECT_ID/locations/REGION/jobs/dunning-inference:run`. (2) **Trigger:** schedule `0 * * * *` (hourly), URI `https://run.googleapis.com/v2/projects/PROJECT_ID/locations/REGION/jobs/dunning-trigger:run`. Use OAuth with a service account that has **Cloud Run Invoker** on both Jobs. | Part 7 |
| 10 | **Test with DRY_RUN** | Run the **trigger** Job manually (e.g. from Console or `gcloud run jobs execute dunning-trigger`). With `DRY_RUN=1`, it should write to `dunning_retry_trigger_log` but **not** call the Retry API. Check logs and BQ. Run the **inference** Job once and confirm rows in `production_dunning_schedule`. | Part 9 #9 |
| 11 | **Go live** | Set `DRY_RUN=0` (or remove it) on the trigger Job. Optionally start with `TRAFFIC_SPLIT_MODEL_PCT=10` and increase after validating recovery. Monitor trigger log and Retry API outcomes; roll back by reverting the model URI or setting traffic split to 0. | Part 9 #10, #11 |

### Summary flow

```
Shadow dunning done → BQ tables + permissions → Implement trigger stubs + secrets
  → Upload model to GCS → Build & push image → Create inference Job → Create trigger Job
  → Create Scheduler (inference every 6h, trigger hourly) → Test with DRY_RUN=1 → Go live (optionally with traffic split)
```

---

## Part 5 — Building and Pushing the Image

1. **Build from the deploy folder** (no parent repo):

   ```bash
   cd aa_dunning_modeling/deploy
   docker build -t dunning-prod .
   ```

2. **Tag and push to Artifact Registry** (replace region and repo):

   ```bash
   export REGION=us-central1
   export REPO=your-artifact-repo
   docker tag dunning-prod ${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${REPO}/dunning-prod:latest
   docker push ${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${REPO}/dunning-prod:latest
   ```

3. Ensure the Cloud Run service account (or the one you assign to the Jobs) can pull from this Artifact Registry repo.

---

## Part 6 — Cloud Run Jobs

### 6.1 Inference Job

- **Image:** The image you built from `deploy/` (e.g. `us-central1-docker.pkg.dev/PROJECT/REPO/dunning-prod:latest`).
- **Command:** `python` **Args:** `inference_job/main.py` (override the default CMD so the container runs this).
- **Env vars (required):**  
  `BQ_PROJECT`, `BQ_DATASET`, `BQ_SOURCE_TABLE` (e.g. `billing_dm.MISc_vw_txn_enriched_subID_fallback`),  
  `GCS_MODEL_URI` (e.g. `gs://your-bucket/dunning/models/catboost_dunning_calibrated_20260224.joblib`) or `DUNNING_MODEL_PATH`.
- **Env vars (optional):**  
  `SCHEDULE_TABLE`, `FEATURE_LOG_TABLE`, `FEATURE_LOG_SAMPLE_PCT`, `DUNNING_MODEL_VERSION`, `BQ_LOCATION`.
- **Service account:** One that can read the BQ source and write to the BQ dataset (and read from GCS if using `GCS_MODEL_URI`).
- **Region:** Same as your BQ dataset (e.g. `europe-west1`) to avoid cross-region BQ latency.

**Create via gcloud (example):**

```bash
gcloud run jobs create dunning-inference \
  --image=us-central1-docker.pkg.dev/PROJECT/REPO/dunning-prod:latest \
  --region=REGION \
  --set-env-vars="BQ_PROJECT=PROJECT,BQ_DATASET=DATASET,BQ_SOURCE_TABLE=billing_dm.MISc_vw_txn_enriched_subID_fallback,GCS_MODEL_URI=gs://BUCKET/dunning/models/catboost_dunning_calibrated_20260224.joblib" \
  --command="python" \
  --args="inference_job/main.py" \
  --service-account=YOUR_SERVICE_ACCOUNT
```

### 6.2 Trigger Job

- **Image:** Same image as the inference job.
- **Command:** `python` **Args:** `trigger_job/main.py`.
- **Env vars (required):**  
  `BQ_PROJECT`, `BQ_DATASET`, `SCHEDULE_TABLE`, `TRIGGER_LOG_TABLE`.
- **Env vars (optional):**  
  `RATE_LIMIT_PER_MIN`, `JITTER_MAX_SECONDS`, `DRY_RUN`, `TRAFFIC_SPLIT_MODEL_PCT`.  
  You will also set the Resolver and Retry API URL (and optionally mount secrets for API keys) once you implement `resolve_invoice_status` and `call_retry_api` in `deploy/trigger_job/main.py`.
- **Secrets:** Mount Chargebee/Billing API key from Secret Manager as an env var (e.g. `CHARGEBEE_API_KEY`).

**Create via gcloud (example):**

```bash
gcloud run jobs create dunning-trigger \
  --image=us-central1-docker.pkg.dev/PROJECT/REPO/dunning-prod:latest \
  --region=REGION \
  --set-env-vars="BQ_PROJECT=PROJECT,BQ_DATASET=DATASET,SCHEDULE_TABLE=production_dunning_schedule,TRIGGER_LOG_TABLE=dunning_retry_trigger_log" \
  --set-secrets="CHARGEBEE_API_KEY=chargebee-api-key:latest" \
  --command="python" \
  --args="trigger_job/main.py" \
  --service-account=YOUR_SERVICE_ACCOUNT
```

### 6.3 Cold Start and Timeouts

- Cloud Run Jobs start a new container on each run; cold start is typically 10–30 s, which is acceptable for batch inference and trigger.
- Set **timeout** (e.g. 3600 s for inference if you have many invoices) in the Job configuration so long runs do not get cut off.

---

## Part 7 — Cloud Scheduler

Create two schedules that invoke the Cloud Run Jobs via the [Cloud Run run API](https://cloud.google.com/run/docs/execute/jobs-on-schedule). The URI format is:

`https://run.googleapis.com/v2/projects/PROJECT_ID/locations/REGION/jobs/JOB_NAME:run`

1. **Inference (every 6 hours):**

   ```bash
   gcloud scheduler jobs create http dunning-inference-schedule \
     --location=REGION \
     --schedule="0 */6 * * *" \
     --uri="https://run.googleapis.com/v2/projects/PROJECT_ID/locations/REGION/jobs/dunning-inference:run" \
     --http-method=POST \
     --oauth-service-account-email=YOUR_SCHEDULER_SERVICE_ACCOUNT
   ```

2. **Trigger (hourly):**

   ```bash
   gcloud scheduler jobs create http dunning-trigger-schedule \
     --location=REGION \
     --schedule="0 * * * *" \
     --uri="https://run.googleapis.com/v2/projects/PROJECT_ID/locations/REGION/jobs/dunning-trigger:run" \
     --http-method=POST \
     --oauth-service-account-email=YOUR_SCHEDULER_SERVICE_ACCOUNT
   ```

The Scheduler service account must have **Cloud Run Invoker** (`roles/run.invoker`) on the Cloud Run Jobs (or the service account that runs them), and the **Service Account User** role so it can act as that identity when invoking. Alternatively, use the **Triggers** tab in the Cloud Run Job (Console) to add a Scheduler trigger with the desired schedule.

---

## Part 8 — APIs and Contracts

### 8.1 Retry API

- **Request:** `POST` with body `{ "invoice_id": "<linked_invoice_id>" }` and header `Idempotency-Key: <key>`.
- **Key format:** `{invoice_id}|{attempt_number}|{scheduled_hour_iso}` (e.g. `inv_xyz|3|2026-02-28T10:00:00Z`).
- **Response:** 200 success; 409 if key already used; 4xx/5xx with clear error.
- Implement the HTTP client in `deploy/trigger_job/main.py` → `call_retry_api(invoice_id, idempotency_key)`; use `RETRY_API_URL` and auth from env/Secret Manager.

### 8.2 Invoice State Resolver (Pre-Flight)

- **Request:** e.g. `POST` with body `{ "invoice_ids": ["id1", "id2", ...] }`.
- **Response:** e.g. `{ "id1": "unpaid", "id2": "paid", ... }`.
- Trigger keeps only `unpaid`/`open`. Implement in `deploy/trigger_job/main.py` → `resolve_invoice_status(invoice_ids)`; use `INVOICE_STATE_RESOLVER_URL` and auth.

---

## Part 9 — Step-by-Step Checklist (Cloud Run)

| # | Step | Notes |
|---|------|--------|
| 1 | Create BQ tables | Run `deploy/bq_schema/production_dunning_schedule.sql` and `dunning_retry_trigger_log.sql` (replace PROJECT_ID, DATASET). |
| 2 | Implement Invoice State Resolver | Implement `resolve_invoice_status()` in `deploy/trigger_job/main.py`; add Resolver URL and auth (env or Secret Manager). |
| 3 | Implement Retry API client | Implement `call_retry_api()` in `deploy/trigger_job/main.py`; add Retry API URL and auth. |
| 4 | Upload model to GCS | e.g. `gs://your-bucket/dunning/models/catboost_dunning_calibrated_20260224.joblib`. |
| 5 | Build and push image | From `deploy/`: `docker build`, tag, push to Artifact Registry. |
| 6 | Create inference Cloud Run Job | Set env: BQ_*, GCS_MODEL_URI (or DUNNING_MODEL_PATH). Command: `python inference_job/main.py`. |
| 7 | Create trigger Cloud Run Job | Set env: BQ_*, SCHEDULE_TABLE, TRIGGER_LOG_TABLE; mount secrets for API keys. Command: `python trigger_job/main.py`. |
| 8 | Create Cloud Scheduler jobs | One for inference (e.g. every 6 h), one for trigger (hourly). Point to Cloud Run Jobs run endpoint. |
| 9 | Test with DRY_RUN | Set `DRY_RUN=1` on the trigger Job and run manually; confirm logs and no real Retry API calls. |
| 10 | Traffic split (optional) | Set `TRAFFIC_SPLIT_MODEL_PCT=10`; increase after validating recovery rate. |
| 11 | Monitoring and rollback | Model vs. Reality dashboard; rollback by updating `GCS_MODEL_URI` to a previous model or setting `TRAFFIC_SPLIT_MODEL_PCT=0`. |

---

## Part 10 — Timezone and Idempotency

- **Inference and schedule:** All times in **UTC** (`optimal_retry_at_utc`, `inference_run_at`).
- **Trigger:** Query schedule for the **current hour UTC** (e.g. 08:00–08:59).
- **Idempotency key:** Use the same hour truncation (e.g. `2026-02-28T08:00:00Z`) so the key is deterministic for that hour.

---

## Part 11 — Feature Consistency and Dependency Pinning

- The container must use the **same** `scikit-learn` and `catboost` versions as the training environment so the calibrated model loads and behaves correctly. Pin them in `deploy/requirements-deploy.txt` and document in a runbook.
- The deploy code includes a stub `train_dunning_v2_20260206.py` so joblib can unpickle the `IsotonicCalibratedClassifier` saved at training time.

---

This guide is tailored for **Google Cloud Run**: two Jobs (inference + trigger), Cloud Scheduler, BigQuery, GCS for the model, and Secret Manager for API keys. The **`deploy/`** folder is self-contained; upload only that folder to build and run on Cloud Run.
