# Deploy Dunning Jobs on Your Team's Existing Airflow

This guide explains how to add the dunning **inference** and **trigger** jobs to an Airflow environment that already has its own `requirements.txt`, `Dockerfile`, and `docker-compose.yml`. You have two ways to run the jobs: **same worker image** (BashOperator) or **dedicated image** (DockerOperator). Steps are ordered so you can do prerequisites once, then choose the run approach.

---

## Overview

- **Two DAGs:** `dunning_inference` (every 6 h) and `dunning_trigger` (hourly).
- **Code:** Everything runs from the `deploy/` folder; entrypoints are `python inference_job/main.py` and `python trigger_job/main.py`.
- **Config:** Airflow Variables (and Connections for Chargebee API key). No `.env` in production; pass env from Variables into the tasks.

---

## Step 1 — Prerequisites (do once)

1. **BigQuery**
   - Create tables using `deploy/bq_schema/production_dunning_schedule.sql` and `dunning_retry_trigger_log.sql` (replace project/dataset).
   - If the schedule table already exists without `model_version_id`:  
     `ALTER TABLE \`project.dataset.production_dunning_schedule\` ADD COLUMN IF NOT EXISTS model_version_id STRING;`
   - Ensure the Airflow worker identity has:
     - **Read** on `billing_dm.v_subscription_customer_flat` and `billing_stg.stg_cb_transactions_enriched_tbl`
     - **Write** on the dataset that contains the schedule and trigger log tables.

2. **GCS model**
   - Upload the calibrated model (e.g. `catboost_dunning_calibrated_YYYYMMDD.joblib`) to a GCS bucket, e.g.  
     `gs://YOUR_BUCKET/dunning/models/catboost_dunning_calibrated_YYYYMMDD.joblib`
   - Grant the worker identity **Storage Object Viewer** on that bucket.

3. **Chargebee (for trigger)**
   - You will need `CHARGEBEE_SITE` and `CHARGEBEE_API_KEY`; store the key in an Airflow Connection or encrypted Variable.

---

## Step 2 — Make deploy code available to Airflow

Your workers must be able to run `python inference_job/main.py` and `python trigger_job/main.py` **with working directory = the folder that contains `inference_job/` and `lib/`** (i.e. the `deploy/` folder).

**Option A — Repo mounted / synced with DAGs**

- If your docker-compose (or deployment) mounts a repo into the Airflow workers (e.g. under `/opt/airflow` or `~/airflow`), add the `aa_dunning_modeling` repo (or at least the `deploy/` folder) there.
- Example layout:
  - `.../airflow/dags/` — your DAGs
  - `.../airflow/repos/aa_dunning_modeling/deploy/` — deploy code
- In the DAG, set the task’s working directory to that path (e.g. `cwd='/opt/airflow/repos/aa_dunning_modeling/deploy'`) and run `python inference_job/main.py` / `python trigger_job/main.py`.

**Option B — Bake deploy into the worker image**

- In your existing **Dockerfile**, copy the `deploy/` folder into the image (e.g. `COPY aa_dunning_modeling/deploy /app/dunning/deploy`) and set `WORKDIR` for the dunning task to `/app/dunning/deploy`, or
- Build a **separate image** from `deploy/Dockerfile` and use DockerOperator for the two dunning tasks only (Step 4B).

---

## Step 3 — Satisfy deploy dependencies

The dunning jobs need specific packages. Either merge them into your existing setup or use a dedicated image.

**Option A — Same worker image**

- Add the contents of `deploy/requirements-deploy.txt` to your team’s **requirements.txt** (or install them in your **Dockerfile**). Important:
  - **scikit-learn>=1.8,<1.9** (must match training so the model unpickles)
  - **pandas-gbq>=0.26.1**
  - **catboost**, **google-cloud-bigquery**, **google-cloud-storage**, **timezonefinder**, **pgeocode**, **python-dotenv**, etc.
- Rebuild your worker image and redeploy (e.g. `docker-compose build` and bring workers up again).

**Option B — Dedicated image**

- Use the image built from `deploy/Dockerfile` (see Step 4B). That image already has `requirements-deploy.txt` installed; no change to your main `requirements.txt` or Dockerfile.

---

## Step 4 — Configure Airflow Variables and Connections

Create **Airflow Variables** (and optionally a **Connection** for the Chargebee API key) and pass them as env into the dunning tasks.

**Inference task**

| Variable | Example | Required |
|----------|---------|----------|
| `BQ_PROJECT` | `aa-datamart` | Yes |
| `BQ_DATASET` | `dunning_modeling` | Yes |
| `GCS_MODEL_URI` | `gs://your-bucket/dunning/models/catboost_dunning_calibrated_20260301.joblib` | Yes (or `DUNNING_MODEL_PATH` for local) |
| `SCHEDULE_TABLE` | `production_dunning_schedule` | No (default) |
| `FEATURE_LOG_TABLE` | `dunning_inference_feature_log` | No |
| `FEATURE_LOG_SAMPLE_PCT` | `0.1` | No |
| `DUNNING_MODEL_VERSION` | `production` | No |
| `BQ_LOCATION` | `europe-west1` | No |

**Trigger task**

| Variable | Example | Required |
|----------|---------|----------|
| `BQ_PROJECT` | `aa-datamart` | Yes |
| `BQ_DATASET` | `dunning_modeling` | Yes |
| `SCHEDULE_TABLE` | `production_dunning_schedule` | No |
| `TRIGGER_LOG_TABLE` | `dunning_retry_trigger_log` | No |
| `CHARGEBEE_SITE` | `aicreative` | Yes for Chargebee |
| `CHARGEBEE_API_KEY` | (secret) | Yes for Chargebee — prefer **Connection** or encrypted Variable |
| `RATE_LIMIT_PER_MIN` | `50` | No |
| `JITTER_MAX_SECONDS` | `1800` | No |
| `DRY_RUN` | `1` then `0` | No (use `1` for testing) |
| `TRAFFIC_SPLIT_MODEL_PCT` | `100` | No |
| `VELOCITY_CAP_DAYS` | `7` | No |
| `VELOCITY_CAP_MAX_RETRIES_7D` | `3` | No |

Use `deploy/env.example` as the full reference.

---

## Step 5 — Create the two DAGs

Create two DAG files in your Airflow `dags/` folder (or wherever your docker-compose mounts DAGs). Below are patterns for **BashOperator** (same worker image) and **DockerOperator** (dedicated image).

**Paths to set**

- `DEPLOY_DIR`: path to the `deploy/` folder on the worker (e.g. `/opt/airflow/repos/aa_dunning_modeling/deploy`).
- For DockerOperator: `DUNNING_IMAGE`: your registry image that contains the deploy code (e.g. built from `deploy/Dockerfile`).

### 5A — Using BashOperator (same worker image)

Assume you have a helper that builds env from Variables (and Connection for `CHARGEBEE_API_KEY`). Then:

**DAG: dunning_inference**

- **Schedule:** `schedule_interval='0 */6 * * *'` (every 6 hours).
- **Task:** BashOperator (or PythonOperator that runs the same command).
  - Command: `python inference_job/main.py`
  - **cwd:** `DEPLOY_DIR` (so `inference_job/main.py` and `lib/` resolve).
  - **env:** Pass all inference Variables above (e.g. `BQ_PROJECT`, `BQ_DATASET`, `GCS_MODEL_URI`, …).
  - **execution_timeout:** e.g. 3600 seconds.

**DAG: dunning_trigger**

- **Schedule:** `schedule_interval='0 * * * *'` (hourly).
- **Task:** BashOperator.
  - Command: `python trigger_job/main.py`
  - **cwd:** `DEPLOY_DIR`
  - **env:** Pass all trigger Variables/Connection (including `CHARGEBEE_SITE`, `CHARGEBEE_API_KEY`).

Example (pseudo-code; adapt to your Variable/Connection API):

```python
# dags/dunning_inference_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import timedelta

DEPLOY_DIR = "/opt/airflow/repos/aa_dunning_modeling/deploy"  # or get from Variable

def get_inference_env():
    return {
        "BQ_PROJECT": Variable.get("BQ_PROJECT"),
        "BQ_DATASET": Variable.get("BQ_DATASET"),
        "GCS_MODEL_URI": Variable.get("GCS_MODEL_URI"),
        "SCHEDULE_TABLE": Variable.get("SCHEDULE_TABLE", default_var="production_dunning_schedule"),
        "BQ_LOCATION": Variable.get("BQ_LOCATION", default_var="europe-west1"),
        # add optional: FEATURE_LOG_TABLE, FEATURE_LOG_SAMPLE_PCT, DUNNING_MODEL_VERSION
    }

with DAG(
    "dunning_inference",
    schedule_interval="0 */6 * * *",
    default_args={"retries": 1},
    tags=["dunning"],
) as dag:
    BashOperator(
        task_id="run_inference",
        bash_command="python inference_job/main.py",
        cwd=DEPLOY_DIR,
        env=get_inference_env(),
        execution_timeout=timedelta(seconds=3600),
    )
```

```python
# dags/dunning_trigger_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook

DEPLOY_DIR = "/opt/airflow/repos/aa_dunning_modeling/deploy"

def get_trigger_env():
    env = {
        "BQ_PROJECT": Variable.get("BQ_PROJECT"),
        "BQ_DATASET": Variable.get("BQ_DATASET"),
        "SCHEDULE_TABLE": Variable.get("SCHEDULE_TABLE", default_var="production_dunning_schedule"),
        "TRIGGER_LOG_TABLE": Variable.get("TRIGGER_LOG_TABLE", default_var="dunning_retry_trigger_log"),
        "CHARGEBEE_SITE": Variable.get("CHARGEBEE_SITE"),
        "DRY_RUN": Variable.get("DRY_RUN", default_var="1"),  # set to 0 for production
        "RATE_LIMIT_PER_MIN": Variable.get("RATE_LIMIT_PER_MIN", default_var="50"),
        "JITTER_MAX_SECONDS": Variable.get("JITTER_MAX_SECONDS", default_var="1800"),
        "VELOCITY_CAP_DAYS": Variable.get("VELOCITY_CAP_DAYS", default_var="7"),
        "VELOCITY_CAP_MAX_RETRIES_7D": Variable.get("VELOCITY_CAP_MAX_RETRIES_7D", default_var="3"),
    }
    # If you store API key in a Connection (e.g. connection_id="chargebee_api"):
    try:
        conn = BaseHook.get_connection("chargebee_api")
        env["CHARGEBEE_API_KEY"] = conn.password
    except Exception:
        env["CHARGEBEE_API_KEY"] = Variable.get("CHARGEBEE_API_KEY")
    return env

with DAG(
    "dunning_trigger",
    schedule_interval="0 * * * *",
    default_args={"retries": 0},  # avoid double trigger on retry
    tags=["dunning"],
) as dag:
    BashOperator(
        task_id="run_trigger",
        bash_command="python trigger_job/main.py",
        cwd=DEPLOY_DIR,
        env=get_trigger_env(),
    )
```

### 5B — Using DockerOperator (dedicated image)

If you build an image from `deploy/Dockerfile` and push it as `DUNNING_IMAGE`:

- **Inference task:** DockerOperator with image `DUNNING_IMAGE`, command `python`, arguments `inference_job/main.py`, env from Variables, working dir `/app` (as in the Dockerfile).
- **Trigger task:** Same image, arguments `trigger_job/main.py`, env from Variables/Connection.

Your existing `docker-compose` and main Dockerfile stay as they are; only the dunning tasks use this second image.

---

## Step 6 — Test and go live

1. **Test inference**
   - Run the inference DAG once manually. Check that rows appear in `production_dunning_schedule` and that logs show “Model loaded from …” (or fallback message).

2. **Test trigger with DRY_RUN**
   - Set Variable `DRY_RUN=1`. Run the trigger DAG. Confirm logs and rows in `dunning_retry_trigger_log` with `error_message='DRY_RUN'` and no real Chargebee calls.

3. **Go live**
   - Set `DRY_RUN=0` (or remove it). Optionally start with `TRAFFIC_SPLIT_MODEL_PCT=10` and increase after validating. Monitor trigger log and Chargebee; roll back by updating `GCS_MODEL_URI` to a previous model or setting traffic split to 0.

---

## Summary checklist

| # | Step |
|---|------|
| 1 | Create BQ tables (schedule + trigger log); add `model_version_id` if needed; grant worker read/write and GCS model read. |
| 2 | Put `deploy/` on the worker (mount or copy into image) so `cwd=DEPLOY_DIR` and `python inference_job/main.py` / `trigger_job/main.py` work. |
| 3 | Install deploy dependencies: merge `requirements-deploy.txt` into your requirements/Dockerfile (Option A) or use the image from `deploy/Dockerfile` (Option B). |
| 4 | Create Airflow Variables (and Connection for Chargebee key). |
| 5 | Add two DAGs (inference + trigger) with correct schedule and env; use BashOperator with `cwd=DEPLOY_DIR` or DockerOperator with the dunning image. |
| 6 | Test inference, then trigger with DRY_RUN=1, then set DRY_RUN=0 and monitor. |

For full guardrails, BigQuery details, and env reference, see **PRODUCTION_DEPLOYMENT_GUIDE.md** and **deploy/env.example**.
