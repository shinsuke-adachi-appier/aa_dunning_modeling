# Local dry run and testing (before deploying to Airflow)

Run the inference and trigger jobs on your machine with the same code as production Airflow tasks. Use **DRY_RUN=1** for the trigger so it never calls Chargebee `collect_payment` (only logs what it would do).

---

## 1. Prerequisites

- **Python 3.11+** and dependencies from `deploy/requirements-deploy.txt`
- **Google Application Default Credentials** so BigQuery works without a service account key:
  ```bash
  gcloud auth application-default login
  ```
- **Env vars** (copy `deploy/env.example` to `deploy/.env` and fill; do not commit `.env`). For the trigger job dry run, `CHARGEBEE_SITE` and `CHARGEBEE_API_KEY` are required (see env.example).

---

## 2. Load `.env` when running locally

The jobs read only `os.environ`; they do not load `.env` by default. Use either:

**Option A — Export in the shell (Unix/macOS):**
```bash
cd aa_dunning_modeling/deploy
set -a
source .env
set +a
python inference_job/main.py
```

**Option B — Use python-dotenv (one-off install):**
```bash
cd aa_dunning_modeling/deploy
pip install python-dotenv
python -c "
import dotenv
dotenv.load_dotenv()
exec(open('inference_job/main.py').read())
"
```
Or add at the **very top** of `inference_job/main.py` and `trigger_job/main.py` (only when you want local .env loading):

```python
if __name__ != "__main__" or not os.environ.get("BQ_PROJECT"):
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass
```
Then run: `python inference_job/main.py` (and same for trigger).

**Option C — Pass vars explicitly:**
```bash
export BQ_PROJECT=myproject BQ_DATASET=mydataset
export DUNNING_MODEL_PATH=/path/to/model.joblib
python inference_job/main.py
```

---

## 3. Install dependencies

From the **deploy** folder:

```bash
cd aa_dunning_modeling/deploy
pip install -r requirements-deploy.txt
```

---

## 4. Run inference job locally

- Uses your **Application Default Credentials** for BigQuery (and GCS if you use `GCS_MODEL_URI`).
- Prefer **DUNNING_MODEL_PATH** pointing to a local `.joblib` file so you don’t need GCS locally.

```bash
cd aa_dunning_modeling/deploy

# Load .env (Option A)
set -a && source .env && set +a

# Or set key vars by hand:
# export BQ_PROJECT=... BQ_DATASET=...
# export DUNNING_MODEL_PATH=/path/to/catboost_dunning_calibrated_YYYYMMDD.joblib

python inference_job/main.py
```

- **Expected:** Fetches active dunning from BQ, runs model (or fallback), writes to `production_dunning_schedule` (and optionally feature log). Check the dataset in BigQuery after the run.
- **Optional:** Use a dedicated dataset or table for local tests (e.g. `SCHEDULE_TABLE=local_dunning_schedule`) so you don’t mix with production.

---

## 5. Run trigger job locally (dry run)

With **DRY_RUN=1** the trigger job:
- Reads from BigQuery (schedule for current UTC hour)
- **Does not** call Chargebee **collect_payment**; it only appends rows to `dunning_retry_trigger_log` with `error_message="DRY_RUN"`

So you need real **BQ** and **Chargebee** credentials; no payment is triggered.

```bash
cd aa_dunning_modeling/deploy

set -a && source .env && set +a
export DRY_RUN=1

python trigger_job/main.py
```

- **Expected:** Logs how many rows were “triggered” (dry run). Check `dunning_retry_trigger_log` in BQ for rows with `error_message='DRY_RUN'`.
- **Note:** For the trigger to find rows, there must be schedule rows with `optimal_retry_at_utc` in the **current UTC hour**. Run inference first, or temporarily insert test rows into the schedule table for that hour.

---

## 6. End-to-end local test (inference → trigger dry run)

1. **Create BQ tables** (if not already): run `deploy/bq_schema/production_dunning_schedule.sql` and `dunning_retry_trigger_log.sql` (use a dev dataset if you prefer).
2. **Run inference** once (step 4). Confirm rows in `production_dunning_schedule` with `optimal_retry_at_utc` in the next hour (or current hour).
3. **Run trigger with DRY_RUN=1** (step 5). Confirm it picks up those rows and writes to `dunning_retry_trigger_log` with `DRY_RUN`, and that it **does not** call `collect_payment`.

---

## 7. Run with Docker (same image as used in Airflow with DockerOperator)

Build the image from the deploy folder and run with env from a file. Use the same pattern when running via Airflow’s DockerOperator.

```bash
cd aa_dunning_modeling/deploy
docker build -t dunning-deploy .

# Inference (use your .env; ensure no secrets are committed)
docker run --rm --env-file .env -e DUNNING_MODEL_PATH=/path/to/model.joblib -v /path/to/model.joblib:/path/to/model.joblib:ro dunning-deploy python inference_job/main.py

# Trigger dry run
docker run --rm --env-file .env -e DRY_RUN=1 dunning-deploy python trigger_job/main.py
```

For inference with a **local** model in Docker, mount the file and set `DUNNING_MODEL_PATH` to the path **inside** the container (e.g. `/tmp/model.joblib` and `-v $(pwd)/model.joblib:/tmp/model.joblib:ro`).

---

## 8. Checklist before deploying to Airflow

| Check | How |
|-------|-----|
| Inference runs and writes schedule | Run step 4; query `production_dunning_schedule` in BQ. |
| Trigger dry run does not call collect_payment | Run step 5 with `DRY_RUN=1`; confirm log rows have `error_message='DRY_RUN'`. |
| No .env committed | Ensure `.env` is in `.gitignore`; only `env.example` is committed. |
| Chargebee key not in code | Use env / Airflow Variables or Connections only; never hardcode. |
