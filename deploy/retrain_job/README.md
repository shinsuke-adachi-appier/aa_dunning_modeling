# Retrain job (Airflow / cron)

This job **retrains** the dunning recovery model with the most recent data, uploads the calibrated model to GCS **under the model filename** (`catboost_dunning_calibrated_{SUFFIX}.joblib`), and **exports the training log to BigQuery**.

Dates are **relative to today** by default (validation through yesterday, holdout from today).

## What it does

1. Resolves the repo root (parent of `deploy/` or `REPO_ROOT` / cwd).
2. Sets date windows relative to today:
   - `VAL_END` = yesterday, `HOLDOUT_START` = today
   - `VAL_START` = today − 10 days, `CAL_END` = today − 11 days, `CAL_START` = today − 25 days, `TRAIN_END` = today − 26 days
   - Override any with env vars (`TRAIN_END`, `CAL_START`, etc.).
3. Runs `scripts/train_dunning_v2_20260301.py` with `FORCE_QUERY=1` and `EXPORT_TRAINING_LOG=1` (training script writes `models/training_run_{SUFFIX}.json`).
4. Uploads `models/catboost_dunning_calibrated_{SUFFIX}.joblib` to GCS at **`{GCS_RETRAIN_BASE_URI}/{filename}`** (so the object name is the file name).
5. Reads the training run JSON and inserts **one row** into the BigQuery **training log** table (`BQ_PROJECT.BQ_DATASET.dunning_training_log`).

## Requirements

- Run from **aa_dunning_modeling** repo root.
- **BigQuery**: credentials and `BQ_PROJECT`, `BQ_DATASET` (same as inference job). Create the log table once from `deploy/bq_schema/dunning_training_log.sql`.
- **GCS**: credentials for upload; set `GCS_RETRAIN_BASE_URI` or `GCS_MODEL_URI`.
- Python deps: training deps + `google-cloud-storage`, `google-cloud-bigquery`.

## Env vars

| Variable | Description |
|----------|-------------|
| `REPO_ROOT` | Optional. Repo root path; default = cwd. |
| `FORCE_QUERY` | Set to `1` by the job. |
| `GCS_RETRAIN_BASE_URI` | Base GCS path; model uploaded as `{base}/{catboost_dunning_calibrated_{SUFFIX}.joblib}`. E.g. `gs://bucket/dunning/models`. |
| `GCS_MODEL_URI` | If set and no `GCS_RETRAIN_BASE_URI`, used as full object path (e.g. overwrite production). If it ends with `.joblib`, upload to that URI; else treated as base and filename is appended. |
| `SUFFIX` | Model filename suffix; default = today YYYYMMDD. |
| `BQ_PROJECT`, `BQ_DATASET` | Required for training log export. |
| `TRAINING_LOG_TABLE` | Table name for training log; default `dunning_training_log`. |
| `TRAIN_END`, `CAL_START`, `CAL_END`, `VAL_START`, `VAL_END`, `HOLDOUT_START` | Optional date overrides (YYYY-MM-DD). |
| `CALIBRATION_TEMPERATURE` | Pass-through to training script. |

## BigQuery training log table

Create once (replace `PROJECT_ID` and `DATASET`):

```bash
# Edit deploy/bq_schema/dunning_training_log.sql then:
bq mk --table PROJECT_ID:DATASET.dunning_training_log deploy/bq_schema/dunning_training_log.sql
```

Columns: `run_at`, `suffix`, date window fields, `n_train`, `n_cal`, `n_val`, `auc_val`, `pr_auc_val`, `brier_val`, `ece_val`, `mce_val`, `calibration_temperature`, `model_gcs_uri`.

## Run locally (from repo root)

```bash
cd /path/to/aa_dunning_modeling
export GCS_RETRAIN_BASE_URI=gs://YOUR_BUCKET/dunning/models
export BQ_PROJECT=your-project
export BQ_DATASET=your_dataset
python deploy/retrain_job/main.py
```

## Airflow DAG example

```python
retrain = BashOperator(
    task_id="retrain_dunning",
    bash_command="cd /path/to/aa_dunning_modeling && python deploy/retrain_job/main.py",
    env={
        "GCS_RETRAIN_BASE_URI": "gs://YOUR_BUCKET/dunning/models",
        "BQ_PROJECT": "your-project",
        "BQ_DATASET": "your_dataset",
    },
)
```

## Cron

```bash
0 2 * * 0 cd /path/to/aa_dunning_modeling && GCS_RETRAIN_BASE_URI=gs://BUCKET/dunning/models BQ_PROJECT=proj BQ_DATASET=ds python deploy/retrain_job/main.py
```

## After upload

- Model is at `gs://{bucket}/{path}/catboost_dunning_calibrated_{SUFFIX}.joblib`. Point inference job’s `GCS_MODEL_URI` to that object or to a stable path you copy it to.
