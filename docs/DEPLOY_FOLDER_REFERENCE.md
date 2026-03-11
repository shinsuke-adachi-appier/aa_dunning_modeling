# Deploy folder reference

This document describes all files under **`deploy/`** (shared library, schemas, config, and build). For step-by-step flows see:

- **Inference job:** [INFERENCE_JOB_FLOW.md](INFERENCE_JOB_FLOW.md)
- **Trigger job:** [TRIGGER_JOB_FLOW.md](TRIGGER_JOB_FLOW.md)

---

## 1. Folder layout

| Path | Purpose |
|------|--------|
| **lib/** | Shared library: model load, features, slots, BQ fetch, timezone/country data. Used by inference job only. |
| **inference_job/** | Entrypoint `main.py`; writes schedule + optional feature log. |
| **trigger_job/** | Entrypoint `main.py`; reads schedule, Retry API, trigger log. No pre-flight status check. |
| **retrain_job/** | Entrypoint `main.py`; runs training from repo root (Airflow/cron), uploads model to GCS. See `retrain_job/README.md`. |
| **bq_schema/** | BigQuery DDL for schedule, trigger log, and training log tables. |
| **Root** | README, env.example, Dockerfile, requirements-deploy.txt, train_dunning_v2 stub. |

---

## 2. Library (`lib/`)

Self-contained module; no dependency on parent repo. Exports: `model`, `features`, `slots`, `bq_fetch`, `timezone_utils`, `country_timezones`.

### 2.1 `lib/model.py`

| Item | Description |
|------|-------------|
| **Role** | Load calibrated model from GCS or local path; register `IsotonicCalibratedClassifier` so joblib can unpickle. |
| **Class** | `IsotonicCalibratedClassifier` — wraps a fitted classifier, calibrates probabilities with `IsotonicRegression`, exposes `feature_names_` from inner estimator. |
| **Function** | `load_model(model_path: str)` — if path starts with `gs://`, downloads blob to `/tmp` via `google.cloud.storage`; then `joblib.load(path)`. Returns `None` on failure (caller uses fallback). |
| **Unpickle** | `_register_for_unpickle()` sets `IsotonicCalibratedClassifier` on `__main__` so joblib finds it when loading artifacts from training. |

### 2.2 `lib/features.py`

| Item | Description |
|------|-------------|
| **Role** | Build one feature row per invoice for inference; define feature names, schema for optional feature log table. |
| **Constants** | `DELAY_HOURS` = 24–120h step 4 (25 slots); `MODEL_FEATURE_NAMES`; `CAT_FEATURES`; `FEATURE_LOG_METADATA_COLUMNS`, `FEATURE_LOG_OUTPUT_COLUMNS`, `FEATURE_LOG_SCHEMA` (name + BQ type for table creation). |
| **Function** | `build_invoice_row(row, base_timestamp, first_attempt_at, as_of_timestamp=None, timezone=None, as_of_localized=None)` — computes time_since_prev_attempt, cumulative_delay_hours (UTC); cyclic features (hour/dow/day, dist_to_payday) from `as_of_localized` if provided, else from `as_of` converted via `timezone`; other features from row. Returns `pd.Series` with `MODEL_FEATURE_NAMES`. |

### 2.3 `lib/slots.py`

| Item | Description |
|------|-------------|
| **Role** | Generate candidate retry slots (24–120h, 4h steps), score with model, pick best slot. |
| **Constants** | `TEMPORAL_FEATURES` — subset of features recomputed per slot. |
| **Functions** | `generate_candidate_slots(invoice_row, base_timestamp, first_attempt_timestamp=None, delay_hours=None, feature_cols=None, timezone=None)` — one row per slot; temporal features in **local** time when timezone set. `optimal_slot_for_invoice(...)` — runs model, returns (1-based slot index, max P, slots DataFrame). `run_inference_for_invoice(invoice_row, base_timestamp_for_slots, first_attempt_at, model, timezone=None)` — returns `(optimal_retry_at, max_prob, raw_snapshot_dict, probs_by_delay)`. |

### 2.4 `lib/bq_fetch.py`

| Item | Description |
|------|-------------|
| **Role** | Query BigQuery for active dunning invoices; apply renames, timezone features, sanitization. |
| **Env** | `BQ_PROJECT` (required); `BQ_LOCATION` (default `europe-west1`). Source data comes from the embedded pipeline query in `lib/bq_fetch.py`, not from `BQ_SOURCE_TABLE`. |
| **Function** | `fetch_active_dunning()` — SQL: latest row per `linked_invoice_id`, `invoice_success_attempt_no IS NULL`, `invoice_attempt_count < 12`, last 5 days, `Decline_type_for_retry = 'Soft decline'`. Renames `Decline_code_norm` → `prev_decline_code`, `advice_code_group` → `prev_advice_code_group`, `card_status` → `prev_card_status`, `first_attempt_at_calc` → `first_attempt_at`. Calls `add_timezone_features(df)`; sanitizes `prev_decline_code`, `prev_advice_code_group`, and `billing_country` to `"UNKNOWN"`. Returns DataFrame. |

### 2.5 `lib/timezone_utils.py`

| Item | Description |
|------|-------------|
| **Role** | Add timezone and localized timestamp columns (aligned with txn_pipeline). |
| **Dependencies** | `country_timezones.COUNTRY_TZ_MAP`; optional `timezonefinder`, `pgeocode` for zip-based lookup. |
| **Function** | `add_timezone_features(df)` — ensures `updated_at` is UTC; sets `timezone` from `billing_country` (COUNTRY_TZ_MAP); overrides with zip when `fill_zip_code == 'Zip Filled'` (pgeocode + timezonefinder). Adds `localized_time` (naive, in local tz), `local_day_of_month`, `local_hour`, `local_day_of_week`. |

### 2.6 `lib/country_timezones.py`

| Item | Description |
|------|-------------|
| **Role** | Country code (ISO 3166-1 alpha-2) → IANA timezone. |
| **Exports** | `COUNTRY_TZ_OVERRIDES` — preferred primary TZ for key countries; `COUNTRY_TZ_IANA` — full map from zone1970.tab; `COUNTRY_TZ_MAP` — overrides merged over IANA (use `.fillna("UTC")` for unknown). |

---

## 3. BigQuery schemas (`bq_schema/`)

### 3.1 `production_dunning_schedule.sql`

| Item | Description |
|------|-------------|
| **Table** | `PROJECT_ID.DATASET.production_dunning_schedule` — replace placeholders. |
| **Columns** | `invoice_id`, `optimal_retry_at_utc`, `attempt_number`, `max_prob`, `inference_run_id`, `created_at`, `status` (e.g. PENDING, TRIGGERED, CANCELLED_PAID). |
| **Partition** | `PARTITION BY DATE(optimal_retry_at_utc)` for efficient current-hour queries by trigger job. |
| **Written by** | Inference job. **Read by** trigger job. |

### 3.2 `dunning_retry_trigger_log.sql`

| Item | Description |
|------|-------------|
| **Table** | `PROJECT_ID.DATASET.dunning_retry_trigger_log` — replace placeholders. |
| **Columns** | `invoice_id`, `optimal_retry_at_utc`, `triggered_at`, `idempotency_key`, `api_response_status` (HTTP status), `error_message`, `status` (Success \| Failed \| Error \| DRY_RUN \| VELOCITY_CAP_7D), `message`, `txn_id`, `attempt_number`, `comment_status`. |
| **Written by** | Trigger job (one row per retry API attempt / dry-run log). |

### 3.3 `dunning_training_log.sql`

| Item | Description |
|------|-------------|
| **Table** | `PROJECT_ID.DATASET.dunning_training_log` — replace placeholders. |
| **Columns** | `run_at`, `suffix`, date window fields (`global_start`, `train_end`, `cal_start`, `cal_end`, `val_start`, `val_end`, `holdout_start`), `n_train`, `n_cal`, `n_val`, `auc_val`, `pr_auc_val`, `brier_val`, `ece_val`, `mce_val`, `calibration_temperature`, `model_gcs_uri`. |
| **Written by** | Retrain job (one row per retrain run). |

---

## 4. Root files

### 4.1 `README.md`

| Item | Description |
|------|-------------|
| **Purpose** | Deploy-folder overview: layout, build/run commands, required env for inference and trigger, what to implement (BQ, `call_retry_api`, secrets). Points to PRODUCTION_DEPLOYMENT_GUIDE for full architecture. |

### 4.2 `env.example`

| Item | Description |
|------|-------------|
| **Purpose** | Template for env vars. Copy to `.env` or set in Cloud Run. |
| **Sections** | BigQuery (project, dataset, source table, location, schedule/trigger log table names, feature log table); model (GCS URI or local path, version); feature log sample %; trigger stubs (resolver URL, retry API URL); trigger behaviour (rate limit, jitter, dry run, traffic split %). |

### 4.3 `Dockerfile`

| Item | Description |
|------|-------------|
| **Base** | `python:3.11-slim`. |
| **Workdir** | `/app`; `COPY . /app` (build from deploy folder). |
| **Install** | `pip install -r requirements-deploy.txt`. |
| **Command** | Override at run time: `python inference_job/main.py` or `python trigger_job/main.py`. |

### 4.4 `requirements-deploy.txt`

| Item | Description |
|------|-------------|
| **Purpose** | Pinned production dependencies; align with training env (see PRODUCTION_DEPLOYMENT_GUIDE). |
| **Key packages** | numpy, pandas, scikit-learn, catboost, joblib, google-cloud-bigquery, google-cloud-storage, timezonefinder, pgeocode. |

### 4.5 `train_dunning_v2_20260206.py`

| Item | Description |
|------|-------------|
| **Purpose** | Stub module so joblib can unpickle models saved from the training repo. |
| **Content** | Defines `IsotonicCalibratedClassifier` with same interface as in `lib/model.py` (estimator, calibrator, `predict_proba`, `feature_names_`). Models reference this module name; this file is only needed if the training repo uses that module path when saving. |

---

## 5. Job entrypoints (summary)

| Job | Entrypoint | Reads | Writes |
|-----|------------|-------|--------|
| **Inference** | `python inference_job/main.py` | Env (BQ, model path); BQ source table via `lib.bq_fetch.fetch_active_dunning()` | `production_dunning_schedule`; optional `dunning_inference_feature_log` |
| **Trigger** | `python trigger_job/main.py` | Env (BQ); `production_dunning_schedule` for current UTC hour | Retry API (you implement `call_retry_api`); `dunning_retry_trigger_log`; no pre-flight |

---

## 6. Dependency graph (library)

```
lib/__init__.py
  → model      (standalone; joblib, sklearn, optional GCS)
  → features   (numpy, pandas; no other lib)
  → slots      (features, numpy, pandas)
  → bq_fetch   (timezone_utils, bigquery, pandas)
  → timezone_utils (country_timezones, pandas; optional timezonefinder, pgeocode)
  → country_timezones (data only)
```

Inference job: `bq_fetch` → `features` → `slots` → `model`. Trigger job: no `lib` imports; uses BigQuery and its own helpers in `main.py`.
