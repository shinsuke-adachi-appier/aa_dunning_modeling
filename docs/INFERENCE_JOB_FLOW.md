# Production inference job: step-by-step flow

Entrypoint: **`inference_job/main.py`** → `run()`.

---

## 1. Setup and config

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 1.1 | Add deploy folder to `sys.path` so `lib` is importable | `inference_job/main.py` (lines 18–22) |
| 1.2 | Import `bq_fetch`, `features`, `model`, `slots` from `lib` | `inference_job/main.py` (line 23) |
| 1.3 | Read `GCS_MODEL_URI` or `DUNNING_MODEL_PATH`; exit if unset | `inference_job/main.py` (lines 70–74) |
| 1.4 | Set `inference_run_at` (timezone-aware UTC now, `pd.Timestamp.now("UTC")`), `inference_run_id` (UUID), `model_version_id` | `inference_job/main.py` (lines 83–85) |

---

## 2. Load model

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 2.1 | Call `model.load_model(model_path)` | `inference_job/main.py` (line 78) |
| 2.2 | Register `IsotonicCalibratedClassifier` for joblib unpickling | `lib/model.py` → `_register_for_unpickle()` |
| 2.3 | If path is `gs://...`, download blob to `/tmp` via `google.cloud.storage` | `lib/model.py` → `load_model()` |
| 2.4 | Load artifact with `joblib.load(path)`; return `None` on failure | `lib/model.py` → `load_model()` |
| 2.5 | Set `use_fallback_global = (model is None)`, `model_version_id = 'fallback_24h'` if no model | `inference_job/main.py` (lines 86–88) |

---

## 3. Fetch active dunning from BigQuery

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 3.1 | Call `bq_fetch.fetch_active_dunning()` | `inference_job/main.py` (line 83) |
| 3.2 | Read env: `BQ_PROJECT`, `BQ_SOURCE_TABLE`, `BQ_LOCATION` | `lib/bq_fetch.py` |
| 3.3 | Run SQL: latest row per `linked_invoice_id`, active dunning (last 5 days, &lt; 12 attempts, Soft decline) | `lib/bq_fetch.py` → `fetch_active_dunning()` |
| 3.4 | Rename columns: `Decline_code_norm` → `prev_decline_code`, `card_status` → `prev_card_status`, `first_attempt_at_calc` → `first_attempt_at` | `lib/bq_fetch.py` |
| 3.5 | **add_timezone_features:** timezone from country (COUNTRY_TZ_MAP), optional zip override; `localized_time`, `local_*`; ensure `updated_at` UTC | `lib/timezone_utils.py` (via `lib/bq_fetch.py`) |
| 3.6 | Sanitize `prev_decline_code` and `billing_country`: fillna + strip → `"UNKNOWN"` | `lib/bq_fetch.py` |
| 3.7 | Return DataFrame; exit early if empty | `inference_job/main.py` (lines 90–93) |

---

## 4. Prepare invoice list

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 4.1 | Dedupe to one row per invoice (`linked_invoice_id`, keep last) | `inference_job/main.py` (lines 98–101) |
| 4.2 | Set feature-log flags from `FEATURE_LOG_SAMPLE_PCT` and `FEATURE_LOG_TABLE` | `inference_job/main.py` (lines 104–105) |

---

## 5. Per-invoice loop (for each active dunning invoice)

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 5.1 | Get `invoice_id`, `base_ts` (= `updated_at`), `first_ts` (= `first_attempt_at` or `base_ts`), `attempt_number` | `inference_job/main.py` (lines 107–112) |
| 5.2 | Default row: `optimal_retry_at_utc = _to_utc_ts(inference_run_at + 24h)`, `version_for_row = 'fallback_24h'` | `inference_job/main.py` (lines 114–115) |
| 5.3 | **If model loaded:** build one feature row for this invoice | |
| 5.3a | Call `features.build_invoice_row(raw, base_ts, first_ts, as_of_timestamp=inference_run_at, timezone=raw["timezone"], as_of_localized=raw["localized_time"])` | `lib/features.py` → `build_invoice_row()` |
| 5.3b | Time features use **localized timestamp column** (`localized_time`) when provided for cyclic features (hour/dow/day, dist_to_payday); else `as_of` converted to local via `timezone`; `time_since_prev_attempt` / `cumulative_delay_hours` stay UTC-based | `lib/features.py` |
| 5.4 | **If build succeeded:** run slot inference | |
| 5.4a | Call `slots.run_inference_for_invoice(..., timezone=raw["timezone"])` → `optimal_slot_for_invoice(..., timezone=...)` → `generate_candidate_slots(..., timezone=...)` | `lib/slots.py` |
| 5.4b | Slot temporal features (hour_sin, dow_sin, day_sin, dist_to_payday) computed in **local** time when timezone set; score with `model.predict_proba(slots)`; pick best slot | `lib/slots.py` |
| 5.4c | Return `optimal_retry_at`, `max_prob`, `raw_snapshot` (best-slot features), `probs_by_delay`; main normalizes `optimal_retry_at_utc = _to_utc_ts(optimal_retry_at)` so output is always **UTC** | `inference_job/main.py`, `lib/slots.py` |
| 5.5 | On any exception in 5.3/5.4, keep fallback 24h and `version_for_row = 'fallback_24h'` | `inference_job/main.py` (lines 125–126, 153–154) |
| 5.6 | If feature log enabled and random sample: build feature-log row from `raw_snapshot` + metadata; `created_at` and `optimal_retry_at_utc` set via `_to_utc_ts()`; append to `feature_log_rows` | `inference_job/main.py` (lines 138–150), uses `features.MODEL_FEATURE_NAMES` |
| 5.7 | Append one schedule row: `invoice_id`, `optimal_retry_at_utc` (UTC), `attempt_number`, `model_version_id`, `max_prob`, `inference_run_id`, `created_at` (UTC via `_to_utc_ts`), `status=PENDING` | `inference_job/main.py` (lines 156–165) |

---

## 6. Write outputs to BigQuery

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 6.1 | Call `write_schedule_to_bq(schedule_rows)` | `inference_job/main.py` (line 167) |
| 6.2 | Build `project.dataset.SCHEDULE_TABLE` from env; `LoadJobConfig(write_disposition=WRITE_APPEND)`; `client.load_table_from_dataframe(df, table_id, job_config).result()` | `inference_job/main.py` → `write_schedule_to_bq()`, `google.cloud.bigquery` |
| 6.3 | If `feature_log_rows` non-empty: call `write_feature_log_to_bq(feature_log_rows)` | `inference_job/main.py` (lines 168–169) |
| 6.4 | Resolve `FEATURE_LOG_TABLE`; if table missing, create it from `features.FEATURE_LOG_SCHEMA`; then append DataFrame | `inference_job/main.py` → `write_feature_log_to_bq()`, `lib/features.py` (`FEATURE_LOG_SCHEMA`) |
| 6.5 | Print summary (schedule row count, feature log row count) | `inference_job/main.py` (line 170) |

---

## File / module summary

| Path | Role |
|------|------|
| **inference_job/main.py** | Entrypoint; env/config; per-invoice loop; `_to_utc_ts()` for UTC output; schedule/feature-log BQ writes; fallback 24h |
| **lib/bq_fetch.py** | BigQuery client; SQL for active dunning; column renames; **add_timezone_features**; **prev_decline_code / billing_country sanitization** |
| **lib/model.py** | `IsotonicCalibratedClassifier`; GCS download; `joblib.load` |
| **lib/features.py** | `MODEL_FEATURE_NAMES`, `CAT_FEATURES`, `DELAY_HOURS`, `FEATURE_LOG_SCHEMA`; `build_invoice_row(..., timezone=..., as_of_localized=...)` — time features use **localized timestamp column** when provided, else local conversion via timezone |
| **lib/slots.py** | `generate_candidate_slots(..., timezone=...)`, `optimal_slot_for_invoice(..., timezone=...)`, `run_inference_for_invoice(..., timezone=...)` — slot temporal features in **local** time when timezone set |
| **lib/timezone_utils.py** | `add_timezone_features(df)` — country (COUNTRY_TZ_MAP) + optional zip (pgeocode, timezonefinder); localized_time, local_* |
| **lib/country_timezones.py** | COUNTRY_TZ_MAP (country code → IANA timezone) |
| **train_dunning_v2_20260206.py** | Stub defining `IsotonicCalibratedClassifier` for joblib unpickle (same module name as training) |
| **google.cloud.bigquery** | Client, LoadJobConfig, WRITE_APPEND, get_table, create_table |
| **google.cloud.storage** | Optional: download model from GCS (in `lib/model.py`) |
| **google.cloud.exceptions.NotFound** | Used when creating feature-log table if missing |

---

## Data flow (high level)

```
Env (GCS_MODEL_URI, BQ_*, FEATURE_LOG_*)
    → model.load_model() → calibrated model or None
    → bq_fetch.fetch_active_dunning() → active_df (with localized_time, timezone)
    → For each invoice:
          features.build_invoice_row(..., as_of_localized=raw["localized_time"]) → invoice_row
          slots.run_inference_for_invoice() → optimal_retry_at, max_prob, raw_snapshot
          optimal_retry_at_utc = _to_utc_ts(optimal_retry_at); created_at = _to_utc_ts(inference_run_at)
          (optional) feature_log row from raw_snapshot
          schedule row (optimal_retry_at_utc and created_at in UTC)
    → write_schedule_to_bq(schedule_rows)
    → write_feature_log_to_bq(feature_log_rows)  [if any and FEATURE_LOG_TABLE set]
```
