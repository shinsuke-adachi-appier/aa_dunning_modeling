# Production trigger job: step-by-step flow

Entrypoint: **`trigger_job/main.py`** → `run()`.

The trigger job runs on a schedule (e.g. hourly). It reads **production_dunning_schedule** for the current UTC hour, applies jitter and rate limiting, enforces the velocity cap, calls the Retry API with an idempotency key, and logs each attempt to **dunning_retry_trigger_log**. There is no pre-flight status check; if an invoice is already paid, the Retry API returns an error and we log it.

---

## 1. Setup and config

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 1.1 | Add deploy folder to `sys.path` so `lib` is importable (if needed) | `trigger_job/main.py` (lines 23–25) |
| 1.2 | Read `DRY_RUN` (1/true/yes → no API calls, only log); `RATE_LIMIT_PER_MIN` (default 50); `JITTER_MAX_SECONDS` (default 1800); `TRAFFIC_SPLIT_MODEL_PCT` (default 100) | `trigger_job/main.py` (lines 112–115) |

---

## 2. Query schedule for current hour

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 2.1 | Call `get_schedule_for_current_hour()` | `trigger_job/main.py` (line 117) |
| 2.2 | Read env: `BQ_PROJECT`, `BQ_DATASET`, `SCHEDULE_TABLE` (default `production_dunning_schedule`) | `trigger_job/main.py` → `get_schedule_for_current_hour()` |
| 2.3 | Compute current UTC time; truncate to hour start (`minute=0`, `second=0`, `microsecond=0`) | `trigger_job/main.py` (lines 62–63) |
| 2.4 | Run parameterized BigQuery: `SELECT invoice_id, optimal_retry_at_utc, attempt_number` where `DATE(optimal_retry_at_utc) = @date`, `TIMESTAMP_TRUNC(optimal_retry_at_utc, HOUR) = @hour_start`, `COALESCE(status, 'PENDING') = 'PENDING'` | `trigger_job/main.py` |
| 2.5 | Return list of dicts (one per row); exit early if empty | `trigger_job/main.py` (lines 118–120) |

---

## 3. Traffic split and shuffle

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 3.1 | If `TRAFFIC_SPLIT_MODEL_PCT` &lt; 100: shuffle schedule rows, take first `max(1, len * traffic_split_pct / 100)` rows (canary/ramp) | `trigger_job/main.py` |
| 3.2 | Shuffle the (possibly reduced) list to spread load within the hour | `trigger_job/main.py` |

---

## 3.5 Velocity cap (rolling window)

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 4.5.1 | Read `VELOCITY_CAP_DAYS` (default 7) and `VELOCITY_CAP_MAX_RETRIES_7D` (default 3) from env | `trigger_job/main.py` |
| 4.5.2 | Call `get_retry_count_last_n_days(invoice_ids, days=velocity_cap_days)` → BQ count of **real** Retry API attempts per invoice in the last N days (excludes `DRY_RUN`, `VELOCITY_CAP_7D`, `call_retry_api not implemented`) | `trigger_job/main.py` |
| 4.5.3 | In the per-invoice loop: if `retry_count_7d[invoice_id] >= velocity_cap_max`, append log row with `error_message="VELOCITY_CAP_7D"`, do **not** call the Retry API, and continue | `trigger_job/main.py` |

This enforces **no more than N retries in any rolling N-day window** per invoice to avoid being aggressive and being flagged by the payment provider.

---

## 4. Per-invoice loop: jitter, rate limit, Retry API, log

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 5.1 | Compute `min_interval = 60 / rate_limit_per_min` seconds between API calls | `trigger_job/main.py` (lines 139–140) |
| 5.2 | **Velocity cap:** If this invoice already has ≥ `VELOCITY_CAP_MAX_RETRIES_7D` real retries in the last `VELOCITY_CAP_DAYS` days → log row with `error_message="VELOCITY_CAP_7D"`, skip API, continue | `trigger_job/main.py` |
| 5.3 | For each row: get `invoice_id`, `attempt_number`, `optimal_retry_at_utc` | `trigger_job/main.py` |
| 5.4 | Build idempotency key: `build_idempotency_key(invoice_id, attempt_number, optimal_retry_at_utc)` → `"{invoice_id}|{attempt_number}|{YYYY-MM-DDTHH:00:00Z}"` | `trigger_job/main.py` (lines 81–88, 146) |
| 5.5 | **Jitter:** sleep random `[0, jitter_max_seconds]` before this invoice (spreads calls over the hour) | `trigger_job/main.py` (lines 150–152) |
| 5.6 | **Rate limit:** sleep `min_interval` seconds before calling API | `trigger_job/main.py` (line 153) |
| 5.7 | **If DRY_RUN:** append log row with `api_response_status=None`, `error_message="DRY_RUN"`; skip API; continue | `trigger_job/main.py` (lines 155–164) |
| 5.8 | **If not DRY_RUN:** call `call_retry_api(invoice_id, idempotency_key)` → `(http_status_code, error_message)` | `trigger_job/main.py` (lines 166–179) |
| 5.9 | If `call_retry_api` raises `NotImplementedError`: log row with `error_message="call_retry_api not implemented"`; continue | `trigger_job/main.py` (lines 169–179) |
| 5.10 | If other exception: set `status_code = -1`, `err_msg = str(e)` and log | `trigger_job/main.py` (lines 180–181) |
| 5.11 | Append one log row: `invoice_id`, `optimal_retry_at_utc`, `triggered_at` (UTC now), `idempotency_key`, `api_response_status`, `error_message`, `status`, `message`, `txn_id`, `attempt_number`, `comment_status` | `trigger_job/main.py` |

---

## 5. Write trigger log to BigQuery

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 5.1 | Call `write_trigger_log_to_bq(log_rows)` | `trigger_job/main.py` |
| 5.2 | Read env: `BQ_PROJECT`, `BQ_DATASET`, `TRIGGER_LOG_TABLE` (default `dunning_retry_trigger_log`) | `trigger_job/main.py` → `write_trigger_log_to_bq()` |
| 5.3 | If project or dataset unset, return without writing | `trigger_job/main.py` |
| 5.4 | `client.load_table_from_dataframe(df, table_id, write_disposition=WRITE_APPEND).result()` | `trigger_job/main.py`, `google.cloud.bigquery` |
| 5.5 | Print summary: number of rows logged and `dry_run` flag | `trigger_job/main.py` |

---

## Stubs you must implement

| Function | Contract | Purpose |
|----------|----------|---------|
| **`call_retry_api(invoice_id: str, idempotency_key: str)`** | POST to Retry API with body `{"invoice_id": "<id>"}` and header `Idempotency-Key`; return `(http_status_code, error_message)`. If the invoice is already paid, the API returns an error; we log it and continue. | Execute the retry (e.g. Chargebee / payment gateway); idempotency key prevents duplicate charges. |

---

## File / module summary

| Path | Role |
|------|------|
| **trigger_job/main.py** | Entrypoint; env config; `get_schedule_for_current_hour()`; `build_idempotency_key()`; `get_retry_count_last_n_days()`; `write_trigger_log_to_bq()`; jitter, rate limit, velocity cap, Retry API loop; **stub:** `call_retry_api` |
| **google.cloud.bigquery** | Client; parameterized query; LoadJobConfig WRITE_APPEND |
| **bq_schema/dunning_retry_trigger_log.sql** | Schema for trigger audit log table |

---

## Data flow (high level)

```
Env (BQ_PROJECT, BQ_DATASET, SCHEDULE_TABLE, TRIGGER_LOG_TABLE, DRY_RUN, RATE_LIMIT_PER_MIN, JITTER_MAX_SECONDS, TRAFFIC_SPLIT_MODEL_PCT, VELOCITY_CAP_DAYS, VELOCITY_CAP_MAX_RETRIES_7D)
    → get_schedule_for_current_hour() → rows (PENDING, optimal_retry_at_utc in current UTC hour)
    → optional traffic split (TRAFFIC_SPLIT_MODEL_PCT); shuffle
    → get_retry_count_last_n_days(invoice_ids) → retry_count_7d
    → For each row:
          if retry_count_7d[invoice_id] >= cap → log VELOCITY_CAP_7D and skip
          else build_idempotency_key(...)
          jitter sleep → rate-limit sleep
          DRY_RUN → log row only | else call_retry_api(invoice_id, idempotency_key) → log row
    → write_trigger_log_to_bq(log_rows)
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BQ_PROJECT` | (required) | BigQuery project for schedule and trigger log |
| `BQ_DATASET` | (required) | BigQuery dataset |
| `SCHEDULE_TABLE` | `production_dunning_schedule` | Table written by inference job; read by trigger job |
| `TRIGGER_LOG_TABLE` | `dunning_retry_trigger_log` | Audit log of retry API attempts |
| `VELOCITY_CAP_DAYS` | `7` | Rolling window (days) for velocity cap |
| `VELOCITY_CAP_MAX_RETRIES_7D` | `3` | Max retries per invoice in that window; above this we skip and log `VELOCITY_CAP_7D` |
| `DRY_RUN` | (empty) | Set to `1`, `true`, or `yes` to skip API calls and only write log rows |
| `RATE_LIMIT_PER_MIN` | `50` | Max API calls per minute (sleep between calls) |
| `JITTER_MAX_SECONDS` | `1800` | Max random delay (seconds) before each call to spread load |
| `TRAFFIC_SPLIT_MODEL_PCT` | `100` | Percentage of eligible invoices to trigger (e.g. 10 for canary); 100 = all |
