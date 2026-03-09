# Production trigger job: step-by-step flow

Entrypoint: **`trigger_job/main.py`** → `run()`.

The trigger job runs on a schedule (e.g. hourly). It reads **production_dunning_schedule** for the current UTC hour, checks invoice status (Pre-Flight), applies jitter and rate limiting, calls the Retry API with an idempotency key, and logs each attempt to **dunning_retry_trigger_log**.

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
| 2.4 | Run parameterized BigQuery: `SELECT invoice_id, optimal_retry_at_utc, attempt_number, model_version_id` where `DATE(optimal_retry_at_utc) = @date`, `TIMESTAMP_TRUNC(optimal_retry_at_utc, HOUR) = @hour_start`, `COALESCE(status, 'PENDING') = 'PENDING'` | `trigger_job/main.py` (lines 64–77) |
| 2.5 | Return list of dicts (one per row); exit early if empty | `trigger_job/main.py` (lines 118–120) |

---

## 3. Pre-Flight: resolve invoice status

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 3.1 | Collect `invoice_ids` from schedule rows | `trigger_job/main.py` (lines 123–124) |
| 3.2 | Call `resolve_invoice_status(invoice_ids)` → `dict[invoice_id, status]` (e.g. `'unpaid'`, `'open'`, `'paid'`, `'cancelled'`) | `trigger_job/main.py` (lines 125–128) |
| 3.3 | If `resolve_invoice_status` raises `NotImplementedError`, print message and return (no triggers) | `trigger_job/main.py` (lines 126–128) |
| 3.4 | Filter to rows where status is `'unpaid'` or `'open'` only; others are skipped for this run | `trigger_job/main.py` (line 130) |

---

## 4. Traffic split and shuffle

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 4.1 | If `TRAFFIC_SPLIT_MODEL_PCT` &lt; 100: shuffle unpaid list, take first `max(1, len * traffic_split_pct / 100)` rows (canary/ramp) | `trigger_job/main.py` (lines 131–134) |
| 4.2 | Shuffle the (possibly reduced) unpaid list to spread load within the hour | `trigger_job/main.py` (line 135) |

---

## 5. Per-invoice loop: jitter, rate limit, Retry API, log

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 5.1 | Compute `min_interval = 60 / rate_limit_per_min` seconds between API calls | `trigger_job/main.py` (lines 139–140) |
| 5.2 | For each row in unpaid list: get `invoice_id`, `attempt_number`, `optimal_retry_at_utc`, `model_version_id` | `trigger_job/main.py` (lines 142–145) |
| 5.3 | Build idempotency key: `build_idempotency_key(invoice_id, attempt_number, optimal_retry_at_utc)` → `"{invoice_id}|{attempt_number}|{YYYY-MM-DDTHH:00:00Z}"` | `trigger_job/main.py` (lines 81–88, 146) |
| 5.4 | **Jitter:** sleep random `[0, jitter_max_seconds]` before this invoice (spreads calls over the hour) | `trigger_job/main.py` (lines 150–152) |
| 5.5 | **Rate limit:** sleep `min_interval` seconds before calling API | `trigger_job/main.py` (line 153) |
| 5.6 | **If DRY_RUN:** append log row with `api_response_status=None`, `error_message="DRY_RUN"`; skip API; continue | `trigger_job/main.py` (lines 155–164) |
| 5.7 | **If not DRY_RUN:** call `call_retry_api(invoice_id, idempotency_key)` → `(http_status_code, error_message)` | `trigger_job/main.py` (lines 166–179) |
| 5.8 | If `call_retry_api` raises `NotImplementedError`: log row with `error_message="call_retry_api not implemented"`; continue | `trigger_job/main.py` (lines 169–179) |
| 5.9 | If other exception: set `status_code = -1`, `err_msg = str(e)` and log | `trigger_job/main.py` (lines 180–181) |
| 5.10 | Append one log row: `invoice_id`, `optimal_retry_at_utc`, `triggered_at` (UTC now), `idempotency_key`, `api_response_status`, `model_version_id`, `error_message` | `trigger_job/main.py` (lines 183–191) |

---

## 6. Write trigger log to BigQuery

| Step | What happens | File / module |
|------|-------------------------------|---------------|
| 6.1 | Call `write_trigger_log_to_bq(log_rows)` | `trigger_job/main.py` (line 193) |
| 6.2 | Read env: `BQ_PROJECT`, `BQ_DATASET`, `TRIGGER_LOG_TABLE` (default `dunning_retry_trigger_log`) | `trigger_job/main.py` → `write_trigger_log_to_bq()` |
| 6.3 | If project or dataset unset, return without writing | `trigger_job/main.py` (lines 98–99) |
| 6.4 | `client.load_table_from_dataframe(df, table_id, write_disposition=WRITE_APPEND).result()` | `trigger_job/main.py` (lines 104–108), `google.cloud.bigquery` |
| 6.5 | Print summary: number of rows logged and `dry_run` flag | `trigger_job/main.py` (line 194) |

---

## Stubs you must implement

| Function | Contract | Purpose |
|----------|----------|---------|
| **`resolve_invoice_status(invoice_ids: list[str])`** | Return `dict[str, str]`: invoice_id → status (`'unpaid'`, `'open'`, `'paid'`, `'cancelled'`, etc.). Only `'unpaid'` and `'open'` are triggered. | Pre-Flight: call your Invoice State Resolver (Billing API or internal service) so only eligible invoices get a retry. |
| **`call_retry_api(invoice_id: str, idempotency_key: str)`** | POST to Retry API with body `{"invoice_id": "<id>"}` and header `Idempotency-Key`; return `(http_status_code, error_message)`. | Execute the retry (e.g. Chargebee / payment gateway); idempotency key prevents duplicate charges. |

---

## File / module summary

| Path | Role |
|------|------|
| **trigger_job/main.py** | Entrypoint; env config; `get_schedule_for_current_hour()`; `build_idempotency_key()`; `write_trigger_log_to_bq()`; Pre-Flight, jitter, rate limit, Retry API loop; **stubs:** `resolve_invoice_status`, `call_retry_api` |
| **google.cloud.bigquery** | Client; parameterized query; LoadJobConfig WRITE_APPEND |
| **bq_schema/dunning_retry_trigger_log.sql** | Schema for trigger audit log table |

---

## Data flow (high level)

```
Env (BQ_PROJECT, BQ_DATASET, SCHEDULE_TABLE, TRIGGER_LOG_TABLE, DRY_RUN, RATE_LIMIT_PER_MIN, JITTER_MAX_SECONDS, TRAFFIC_SPLIT_MODEL_PCT)
    → get_schedule_for_current_hour() → rows (PENDING, optimal_retry_at_utc in current UTC hour)
    → resolve_invoice_status(invoice_ids) → status_map
    → filter rows to status in ('unpaid', 'open')
    → optional traffic split (TRAFFIC_SPLIT_MODEL_PCT); shuffle
    → For each row:
          build_idempotency_key(invoice_id, attempt_number, optimal_retry_at_utc)
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
| `DRY_RUN` | (empty) | Set to `1`, `true`, or `yes` to skip API calls and only write log rows |
| `RATE_LIMIT_PER_MIN` | `50` | Max API calls per minute (sleep between calls) |
| `JITTER_MAX_SECONDS` | `1800` | Max random delay (seconds) before each call to spread load |
| `TRAFFIC_SPLIT_MODEL_PCT` | `100` | Percentage of eligible invoices to trigger (e.g. 10 for canary); 100 = all |
