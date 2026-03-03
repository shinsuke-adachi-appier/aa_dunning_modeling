# Dunning Model: Production Deployment Guide (Revised)

Expert-level guide to move the dunning retry-time model from shadow monitoring to **production**: predict the hour with highest success probability per invoice and **send an API request (e.g. Chargebee retry) at that hour**. This revision incorporates critical guardrails: **pre-flight checks**, **rate limiting**, **idempotency**, **feature drift**, **model fallback**, and **shadow-to-live** rollout.

---

## Part 1 — Current Inference Setup (Analysis)

### 1.1 What Exists Today

| Component | Implementation | Notes |
|-----------|----------------|-------|
| **Data source** | BigQuery `MISc_vw_txn_enriched_subID_fallback` | Active dunning: `invoice_success_attempt_no IS NULL`, `invoice_attempt_count < 12`, last 5 days; latest row per `linked_invoice_id`. **Table is updated daily** — see [1.4](#14-bigquery-table-refresh-frequency) for refresh-frequency guidance. |
| **Features** | 20 inputs (prev_decline_code, hour/dow/day sin/cos, dist_to_payday, log_charge_amount, is_debit, amt_per_attempt, time_since_prev_attempt, cumulative_delay_hours, billing_country, gateway, funding_type_norm, card_brand, prev_card_status, Domain_category, invoice_attempt_no) | Time-aware features recomputed **per candidate slot**. |
| **Slots** | 24–120h from **inference run time**, 4-hour resolution | 25 slots: 24, 28, 32, …, 120. `ranking_backtest.generate_candidate_slots` builds one feature row per slot; model scores each; argmax → optimal delay. |
| **Model** | Isotonic-calibrated CatBoost (`catboost_dunning_calibrated_20260224.joblib`) | Custom `IsotonicCalibratedClassifier` wrapper; must be available at load time for joblib. |
| **Output** | Per invoice: `optimal_retry_at` (rounded to hour), `suggested_max_prob`, full vector `prob_24h … prob_120h` | `optimal_retry_at = inference_run_at + best_delay_hours`. |

| **Execution** | Batch script: fetch from BQ → build features → score all slots → append to CSV | No real-time API; no "trigger retry" call. An **inference API** can be exposed (e.g. on Cloud Run) and called with **invoice_id** for on-demand scoring. |

### 1.4 BigQuery Table Refresh Frequency

The table **`MISc_vw_txn_enriched_subID_fallback`** is updated **daily**. For production:

- **Does it need to be updated more frequently?**  
  **Recommended: yes**, if feasible. Inference runs every 4–6 hours and needs (1) the current set of active dunning invoices and (2) the **latest attempt** per invoice (for `prev_decline_code`, `invoice_attempt_no`, `updated_at`). With a **daily** refresh:
  - **New failures** that occur after the last refresh may not appear until the next day → those invoices are **missed** for scoring until the next refresh. Updating the table every **4–6 hours** (or at least twice daily) would align with inference and reduce missed coverage.
  - **Paid invoices** may still appear as "in dunning" until the next refresh, but the **Pre-Flight check** at trigger time (Billing API / Invoice State Resolver) ensures we do **not** call the Retry API for them. So double-retry is avoided; the main cost of daily refresh is **latency** (new failures not scored for up to ~24 h).

- **If daily is the only option:** Run the inference job **shortly after** the daily refresh (e.g. 1–2 hours after the refresh completes) so the snapshot is as fresh as possible. Document in the runbook that "active dunning" is as of the last refresh; consider alerting if refresh fails or is delayed.

- **Summary:** For production, prefer refreshing the table **at least every 6–12 hours** (or every 4–6 h to match inference). If the table must stay daily, accept up to ~24 h delay for new failures to enter the schedule and run inference after the refresh.

### 1.2 Gaps for Production

1. **No “retry at this hour” trigger** — Today we only **log** the best hour; we do not call Chargebee (or any billing API) at that time.
2. **No inference API** — Everything is batch (BigQuery + Python script). For production you either keep batch and add a **trigger pipeline**, or add an **inference API** for on-demand scoring.
3. **Optimal time is relative to “now”** — Each run computes “best slot in the next 5 days from now.” For production you need a **scheduler** that, at the **optimal hour**, performs the retry (one API request per invoice).
4. **Credentials & env** — BQ and (for retry) Chargebee/billing API keys must be available in the deployment environment; no explicit retry client in repo yet.

### 1.3 Production Objective (Clarified)

- **Goal:** For each invoice in dunning, send **one API request at the hour with the highest predicted success probability** (the current “optimal retry” hour).
- **Flow:**  
  - **Inference:** Who is in dunning now, and what is each invoice’s **optimal_retry_at** (and optionally full prob vector)?  
  - **Trigger:** At that hour, find all invoices whose `optimal_retry_at` falls in the current hour; **pre-flight check** each; then call the retry API with **jitter and rate limiting**.

---

## Part 2 — Critical Risks and Mandatory Fixes

### 2.1 CRITICAL: Data Staleness — Pre-Flight Check (Mandatory)

**Problem:** Inference runs every 4–6 hours but schedules triggers up to **120 hours (5 days)** into the future. If an invoice fails Monday 8 AM, the 12 PM inference might schedule a retry for Friday 10 AM. If the customer **manually pays on Tuesday**, the trigger job (which only reads the schedule table) would still fire that retry on Friday unless you re-validate state.

**Risk:** Charging or retrying a customer who has already paid — the most common dunning horror story.

**Fix (mandatory):** The **Trigger Job MUST perform a Pre-Flight Check** before calling `POST /retry`. It must query the **source-of-truth** (Billing API or real-time DB) to confirm the invoice is still **OPEN/UNPAID**. Do **not** rely solely on a BigQuery table populated hours or days ago.

- **Logic:** `if (invoice_status != 'unpaid' | 'open') → mark schedule row as CANCELLED_PAID → SKIP (do not call Retry API).`
- **Implementation:** Before Phase 3 (Retry API), define an **Invoice State Resolver** service or API that returns current status for a list of `invoice_id`s. The trigger job calls this first, filters to only still-unpaid invoices, then calls the Retry API only for those.

### 2.2 CRITICAL: Thundering Herd — Jitter and Rate Limiting (Mandatory)

**Problem:** If the model says “Monday 9 AM” is optimal for **5,000 invoices**, an hourly trigger at 09:00:01 would fire 5,000 API calls at once. This will likely cause **429 Too Many Requests** or gateway timeouts from Chargebee (or your billing gateway).

**Fix (mandatory):** Implement **randomized jitter** and **rate limiting** in the Trigger Job.

- **Spread calls over the hour:** e.g. spread the N due invoices across the **first 15–30 minutes** of the hour (e.g. random delay 0–30 min per invoice, or batch and throttle).
- **Rate limit:** Cap concurrent or per-second requests to the Retry API (e.g. 50–100/min depending on Chargebee limits). Use a queue or token bucket; retry 429s with backoff.
- **Implementation:** After querying the schedule for the current hour, shuffle the list of invoice_ids, then for each invoice (after pre-flight): apply a random delay in `[0, 900]` seconds (0–15 min) or `[0, 1800]` (0–30 min), then call the API; enforce max QPS.

### 2.3 Idempotency Key (Mandatory)

**Requirement:** To prevent **double-billing** or double-retry if a Cloud Run job retries, a cron double-fires, or the trigger runs twice, you need a **deterministic idempotency key** that the billing gateway (or your Retry API) accepts.

**Fix:** Use a **composite key** that uniquely identifies one “intent” to retry:

- **Key:** `invoice_id` + `attempt_number` + `scheduled_hour`
  - `invoice_id`: linked_invoice_id.
  - `attempt_number`: the dunning attempt number at schedule time (e.g. from inference run); or use a hash of (invoice_id, optimal_retry_at_utc) if attempt_number is not available at trigger time.
  - `scheduled_hour`: e.g. `DATE_TRUNC(optimal_retry_at_utc, HOUR)` or `YYYY-MM-DDTHH:00:00Z` so the same hour always yields the same key.

**Example:** `idempotency_key = f"{invoice_id}|{attempt_no}|{scheduled_hour_iso}"`. Send this in a header (e.g. `Idempotency-Key`) to your Retry API / Chargebee so duplicate requests within the same hour are rejected or no-op.

---

## Part 3 — Pipeline & Architecture Design

### 3.1 High-Level Architecture (Revised)

**Option A — Batch inference + scheduled trigger (recommended)**

- **Inference job** (e.g. every 4–6 hours):  
  - Fetch active dunning from BigQuery; build features; **log feature sample/distributions** (see Feature Drift); run model; on **model failure use fallback** (e.g. 24h default).  
  - Write to `production_dunning_schedule` with `invoice_id`, `optimal_retry_at_utc`, `attempt_number`, `model_version_id`, etc.
- **Trigger job** (every hour):  
  - Query schedule for current hour → **Pre-Flight:** resolve invoice state (Billing API / DB) → filter to still-unpaid only → **apply jitter + rate limit** → for each: send **Idempotency-Key** + `POST /retry` → log to audit table.

### 3.2 Data Flow (Revised — State-Aware)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INFERENCE PIPELINE (every 4–6h)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  BQ (active dunning) → Feature build → [Feature Log sample] → Model (25 slots)│
│  On model error → Fallback: optimal_retry_at = now + 24h                     │
│  Output: (invoice_id, optimal_retry_at_utc, attempt_number, model_version_id,  │
│          max_prob, inference_run_id, created_at) → production_dunning_schedule │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. PRE-FLIGHT (Invoice State Resolver) — BEFORE trigger                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input: list of invoice_id from schedule for current hour                     │
│  Query: Billing API or real-time DB → current status per invoice             │
│  If status != OPEN/UNPAID → mark schedule as CANCELLED_PAID → SKIP          │
│  Output: list of invoice_id still unpaid → pass to Trigger                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRIGGER PIPELINE (every hour, with jitter + rate limit)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  For each invoice_id (still unpaid):                                         │
│    idempotency_key = invoice_id | attempt_number | scheduled_hour            │
│    Random delay 0–30 min (or batch + throttle) → POST /retry with key         │
│  Log: (invoice_id, triggered_at, api_response, idempotency_key, model_version) │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Component Design (Revised)

| Component | Responsibility | Inputs | Outputs |
|-----------|----------------|--------|--------|
| **Inference job** | BQ → features → feature log sample → model (or fallback) → schedule table. | BQ view, model artifact, config. | Rows in `production_dunning_schedule`; optional feature log table. |
| **Invoice State Resolver** | Return current paid/unpaid status for a list of invoice_ids. | List of invoice_id. | Status per invoice; trigger uses only still-unpaid. |
| **Trigger job** | Query schedule → Pre-Flight filter → jitter + rate limit → POST /retry with idempotency key → audit log. | Schedule table, current hour, Resolver, Retry API. | API calls + `dunning_retry_trigger_log`; schedule rows updated to CANCELLED_PAID where applicable. |
| **Retry API** | Accept idempotency key + invoice_id; perform retry; idempotent. | invoice_id, Idempotency-Key header. | Success/failure; duplicate key → 200 no-op or 409. |

### 3.4 Schedule Table and State Handling

- **Relationship between BigQuery and API sink:** The schedule table is **stale by design** (updated every 4–6h). The **source of truth for “should we retry?”** at trigger time is the **Billing API / real-time DB**, not the schedule. The trigger must **never** call the Retry API without confirming current state.
- **State change:** When an invoice is paid (or cancelled), that state lives in billing; the schedule row may still say “retry at Friday 10 AM.” Pre-Flight ensures we skip that row and optionally mark it `CANCELLED_PAID` so analytics are clear.

---

## Part 4 — API and Interface Design

### 4.1 Inference Output and Schedule Schema

| Column | Type | Description |
|--------|------|-------------|
| `invoice_id` | STRING | Linked invoice id. |
| `optimal_retry_at_utc` | TIMESTAMP | Hour (UTC) to trigger retry. |
| `attempt_number` | INT64 | Dunning attempt number at inference time (for idempotency key). |
| `model_version_id` | STRING | e.g. `20260224`. |
| `max_prob` | FLOAT | P(success) at best slot. |
| `inference_run_id` | STRING | Run identifier. |
| `created_at` | TIMESTAMP | When the row was produced. |
| `status` | STRING | Optional: `PENDING` → `TRIGGERED` or `CANCELLED_PAID` by trigger. |

Partition by `DATE(optimal_retry_at_utc)` for efficient current-hour queries.

### 4.2 Idempotency Key (Sent to Retry API)

- **Header:** `Idempotency-Key: <value>`
- **Value:** Deterministic composite: `{invoice_id}|{attempt_number}|{scheduled_hour}`  
  Example: `inv_xyz|3|2026-02-28T10:00:00Z`.  
  Same invoice + same attempt + same hour → same key; duplicate requests are no-op or 409.

### 4.3 Retry API Contract

- **Endpoint:** `POST /retry` (or Chargebee equivalent).
- **Request:** The API is called with **invoice_id** in the body: `{ "invoice_id": "<linked_invoice_id>" }` plus header `Idempotency-Key: <key>`. The billing system uses `invoice_id` to identify the invoice to retry.
- **Response:** 200 + success flag; 409 if key already used; 4xx/5xx with clear error.
- **Pre-Flight:** Not part of Retry API; trigger calls **Invoice State Resolver** first (separate endpoint or batch status call).

### 4.4 Invoice State Resolver (Pre-Flight)

- **Endpoint:** `POST /invoices/status` or equivalent (Billing API / internal).
- **Request:** `{ "invoice_ids": ["id1", "id2", ...] }`.
- **Response:** `{ "id1": "unpaid", "id2": "paid", ... }` (or open/closed).  
  Trigger keeps only `unpaid`/`open` and skips the rest; optionally write back to schedule: `status = CANCELLED_PAID` for paid ones.

---

## Part 5 — Implementation Gaps Addressed

### 5.1 Feature Drift — Feature Logging (Inference Job)

**Gap:** If the model starts failing in production, you need to know whether **feature distributions** (e.g. `dist_to_payday`, `prev_decline_code`) have shifted.

**Action:** Add a **Feature Logging** step in the Inference Job.

- Log a **sample** of feature rows (e.g. 1% or 500 rows per run) to a BigQuery table: `project.dataset.dunning_inference_feature_log` with columns: `inference_run_id`, `created_at`, `invoice_id`, plus all model input features (and optionally `max_prob`, `optimal_retry_at_utc`).
- Optionally log **aggregates** per run: min/max/mean of numeric features, value_counts of categoricals (or top-K). This allows comparison over time (e.g. drift dashboards).
- **Checklist:** Implement feature log write in the same code path that builds features and runs the model; schedule and retain (e.g. 90 days).

### 5.2 Model Fallback — Hardcoded Baseline

**Gap:** If the Inference Job fails (e.g. model load error, runtime error) or the model returns an error for an invoice, you need a **fail-safe** so dunning still runs.

**Action:** Define a **Hardcoded Baseline**.

- **Rule:** If model is unavailable or inference fails for an invoice, set `optimal_retry_at_utc = inference_run_at + 24 hours` (or match Chargebee’s default retry interval). Write this to the schedule table with `model_version_id = 'fallback_24h'` so monitoring can distinguish model vs fallback.
- **Scope:** Per-invoice errors (e.g. feature build failure): use 24h for that invoice. Global failure (e.g. model file missing): use 24h for all invoices in that run and alert.

### 5.3 Monitoring — “Model vs. Reality” Dashboard

**Gap:** Partial monitoring; need to tie **model predictions** to **actual outcomes**.

**Action:** Build a **Model vs. Reality** dashboard.

- **Data:** Join trigger log (and schedule) with **actual outcomes** from BQ (e.g. same source as `compare_shadow_vs_actual`): for each triggered invoice, did it recover? when?
- **Metrics:**
  - **Delta:** For triggered invoices, compare `suggested_max_prob` (or prob bucket) to **actual success rate** (e.g. by prob decile). Expect roughly increasing recovery rate with higher prob; large deviation suggests calibration drift.
  - **Recovery rate:** % of triggered invoices that recovered within 7 days (or your window).
  - **TTR:** Among recovered, time from `optimal_retry_at_utc` to recovery.
- **Alert:** If actual success rate in a prob bucket drops significantly vs historical, or recovery rate drops, trigger investigation (feature drift, model version, or external factor).

---

## Part 6 — Deployment Options and Model Artifact

### 6.1 Where to Run

| Option | Inference job | Trigger job | Pros | Cons |
|--------|----------------|-------------|------|-----|
| **Airflow** | DAG: BQ → model → schedule + feature log | DAG: query → Pre-Flight → jitter/rate limit → Retry API → log | Central scheduling, retries | Need Airflow; same worker/env for model |
| **Cloud Run + Cloud Scheduler** | Scheduler → Cloud Run **Job** (inference, read BQ, write schedule) | Scheduler (hourly) → Cloud Run **Job** (Pre-Flight + trigger + log) | Serverless, no VM; scale to zero | Cold start; model in GCS or image; see [6.3](#63-google-cloud-run-recommended) |
| **GKE / VM + cron** | CronJob / cron script | CronJob / cron script | Full control | Operate cluster/VM |

### 6.2 Feature Consistency — The “Pickle” Trap

**Warning:** The production environment (Cloud Run, Airflow worker, etc.) must use the **exact same versions** of `scikit-learn` and `catboost` as the **training environment**. CatBoost models are generally robust across patch versions, but **scikit-learn’s IsotonicRegression** can have breaking or behavioral changes between minor versions. Mismatches can cause load failures or silent mis-calibration.

**Action:**

- Pin in `requirements.txt` or in the container image: e.g. `scikit-learn==1.3.x`, `catboost==1.2.x` (match your training env exactly).
- Document the training runtime (Python version, sklearn, catboost) in the model card or config; CI or deployment checks should verify production matches.
- Keep **IsotonicCalibratedClassifier** in the loading path (same module as in `shadow_monitoring_20260206.py`) so joblib can unpickle.

### 6.3 Google Cloud Run (Recommended)

If you deploy on **Google Cloud Run**, use the following pattern.

**Inference and trigger as Cloud Run Jobs**

- **Inference job:** A Cloud Run **Job** that (1) reads active dunning from BigQuery, (2) loads the model (from GCS or from the container image), (3) runs inference, (4) writes to `production_dunning_schedule` (and optional feature log). Invoked by **Cloud Scheduler** every 4–6 hours (e.g. `0 */6 * * *` for every 6 hours).
- **Trigger job:** A Cloud Run **Job** that (1) queries the schedule for the current hour, (2) calls Invoice State Resolver (Pre-Flight), (3) applies jitter and rate limiting, (4) calls the Retry API with **invoice_id** and Idempotency-Key for each still-unpaid invoice, (5) writes to `dunning_retry_trigger_log`. Invoked by **Cloud Scheduler** hourly (e.g. `0 * * * *`).

**Model artifact**

- Do **not** bundle the model in the container if it is large or changes often. Store the calibrated model in **Cloud Storage** (e.g. `gs://your-bucket/dunning/models/catboost_dunning_calibrated_20260224.joblib`). At job start, download the model (e.g. with `google-cloud-storage`) and load with joblib. Set `DUNNING_MODEL_PATH` or `GCS_MODEL_URI` in the Job's environment so the code knows where to load from.
- Alternatively, bake the model into the container image if you want to avoid GCS at runtime; then redeploy the image when you promote a new model.

**Cold start**

- Cloud Run Jobs start a new container when scheduled. Cold start adds latency (often 10–30 s). For **batch** inference and trigger jobs this is acceptable. If you add an **on-demand inference API** (e.g. "score this invoice_id now"), deploy it as a Cloud Run **Service** and consider setting **minimum instances** (e.g. 1) to avoid cold start on first request, or accept cold start for low traffic.

**Environment and secrets**

- Use Cloud Run Job/Service **environment variables** for: `BQ_PROJECT`, `BQ_TABLE`, `DUNNING_MODEL_PATH` or `GCS_MODEL_URI`, `TRAFFIC_SPLIT_MODEL_PCT`, Retry API base URL.
- Use **Secret Manager** for: Billing/Chargebee API keys, BigQuery credentials (or rely on the default service account with BQ permissions). Mount secrets as env vars or volume in the Job/Service.

**Optional: Inference API (invoice_id)**

- To support **on-demand** scoring (e.g. "get optimal retry for this invoice_id"), deploy a Cloud Run **Service** that exposes `POST /score` or `GET /score?invoice_id=<id>`. Request body or query: **invoice_id**. The service (1) fetches that invoice's latest state from BigQuery (or from an internal API), (2) builds features, (3) runs the model for all candidate slots, (4) returns `optimal_retry_at`, `max_prob`, and optionally the full prob vector. This is optional; the main production path is batch inference → schedule → hourly trigger.

---

## Part 7 — Refined Step-by-Step for ML/DevOps

### Phase 1 — Pre-Flight Service (Before Retry API)

1. **Define Invoice State Resolver**  
   - Implement or wire an API (or BQ view with minimal latency) that returns current status for a list of `invoice_id`: e.g. `unpaid` / `paid` / `cancelled`. Source: Billing API or real-time DB (not the batch BQ snapshot used for inference).
2. **Contract for Trigger**  
   - Trigger job calls Resolver with all `invoice_id` in the current hour’s schedule; receives status map; filters to `status in ('unpaid', 'open')`. For any other status, update schedule row to `status = 'CANCELLED_PAID'` (or similar) and **SKIP** Retry API.
3. **No Retry API call without Pre-Flight**  
   - Do not proceed to Phase 3 (Retry API integration) until Pre-Flight is in place and tested (e.g. unit test: paid invoice → not in “to trigger” list).

### Phase 2 — Schema, Storage, and Feature Consistency

4. **Create BigQuery tables**  
   - `production_dunning_schedule`: columns including `invoice_id`, `optimal_retry_at_utc`, `attempt_number`, `model_version_id`, `max_prob`, `inference_run_id`, `created_at`, `status`. Partition by `DATE(optimal_retry_at_utc)`.  
   - `dunning_retry_trigger_log`: `invoice_id`, `optimal_retry_at_utc`, `triggered_at`, `idempotency_key`, `api_response_status`, `model_version_id`, `error_message`.  
   - `dunning_inference_feature_log`: optional; sample of features + run id + timestamp for drift analysis.
5. **Pin dependencies**  
   - Pin `scikit-learn` and `catboost` to training versions; document in runbook; verify in production image or venv.

### Phase 3 — Inference Pipeline (with Feature Log and Fallback)

6. **Refactor inference**  
   - Write to `production_dunning_schedule` (upsert by invoice_id or overwrite for upcoming window). Include `attempt_number` from BQ (e.g. `invoice_attempt_no`).  
   - **Feature log:** Write sample of feature rows (+ run id, timestamp) to `dunning_inference_feature_log`.  
   - **Fallback:** On model load failure or per-invoice inference error, set `optimal_retry_at_utc = now + 24h`, `model_version_id = 'fallback_24h'`; still write to schedule.
7. **Schedule inference job**  
   - Every 6 h (Airflow or Cloud Scheduler + Cloud Run). Model from GCS; BQ read + write; GCS read for model.

### Phase 4 — Retry API and Trigger (with Jitter, Rate Limit, Idempotency)

8. **Implement Retry API**  
   - Accept `invoice_id` + `Idempotency-Key` header; call Chargebee (or internal); return 200/409/4xx/5xx. Enforce idempotency key server-side.
9. **Trigger job logic**  
   - Query schedule for current hour (UTC).  
   - **Pre-Flight:** Call Invoice State Resolver; keep only unpaid; mark others CANCELLED_PAID.  
   - For each remaining invoice: build `idempotency_key = f"{invoice_id}|{attempt_number}|{scheduled_hour_iso}"`.  
   - **Jitter:** Shuffle list; assign each a random delay in [0, 1800] seconds (or [0, 900]).  
   - **Rate limit:** Execute API calls with a cap (e.g. 50/min); on 429, backoff and retry.  
   - Log every attempt to `dunning_retry_trigger_log` (including idempotency_key, status, error).
10. **Schedule trigger job**  
    - Hourly (e.g. at :00). Ensure Pre-Flight and jitter/rate limit are part of the same job so no “naked” trigger without state check.

### Phase 5 — Shadow-to-Live Transition (Traffic Splitter)

11. **Do not go 100% live initially**  
    - Implement a **Traffic Splitter**: e.g. only **10%** of “optimal retries” (by invoice_id hash or random seed) are sent to the Retry API; **90%** continue to follow default Chargebee logic (no override).  
    - Log both cohorts (model-driven vs default) with a flag; after **14 days** compare **recovery rate** and TTR between the two. If model cohort performs better (or non-inferior), increase to 50% then 100%.
12. **Configuration**  
    - e.g. `TRAFFIC_SPLIT_MODEL_PCT = 10` in env; trigger job only calls Retry API for invoice_id where `hash(invoice_id) % 100 < TRAFFIC_SPLIT_MODEL_PCT`; others are logged but not sent to API (or sent with a “no-op” path that still uses default Chargebee schedule).

### Phase 6 — Safety and Observability

13. **Dry run**  
    - Trigger job supports `DRY_RUN=1`: log “would trigger invoice_id at … with key …” without calling Retry API.  
14. **Model vs. Reality dashboard**  
    - Join trigger log + schedule with actual outcomes; plot success rate by prob bucket; recovery rate; TTR; alert on drift.  
15. **Rollback**  
    - Switch model version to previous artifact and re-run inference; or set `TRAFFIC_SPLIT_MODEL_PCT = 0` to stop model-driven retries; document in runbook.

---

## Part 8 — Checklist Summary (Revised)

| # | Step | Owner |
|---|------|--------|
| 1 | Create BQ tables: `production_dunning_schedule`, `dunning_retry_trigger_log`, optional `dunning_inference_feature_log` | Data Eng |
| 2 | Define and implement **Invoice State Resolver** (Pre-Flight); contract: list of invoice_id → status map | Backend |
| 3 | Implement **Pre-Flight** in trigger: filter to unpaid only; mark CANCELLED_PAID for paid | Backend/ML |
| 4 | Refactor inference: write schedule to BQ; include **attempt_number**; add **Feature Log** sample; add **Model Fallback** (24h) | ML Eng |
| 5 | Pin **scikit-learn** and **catboost** versions; document; verify in production image | ML Eng / DevOps |
| 6 | Deploy model to GCS; set model version in config | ML Eng |
| 7 | Implement **Retry API** with **Idempotency-Key** (invoice_id + attempt_number + scheduled_hour) | Backend |
| 8 | Implement trigger: query → Pre-Flight → **jitter** (0–30 min) + **rate limit** → POST /retry with idempotency key → log | Backend/ML |
| 9 | Schedule inference job (every 6 h) and trigger job (hourly) | DevOps |
| 10 | Implement **Traffic Splitter** (10% model, 90% default); compare recovery rate after 14 days | ML Eng |
| 11 | **Model vs. Reality** dashboard + alerts; dry-run mode; runbook and rollback | DevOps / ML |

---

## Part 9 — Timezone and “At the Hour”

- **Inference:** `inference_run_at` and `optimal_retry_at` in UTC; store `optimal_retry_at_utc` in schedule.  
- **Trigger:** Query in UTC (e.g. current hour window 08:00–08:59 UTC).  
- **Idempotency key:** Use the same hour truncation (e.g. `2026-02-28T08:00:00Z`) so the key is deterministic for that hour.

---

This revised guide ensures: **no retry without Pre-Flight**, **no thundering herd** (jitter + rate limit), **deterministic idempotency**, **feature logging**, **model fallback**, **Model vs. Reality** monitoring, and **shadow-to-live** rollout with a traffic splitter.
