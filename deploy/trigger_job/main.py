"""
Production trigger job: query schedule for current hour (UTC), jitter + rate limit,
velocity cap, call Retry API with idempotency key, log to dunning_retry_trigger_log.

No pre-flight status API: if an invoice is already paid, the Retry API returns an error
and we log it.

Implements Chargebee API:
- call_retry_api: POST /invoices/{id}/collect_payment with Idempotency-Key header.

Env: CHARGEBEE_SITE (e.g. aicreative), CHARGEBEE_API_KEY (or mount from Secret Manager).
"""
from __future__ import annotations

import base64
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# App root = directory containing lib/ and trigger_job/ (the deploy folder when run on Cloud Run).
_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))


def _get_chargebee_auth() -> tuple[str, str]:
    """Return (site, api_key). Raises if not configured."""
    site = (os.environ.get("CHARGEBEE_SITE") or "").strip()
    key = (os.environ.get("CHARGEBEE_API_KEY") or "").strip()
    if not site or not key:
        raise RuntimeError(
            "CHARGEBEE_SITE and CHARGEBEE_API_KEY must be set (or mount API key from Secret Manager)."
        )
    return site, key


def _chargebee_fetch(
    url: str,
    method: str = "GET",
    idempotency_key: str | None = None,
    max_retries: int = 3,
    base_sleep_sec: float = 0.8,
) -> tuple[int, str]:
    """
    Call Chargebee API with Basic auth and optional retry on 429/5xx.
    Returns (http_status_code, response_text).
    """
    site, api_key = _get_chargebee_auth()
    raw = f"{api_key}:"
    auth_header = "Basic " + base64.b64encode(raw.encode()).decode()

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, method=method)
            req.add_header("Authorization", auth_header)
            req.add_header("Content-Type", "application/json")
            if idempotency_key:
                req.add_header("Idempotency-Key", idempotency_key)
            with urllib.request.urlopen(req, timeout=60) as resp:
                code = resp.getcode()
                text = resp.read().decode("utf-8", errors="replace")
                return (code, text)
        except urllib.error.HTTPError as e:
            code = e.code
            text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if (code == 429 or (500 <= code <= 599)) and attempt < max_retries:
                time.sleep(base_sleep_sec * (2**attempt))
                continue
            return (code, text)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(base_sleep_sec * (2**attempt))
                continue
            return (-1, str(e))
    return (-1, str(last_err) if last_err else "Unknown error")


def _normalize_error_message(code: int, text: str, fallback: str = "N/A") -> str:
    """Extract error message from Chargebee JSON or use text/fallback."""
    try:
        j = json.loads(text)
        if isinstance(j.get("message"), str):
            return j["message"]
        if isinstance(j.get("transaction"), dict) and isinstance(j["transaction"].get("error_text"), str):
            return j["transaction"]["error_text"]
    except (json.JSONDecodeError, TypeError):
        pass
    if text:
        return text[:500]
    return fallback


def _parse_collect_payment_response(code: int, text: str) -> tuple[str, str]:
    """
    Parse collect_payment response. Returns (message, txn_id).
    - message: normalized error message, or 'Payment collection success.' when code==200.
    - txn_id: transaction.id from response when code==200, else ''.
    """
    if code == 200:
        try:
            j = json.loads(text)
            txn_id = ""
            if isinstance(j.get("transaction"), dict) and j["transaction"].get("id"):
                txn_id = str(j["transaction"]["id"]).strip()
            return ("Payment collection success.", txn_id)
        except (json.JSONDecodeError, TypeError, KeyError):
            return ("Payment collection success.", "")
    msg = _normalize_error_message(code, text, f"HTTP {code}")
    return (msg, "")


def call_retry_api(invoice_id: str, idempotency_key: str) -> tuple[int, str, str]:
    """
    Chargebee collect_payment: POST /invoices/{invoice_id}/collect_payment
    with Idempotency-Key header.
    Returns (http_status_code, message, txn_id).
    - message: normalized error or 'Payment collection success.'
    - txn_id: transaction id when 200, else empty string.
    """
    site, _ = _get_chargebee_auth()
    url = f"https://{site}.chargebee.com/api/v2/invoices/{urllib.parse.quote(str(invoice_id).strip(), safe='')}/collect_payment"
    code, text = _chargebee_fetch(url, method="POST", idempotency_key=idempotency_key)
    message, txn_id = _parse_collect_payment_response(code, text)
    return (code, message, txn_id)


def get_schedule_for_current_hour() -> list[dict]:
    """Query production_dunning_schedule for current hour (UTC). Env: BQ_PROJECT, BQ_DATASET, SCHEDULE_TABLE."""
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table = os.environ.get("SCHEDULE_TABLE", "production_dunning_schedule")
    if not project or not dataset:
        raise RuntimeError("BQ_PROJECT and BQ_DATASET must be set.")
    print(f"[trigger_job] Querying schedule: {project}.{dataset}.{table}", file=sys.stderr)
    client = bigquery.Client(project=project)
    now = datetime.now(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    query = f"""
        SELECT invoice_id, optimal_retry_at_utc, attempt_number
        FROM `{project}.{dataset}.{table}`
        WHERE DATE(optimal_retry_at_utc) = @date
          AND TIMESTAMP_TRUNC(optimal_retry_at_utc, HOUR) = @hour_start
          AND COALESCE(status, 'PENDING') = 'PENDING'
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date", "DATE", hour_start.date()),
            bigquery.ScalarQueryParameter("hour_start", "TIMESTAMP", hour_start.isoformat()),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    rows = df.to_dict("records")
    print(f"[trigger_job] Fetched {len(rows)} PENDING rows for hour {hour_start.isoformat()} UTC", file=sys.stderr)
    return rows


def build_idempotency_key(invoice_id: str, attempt_number: int, optimal_retry_at_utc) -> str:
    """Deterministic key: invoice_id|attempt_number|scheduled_hour_iso."""
    if hasattr(optimal_retry_at_utc, "strftime"):
        hour_iso = optimal_retry_at_utc.strftime("%Y-%m-%dT%H:00:00Z")
    else:
        hour_iso = str(optimal_retry_at_utc)[:13].replace(" ", "T") + ":00:00Z"
    return f"{invoice_id}|{attempt_number}|{hour_iso}"


def get_retry_count_last_n_days(
    invoice_ids: list[str],
    days: int = 7,
    as_of_utc: datetime | None = None,
) -> dict[str, int]:
    """
    Return per-invoice count of retry API attempts in the last `days` days (rolling window).
    Only counts real Retry API calls: excludes DRY_RUN and VELOCITY_CAP_7D log rows.
    Used for velocity cap: no more than N retries in any rolling 7-day window.
    """
    if not invoice_ids:
        return {}
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table = os.environ.get("TRIGGER_LOG_TABLE", "dunning_retry_trigger_log")
    if not project or not dataset:
        return {inv_id: 0 for inv_id in invoice_ids}
    now = as_of_utc or datetime.now(timezone.utc)
    since = now - timedelta(days=days)

    client = bigquery.Client(project=project)
    query = f"""
        SELECT invoice_id, COUNT(1) AS cnt
        FROM `{project}.{dataset}.{table}`
        WHERE triggered_at >= @since
          AND triggered_at <= @now
          AND invoice_id IN UNNEST(@invoice_ids)
          AND (error_message IS NULL OR (
              error_message != 'DRY_RUN'
              AND error_message != 'VELOCITY_CAP_7D'
              AND error_message != 'call_retry_api not implemented'
          ))
        GROUP BY invoice_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("since", "TIMESTAMP", since.isoformat()),
            bigquery.ScalarQueryParameter("now", "TIMESTAMP", now.isoformat()),
            bigquery.ArrayQueryParameter("invoice_ids", "STRING", list(invoice_ids)),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    count_map = dict(zip(df["invoice_id"], df["cnt"]))
    return {inv_id: int(count_map.get(inv_id, 0)) for inv_id in invoice_ids}


def write_trigger_log_to_bq(rows: list[dict]) -> None:
    """Insert into dunning_retry_trigger_log. Env: BQ_PROJECT, BQ_DATASET, TRIGGER_LOG_TABLE."""
    if not rows:
        return
    import pandas as pd
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table = os.environ.get("TRIGGER_LOG_TABLE", "dunning_retry_trigger_log")
    if not project or not dataset:
        return
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    df = pd.DataFrame(rows)
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()


def run() -> None:
    """Query schedule → jitter + rate limit → velocity cap → Retry API → log.
    No pre-flight status check: if an invoice is already paid, the Retry API returns an error and we log it."""
    dry_run = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
    rate_limit_per_min = int(os.environ.get("RATE_LIMIT_PER_MIN", "50") or 50)
    jitter_max_seconds = int(os.environ.get("JITTER_MAX_SECONDS", "1800") or 1800)
    traffic_split_pct = int(os.environ.get("TRAFFIC_SPLIT_MODEL_PCT", "100") or 100)
    velocity_cap_days = int(os.environ.get("VELOCITY_CAP_DAYS", "7") or 7)
    velocity_cap_max = int(os.environ.get("VELOCITY_CAP_MAX_RETRIES_7D", "3") or 3)

    print(
        f"[trigger_job] Starting run: dry_run={dry_run}, rate_limit={rate_limit_per_min}/min, "
        f"traffic_split={traffic_split_pct}%, velocity_cap={velocity_cap_max} retries/{velocity_cap_days}d",
        file=sys.stderr,
    )
    rows = get_schedule_for_current_hour()
    if not rows:
        print("No schedule rows for current hour.", file=sys.stderr)
        return

    n_before_split = len(rows)
    if traffic_split_pct < 100:
        random.shuffle(rows)
        n = max(1, int(len(rows) * traffic_split_pct / 100))
        rows = rows[:n]
        print(f"[trigger_job] Traffic split: {n_before_split} -> {len(rows)} rows ({traffic_split_pct}%)", file=sys.stderr)
    random.shuffle(rows)

    now = datetime.now(timezone.utc)
    retry_count_7d = get_retry_count_last_n_days(
        [r["invoice_id"] for r in rows],
        days=velocity_cap_days,
        as_of_utc=now,
    )
    n_capped = sum(1 for r in rows if retry_count_7d.get(r["invoice_id"], 0) >= velocity_cap_max)
    if n_capped > 0:
        print(f"[trigger_job] Velocity cap: {n_capped} invoice(s) already at {velocity_cap_max} retries in last {velocity_cap_days}d", file=sys.stderr)

    log_rows = []
    min_interval = 60.0 / rate_limit_per_min if rate_limit_per_min else 0
    n_dry_run = 0
    n_velocity_cap = 0
    n_success = 0
    n_failed = 0

    for r in rows:
        invoice_id = r["invoice_id"]
        if retry_count_7d.get(invoice_id, 0) >= velocity_cap_max:
            n_velocity_cap += 1
            log_rows.append({
                "invoice_id": invoice_id,
                "optimal_retry_at_utc": r.get("optimal_retry_at_utc"),
                "triggered_at": now,
                "idempotency_key": build_idempotency_key(
                    invoice_id, int(r.get("attempt_number", 0)), r.get("optimal_retry_at_utc")
                ),
                "api_response_status": None,
                "error_message": "VELOCITY_CAP_7D",
                "status": "VELOCITY_CAP_7D",
                "message": "VELOCITY_CAP_7D",
                "txn_id": None,
                "attempt_number": int(r.get("attempt_number", 0)),
                "comment_status": None,
            })
            continue

        attempt_number = int(r.get("attempt_number", 0))
        optimal_retry_at_utc = r.get("optimal_retry_at_utc")
        idempotency_key = build_idempotency_key(invoice_id, attempt_number, optimal_retry_at_utc)

        delay = random.randint(0, jitter_max_seconds) if jitter_max_seconds else 0
        if delay > 0:
            time.sleep(delay)
        time.sleep(min_interval)

        if dry_run:
            n_dry_run += 1
            print(f"[trigger_job] DRY_RUN: would trigger invoice_id={invoice_id}", file=sys.stderr)
            log_rows.append({
                "invoice_id": invoice_id,
                "optimal_retry_at_utc": optimal_retry_at_utc,
                "triggered_at": now,
                "idempotency_key": idempotency_key,
                "api_response_status": None,
                "error_message": "DRY_RUN",
                "status": "DRY_RUN",
                "message": "DRY_RUN",
                "txn_id": None,
                "attempt_number": attempt_number,
                "comment_status": None,
            })
            continue

        try:
            print(f"[trigger_job] Calling Retry API: invoice_id={invoice_id}", file=sys.stderr)
            status_code, api_message, api_txn_id = call_retry_api(invoice_id, idempotency_key)
            if status_code == 200:
                n_success += 1
                print(f"[trigger_job] Success: invoice_id={invoice_id} txn_id={api_txn_id or 'N/A'}", file=sys.stderr)
            else:
                n_failed += 1
                print(f"[trigger_job] Failed: invoice_id={invoice_id} status={status_code} message={api_message[:80]}", file=sys.stderr)
        except NotImplementedError:
            n_failed += 1
            print(f"[trigger_job] NotImplemented: invoice_id={invoice_id} (call_retry_api not implemented)", file=sys.stderr)
            log_rows.append({
                "invoice_id": invoice_id,
                "optimal_retry_at_utc": optimal_retry_at_utc,
                "triggered_at": now,
                "idempotency_key": idempotency_key,
                "api_response_status": None,
                "error_message": "call_retry_api not implemented",
                "status": "Error",
                "message": "call_retry_api not implemented",
                "txn_id": None,
                "attempt_number": attempt_number,
                "comment_status": None,
            })
            continue
        except Exception as e:
            status_code = -1
            api_message = str(e)
            api_txn_id = None
            n_failed += 1
            print(f"[trigger_job] Exception: invoice_id={invoice_id} error={e}", file=sys.stderr)

        status_label = "Success" if status_code == 200 else "Failed"
        log_rows.append({
            "invoice_id": invoice_id,
            "optimal_retry_at_utc": optimal_retry_at_utc,
            "triggered_at": now,
            "idempotency_key": idempotency_key,
            "api_response_status": status_code,
            "error_message": None if status_code == 200 else api_message,
            "status": status_label,
            "message": api_message,
            "txn_id": api_txn_id or None,
            "attempt_number": attempt_number,
            "comment_status": None,  # Trigger job does not create transaction comment; use "Skipped" if preferred
        })

    write_trigger_log_to_bq(log_rows)
    print(f"[trigger_job] Wrote {len(log_rows)} rows to dunning_retry_trigger_log", file=sys.stderr)
    print(
        f"[trigger_job] Summary: total={len(log_rows)} | dry_run={n_dry_run} | velocity_cap={n_velocity_cap} | "
        f"success={n_success} | failed={n_failed}",
        file=sys.stderr,
    )
    print(f"Trigger run: {len(log_rows)} logged (dry_run={dry_run}).")


if __name__ == "__main__":
    run()
