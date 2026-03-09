"""
Production trigger job: query schedule for current hour (UTC), Pre-Flight (invoice status),
jitter + rate limit, call Retry API with idempotency key, log to dunning_retry_trigger_log.

Implements Chargebee API:
- resolve_invoice_status: GET /invoices/{id} per invoice, map status to unpaid|open|paid|cancelled.
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
from datetime import datetime, timezone
from pathlib import Path

# App root = directory containing lib/ and trigger_job/ (the deploy folder when run on Cloud Run).
_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

# Chargebee status -> trigger job canonical (only 'unpaid' and 'open' are retried)
_CB_STATUS_TO_CANONICAL = {
    "payment_due": "unpaid",
    "not_paid": "unpaid",
    "unpaid": "unpaid",
    "open": "open",
    "paid": "paid",
    "voided": "cancelled",
    "cancelled": "cancelled",
}


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


def resolve_invoice_status(invoice_ids: list[str]) -> dict[str, str]:
    """
    Return current status per invoice_id via Chargebee GET /invoices/{id}.
    Maps Chargebee status to 'unpaid'|'open'|'paid'|'cancelled'. Only 'unpaid' and 'open' are retried.
    """
    if not invoice_ids:
        return {}
    site, _ = _get_chargebee_auth()
    base_url = f"https://{site}.chargebee.com/api/v2/invoices"
    result: dict[str, str] = {}
    for i, inv_id in enumerate(invoice_ids):
        inv_id = str(inv_id).strip()
        if not inv_id:
            continue
        url = f"{base_url}/{urllib.parse.quote(inv_id, safe='')}"
        code, text = _chargebee_fetch(url, method="GET")
        if code == 200:
            try:
                data = json.loads(text)
                cb_status = (data.get("invoice") or {}).get("status", "")
                result[inv_id] = _CB_STATUS_TO_CANONICAL.get(
                    (cb_status or "").lower().replace("-", "_"), "unknown"
                )
            except (json.JSONDecodeError, TypeError):
                result[inv_id] = "unknown"
        elif code == 404:
            result[inv_id] = "cancelled"
        else:
            result[inv_id] = "unknown"
        if i < len(invoice_ids) - 1:
            time.sleep(0.2)
    return result


def call_retry_api(invoice_id: str, idempotency_key: str) -> tuple[int, str]:
    """
    Chargebee collect_payment: POST /invoices/{invoice_id}/collect_payment
    with Idempotency-Key header. Returns (http_status_code, error_message or "").
    """
    site, _ = _get_chargebee_auth()
    url = f"https://{site}.chargebee.com/api/v2/invoices/{urllib.parse.quote(str(invoice_id).strip(), safe='')}/collect_payment"
    code, text = _chargebee_fetch(url, method="POST", idempotency_key=idempotency_key)
    if code == 200:
        return (200, "")
    return (code, _normalize_error_message(code, text, f"HTTP {code}"))


def get_schedule_for_current_hour() -> list[dict]:
    """Query production_dunning_schedule for current hour (UTC). Env: BQ_PROJECT, BQ_DATASET, SCHEDULE_TABLE."""
    from google.cloud import bigquery

    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table = os.environ.get("SCHEDULE_TABLE", "production_dunning_schedule")
    if not project or not dataset:
        raise RuntimeError("BQ_PROJECT and BQ_DATASET must be set.")
    client = bigquery.Client(project=project)
    now = datetime.now(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    query = f"""
        SELECT invoice_id, optimal_retry_at_utc, attempt_number, model_version_id
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
    return df.to_dict("records")


def build_idempotency_key(invoice_id: str, attempt_number: int, optimal_retry_at_utc) -> str:
    """Deterministic key: invoice_id|attempt_number|scheduled_hour_iso."""
    if hasattr(optimal_retry_at_utc, "strftime"):
        hour_iso = optimal_retry_at_utc.strftime("%Y-%m-%dT%H:00:00Z")
    else:
        hour_iso = str(optimal_retry_at_utc)[:13].replace(" ", "T") + ":00:00Z"
    return f"{invoice_id}|{attempt_number}|{hour_iso}"


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
    """Query schedule → Pre-Flight → jitter + rate limit → Retry API → log."""
    dry_run = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
    rate_limit_per_min = int(os.environ.get("RATE_LIMIT_PER_MIN", "50") or 50)
    jitter_max_seconds = int(os.environ.get("JITTER_MAX_SECONDS", "1800") or 1800)
    traffic_split_pct = int(os.environ.get("TRAFFIC_SPLIT_MODEL_PCT", "100") or 100)

    rows = get_schedule_for_current_hour()
    if not rows:
        print("No schedule rows for current hour.")
        return

    invoice_ids = [r["invoice_id"] for r in rows]
    try:
        status_map = resolve_invoice_status(invoice_ids)
    except NotImplementedError:
        print("resolve_invoice_status not implemented; skipping trigger.", file=sys.stderr)
        return

    unpaid = [r for r in rows if status_map.get(r["invoice_id"], "").lower() in ("unpaid", "open")]
    if traffic_split_pct < 100:
        random.shuffle(unpaid)
        n = max(1, int(len(unpaid) * traffic_split_pct / 100))
        unpaid = unpaid[:n]
    random.shuffle(unpaid)

    log_rows = []
    now = datetime.now(timezone.utc)
    min_interval = 60.0 / rate_limit_per_min if rate_limit_per_min else 0

    for r in unpaid:
        invoice_id = r["invoice_id"]
        attempt_number = int(r.get("attempt_number", 0))
        optimal_retry_at_utc = r.get("optimal_retry_at_utc")
        model_version_id = r.get("model_version_id", "")
        idempotency_key = build_idempotency_key(invoice_id, attempt_number, optimal_retry_at_utc)

        delay = random.randint(0, jitter_max_seconds) if jitter_max_seconds else 0
        if delay > 0:
            time.sleep(delay)
        time.sleep(min_interval)

        if dry_run:
            log_rows.append({
                "invoice_id": invoice_id,
                "optimal_retry_at_utc": optimal_retry_at_utc,
                "triggered_at": now,
                "idempotency_key": idempotency_key,
                "api_response_status": None,
                "model_version_id": model_version_id,
                "error_message": "DRY_RUN",
            })
            continue

        try:
            status_code, err_msg = call_retry_api(invoice_id, idempotency_key)
        except NotImplementedError:
            log_rows.append({
                "invoice_id": invoice_id,
                "optimal_retry_at_utc": optimal_retry_at_utc,
                "triggered_at": now,
                "idempotency_key": idempotency_key,
                "api_response_status": None,
                "model_version_id": model_version_id,
                "error_message": "call_retry_api not implemented",
            })
            continue
        except Exception as e:
            status_code = -1
            err_msg = str(e)

        log_rows.append({
            "invoice_id": invoice_id,
            "optimal_retry_at_utc": optimal_retry_at_utc,
            "triggered_at": now,
            "idempotency_key": idempotency_key,
            "api_response_status": status_code,
            "model_version_id": model_version_id,
            "error_message": err_msg or None,
        })

    write_trigger_log_to_bq(log_rows)
    print(f"Trigger run: {len(log_rows)} logged (dry_run={dry_run}).")


if __name__ == "__main__":
    run()
