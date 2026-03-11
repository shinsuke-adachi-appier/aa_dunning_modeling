"""
Production inference job (standalone). Fetch active dunning from BQ, run model, write
to production_dunning_schedule and optional dunning_inference_feature_log.
All code lives in deploy/ (lib); no parent repo.
"""
from __future__ import annotations

import os
import random
import sys
import uuid
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# App root = directory containing lib/ and inference_job/ (the deploy folder when run on Cloud Run).
_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from lib import bq_fetch, features, model, slots


def _to_utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """Ensure timestamp is timezone-aware UTC for BQ output."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def write_schedule_to_bq(rows: list[dict]) -> None:
    if not rows:
        return
    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table = os.environ.get("SCHEDULE_TABLE", "production_dunning_schedule")
    if not project or not dataset:
        print("BQ_PROJECT and BQ_DATASET must be set to write schedule.", file=sys.stderr)
        return
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    df = pd.DataFrame(rows)
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"Written {len(rows)} rows to BigQuery {table_id}.", file=sys.stderr)


def write_feature_log_to_bq(rows: list[dict]) -> None:
    """Write feature log rows. Table is created from lib.features.FEATURE_LOG_SCHEMA if it does not exist."""
    if not rows:
        return
    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table = os.environ.get("FEATURE_LOG_TABLE", "dunning_inference_feature_log")
    if not project or not dataset:
        return
    try:
        client = bigquery.Client(project=project)
        table_ref = f"{project}.{dataset}.{table}"
        try:
            client.get_table(table_ref)
        except NotFound:
            # Table does not exist: create from Python schema (lib.features.FEATURE_LOG_SCHEMA)
            schema = [
                bigquery.SchemaField(name, bq_type, mode="NULLABLE")
                for name, bq_type in features.FEATURE_LOG_SCHEMA
            ]
            client.create_table(bigquery.Table(table_ref, schema=schema))
        df = pd.DataFrame(rows)
        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
        client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()
    except Exception as e:
        print(f"Feature log write failed (non-fatal): {e}", file=sys.stderr)


def run() -> None:
    # Prefer local model when path exists (avoids GCS with placeholder bucket in local dev).
    model_path = None
    local_path = os.environ.get("DUNNING_MODEL_PATH")
    if local_path and Path(local_path).exists():
        model_path = local_path
    if not model_path:
        model_path = os.environ.get("GCS_MODEL_URI")
    if not model_path:
        print("Set GCS_MODEL_URI or DUNNING_MODEL_PATH.", file=sys.stderr)
        sys.exit(1)
    inference_run_at = pd.Timestamp.now("UTC")
    inference_run_id = str(uuid.uuid4())
    model_version_id = os.environ.get("DUNNING_MODEL_VERSION") or "production"
    calibrated = model.load_model(model_path)
    use_fallback_global = calibrated is None
    if calibrated is None:
        model_version_id = "fallback_24h"
        print("Model not loaded; using fallback_24h for all rows.", file=sys.stderr)
    else:
        print(f"Model loaded from {model_path}.", file=sys.stderr)

    active_df = bq_fetch.fetch_active_dunning()
    if active_df.empty:
        print("No active dunning invoices.")
        return

    time_col = "updated_at"
    first_col = "first_attempt_at"
    invoices = active_df.drop_duplicates(subset=["linked_invoice_id"], keep="last")
    if "linked_invoice_id" not in invoices.columns and "invoice_id" in invoices.columns:
        invoices = invoices.rename(columns={"invoice_id": "linked_invoice_id"})

    schedule_rows = []
    feature_log_rows = []
    feature_log_sample_pct = float(os.environ.get("FEATURE_LOG_SAMPLE_PCT", "1") or 1)
    do_feature_log = feature_log_sample_pct > 0 and os.environ.get("FEATURE_LOG_TABLE")

    for _, raw in invoices.iterrows():
        invoice_id = raw.get("linked_invoice_id") or raw.get("invoice_id")
        base_ts = pd.Timestamp(raw[time_col])
        first_ts = raw.get(first_col)
        first_ts = pd.Timestamp(first_ts) if first_ts is not None and not pd.isna(first_ts) else base_ts
        attempt_number = int(raw.get("invoice_attempt_no", 0) or 0)

        optimal_retry_at_utc = _to_utc_ts((inference_run_at + pd.Timedelta(hours=24)).round("h"))
        max_prob = None
        version_for_row = "fallback_24h"

        if not use_fallback_global:
            try:
                tz = str(raw.get("timezone", "") or "").strip() or None
                invoice_row = features.build_invoice_row(
                    raw, base_ts, first_ts, as_of_timestamp=inference_run_at, timezone=tz,
                    as_of_localized=raw.get("localized_time"),
                )
            except Exception as e:
                version_for_row = "fallback_24h"
                if len(schedule_rows) == 0:  # log first failure only
                    print(f"First fallback (build_invoice_row): {e}", file=sys.stderr)
            else:
                try:
                    optimal_retry_at, max_prob, raw_snapshot, _ = slots.run_inference_for_invoice(
                        invoice_row,
                        base_timestamp_for_slots=inference_run_at,
                        first_attempt_at=first_ts,
                        model=calibrated,
                        timezone=tz,
                    )
                    optimal_retry_at_utc = _to_utc_ts(optimal_retry_at)
                    version_for_row = model_version_id
                    if do_feature_log and random.random() * 100 < feature_log_sample_pct:
                        fl = {
                            "inference_run_id": inference_run_id,
                            "created_at": _to_utc_ts(inference_run_at),
                            "invoice_id": invoice_id,
                            "model_version_id": model_version_id,
                        }
                        for k, v in raw_snapshot.items():
                            if k in features.MODEL_FEATURE_NAMES:
                                fl[k] = v
                        fl["max_prob"] = max_prob
                        fl["optimal_retry_at_utc"] = optimal_retry_at_utc
                        feature_log_rows.append(fl)
                except Exception as e:
                    version_for_row = "fallback_24h"
                    if len(schedule_rows) == 0:
                        print(f"First fallback (run_inference_for_invoice): {e}", file=sys.stderr)

        schedule_rows.append({
            "invoice_id": invoice_id,
            "optimal_retry_at_utc": optimal_retry_at_utc,
            "attempt_number": attempt_number,
            "max_prob": max_prob,
            "inference_run_id": inference_run_id,
            "created_at": _to_utc_ts(inference_run_at),
            "status": "PENDING",
            "model_version_id": version_for_row,
        })

    write_schedule_to_bq(schedule_rows)
    if feature_log_rows:
        write_feature_log_to_bq(feature_log_rows)
    print(f"Wrote {len(schedule_rows)} rows to production_dunning_schedule; feature_log={len(feature_log_rows)} rows.")
    print(pd.DataFrame(schedule_rows).head())

if __name__ == "__main__":
    run()
