"""
One-off: insert a few test rows into production_dunning_schedule for the current hour (UTC)
so the trigger job has something to process when testing locally.

Usage (from deploy/):
  python scripts/insert_test_schedule_rows.py

Uses deploy/.env for BQ_PROJECT, BQ_DATASET, SCHEDULE_TABLE.
Optional: set NUM_ROWS=5 (default 3) to insert more rows.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# Load deploy/.env
_APP_ROOT = Path(__file__).resolve().parent.parent
_env = _APP_ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"").strip()
            if k and k not in os.environ:
                os.environ[k] = v

project = os.environ.get("BQ_PROJECT")
dataset = os.environ.get("BQ_DATASET")
table = os.environ.get("SCHEDULE_TABLE", "production_dunning_schedule")
if not project or not dataset:
    print("BQ_PROJECT and BQ_DATASET must be set (e.g. in deploy/.env).", file=sys.stderr)
    sys.exit(1)

num_rows = int(os.environ.get("NUM_ROWS", "3"))
now_utc = pd.Timestamp.now("UTC")
hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
inference_run_id = f"test-insert-{uuid.uuid4().hex[:8]}"

rows = [
    {
        "invoice_id": f"TEST_SCHED_{i}",
        "optimal_retry_at_utc": hour_start,
        "attempt_number": 1,
        "max_prob": 0.05 + i * 0.01,
        "inference_run_id": inference_run_id,
        "created_at": now_utc,
        "status": "PENDING",
        "model_version_id": "test",
    }
    for i in range(1, num_rows + 1)
]

table_id = f"{project}.{dataset}.{table}"
client = bigquery.Client(project=project)
df = pd.DataFrame(rows)
job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
print(f"Inserted {len(rows)} test rows into {table_id} for hour {hour_start} UTC.")
print("Run the trigger job now; it will pick these up (invoice_id like TEST_SCHED_*).")
