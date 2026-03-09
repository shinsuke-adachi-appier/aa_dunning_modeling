"""
Fetch active dunning invoices from BigQuery (standalone; no txn_pipeline).
Uses BQ_SOURCE_TABLE. Applies column renames, timezone/localized_time processing, and prev_decline_code sanitization.
"""
from __future__ import annotations

import os

import pandas as pd
from google.cloud import bigquery

from .timezone_utils import add_timezone_features


def fetch_active_dunning() -> pd.DataFrame:
    """
    Query BigQuery for active dunning: invoice_success_attempt_no IS NULL,
    invoice_attempt_count < 12, last 5 days; latest row per linked_invoice_id.
    Renames Decline_code_norm -> prev_decline_code, card_status -> prev_card_status,
    first_attempt_at_calc -> first_attempt_at.
    """
    project = os.environ.get("BQ_PROJECT")
    table = os.environ.get("BQ_SOURCE_TABLE")
    location = os.environ.get("BQ_LOCATION", "europe-west1")
    if not project or not table:
        raise RuntimeError("Set BQ_PROJECT and BQ_SOURCE_TABLE (e.g. billing_dm.MISc_vw_txn_enriched_subID_fallback).")

    if "." not in table:
        raise ValueError("BQ_SOURCE_TABLE must be dataset.table or project.dataset.table")
    if table.count(".") == 1:
        full_table = f"`{project}.{table}`"
    else:
        full_table = f"`{table}`"

    query = f"""
        WITH LatestState AS (
            SELECT *,
                ROW_NUMBER() OVER(PARTITION BY linked_invoice_id ORDER BY updated_at DESC) AS latest_row,
                MIN(updated_at) OVER(PARTITION BY linked_invoice_id) AS first_attempt_at_calc
            FROM {full_table}
            WHERE invoice_success_attempt_no IS NULL
              AND invoice_attempt_count < 12
              AND updated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 DAY)
              AND Decline_type_for_retry = 'Soft decline'
        )
        SELECT * EXCEPT(latest_row)
        FROM LatestState
        WHERE latest_row = 1
        ORDER BY updated_at ASC
    """
    client = bigquery.Client(project=project, location=location)
    df = client.query(query).to_dataframe()

    rename_map = {}
    if "Decline_code_norm" in df.columns:
        rename_map["Decline_code_norm"] = "prev_decline_code"
    if "card_status" in df.columns:
        rename_map["card_status"] = "prev_card_status"
    if "first_attempt_at" not in df.columns and "first_attempt_at_calc" in df.columns:
        rename_map["first_attempt_at_calc"] = "first_attempt_at"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Align with txn_pipeline: timezone and localized timestamps from country (and optional zip)
    df = add_timezone_features(df)

    # Sanitize prev_decline_code for model (same as training: fillna UNKNOWN)
    if "prev_decline_code" in df.columns:
        df["prev_decline_code"] = df["prev_decline_code"].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    if "billing_country" in df.columns:
        df["billing_country"] = df["billing_country"].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")

    return df
