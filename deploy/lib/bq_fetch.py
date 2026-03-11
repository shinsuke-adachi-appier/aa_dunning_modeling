"""
Fetch active dunning invoices from BigQuery (standalone; no txn_pipeline).
Uses the enriched subscription+transaction pipeline query. Applies column renames,
timezone/localized_time processing, and prev_decline_code sanitization.
"""
from __future__ import annotations

import os

import pandas as pd
from google.cloud import bigquery

from .timezone_utils import add_timezone_features


# Full pipeline: sub_flat_best -> eff_txn -> attempt stats/rank -> base_raw -> mapped -> final_enriched -> LatestState.
# first_attempt_at_calc is added so inference can compute cumulative_delay_hours.
ACTIVE_DUNNING_QUERY = """
WITH sub_flat_best AS (
  SELECT * EXCEPT (rn)
  FROM (
    SELECT
      sf.*,
      ROW_NUMBER() OVER (
        PARTITION BY sf.customer_id
        ORDER BY
          IF(LOWER(sf.status) = 'active', 1, 0) DESC,
          sf.create_ts DESC NULLS LAST,
          sf.create_date DESC NULLS LAST,
          sf.current_term_end DESC NULLS LAST,
          sf.subscription_id DESC
      ) AS rn
    FROM `aa-datamart.billing_dm.v_subscription_customer_flat` sf
    WHERE sf.customer_id IS NOT NULL
  )
  WHERE rn = 1
),

eff_txn AS (
  SELECT
    t.*,
    COALESCE(sf.subscription_id, t.subscription_id) AS effective_subscription_id,
    JSON_EXTRACT_SCALAR(t.raw_json, '$.linked_invoices[0].invoice_id')      AS linked_invoice_id,
    JSON_EXTRACT_SCALAR(t.raw_json, '$.linked_invoices[0].invoice_status') AS linked_invoice_status,
    TIMESTAMP_SECONDS(SAFE_CAST(JSON_EXTRACT_SCALAR(t.raw_json, '$.linked_invoices[0].applied_at') AS INT64)) AS linked_invoice_applied_at,
    TIMESTAMP_SECONDS(SAFE_CAST(JSON_EXTRACT_SCALAR(t.raw_json, '$.linked_invoices[0].invoice_date') AS INT64)) AS linked_invoice_date
  FROM `aa-datamart.billing_stg.stg_cb_transactions_enriched_tbl` t
  LEFT JOIN sub_flat_best sf ON sf.customer_id = t.customer_id
  WHERE DATE(t.date) > DATE '2025-01-01'
),

invoice_attempt_stats AS (
  SELECT
    linked_invoice_id,
    COUNT(*) AS invoice_attempt_count
  FROM eff_txn
  WHERE linked_invoice_id IS NOT NULL
  GROUP BY linked_invoice_id
),

invoice_attempt_rank_base AS (
  SELECT
    id AS txn_id,
    linked_invoice_id,
    status,
    ROW_NUMBER() OVER (
      PARTITION BY linked_invoice_id
      ORDER BY linked_invoice_applied_at ASC NULLS LAST, updated_at ASC NULLS LAST, id ASC
    ) AS invoice_attempt_no
  FROM eff_txn
  WHERE linked_invoice_id IS NOT NULL
),

invoice_attempt_rank AS (
  SELECT
    txn_id,
    linked_invoice_id,
    invoice_attempt_no,
    MIN(IF(LOWER(status) = 'success', invoice_attempt_no, NULL))
      OVER (PARTITION BY linked_invoice_id) AS invoice_success_attempt_no
  FROM invoice_attempt_rank_base
),

base_raw AS (
  SELECT
    Txn.updated_at,
    Txn.id AS txn_id,
    Txn.amount,
    Txn.gateway,
    Txn.linked_invoice_id,
    sf.card_status,
    sf.billing_region AS billing_country,
    CASE
      WHEN REGEXP_CONTAINS(LOWER(REGEXP_EXTRACT(sf.email, r'@(.+)$')), r'gmail')   THEN 'Gmail'
      WHEN REGEXP_CONTAINS(LOWER(REGEXP_EXTRACT(sf.email, r'@(.+)$')), r'hotmail') THEN 'Hotmail'
      WHEN REGEXP_CONTAINS(LOWER(REGEXP_EXTRACT(sf.email, r'@(.+)$')), r'icloud')  THEN 'iCloud'
      WHEN REGEXP_CONTAINS(LOWER(REGEXP_EXTRACT(sf.email, r'@(.+)$')), r'outlook') THEN 'Outlook'
      WHEN REGEXP_CONTAINS(LOWER(REGEXP_EXTRACT(sf.email, r'@(.+)$')), r'yahoo')   THEN 'Yahoo'
      ELSE 'Work'
    END AS Domain_category,
    CASE
      WHEN LOWER(Txn.gateway_name) LIKE '%adyen%'
       AND LOWER(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.funding_type')) NOT IN ('credit','debit')
        THEN 'prepaid'
      ELSE JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.funding_type')
    END AS funding_type_norm,
    CASE
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('visa') THEN 'visa'
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('mc','mastercard') THEN 'mc'
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('amex','american_express') THEN 'amex'
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('discover') THEN 'discover'
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('jcb') THEN 'jcb'
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('diners','diners_club') THEN 'diners'
      WHEN LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand'))) IN ('unionpay','cup') THEN 'unionpay'
      ELSE LOWER(TRIM(JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.brand')))
    END AS card_brand,
    JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.billing_zip') AS billing_zip,
    CASE
      WHEN JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.billing_zip') IS NOT NULL
       AND JSON_EXTRACT_SCALAR(Txn.payment_method_details, '$.card.billing_zip') != ''
        THEN 'Zip Filled'
      ELSE 'Zip not filled'
    END AS fill_zip_code,
    LOWER(CASE WHEN LOWER(Txn.gateway)='adyen' THEN JSON_EXTRACT_SCALAR(Txn.error_detail_json,'$.decline_message') ELSE Txn.decline_code END) AS decline_code_norm_raw
  FROM eff_txn Txn
  LEFT JOIN sub_flat_best sf ON sf.customer_id = Txn.customer_id
),

mapped AS (
  SELECT
    b.*,
    CASE b.decline_code_norm_raw
      WHEN 'insufficient_funds'        THEN 'insufficient_funds'
      WHEN 'not enough balance'        THEN 'insufficient_funds'
      WHEN 'do_not_honor'              THEN 'do_not_honor'
      WHEN 'refused'                   THEN 'do_not_honor'
      WHEN 'generic_decline'           THEN 'generic_decline'
      WHEN 'declined non generic'      THEN 'generic_decline'
      WHEN 'issuer_not_available'      THEN 'issuer_unavailable'
      WHEN 'issuer unavailable'        THEN 'issuer_unavailable'
      WHEN 'try_again_later'           THEN 'try_again_later'
      WHEN 'acquirer error'            THEN 'acquirer_error'
      WHEN 'transaction_not_allowed'   THEN 'transaction_not_allowed'
      WHEN 'transaction not permitted' THEN 'transaction_not_allowed'
      WHEN 'card_not_supported'        THEN 'card_not_supported'
      WHEN 'not supported'             THEN 'not_supported'
      WHEN 'card_velocity_exceeded'          THEN 'velocity_exceeded'
      WHEN 'withdrawal_count_limit_exceeded' THEN 'velocity_exceeded'
      WHEN 'withdrawal count exceeded'       THEN 'velocity_exceeded'
      WHEN 'withdrawal amount exceeded'      THEN 'velocity_exceeded'
      WHEN 'fraudulent'                THEN 'fraudulent'
      WHEN 'fraud'                     THEN 'fraud'
      WHEN 'issuer suspected fraud'    THEN 'issuer_suspected_fraud'
      WHEN 'incorrect_number'          THEN 'invalid_card'
      WHEN 'invalid_account'           THEN 'invalid_card'
      WHEN 'invalid card number'       THEN 'invalid_card'
      WHEN 'lost_card'                 THEN 'lost_stolen_card'
      WHEN 'stolen_card'               THEN 'lost_stolen_card'
      WHEN 'pickup_card'               THEN 'lost_stolen_card'
      WHEN 'restricted_card'           THEN 'restricted_card'
      WHEN 'restricted card'           THEN 'restricted_card'
      WHEN 'blocked card'              THEN 'blocked_card'
      WHEN 'expired_card'              THEN 'expired_card'
      WHEN 'expired card'              THEN 'expired_card'
      WHEN 'incorrect_cvc'             THEN 'invalid_cvc'
      WHEN 'invalid_cvc'               THEN 'invalid_cvc'
      WHEN 'cvc declined'              THEN 'invalid_cvc'
      WHEN 'invalid_pin'               THEN 'invalid_pin'
      WHEN 'invalid pin'               THEN 'invalid_pin'
      WHEN 'pin tries exceeded'        THEN 'invalid_pin'
      WHEN 'invalid_amount'            THEN 'invalid_amount'
      WHEN 'invalid amount'            THEN 'invalid_amount'
      WHEN 'revocation_of_authorization'      THEN 'revocation'
      WHEN 'revocation_of_all_authorizations' THEN 'revocation'
      WHEN 'revocation of auth'               THEN 'revocation'
      WHEN 'authentication_required'   THEN 'authentication_required'
      WHEN 'authentication required'   THEN 'authentication_required'
      WHEN 'processing_error'          THEN 'processing_error'
      WHEN 'reenter_transaction'       THEN 'processing_error'
      WHEN 'call_issuer'               THEN 'call_issuer'
      WHEN 'referral'                  THEN 'call_issuer'
      WHEN 'stop_payment_order'              THEN 'stop_payment'
      WHEN 'invalid_expiry_year'             THEN 'invalid_expiry'
      WHEN 'payment_method_not_available'    THEN 'payment_method_unavailable'
      ELSE COALESCE(b.decline_code_norm_raw, 'unknown')
    END AS Decline_code_norm
  FROM base_raw b
),

final_enriched AS (
  SELECT
    m.*,
    iar.invoice_success_attempt_no,
    iar.invoice_attempt_no,
    ias.invoice_attempt_count,
    CASE
      WHEN m.Decline_code_norm IN (
        'transaction_not_allowed', 'card_not_supported', 'not_supported', 'fraudulent', 'fraud',
        'issuer_suspected_fraud', 'invalid_card', 'lost_stolen_card', 'restricted_card', 'blocked_card',
        'expired_card', 'invalid_cvc', 'invalid_pin', 'invalid_amount', 'revocation', 'stop_payment',
        'invalid_expiry', 'payment_method_unavailable'
      ) THEN 'Hard decline'
      ELSE 'Soft decline'
    END AS Decline_type_for_retry
  FROM mapped m
  JOIN invoice_attempt_rank iar ON iar.txn_id = m.txn_id
  JOIN invoice_attempt_stats ias ON ias.linked_invoice_id = m.linked_invoice_id
),

LatestState AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY linked_invoice_id ORDER BY updated_at DESC) AS latest_row,
    MIN(updated_at) OVER(PARTITION BY linked_invoice_id) AS first_attempt_at_calc
  FROM final_enriched
  WHERE invoice_success_attempt_no IS NULL
    AND invoice_attempt_count < 12
    AND updated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 DAY)
    AND Decline_type_for_retry = 'Soft decline'
)

SELECT
  linked_invoice_id,
  updated_at,
  first_attempt_at_calc,
  invoice_success_attempt_no,
  invoice_attempt_count,
  Decline_type_for_retry,
  Decline_code_norm,
  card_status,
  invoice_attempt_no,
  amount,
  billing_country,
  gateway,
  funding_type_norm,
  card_brand,
  Domain_category,
  fill_zip_code,
  billing_zip
FROM LatestState
WHERE latest_row = 1
ORDER BY updated_at ASC
"""


def fetch_active_dunning() -> pd.DataFrame:
    """
    Query BigQuery for active dunning using the subscription+transaction pipeline.
    Returns latest row per linked_invoice_id (soft-decline, < 12 attempts, last 5 days).
    Renames Decline_code_norm -> prev_decline_code, card_status -> prev_card_status,
    first_attempt_at_calc -> first_attempt_at. Applies timezone and sanitization.
    """
    project = os.environ.get("BQ_PROJECT")
    location = os.environ.get("BQ_LOCATION", "europe-west1")
    if not project:
        raise RuntimeError("Set BQ_PROJECT.")

    client = bigquery.Client(project=project, location=location)
    df = client.query(ACTIVE_DUNNING_QUERY).to_dataframe()

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
