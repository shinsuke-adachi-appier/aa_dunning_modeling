-- Trigger audit log: one row per retry API call attempt.
-- Aligned with Apps Script force_dunning log (D~I): status, message, txn_id, attempt_number, comment_status.
-- Replace PROJECT_ID and DATASET with your GCP project and dataset.
--
-- If the table already exists (old schema), add the five new columns in BigQuery:
--   ALTER TABLE `PROJECT_ID.DATASET.dunning_retry_trigger_log` ADD COLUMN status STRING;
--   ALTER TABLE `PROJECT_ID.DATASET.dunning_retry_trigger_log` ADD COLUMN message STRING;
--   (and similarly txn_id STRING, attempt_number INT64, comment_status STRING)

CREATE TABLE IF NOT EXISTS `aa-datamart.dunning_modeling.dunning_retry_trigger_log` (
  invoice_id            STRING    NOT NULL,
  optimal_retry_at_utc  TIMESTAMP NOT NULL,
  triggered_at          TIMESTAMP NOT NULL,
  idempotency_key       STRING    NOT NULL,
  api_response_status   INT64,    -- HTTP status (200, 409, 429, etc.)
  error_message         STRING,  -- Special sentinels: DRY_RUN, VELOCITY_CAP_7D; used for velocity cap filter
  -- Columns aligned with Apps Script sheet (status D, message E, txn_id G, attempt H, comment I)
  status                STRING,   -- Success | Failed | Error | DRY_RUN | VELOCITY_CAP_7D
  message               STRING,   -- Human-readable: error text or "Payment collection success."
  txn_id                STRING,   -- Chargebee transaction id from collect_payment response (when 200)
  attempt_number        INT64,    -- Attempt number at trigger time (from schedule)
  comment_status        STRING    -- Comment on transaction: Success | Failed (code) | Skipped | NULL
)
OPTIONS(
  description = "Audit log of retry API calls from trigger job; format aligned with Apps Script force_dunning"
);
