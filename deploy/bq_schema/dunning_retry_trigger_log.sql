-- Trigger audit log: one row per retry API call attempt.
-- Replace PROJECT_ID and DATASET with your GCP project and dataset.

CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET.dunning_retry_trigger_log` (
  invoice_id            STRING    NOT NULL,
  optimal_retry_at_utc  TIMESTAMP NOT NULL,
  triggered_at          TIMESTAMP NOT NULL,
  idempotency_key       STRING    NOT NULL,
  api_response_status   INT64,   -- HTTP status (200, 409, 429, etc.)
  model_version_id      STRING,
  error_message         STRING
)
OPTIONS(
  description = "Audit log of retry API calls from trigger job"
);
