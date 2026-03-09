-- Production dunning schedule: one row per invoice with optimal_retry_at_utc.
-- Partition by date for efficient current-hour queries.
-- Replace PROJECT_ID and DATASET with your GCP project and dataset.

CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET.production_dunning_schedule` (
  invoice_id           STRING      NOT NULL,
  optimal_retry_at_utc TIMESTAMP   NOT NULL,
  attempt_number       INT64       NOT NULL,
  model_version_id     STRING      NOT NULL,
  max_prob             FLOAT64,
  inference_run_id     STRING,
  created_at           TIMESTAMP   NOT NULL,
  status               STRING      -- PENDING | TRIGGERED | CANCELLED_PAID
)
PARTITION BY DATE(optimal_retry_at_utc)
OPTIONS(
  description = "Production dunning retry schedule from inference job"
);
