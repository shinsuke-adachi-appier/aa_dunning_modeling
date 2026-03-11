-- Production dunning schedule: one row per invoice with optimal_retry_at_utc.
-- Partition by date for efficient current-hour queries.
-- Replace PROJECT_ID and DATASET with your GCP project and dataset.

CREATE TABLE IF NOT EXISTS `aa-datamart.dunning_modeling.production_dunning_schedule` (
  invoice_id           STRING      NOT NULL,
  optimal_retry_at_utc TIMESTAMP   NOT NULL,
  attempt_number       INT64       NOT NULL,
  max_prob             FLOAT64,
  inference_run_id     STRING,
  created_at           TIMESTAMP   NOT NULL,
  status               STRING,     -- PENDING | TRIGGERED | CANCELLED_PAID
  model_version_id     STRING      -- e.g. production | fallback_24h
)
PARTITION BY DATE(optimal_retry_at_utc)
OPTIONS(
  description = "Production dunning retry schedule from inference job"
);
