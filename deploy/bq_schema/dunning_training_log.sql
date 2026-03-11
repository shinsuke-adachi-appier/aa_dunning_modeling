-- Dunning model training log: one row per retrain run (from retrain_job).
-- Replace PROJECT_ID and DATASET with your GCP project and dataset.

CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET.dunning_training_log` (
  run_at                   TIMESTAMP   NOT NULL,  -- UTC
  suffix                   STRING      NOT NULL,  -- e.g. 20260315
  global_start             STRING      NOT NULL,
  train_end                STRING      NOT NULL,
  cal_start                STRING      NOT NULL,
  cal_end                  STRING      NOT NULL,
  val_start                STRING      NOT NULL,
  val_end                  STRING      NOT NULL,
  holdout_start            STRING      NOT NULL,
  n_train                  INT64       NOT NULL,
  n_cal                    INT64       NOT NULL,
  n_val                    INT64       NOT NULL,
  auc_val                  FLOAT64     NOT NULL,
  pr_auc_val               FLOAT64     NOT NULL,
  brier_val                FLOAT64     NOT NULL,
  ece_val                  FLOAT64     NOT NULL,
  mce_val                  FLOAT64     NOT NULL,
  calibration_temperature  FLOAT64,
  model_gcs_uri            STRING     -- gs://bucket/.../catboost_dunning_calibrated_{suffix}.joblib
)
OPTIONS(
  description = "Dunning model retrain run log from deploy/retrain_job"
);
