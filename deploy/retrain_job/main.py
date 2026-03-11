"""
Retrain job for dunning recovery model. Intended for Airflow/cron: run from repo root
to pull latest data, retrain, upload the calibrated model to GCS (under the model filename),
and export the training log to BigQuery.

Dates are set relative to today by default:
  VAL_END = yesterday, HOLDOUT_START = today
  VAL_START = today - 10 days, CAL_END = today - 11 days, CAL_START = today - 25 days, TRAIN_END = today - 26 days

Usage (from aa_dunning_modeling repo root):
  python deploy/retrain_job/main.py

Env:
  REPO_ROOT              - Path to repo root (default: cwd).
  FORCE_QUERY            - Set to 1 for fresh BQ data (default: 1).
  GCS_RETRAIN_BASE_URI   - Base GCS path; model is uploaded as {base_uri}/{filename} (e.g. gs://bucket/dunning/models/).
  SUFFIX                 - Model filename suffix (default: YYYYMMDD).
  TRAIN_END, CAL_* , VAL_* , HOLDOUT_START - Override date windows (YYYY-MM-DD).
  BQ_PROJECT, BQ_DATASET - Required for training log export. Same as inference job.
  TRAINING_LOG_TABLE     - Table name for training log (default: dunning_training_log).
  CALIBRATION_TEMPERATURE - Pass-through to training script.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


# Default window lengths (days back from today)
VAL_DAYS = 10
CAL_DAYS = 14
TRAIN_END_OFFSET_DAYS = CAL_DAYS + VAL_DAYS + 1  # day before cal start


def _repo_root() -> Path:
    """Resolve repo root: parent of deploy/ when this file is deploy/retrain_job/main.py."""
    this_file = Path(__file__).resolve()
    deploy_dir = this_file.parent.parent
    if (deploy_dir / "lib").exists() and (deploy_dir / "inference_job").exists():
        return deploy_dir.parent
    return Path(os.environ.get("REPO_ROOT", os.getcwd()))


def _default_suffix() -> str:
    return date.today().strftime("%Y%m%d")


def _dates_relative_to_today() -> dict[str, str]:
    """Set all date windows relative to today. Override any via env."""
    today = date.today()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    val_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    val_start = (today - timedelta(days=VAL_DAYS)).strftime("%Y-%m-%d")
    cal_end = (today - timedelta(days=VAL_DAYS + 1)).strftime("%Y-%m-%d")
    cal_start = (today - timedelta(days=VAL_DAYS + 1 + CAL_DAYS)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=TRAIN_END_OFFSET_DAYS)).strftime("%Y-%m-%d")
    return {
        "GLOBAL_START": os.environ.get("GLOBAL_START", "2025-01-02"),
        "TRAIN_END": os.environ.get("TRAIN_END", train_end),
        "CAL_START": os.environ.get("CAL_START", cal_start),
        "CAL_END": os.environ.get("CAL_END", cal_end),
        "VAL_START": os.environ.get("VAL_START", val_start),
        "VAL_END": os.environ.get("VAL_END", val_end),
        "HOLDOUT_START": os.environ.get("HOLDOUT_START", today_str),
    }


def run_training(repo_root: Path) -> str:
    """Run training script with EXPORT_TRAINING_LOG=1 and dates relative to today; return SUFFIX."""
    script = repo_root / "scripts" / "train_dunning_v2_20260301.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Training script not found: {script}. Run this job from the aa_dunning_modeling repo root."
        )
    env = os.environ.copy()
    env["FORCE_QUERY"] = "1"
    env["EXPORT_TRAINING_LOG"] = "1"
    suffix = os.environ.get("SUFFIX", _default_suffix())
    env["SUFFIX"] = suffix
    for k, v in _dates_relative_to_today().items():
        env[k] = v
    cmd = [sys.executable, str(script)]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    print(f"  Dates: TRAIN_END={env['TRAIN_END']} CAL={env['CAL_START']}..{env['CAL_END']} VAL={env['VAL_START']}..{env['VAL_END']} HOLDOUT_START={env['HOLDOUT_START']}", file=sys.stderr)
    subprocess.run(cmd, env=env, cwd=str(repo_root), check=True)
    return suffix


def upload_to_gcs(local_path: Path, gs_uri: str) -> None:
    """Upload local file to GCS."""
    if not gs_uri.startswith("gs://"):
        print(f"Skip upload (not a gs:// URI): {gs_uri}", file=sys.stderr)
        return
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed; skip upload.", file=sys.stderr)
        return
    from urllib.parse import urlparse
    parsed = urlparse(gs_uri)
    bucket_name, blob_path = parsed.netloc, parsed.path.lstrip("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path), content_type="application/octet-stream")
    print(f"Uploaded to {gs_uri}", file=sys.stderr)


def write_training_log_to_bq(repo_root: Path, suffix: str, model_gcs_uri: str | None) -> None:
    """Read training_run_{suffix}.json and append one row to BigQuery training log table."""
    log_path = repo_root / "models" / f"training_run_{suffix}.json"
    if not log_path.exists():
        print(f"Training log not found: {log_path}; skip BQ export.", file=sys.stderr)
        return
    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    table_name = os.environ.get("TRAINING_LOG_TABLE", "dunning_training_log")
    if not project or not dataset:
        print("BQ_PROJECT and BQ_DATASET required for training log export.", file=sys.stderr)
        return
    with open(log_path) as f:
        data = json.load(f)
    try:
        from google.cloud import bigquery
    except ImportError:
        print("google-cloud-bigquery not installed; skip training log export.", file=sys.stderr)
        return
    table_id = f"{project}.{dataset}.{table_name}"
    row = {
        "run_at": data["run_at"],
        "suffix": data["suffix"],
        "global_start": data["global_start"],
        "train_end": data["train_end"],
        "cal_start": data["cal_start"],
        "cal_end": data["cal_end"],
        "val_start": data["val_start"],
        "val_end": data["val_end"],
        "holdout_start": data["holdout_start"],
        "n_train": data["n_train"],
        "n_cal": data["n_cal"],
        "n_val": data["n_val"],
        "auc_val": data["auc_val"],
        "pr_auc_val": data["pr_auc_val"],
        "brier_val": data["brier_val"],
        "ece_val": data["ece_val"],
        "mce_val": data["mce_val"],
        "calibration_temperature": data.get("calibration_temperature"),
        "model_gcs_uri": model_gcs_uri,
    }
    client = bigquery.Client(project=project)
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"BigQuery insert failed: {errors}", file=sys.stderr)
        raise RuntimeError(f"Training log insert failed: {errors}")
    print(f"Training log written to {table_id}", file=sys.stderr)


def main() -> None:
    repo_root = _repo_root()
    suffix = run_training(repo_root)

    filename = f"catboost_dunning_calibrated_{suffix}.joblib"
    calibrated_path = repo_root / "models" / filename
    if not calibrated_path.exists():
        print(f"Calibrated model not found: {calibrated_path}", file=sys.stderr)
        sys.exit(1)

    base_uri = (os.environ.get("GCS_RETRAIN_BASE_URI") or os.environ.get("GCS_MODEL_URI") or "").strip()
    model_gcs_uri = None
    if base_uri:
        # Upload under the file name: {base_uri}/{filename}
        if base_uri.startswith("gs://"):
            base_uri = base_uri.rstrip("/")
            if base_uri.endswith(".joblib"):
                # Full object path given; use as-is
                model_gcs_uri = base_uri
            else:
                model_gcs_uri = f"{base_uri}/{filename}"
            upload_to_gcs(calibrated_path, model_gcs_uri)
        else:
            print("GCS_RETRAIN_BASE_URI or GCS_MODEL_URI must be a gs:// URI.", file=sys.stderr)

    write_training_log_to_bq(repo_root, suffix, model_gcs_uri)


if __name__ == "__main__":
    main()
