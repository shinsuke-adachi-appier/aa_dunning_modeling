#!/usr/bin/env bash
# Wrapper for shadow_monitoring_20260206.py — use from cron with correct cwd and env.
# Usage: /path/to/aa_dunning_modeling/run_shadow_cron.sh

set -e
AA_DUNNING_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$AA_DUNNING_ROOT"

# Load optional .env (paths, DUNNING_MODEL_VERSION, GOOGLE_APPLICATION_CREDENTIALS)
if [ -f "$AA_DUNNING_ROOT/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$AA_DUNNING_ROOT/.env"
  set +a
fi

export DUNNING_MODEL_PATH="${DUNNING_MODEL_PATH:-$AA_DUNNING_ROOT/models/catboost_dunning_calibrated_20260224.joblib}"
# Append to canonical shadow outputs unless overridden in .env.
export SHADOW_LOG_PATH="${SHADOW_LOG_PATH:-$AA_DUNNING_ROOT/artifacts/shadow_log.csv}"
export SHADOW_SLOT_LOG_PATH="${SHADOW_SLOT_LOG_PATH:-$AA_DUNNING_ROOT/artifacts/shadow_slot_log.csv}"
export DUNNING_MODEL_VERSION="${DUNNING_MODEL_VERSION:-20260224}"

LOG_DIR="$AA_DUNNING_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/shadow_$(date +%Y%m%d_%H%M%S).log"

if [ -d "$AA_DUNNING_ROOT/.venv" ]; then
  source "$AA_DUNNING_ROOT/.venv/bin/activate"
fi

python shadow_monitoring_20260206.py >> "$LOG" 2>&1
