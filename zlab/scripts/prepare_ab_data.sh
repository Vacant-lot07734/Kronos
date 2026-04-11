#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZON="${1:-${HORIZON:-1}}"

if [[ -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-${ZLAB_KRONOS_PYTHON:-$HOME/miniconda3/envs/kronos/bin/python}}"
export KRONOS_LOCAL_CSV_DIR="${KRONOS_LOCAL_CSV_DIR:-${ZLAB_DATA_ROOT:-$ROOT/zlab/data/daily}}"
export KRONOS_LOOKBACK_WINDOW="${KRONOS_LOOKBACK_WINDOW:-20}"
export KRONOS_PREDICT_WINDOW="$HORIZON"
export KRONOS_DATASET_BEGIN_TIME="${KRONOS_DATASET_BEGIN_TIME:-2021-01-01}"
export KRONOS_DATASET_END_TIME="${KRONOS_DATASET_END_TIME:-2025-12-31}"
export KRONOS_TRAIN_TIME_RANGE="${KRONOS_TRAIN_TIME_RANGE:-2021-01-01,2024-06-30}"
export KRONOS_VAL_TIME_RANGE="${KRONOS_VAL_TIME_RANGE:-2024-07-01,2024-12-31}"
export KRONOS_TEST_TIME_RANGE="${KRONOS_TEST_TIME_RANGE:-2025-01-01,2025-12-31}"
export KRONOS_BACKTEST_TIME_RANGE="${KRONOS_BACKTEST_TIME_RANGE:-2025-01-01,2025-12-31}"
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}"

mkdir -p "$KRONOS_DATASET_PATH"

echo "Preparing datasets for horizon H=${HORIZON}"
echo "CSV dir: $KRONOS_LOCAL_CSV_DIR"
echo "KRONOS_DATASET_PATH=$KRONOS_DATASET_PATH"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/csv_data_preprocess.py" \
  --csv-dir "$KRONOS_LOCAL_CSV_DIR" \
  --output-dir "$KRONOS_DATASET_PATH"
