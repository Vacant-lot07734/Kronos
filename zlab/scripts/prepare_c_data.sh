#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZON="${1:-${HORIZON:-1}}"
AUTO_SOURCE_ENV="${KRONOS_AUTO_SOURCE_ENV:-true}"

if [[ "${AUTO_SOURCE_ENV,,}" != "false" && -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/kronos/bin/python}"
DAILY_CSV_DIR="${KRONOS_LOCAL_CSV_DIR:-$ROOT/zlab/data/daily}"
HOURLY_CSV_DIR="${KRONOS_HOURLY_CSV_DIR:-$ROOT/zlab/data/hourly}"
HOURLY_WINDOW="${KRONOS_HOURLY_WINDOW:-25}"
OUTPUT_DIR="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}_c"

export KRONOS_PREDICT_WINDOW="$HORIZON"

echo "Preparing C-group data, H=${HORIZON}"
echo "  Daily CSV dir:  $DAILY_CSV_DIR"
echo "  Hourly CSV dir: $HOURLY_CSV_DIR"
echo "  Hourly window:  $HOURLY_WINDOW"
echo "  Output dir:     $OUTPUT_DIR"

export KRONOS_DATASET_PATH="$OUTPUT_DIR"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/csv_data_preprocess_c.py" \
  --daily-csv-dir "$DAILY_CSV_DIR" \
  --hourly-csv-dir "$HOURLY_CSV_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --hourly-window "$HOURLY_WINDOW"
