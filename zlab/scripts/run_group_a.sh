#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZON="${1:-${HORIZON:-1}}"

if [[ -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/kronos/bin/python}"
DEVICE="${DEVICE:-auto}"
SAMPLE_COUNT="${KRONOS_INFERENCE_SAMPLE_COUNT:-5}"
TOPK="${KRONOS_EVAL_TOPK:-10}"
RUN_TAG="h${HORIZON}_sc${SAMPLE_COUNT}_topk${TOPK}_t${KRONOS_INFERENCE_T:-0.6}_tp${KRONOS_INFERENCE_TOP_P:-0.9}"
export KRONOS_PREDICT_WINDOW="$HORIZON"
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}"
RESULT_SAVE_PATH="${KRONOS_EVAL_RESULT_PATH_BASE:-$ROOT/zlab/results/evaluations}/h${HORIZON}"
RESULT_NAME_BASE="${RESULT_NAME:-group_a}"
if [[ "$RESULT_NAME_BASE" == *"${RUN_TAG}" ]]; then
  RESULT_NAME="$RESULT_NAME_BASE"
else
  RESULT_NAME="${RESULT_NAME_BASE}_${RUN_TAG}"
fi
RUN_DIR="$RESULT_SAVE_PATH/$RESULT_NAME"
LOG_PATH="$RUN_DIR/inference.log"

mkdir -p "$RUN_DIR"

{
  echo "Running Group A (zero-shot), H=${HORIZON}"
  echo "Dataset path: $KRONOS_DATASET_PATH"
  echo "Result path: $RUN_DIR"
  echo "Requested device: $DEVICE"
  echo "Inference log: $LOG_PATH"

  PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/evaluate_ab.py" \
    --device "$DEVICE" \
    --tokenizer-path "${KRONOS_PRETRAINED_TOKENIZER_PATH}" \
    --model-path "${KRONOS_PRETRAINED_PREDICTOR_PATH}" \
    --data-path "$KRONOS_DATASET_PATH" \
    --result-save-path "$RESULT_SAVE_PATH" \
    --result-name "$RESULT_NAME" \
    --pred-len "$HORIZON" \
    --sample-count "$SAMPLE_COUNT" \
    --batch-size "${KRONOS_EVAL_BATCH_SIZE:-128}" \
    --topk "$TOPK"
} 2>&1 | tee "$LOG_PATH"
