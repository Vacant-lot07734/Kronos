#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZON="${1:-${HORIZON:-1}}"
COMMON_HELPER="$ROOT/zlab/scripts/_group_runner_common.sh"
AUTO_SOURCE_ENV="${KRONOS_AUTO_SOURCE_ENV:-true}"

if [[ "${AUTO_SOURCE_ENV,,}" != "false" && -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
fi
if [[ -f "$COMMON_HELPER" ]]; then
  # shellcheck disable=SC1091
  source "$COMMON_HELPER"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/kronos/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$HOME/miniconda3/envs/kronos/bin/torchrun}"
DEVICE="${DEVICE:-cuda:0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EVAL_ONLY="${KRONOS_EVAL_ONLY:-false}"

# C-group specific settings
HOURLY_WINDOW="${KRONOS_HOURLY_WINDOW:-25}"
HOURLY_LR="${KRONOS_HOURLY_LR:-1e-4}"
HOURLY_ENCODER_LAYERS="${KRONOS_HOURLY_ENCODER_LAYERS:-2}"
BATCH_SIZE="${KRONOS_BATCH_SIZE:-32}"
EPOCHS="${KRONOS_EPOCHS:-15}"
PRED_LR="${KRONOS_PREDICTOR_LR:-5e-5}"
TRAIN_LAST_RATIO="${KRONOS_PREDICTOR_TRAIN_LAST_RATIO:-0.333333}"
SAMPLE_COUNT="${KRONOS_INFERENCE_SAMPLE_COUNT:-10}"
TOPK="${KRONOS_EVAL_TOPK:-10}"

PRED_LR_TAG="$(sanitize_num "$PRED_LR")"
HOURLY_LR_TAG="$(sanitize_num "$HOURLY_LR")"
validate_eval_only_flag "$EVAL_ONLY"

export KRONOS_PREDICT_WINDOW="$HORIZON"
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}_c"
export KRONOS_SAVE_PATH="${KRONOS_SAVE_PATH_BASE:-$ROOT/zlab/results/models}/h${HORIZON}"
export KRONOS_FUTURE_ONLY_LOSS="${KRONOS_FUTURE_ONLY_LOSS:-true}"
export KRONOS_HOURLY_WINDOW="$HOURLY_WINDOW"

RESULT_SAVE_PATH="${KRONOS_EVAL_RESULT_PATH_BASE:-$ROOT/zlab/results/evaluations}/h${HORIZON}"
DEFAULT_NAME="group_c_hlr${HOURLY_LR_TAG}_lr${PRED_LR_TAG}_e${EPOCHS}_bs${BATCH_SIZE}"
SAVE_FOLDER="${KRONOS_PREDICTOR_SAVE_FOLDER_NAME:-$DEFAULT_NAME}"
RESULT_BASE="${RESULT_NAME:-$DEFAULT_NAME}"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="$SAVE_FOLDER"
MODEL_RUN_DIR="${KRONOS_SAVE_PATH}/${SAVE_FOLDER}"
TRAIN_LOG_PATH="${MODEL_RUN_DIR}/train.log"
resolve_checkpoint_paths "$MODEL_RUN_DIR"
EVAL_ROOT_DIR="${RESULT_SAVE_PATH}/${RESULT_BASE}"
INFER_LOG_PATH="${EVAL_ROOT_DIR}/inference.log"

mkdir -p "$MODEL_RUN_DIR" "$EVAL_ROOT_DIR"

# --- Training ---
if [[ "${EVAL_ONLY,,}" != "true" ]]; then
  {
    echo "Running Group C training, H=${HORIZON}"
    echo "Dataset path: $KRONOS_DATASET_PATH"
    echo "Model save: $MODEL_RUN_DIR"
    echo "Device: $DEVICE"
    echo "Hourly window: $HOURLY_WINDOW, Hourly LR: $HOURLY_LR"
    echo "Hourly encoder layers: $HOURLY_ENCODER_LAYERS"

    PYTHONPATH="$ROOT" "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
      "$ROOT/finetune/train_predictor_c.py" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "${KRONOS_NUM_WORKERS:-2}" \
      --predictor-learning-rate "$PRED_LR" \
      --hourly-learning-rate "$HOURLY_LR" \
      --hourly-encoder-layers "$HOURLY_ENCODER_LAYERS" \
      --hourly-window "$HOURLY_WINDOW" \
      --train-last-ratio "$TRAIN_LAST_RATIO" \
      --save-folder-name "$SAVE_FOLDER" \
      --disable-comet
  } 2>&1 | tee "$TRAIN_LOG_PATH"
else
  require_any_checkpoint "$MODEL_RUN_DIR"
fi

: > "$INFER_LOG_PATH"
if [[ -d "$LOSS_MODEL_PATH" ]]; then
  run_checkpoint_eval "cross-entropy" "$LOSS_MODEL_PATH" "$INFER_LOG_PATH" \
    env PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/evaluate_c.py" \
    --device "$DEVICE" \
    --tokenizer-path "${KRONOS_PRETRAINED_TOKENIZER_PATH}" \
    --data-path "$KRONOS_DATASET_PATH" \
    --result-save-path "$EVAL_ROOT_DIR" \
    --result-name "cross-entropy" \
    --pred-len "$HORIZON" \
    --hourly-window "$HOURLY_WINDOW" \
    --sample-count "$SAMPLE_COUNT" \
    --batch-size "${KRONOS_EVAL_BATCH_SIZE:-64}" \
    --topk "$TOPK" \
    --splits test
else
  echo "Skip val-loss checkpoint evaluation: $LOSS_MODEL_PATH not found" | tee -a "$INFER_LOG_PATH"
fi

if [[ -d "$RANKIC_MODEL_PATH" ]]; then
  run_checkpoint_eval "rankIc" "$RANKIC_MODEL_PATH" "$INFER_LOG_PATH" \
    env PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/evaluate_c.py" \
    --device "$DEVICE" \
    --tokenizer-path "${KRONOS_PRETRAINED_TOKENIZER_PATH}" \
    --data-path "$KRONOS_DATASET_PATH" \
    --result-save-path "$EVAL_ROOT_DIR" \
    --result-name "rankIc" \
    --pred-len "$HORIZON" \
    --hourly-window "$HOURLY_WINDOW" \
    --sample-count "$SAMPLE_COUNT" \
    --batch-size "${KRONOS_EVAL_BATCH_SIZE:-64}" \
    --topk "$TOPK" \
    --splits test
else
  echo "Skip val-rankic checkpoint evaluation: $RANKIC_MODEL_PATH not found" | tee -a "$INFER_LOG_PATH"
fi
