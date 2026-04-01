#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZON="${1:-${HORIZON:-1}}"

if [[ -f "$ROOT/zlab/ab_env.sh" ]]; then
  source "$ROOT/zlab/ab_env.sh"
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
RESULT_NAME="${RESULT_NAME:-group_c}"
SAVE_FOLDER="${KRONOS_PREDICTOR_SAVE_FOLDER_NAME:-group_c_predictor}"

export KRONOS_PREDICT_WINDOW="$HORIZON"
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}_c"
export KRONOS_SAVE_PATH="${KRONOS_SAVE_PATH_BASE:-$ROOT/zlab/results/models}/h${HORIZON}"
export KRONOS_FUTURE_ONLY_LOSS="${KRONOS_FUTURE_ONLY_LOSS:-true}"
export KRONOS_HOURLY_WINDOW="$HOURLY_WINDOW"

RESULT_SAVE_PATH="${KRONOS_EVAL_RESULT_PATH_BASE:-$ROOT/zlab/results/evaluations}/h${HORIZON}"
MODEL_RUN_DIR="${KRONOS_SAVE_PATH}/${SAVE_FOLDER}"
EVAL_RUN_DIR="${RESULT_SAVE_PATH}/${RESULT_NAME}"
TRAIN_LOG_PATH="${MODEL_RUN_DIR}/train.log"
INFER_LOG_PATH="${EVAL_RUN_DIR}/inference.log"
MODEL_PATH="${MODEL_RUN_DIR}/checkpoints/best_model"

mkdir -p "$MODEL_RUN_DIR" "$EVAL_RUN_DIR"

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
      --epochs "${KRONOS_EPOCHS:-15}" \
      --batch-size "${KRONOS_BATCH_SIZE:-32}" \
      --num-workers "${KRONOS_NUM_WORKERS:-2}" \
      --predictor-learning-rate "${KRONOS_PREDICTOR_LR:-5e-5}" \
      --hourly-learning-rate "$HOURLY_LR" \
      --hourly-encoder-layers "$HOURLY_ENCODER_LAYERS" \
      --hourly-window "$HOURLY_WINDOW" \
      --train-last-ratio "${KRONOS_PREDICTOR_TRAIN_LAST_RATIO:-0.333333}" \
      --save-folder-name "$SAVE_FOLDER" \
      --disable-comet
  } 2>&1 | tee "$TRAIN_LOG_PATH"
else
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "KRONOS_EVAL_ONLY=true but checkpoint not found: $MODEL_PATH" >&2
    exit 1
  fi
fi

# --- Evaluation ---
{
  echo "Running Group C evaluation, H=${HORIZON}"
  echo "Model path: $MODEL_PATH"
  echo "Result path: $EVAL_RUN_DIR"

  PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/evaluate_c.py" \
    --device "$DEVICE" \
    --tokenizer-path "${KRONOS_PRETRAINED_TOKENIZER_PATH}" \
    --model-path "$MODEL_PATH" \
    --data-path "$KRONOS_DATASET_PATH" \
    --result-save-path "$RESULT_SAVE_PATH" \
    --result-name "$RESULT_NAME" \
    --pred-len "$HORIZON" \
    --hourly-window "$HOURLY_WINDOW" \
    --sample-count "${KRONOS_INFERENCE_SAMPLE_COUNT:-5}" \
    --batch-size "${KRONOS_EVAL_BATCH_SIZE:-64}" \
    --topk "${KRONOS_EVAL_TOPK:-10}"
} 2>&1 | tee "$INFER_LOG_PATH"
