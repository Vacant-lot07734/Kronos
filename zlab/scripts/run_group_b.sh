#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HORIZON="${1:-${HORIZON:-1}}"

if [[ -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/kronos/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$HOME/miniconda3/envs/kronos/bin/torchrun}"
DEVICE="${DEVICE:-cuda:0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RESULT_NAME="${RESULT_NAME:-group_b}"

export KRONOS_PREDICT_WINDOW="$HORIZON"
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}"
export KRONOS_SAVE_PATH="${KRONOS_SAVE_PATH_BASE:-$ROOT/zlab/results/models}/h${HORIZON}"
export KRONOS_FUTURE_ONLY_LOSS="${KRONOS_FUTURE_ONLY_LOSS:-true}"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="${KRONOS_PREDICTOR_SAVE_FOLDER_NAME:-group_b_predictor}"
RESULT_SAVE_PATH="${KRONOS_EVAL_RESULT_PATH_BASE:-$ROOT/zlab/results/evaluations}/h${HORIZON}"
MODEL_RUN_DIR="${KRONOS_SAVE_PATH}/${KRONOS_PREDICTOR_SAVE_FOLDER_NAME}"
EVAL_RUN_DIR="${RESULT_SAVE_PATH}/${RESULT_NAME}"
TRAIN_LOG_PATH="${MODEL_RUN_DIR}/train.log"
INFER_LOG_PATH="${EVAL_RUN_DIR}/inference.log"
TENSORBOARD_LOG_DIR="${MODEL_RUN_DIR}/${KRONOS_TENSORBOARD_SUBDIR:-tensorboard}"

mkdir -p "$KRONOS_SAVE_PATH" "$RESULT_SAVE_PATH" "$MODEL_RUN_DIR" "$EVAL_RUN_DIR"

{
  echo "Running Group B training, H=${HORIZON}"
  echo "Dataset path: $KRONOS_DATASET_PATH"
  echo "Model save root: $KRONOS_SAVE_PATH"
  echo "Requested device: $DEVICE"
  echo "Training log: $TRAIN_LOG_PATH"
  echo "TensorBoard log dir: $TENSORBOARD_LOG_DIR"

  PYTHONPATH="$ROOT" "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT/finetune/train_predictor.py" \
    --use-pretrained-tokenizer \
    --freeze-for-ab \
    --train-last-ratio "${KRONOS_PREDICTOR_TRAIN_LAST_RATIO:-0.333333}" \
    --freeze-embedding \
    --future-only-loss \
    --epochs "${KRONOS_EPOCHS:-10}" \
    --batch-size "${KRONOS_BATCH_SIZE:-32}" \
    --num-workers "${KRONOS_NUM_WORKERS:-2}" \
    --predictor-learning-rate "${KRONOS_PREDICTOR_LR:-5e-5}" \
    --save-folder-name "${KRONOS_PREDICTOR_SAVE_FOLDER_NAME}" \
    --disable-comet
} 2>&1 | tee "$TRAIN_LOG_PATH"

MODEL_PATH="${KRONOS_SAVE_PATH}/${KRONOS_PREDICTOR_SAVE_FOLDER_NAME}/checkpoints/best_model"

{
  echo "Running Group B evaluation, H=${HORIZON}"
  echo "Model path: $MODEL_PATH"
  echo "Result path: $EVAL_RUN_DIR"
  echo "Inference log: $INFER_LOG_PATH"

  PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/evaluate_ab.py" \
    --device "$DEVICE" \
    --tokenizer-path "${KRONOS_PRETRAINED_TOKENIZER_PATH}" \
    --model-path "$MODEL_PATH" \
    --data-path "$KRONOS_DATASET_PATH" \
    --result-save-path "$RESULT_SAVE_PATH" \
    --result-name "$RESULT_NAME" \
    --pred-len "$HORIZON" \
    --sample-count "${KRONOS_INFERENCE_SAMPLE_COUNT:-5}" \
    --batch-size "${KRONOS_EVAL_BATCH_SIZE:-128}" \
    --topk "${KRONOS_EVAL_TOPK:-10}"
} 2>&1 | tee "$INFER_LOG_PATH"
