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
EVAL_ONLY="${KRONOS_EVAL_ONLY:-false}"
BATCH_SIZE="${KRONOS_BATCH_SIZE:-32}"
EPOCHS="${KRONOS_EPOCHS:-10}"
PRED_LR="${KRONOS_PREDICTOR_LR:-5e-5}"
TRAIN_LAST_RATIO="${KRONOS_PREDICTOR_TRAIN_LAST_RATIO:-0.333333}"
SAMPLE_COUNT="${KRONOS_INFERENCE_SAMPLE_COUNT:-5}"
TOPK="${KRONOS_EVAL_TOPK:-10}"

case "${EVAL_ONLY,,}" in
  true|false)
    ;;
  *)
    echo "KRONOS_EVAL_ONLY must be true or false, got: $EVAL_ONLY" >&2
    exit 1
    ;;
esac

export KRONOS_PREDICT_WINDOW="$HORIZON"
export KRONOS_DATASET_PATH="${KRONOS_DATASET_PATH_BASE:-$ROOT/zlab/results/processed_datasets}/h${HORIZON}"
export KRONOS_SAVE_PATH="${KRONOS_SAVE_PATH_BASE:-$ROOT/zlab/results/models}/h${HORIZON}"
export KRONOS_FUTURE_ONLY_LOSS="${KRONOS_FUTURE_ONLY_LOSS:-true}"
RESULT_SAVE_PATH="${KRONOS_EVAL_RESULT_PATH_BASE:-$ROOT/zlab/results/evaluations}/h${HORIZON}"
SAVE_FOLDER_BASE="${KRONOS_PREDICTOR_SAVE_FOLDER_NAME:-group_b_predictor}"
RESULT_NAME_BASE="${RESULT_NAME:-group_b}"
MODEL_SOURCE_HORIZON="${KRONOS_EVAL_MODEL_HORIZON:-$HORIZON}"
if [[ "${EVAL_ONLY,,}" == "true" && -z "${KRONOS_EVAL_MODEL_HORIZON:-}" && "$HORIZON" != "1" ]]; then
  MODEL_SOURCE_HORIZON="1"
fi
MODEL_RUN_TAG="h${MODEL_SOURCE_HORIZON}_lr${PRED_LR}_bs${BATCH_SIZE}_ep${EPOCHS}_tlr${TRAIN_LAST_RATIO}"
EVAL_RUN_TAG="predh${HORIZON}_fromh${MODEL_SOURCE_HORIZON}_lr${PRED_LR}_bs${BATCH_SIZE}_ep${EPOCHS}_tlr${TRAIN_LAST_RATIO}_sc${SAMPLE_COUNT}"
if [[ "$SAVE_FOLDER_BASE" == *"${MODEL_RUN_TAG}" ]]; then
  SAVE_FOLDER="$SAVE_FOLDER_BASE"
else
  SAVE_FOLDER="${SAVE_FOLDER_BASE}_${MODEL_RUN_TAG}"
fi
if [[ "$RESULT_NAME_BASE" == *"${EVAL_RUN_TAG}" ]]; then
  RESULT_BASE="$RESULT_NAME_BASE"
else
  RESULT_BASE="${RESULT_NAME_BASE}_${EVAL_RUN_TAG}"
fi
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="$SAVE_FOLDER"
MODEL_RUN_DIR="${KRONOS_SAVE_PATH}/${SAVE_FOLDER}"
TRAIN_LOG_PATH="${MODEL_RUN_DIR}/train.log"
TENSORBOARD_LOG_DIR="${MODEL_RUN_DIR}/${KRONOS_TENSORBOARD_SUBDIR:-tensorboard}"
MODEL_SOURCE_ROOT="${KRONOS_SAVE_PATH_BASE:-$ROOT/zlab/results/models}/h${MODEL_SOURCE_HORIZON}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_ROOT}/${SAVE_FOLDER}"
LOSS_MODEL_PATH="${MODEL_SOURCE_DIR}/checkpoints/best_model_by_loss"
RANKIC_MODEL_PATH="${MODEL_SOURCE_DIR}/checkpoints/best_model_by_rankic"
LEGACY_MODEL_PATH="${MODEL_SOURCE_DIR}/checkpoints/best_model"
if [[ ! -d "$LOSS_MODEL_PATH" && -d "$LEGACY_MODEL_PATH" ]]; then
  LOSS_MODEL_PATH="$LEGACY_MODEL_PATH"
fi
EVAL_ROOT_DIR="${RESULT_SAVE_PATH}/${RESULT_BASE}"
INFER_LOG_PATH="${EVAL_ROOT_DIR}/inference.log"

mkdir -p "$KRONOS_SAVE_PATH" "$RESULT_SAVE_PATH" "$MODEL_RUN_DIR" "$EVAL_ROOT_DIR"

if [[ "${EVAL_ONLY,,}" != "true" ]]; then
  {
    echo "Running Group B training, H=${HORIZON}"
    echo "Eval only: $EVAL_ONLY"
    echo "Dataset path: $KRONOS_DATASET_PATH"
    echo "Model save root: $KRONOS_SAVE_PATH"
    echo "Model run dir: $MODEL_RUN_DIR"
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
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "${KRONOS_NUM_WORKERS:-2}" \
      --predictor-learning-rate "$PRED_LR" \
      --save-folder-name "$SAVE_FOLDER" \
      --disable-comet
  } 2>&1 | tee "$TRAIN_LOG_PATH"
else
  if [[ ! -d "$LOSS_MODEL_PATH" && ! -d "$RANKIC_MODEL_PATH" ]]; then
    echo "KRONOS_EVAL_ONLY=true, but no compatible checkpoint exists under: $MODEL_SOURCE_DIR/checkpoints" >&2
    echo "Set KRONOS_EVAL_MODEL_HORIZON or KRONOS_PREDICTOR_SAVE_FOLDER_NAME if you want to reuse a different trained model." >&2
    exit 1
  fi
fi

run_eval() {
  local selection="$1"
  local model_path="$2"
  local result_name="${RESULT_BASE}__ckpt-${selection}"
  local result_path="${RESULT_SAVE_PATH}/${result_name}"

  {
    echo "Running Group B evaluation, H=${HORIZON}, checkpoint=${selection}"
    echo "Eval only: $EVAL_ONLY"
    echo "Model source horizon: $MODEL_SOURCE_HORIZON"
    echo "Model path: $model_path"
    echo "Result path: $result_path"
    echo "Inference log: $INFER_LOG_PATH"

    PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/finetune/evaluate_ab.py" \
      --device "$DEVICE" \
      --tokenizer-path "${KRONOS_PRETRAINED_TOKENIZER_PATH}" \
      --model-path "$model_path" \
      --data-path "$KRONOS_DATASET_PATH" \
      --result-save-path "$RESULT_SAVE_PATH" \
      --result-name "$result_name" \
      --pred-len "$HORIZON" \
      --sample-count "$SAMPLE_COUNT" \
      --batch-size "${KRONOS_EVAL_BATCH_SIZE:-128}" \
      --topk "$TOPK"
  } 2>&1 | tee -a "$INFER_LOG_PATH"
}

: > "$INFER_LOG_PATH"
if [[ -d "$LOSS_MODEL_PATH" ]]; then
  run_eval "val-loss" "$LOSS_MODEL_PATH"
else
  echo "Skip val-loss checkpoint evaluation: $LOSS_MODEL_PATH not found" | tee -a "$INFER_LOG_PATH"
fi

if [[ -d "$RANKIC_MODEL_PATH" ]]; then
  run_eval "val-rankic" "$RANKIC_MODEL_PATH"
else
  echo "Skip val-rankic checkpoint evaluation: $RANKIC_MODEL_PATH not found" | tee -a "$INFER_LOG_PATH"
fi
