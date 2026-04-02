#!/usr/bin/env bash

sanitize_num() {
  local value="${1//-/}"
  value="${value//+/}"
  value="${value//./}"
  echo "$value"
}

validate_eval_only_flag() {
  local value="${1,,}"
  case "$value" in
    true|false) ;;
    *)
      echo "KRONOS_EVAL_ONLY must be true or false, got: $1" >&2
      exit 1
      ;;
  esac
}

resolve_checkpoint_paths() {
  local model_source_dir="$1"
  LOSS_MODEL_PATH="${model_source_dir}/checkpoints/best_model_by_loss"
  RANKIC_MODEL_PATH="${model_source_dir}/checkpoints/best_model_by_rankic"
  local legacy_model_path="${model_source_dir}/checkpoints/best_model"
  if [[ ! -d "$LOSS_MODEL_PATH" && -d "$legacy_model_path" ]]; then
    LOSS_MODEL_PATH="$legacy_model_path"
  fi
}

require_any_checkpoint() {
  local model_source_dir="$1"
  if [[ ! -d "$LOSS_MODEL_PATH" && ! -d "$RANKIC_MODEL_PATH" ]]; then
    echo "KRONOS_EVAL_ONLY=true, but no compatible checkpoint exists under: $model_source_dir/checkpoints" >&2
    exit 1
  fi
}

run_checkpoint_eval() {
  local selection="$1"
  local model_path="$2"
  local infer_log_path="$3"
  shift 3

  {
    echo "Running evaluation for checkpoint=${selection}"
    echo "Model path: $model_path"
    "$@" --model-path "$model_path"
  } 2>&1 | tee -a "$infer_log_path"
}
