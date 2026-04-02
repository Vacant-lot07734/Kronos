#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_HELPER="$ROOT/zlab/scripts/_group_runner_common.sh"

if [[ -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
fi
if [[ -f "$COMMON_HELPER" ]]; then
  # shellcheck disable=SC1091
  source "$COMMON_HELPER"
fi

split_items() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  local items=( $raw )
  printf '%s\n' "${items[@]}"
}

bool_or_exit() {
  local name="$1"
  local value="${2,,}"
  case "$value" in
    true|false) ;;
    *)
      echo "$name must be true or false, got: $2" >&2
      exit 1
      ;;
  esac
}

mapfile -t HORIZONS < <(
  if [[ "$#" -gt 0 ]]; then
    printf '%s\n' "$@"
  else
    split_items "${KRONOS_C_SWEEP_HORIZONS:-1}"
  fi
)
mapfile -t PRED_LRS < <(split_items "${KRONOS_C_SWEEP_PRED_LRS:-${KRONOS_PREDICTOR_LR}}")
mapfile -t HOURLY_LRS < <(split_items "${KRONOS_C_SWEEP_HOURLY_LRS:-${KRONOS_HOURLY_LR}}")
mapfile -t BATCH_SIZES < <(split_items "${KRONOS_C_SWEEP_BATCH_SIZES:-${KRONOS_BATCH_SIZE}}")
mapfile -t EPOCH_LIST < <(split_items "${KRONOS_C_SWEEP_EPOCHS_LIST:-${KRONOS_EPOCHS}}")
mapfile -t HOURLY_WINDOWS < <(split_items "${KRONOS_C_SWEEP_HOURLY_WINDOWS:-${KRONOS_HOURLY_WINDOW}}")
mapfile -t ENCODER_LAYERS < <(split_items "${KRONOS_C_SWEEP_ENCODER_LAYERS:-${KRONOS_HOURLY_ENCODER_LAYERS}}")

PREPARE_DATA="${KRONOS_C_SWEEP_PREPARE_DATA:-false}"
CONTINUE_ON_ERROR="${KRONOS_C_SWEEP_CONTINUE_ON_ERROR:-false}"
bool_or_exit "KRONOS_C_SWEEP_PREPARE_DATA" "$PREPARE_DATA"
bool_or_exit "KRONOS_C_SWEEP_CONTINUE_ON_ERROR" "$CONTINUE_ON_ERROR"

if [[ "${#HOURLY_WINDOWS[@]}" -gt 1 && "${PREPARE_DATA,,}" != "true" ]]; then
  echo "Multiple KRONOS_C_SWEEP_HOURLY_WINDOWS were requested, but KRONOS_C_SWEEP_PREPARE_DATA=false." >&2
  echo "Set KRONOS_C_SWEEP_PREPARE_DATA=true so each hourly_window rebuilds its C dataset first." >&2
  exit 1
fi

SWEEP_TAG="${KRONOS_C_SWEEP_TAG:-$(date +%Y%m%dT%H%M%S)}"
SWEEP_DIR="${KRONOS_RESULTS_ROOT:-$ROOT/zlab/results}/sweeps"
SWEEP_LOG_PATH="$SWEEP_DIR/c_sweep_${SWEEP_TAG}.log"
SWEEP_CSV_PATH="$SWEEP_DIR/c_sweep_${SWEEP_TAG}.csv"
mkdir -p "$SWEEP_DIR"

{
  echo "horizon,hourly_window,hourly_lr,predictor_lr,batch_size,epochs,encoder_layers,run_name,status"
} > "$SWEEP_CSV_PATH"

export KRONOS_AUTO_SOURCE_ENV="false"

declare -A PREPARED_KEYS=()
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

echo "C-group sweep started: $SWEEP_TAG" | tee "$SWEEP_LOG_PATH"
echo "Sweep log: $SWEEP_LOG_PATH" | tee -a "$SWEEP_LOG_PATH"
echo "Sweep csv: $SWEEP_CSV_PATH" | tee -a "$SWEEP_LOG_PATH"

for horizon in "${HORIZONS[@]}"; do
  for hourly_window in "${HOURLY_WINDOWS[@]}"; do
    PREP_KEY="${horizon}:${hourly_window}"
    if [[ "${PREPARE_DATA,,}" == "true" && -z "${PREPARED_KEYS[$PREP_KEY]:-}" ]]; then
      export KRONOS_HOURLY_WINDOW="$hourly_window"
      echo "" | tee -a "$SWEEP_LOG_PATH"
      echo "[Prepare] H=$horizon hourly_window=$hourly_window" | tee -a "$SWEEP_LOG_PATH"
      bash "$ROOT/zlab/scripts/prepare_c_data.sh" "$horizon" 2>&1 | tee -a "$SWEEP_LOG_PATH"
      PREPARED_KEYS[$PREP_KEY]=1
    fi

    for hourly_lr in "${HOURLY_LRS[@]}"; do
      for pred_lr in "${PRED_LRS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
          for epochs in "${EPOCH_LIST[@]}"; do
            for encoder_layers in "${ENCODER_LAYERS[@]}"; do
              TOTAL_RUNS=$((TOTAL_RUNS + 1))
              export KRONOS_HOURLY_WINDOW="$hourly_window"
              export KRONOS_HOURLY_LR="$hourly_lr"
              export KRONOS_PREDICTOR_LR="$pred_lr"
              export KRONOS_BATCH_SIZE="$batch_size"
              export KRONOS_EPOCHS="$epochs"
              export KRONOS_HOURLY_ENCODER_LAYERS="$encoder_layers"

              HOURLY_LR_TAG="$(sanitize_num "$hourly_lr")"
              PRED_LR_TAG="$(sanitize_num "$pred_lr")"
              RUN_NAME="group_c_hlr${HOURLY_LR_TAG}_lr${PRED_LR_TAG}_e${epochs}_bs${batch_size}"
              export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="$RUN_NAME"
              export RESULT_NAME="$RUN_NAME"

              echo "" | tee -a "$SWEEP_LOG_PATH"
              echo "[Run $TOTAL_RUNS] H=$horizon, hourly_window=$hourly_window, hourly_lr=$hourly_lr, predictor_lr=$pred_lr, batch_size=$batch_size, epochs=$epochs, encoder_layers=$encoder_layers" | tee -a "$SWEEP_LOG_PATH"
              echo "  run_name=$RUN_NAME" | tee -a "$SWEEP_LOG_PATH"

              if bash "$ROOT/zlab/scripts/run_group_c.sh" "$horizon" 2>&1 | tee -a "$SWEEP_LOG_PATH"; then
                SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
                STATUS="success"
              else
                FAILED_RUNS=$((FAILED_RUNS + 1))
                STATUS="failed"
                if [[ "${CONTINUE_ON_ERROR,,}" != "true" ]]; then
                  echo "$horizon,$hourly_window,$hourly_lr,$pred_lr,$batch_size,$epochs,$encoder_layers,$RUN_NAME,$STATUS" >> "$SWEEP_CSV_PATH"
                  echo "Sweep aborted on failure. Set KRONOS_C_SWEEP_CONTINUE_ON_ERROR=true to continue after failures." | tee -a "$SWEEP_LOG_PATH"
                  exit 1
                fi
              fi

              echo "$horizon,$hourly_window,$hourly_lr,$pred_lr,$batch_size,$epochs,$encoder_layers,$RUN_NAME,$STATUS" >> "$SWEEP_CSV_PATH"
            done
          done
        done
      done
    done
  done
done

echo "" | tee -a "$SWEEP_LOG_PATH"
echo "Sweep finished. total=$TOTAL_RUNS success=$SUCCESS_RUNS failed=$FAILED_RUNS" | tee -a "$SWEEP_LOG_PATH"
