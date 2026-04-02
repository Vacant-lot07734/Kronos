# Auto-generated local experiment config for A/B/C runs.
# Reset experiment-scoped overrides on each `source` to avoid stale shell
# state when previous override lines are later commented out.
unset KRONOS_PREDICTOR_LR
unset KRONOS_BATCH_SIZE
unset KRONOS_EPOCHS
unset KRONOS_PREDICTOR_SAVE_FOLDER_NAME
unset RESULT_NAME
unset KRONOS_EVAL_ONLY
unset KRONOS_EVAL_MODEL_HORIZON
unset KRONOS_HOURLY_WINDOW
unset KRONOS_HOURLY_LR
unset KRONOS_HOURLY_ENCODER_LAYERS

# -------------------------------------------------------------------
# Python / device
# -------------------------------------------------------------------
export PYTHON_BIN="$HOME/miniconda3/envs/kronos/bin/python"
export TORCHRUN_BIN="$HOME/miniconda3/envs/kronos/bin/torchrun"
export DEVICE="${DEVICE:-cuda:0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# -------------------------------------------------------------------
# Data / models
# -------------------------------------------------------------------
export KRONOS_LOCAL_CSV_DIR="/home/yzh/workspace/Kronos-0/zlab/data/daily"
export KRONOS_PRETRAINED_TOKENIZER_PATH="NeoQuasar/Kronos-Tokenizer-base"
export KRONOS_PRETRAINED_PREDICTOR_PATH="NeoQuasar/Kronos-base"

# -------------------------------------------------------------------
# A/B experiment split
# -------------------------------------------------------------------
export KRONOS_DATASET_BEGIN_TIME="2025-06-01"
export KRONOS_DATASET_END_TIME="2026-02-28"
export KRONOS_TRAIN_TIME_RANGE="2025-06-01,2025-11-30"
export KRONOS_VAL_TIME_RANGE="2025-12-01,2025-12-31"
export KRONOS_TEST_TIME_RANGE="2026-01-01,2026-02-28"
export KRONOS_LOOKBACK_WINDOW="20"

# HORIZON is set by each wrapper script: `1` or `5`.

# -------------------------------------------------------------------
# Shared training defaults for groups B/C
# -------------------------------------------------------------------
export KRONOS_BATCH_SIZE="32"
export KRONOS_EPOCHS="10"
export KRONOS_NUM_WORKERS="2"
export KRONOS_PREDICTOR_LR="5e-5"
export KRONOS_USE_COMET="false"
export KRONOS_USE_TENSORBOARD="true"
export KRONOS_TENSORBOARD_SUBDIR="tensorboard"
export KRONOS_FUTURE_ONLY_LOSS="true"
export KRONOS_NORMALIZE_WITH_CONTEXT_ONLY="true"
export KRONOS_FREEZE_PREDICTOR_FOR_AB="true"
export KRONOS_PREDICTOR_TRAIN_LAST_RATIO="0.333333"
export KRONOS_FREEZE_EMBEDDING="true"
export KRONOS_TRAIN_TIME_EMBEDDING="false"
export KRONOS_SKIP_TOKENIZER_FINETUNE="true"
export KRONOS_EVAL_BATCH_SIZE="128"
export KRONOS_EVAL_TOPK="10"
export KRONOS_INFERENCE_SAMPLE_COUNT="10"
export KRONOS_EVAL_ONLY="${KRONOS_EVAL_ONLY:-false}"

# -------------------------------------------------------------------
# Output roots
# -------------------------------------------------------------------
export KRONOS_RESULTS_ROOT="/home/yzh/workspace/Kronos-0/zlab/results"
export KRONOS_DATASET_PATH_BASE="$KRONOS_RESULTS_ROOT/processed_datasets"
export KRONOS_SAVE_PATH_BASE="$KRONOS_RESULTS_ROOT/models"
export KRONOS_EVAL_RESULT_PATH_BASE="$KRONOS_RESULTS_ROOT/evaluations"

# -------------------------------------------------------------------
# Optional overrides for group B
# Uncomment only when needed. These values are reset on each `source`.
# -------------------------------------------------------------------
# export KRONOS_EVAL_ONLY="false"  # true: skip training and only run evaluation
# export KRONOS_PREDICTOR_LR="1e-5"  # predictor learning rate
# export KRONOS_BATCH_SIZE="64"  # per-GPU batch size
# export KRONOS_EPOCHS="15"  # total training epochs
# export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_b_lr1e5_e10_bs64"  # exact model dir name; unset to use script default naming
# export RESULT_NAME="group_b_lr1e5_e10_bs64"  # exact evaluation dir name; results are written under cross-entropy/ and rankIc/

# -------------------------------------------------------------------
# Optional overrides for group C
# These variables are only consumed by `run_group_c.sh`.
# -------------------------------------------------------------------
# export KRONOS_EVAL_ONLY="false"  # true: reuse existing C checkpoint and only evaluate
# export KRONOS_HOURLY_WINDOW="25"  # hourly lookback length L_h, 5 trading days * 5 hourly bars/day
# export KRONOS_HOURLY_LR="1e-4"  # learning rate for hourly encoder + fusion
# export KRONOS_HOURLY_ENCODER_LAYERS="2"  # depth of hourly encoder
# export KRONOS_PREDICTOR_LR="5e-5"  # learning rate for the trainable Kronos upper layers
# export KRONOS_BATCH_SIZE="64"  # per-GPU batch size
# export KRONOS_EPOCHS="10"  # total training epochs
# export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_hlr1e4_lr5e5_e10_bs64"  # exact model dir name; unset to use script default naming
# export RESULT_NAME="group_c_hlr1e4_lr5e5_e10_bs64"  # exact evaluation dir name; results are written under cross-entropy/ and rankIc/

# -------------------------------------------------------------------
# Optional overrides for sequential C-group sweep
# `run_group_c_sweep.sh` reads these after sourcing this file once.
# -------------------------------------------------------------------
# export KRONOS_C_SWEEP_HORIZONS="1 5"  # sweep horizons in sequence; CLI args override this
# export KRONOS_C_SWEEP_PRED_LRS="5e-5 3e-5 2e-5"
# export KRONOS_C_SWEEP_HOURLY_LRS="1e-4 5e-5"
# export KRONOS_C_SWEEP_BATCH_SIZES="64"
# export KRONOS_C_SWEEP_EPOCHS_LIST="10 15"
# export KRONOS_C_SWEEP_HOURLY_WINDOWS="25"
# export KRONOS_C_SWEEP_ENCODER_LAYERS="2"
# export KRONOS_C_SWEEP_PREPARE_DATA="false"  # set true when hourly_window changes or C data must be rebuilt
# export KRONOS_C_SWEEP_CONTINUE_ON_ERROR="false"  # set true to continue after a failed run
