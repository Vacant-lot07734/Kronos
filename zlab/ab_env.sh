# Auto-generated local experiment config for A/B runs.
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
# Training defaults for group B
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

# -------------------------------------------------------------------
# Output roots
# -------------------------------------------------------------------
export KRONOS_RESULTS_ROOT="/home/yzh/workspace/Kronos-0/zlab/results"
export KRONOS_DATASET_PATH_BASE="$KRONOS_RESULTS_ROOT/processed_datasets"
export KRONOS_SAVE_PATH_BASE="$KRONOS_RESULTS_ROOT/models"
export KRONOS_EVAL_RESULT_PATH_BASE="$KRONOS_RESULTS_ROOT/evaluations"
