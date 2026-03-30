#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$ROOT/zlab/ab_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.sh"
elif [[ -f "$ROOT/zlab/ab_env.example" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/zlab/ab_env.example"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/kronos/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$HOME/miniconda3/envs/kronos/bin/torchrun}"
RESULTS_ROOT="${KRONOS_RESULTS_ROOT:-$ROOT/zlab/results}"

echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TORCHRUN_BIN=$TORCHRUN_BIN"
echo "KRONOS_LOCAL_CSV_DIR=${KRONOS_LOCAL_CSV_DIR:-}"
echo "KRONOS_PRETRAINED_TOKENIZER_PATH=${KRONOS_PRETRAINED_TOKENIZER_PATH:-}"
echo "KRONOS_PRETRAINED_PREDICTOR_PATH=${KRONOS_PRETRAINED_PREDICTOR_PATH:-}"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "WARN: nvidia-smi not found"
fi

echo
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found at $PYTHON_BIN"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
mods = ["torch", "pandas", "numpy", "matplotlib", "huggingface_hub", "tqdm", "safetensors"]
missing = []
for mod in mods:
    ok = importlib.util.find_spec(mod) is not None
    print(f"{mod}: {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(mod)
if missing:
    raise SystemExit(1)
PY

echo
if [[ -d "${KRONOS_LOCAL_CSV_DIR:-}" ]]; then
  echo "Local CSV data dir exists: ${KRONOS_LOCAL_CSV_DIR}"
else
  echo "WARN: Local CSV data dir does not exist: ${KRONOS_LOCAL_CSV_DIR:-<unset>}"
fi

mkdir -p "$RESULTS_ROOT"
echo "Results root ready: $RESULTS_ROOT"
