#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cat <<EOF
Per AGENTS.md, this script does not install packages automatically.

Use the existing training environment:

  conda activate kronos

Recommended checks:

  cd "$ROOT"
  python -V
  which python
  python - <<'PY'
import torch, pandas, numpy, matplotlib, huggingface_hub, tqdm, safetensors
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

Then prepare your local config file if needed:

  cp zlab/ab_env.example zlab/ab_env.sh
  sed -n '1,220p' zlab/ab_env.sh

If torch reports cuda_available=False or your torch CUDA version is newer than the server driver supports,
keep DEVICE=auto (recommended) or set DEVICE=cpu in zlab/ab_env.sh.

Finally run:

  zlab/scripts/check_ab_setup.sh
EOF
