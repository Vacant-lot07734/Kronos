#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
  if len(sys.argv) < 3:
    raise SystemExit(
      "Usage: compare_timesfm_kronos.py <kronos_root> <timesfm_root> [--output-dir ...] [--split ...]"
    )

  workspace_root = Path(__file__).resolve().parents[3]
  shared_script = workspace_root / "zlab" / "scripts" / "compare_zero_shot.py"
  if not shared_script.exists():
    raise SystemExit(f"Shared compare script not found: {shared_script}")

  kronos_root = sys.argv[1]
  timesfm_root = sys.argv[2]
  passthrough = sys.argv[3:]
  sys.path.insert(0, str(shared_script.parent))
  sys.argv = [
    str(shared_script),
    f"kronos={kronos_root}",
    f"timesfm={timesfm_root}",
    *passthrough,
  ]
  runpy.run_path(str(shared_script), run_name="__main__")


if __name__ == "__main__":
  main()
