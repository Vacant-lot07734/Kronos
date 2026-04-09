# TimesFM vs Kronos Zero-Shot

Cross-repo zero-shot comparison is now owned by the shared workspace
`/home/yzh/workspace/zlab` layer.

Shared metric contract:

- Primary metrics: `rank_ic`, `ic`, `rank_icir`, `icir`
- Supporting metrics: `da`, `mae`, `rmse`
- Deferred strategy-validation metrics: `AER`, `IR`

Shared references:

- `/home/yzh/workspace/zlab/protocol/zero_shot.md`
- `/home/yzh/workspace/zlab/protocol/evaluation_metrics.md`
- `/home/yzh/workspace/zlab/protocol/evaluation_metrics.json`

## Canonical entrypoints

Run Kronos:

```bash
bash /home/yzh/workspace/zlab/scripts/run_kronos_zero_shot.sh
```

Run TimesFM:

```bash
bash /home/yzh/workspace/zlab/scripts/run_timesfm_zero_shot.sh
```

Compare results:

```bash
python /home/yzh/workspace/zlab/scripts/compare_zero_shot.py \
  kronos=/home/yzh/workspace/zlab/results/zero_shot/kronos \
  timesfm=/home/yzh/workspace/zlab/results/zero_shot/timesfm \
  --output-dir /home/yzh/workspace/zlab/results/comparisons
```

## Repo-local compatibility wrappers

Kronos repo:

```bash
bash zlab/scripts/run_kronos_zero_shot.sh
```

TimesFM repo:

```bash
bash /home/yzh/workspace/timesfm/zlab/financial_forecasting/scripts/run_timesfm_zero_shot.sh
```

Kronos-specific training, fine-tuning and ablation work still stays under
`Kronos-0/zlab`. Only the shared zero-shot orchestration moved to the workspace
root.
