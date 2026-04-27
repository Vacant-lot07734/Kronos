# Kronos Phase C Runtime

This file is the repo-local migration note after the phase C cleanup.

## Canonical entrypoints

Use the shared Python CLI from `workspace/zlab`:

- `python -m zlab.cli.prepare_data`
- `python -m zlab.cli.run_kronos_zero_shot`
- `python -m zlab.cli.run_kronos_finetune`
- `python -m zlab.cli.run_kronos_hourly_zero_shot`
- `python -m zlab.cli.run_backtest`

Repo-local shell entrypoints have been removed on purpose. The supported path is
now the shared Python CLI only.

## Current runtime split

The runtime is intentionally split into two layers:

- `Kronos/` keeps the model and the existing finetune training core
- `workspace/zlab/` owns shared data cache, orchestration, evaluation, and
  backtest execution

This keeps the model code close to upstream while moving the experiment control
path to the workspace layer.

## What stays in Kronos

These files are still part of the supported path:

- `Kronos/model/kronos.py`
- `Kronos/finetune/config.py`
- `Kronos/finetune/dataset.py`
- `Kronos/finetune/train_tokenizer.py`
- `Kronos/finetune/train_predictor.py`
- `Kronos/finetune/qlib_test.py`

## What moved to workspace/zlab

The shared layer now owns:

- cache import and split materialization
- zero-shot orchestration
- finetune launch orchestration
- unified metric export
- shared Qlib Top-K backtest execution

Implementation anchors:

- `workspace/zlab/kronos/legacy_bundle.py`
- `workspace/zlab/kronos/evaluation.py`
- `workspace/zlab/kronos/finetune.py`

## Removed legacy paths

The following repo-local paths were removed during the phase C cleanup:

- grouped A/B/C shell runners
- `ab_env.sh` and related env bootstrap scripts
- C-group-specific preprocessing, dataset, training, and evaluation code
- old repo-local experiment guides that still described the A/B/C workflow

## Current backtest semantics

Backtesting is still enabled by default in the shared path and preserves the
legacy Kronos signal semantics:

- signal set: `last`, `mean`, `max`, `min`
- signal timestamp: history-window last day
- score definition: predicted close delta relative to the last context close

The canonical protocol lives in:

- `/home/yzh/workspace/zlab/protocol/backtest_topk.md`
- `/home/yzh/workspace/zlab/protocol/evaluation_pipeline.md`

## Basket filtering

Both daily and hourly Kronos backtests accept:

- `--inference-basket-path /path/to/basket.csv`
- `--basket-path /path/to/basket.csv`

The basket file is a single-column CSV named `instrument`.

- `--inference-basket-path` restricts prediction and forecast-quality metrics to
  the selected instruments
- `--basket-path` keeps full-universe prediction but filters only the
  downstream backtest layer
