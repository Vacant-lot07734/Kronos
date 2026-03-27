# externalContext.md

## Role of this file

This file is meant to be copied across repositories. It should preserve durable project context and operating assumptions, not task-specific notes. Keep the top-level structure stable and add new projects under the same project layer instead of rewriting the whole file.

## Layer 1: reusable shared context

### Working intent

- Use this file to provide durable context that helps AI or engineers start work in a new repository quickly.
- Keep it focused on project facts, operating assumptions, and directory conventions.
- Do not duplicate detailed task plans or temporary notes from day-to-day work.

### Recommended fixed assets

- `AGENTS.md`: rules, workflow, writing constraints, environment
- `externalContext.md`: reusable project context
- `zlab/`: experiment, research, finetuning, and result analysis assets
- `zdoc/`: code reading notes and architecture walkthroughs when needed

### Reusable operating rules

- Prefer repository-local conventions over generic answers.
- Record behavior changes in project-local docs instead of leaving them only in code.
- Keep runs reproducible: stable names, explicit parameters, explicit output paths.
- Prefer cheap validation first, then expensive validation.
- In shared machines, avoid unnecessary global changes and provide commands for the user to run.

### Directory pattern

This structure works well across many repos:

- runtime code remains in original code directories
- `zlab/` stores research or experiment work
- `zdoc/` stores deep code reading notes
- helper code should stay near the runtime code if it belongs to the product path

## Layer 2: project contexts

Add each project as a peer section under this layer. Do not merge multiple projects into one block.

### Project: Kronos

Kronos models financial time series as a discrete token sequence—turning OHLCVA-style market data into a “language” and learning to predict future market behavior autoregressively with a generative foundation model.

#### Project type

Mixed engineering and quantitative research repository, centered on Kronos-based financial time-series modeling. The active workflow is mostly research-driven: data preparation, finetuning, evaluation, ablation, and experiment analysis.

#### Main logic

The repository currently serves as the model and training core behind the
shared workspace workflow. Shared cache construction, shared evaluation, and
shared backtest orchestration live under `/home/yzh/workspace/zlab`.

Current supported paths:

- zero-shot forecasting
- finetuning on the shared daily cache
- shared Top-K backtesting on exported signals

#### Environment

- Training Python: `/home/yzh/miniconda3/envs/kronos/bin/python`
- Training env activation: `conda activate kronos`
- HF or CLI Python: `/home/yzh/miniconda3/bin/python`
- Base env activation: `conda activate base`

Machine assumption:

- multi-tenant shared Ubuntu environment
- avoid automatic package installs
- avoid unnecessary system-level changes

#### Main directories

- `finetune/`: preprocessing, training, evaluation scripts
- `model/`: Kronos core model code
- `zlab/`: experiment definitions, guides, prompts, analysis, run scripts
- `zlab/results/`: processed datasets, checkpoints, evaluation outputs, comparison figures
- `zdoc/`: optional code reading notes

#### Current task model

Main experiment settings:

- train / val / test split over `2021-01-01` to `2025-12-31`
- shared default split:
  train `2021-01-01 ~ 2024-06-30`,
  val `2024-07-01 ~ 2024-12-31`,
  test `2025-01-01 ~ 2025-12-31`
- shared daily context windows and horizons are controlled from the shared CLI
- canonical sample-rule reference: `/home/yzh/workspace/zlab/protocol/sample_selection.md`

Current key metrics:

- `rank_ic`
- `ic`
- `rank_icir`
- `icir`
- `da`
- `mae`
- `rmse`

Training and evaluation are separated:

- training objective: future-only token cross-entropy
- selection paths: validation loss and validation rankIC
- final comparison: financial metrics on test

#### Important operating details

- canonical runtime entrypoints are the shared Python CLIs under
  `/home/yzh/workspace/zlab/cli/`
- supported Kronos paths are zero-shot, finetune, and shared backtest
- finetune saves the best validation-loss checkpoint under `best_model`
- evaluation outputs are organized by horizon, run name, and selection path
- result comparison reads `zlab/results/evaluations/**/metrics.json`

#### Common risks

- autoregressive sampling can make repeated evaluation unstable
- token loss and downstream financial metrics may diverge
- long multi-run experiments can still consume disk quickly if checkpoints are kept aggressively
- shared GPU environments make long concurrent runs risky

### Project: <new-project-name>

#### Project type

Write one short sentence describing the repository type and main development mode.

#### Main logic

Write the main business or research logic in 2 to 4 lines. Focus on what the repository is trying to do, not on temporary tasks.

#### Environment

- Training or main Python:
- Secondary Python or CLI:
- Main env activation:
- Machine assumptions:

#### Main directories

- runtime code:
- experiment or lab directory:
- note or docs directory:
- outputs:

#### Current task model

- main task groups:
- key data ranges or environments:
- key metrics:
- training / evaluation split if relevant:

#### Important operating details

- main entry scripts:
- naming rules:
- output rules:
- checkpoint or deployment rules:

#### Common risks

- risk 1:
- risk 2:
- risk 3:

## Maintenance rule

When copying this file to another repository:

- keep Layer 1 mostly unchanged
- keep `Project: Kronos` as reference if useful, or remove it if the target repo should stay clean
- add the target repository as another peer project section under Layer 2
- do not convert this file into a temporary work log
