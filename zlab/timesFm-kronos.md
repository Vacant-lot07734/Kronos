# 第一阶段：Zero-shot 对比
严格对比 TimesFM 与 Kronos 的 zero-shot 基础能力
## 实验设置
### 数据区间
train: 2025-06-01 ~ 2025-11-30
val: 2025-12-01 ~ 2025-12-31
test: 2026-01-01 ~ 2026-02-28

虽然 zero-shot 不训练，但仍然保留 train / val / test 划分，原因是：

与后续微调阶段保持统一协议
val 集可用于后面的“最佳情况调参实验”
test 集始终作为最终报告集，避免反复用 test 调参数
### 输入窗口
当前对 Kronos 增加两组 zero-shot 设置：
- context = 20
- context = 32

TimesFM 当前先保留 context = 32 主结果。
### 预测窗口

zero-shot H=1
zero-shot H=5
### 输入输出定义
输入
每只股票在每个评估时点输入：
过去 32 个交易日的日线序列
输出
预测未来 1 日或 5 日日线序列
再从预测结果中提取未来收益信号，做横截面排序评价
收益率定义：
$$r_{T,T+t}=\frac{P_{T+t}}{P_T}-1$$
其中 $P_T$ 是当前收盘价， $P_{T+t}$ 是未来 $t$ 日收盘价。
### 指标
rankIC：主指标，最直接反映横截面排序能力
IC：辅助主指标，反映数值收益相关性
MAE：数值误差指标，用于观察预测值是否更接近真实值，但不作为最终金融效果的唯一依据

### 评估流程
评估按“滚动样本 -> 日度横截面 -> 跨日期聚合”三层进行。

第一层是样本构造。对每只股票、每个可评估日期，都构造一个滚动窗口样本：
- 输入：该日期之前连续 `context` 个交易日的日线序列
- 目标：未来 `H` 个交易日

因此同一只股票会在不同日期产生多个样本，所有股票在同一评估日期也会形成一个横截面。

第二层是模型推理。实现上可以按 batch 做推理，但每个样本都必须保留：
- `instrument`
- `context_end_date`
- `target_end_date`
- `pred_return`
- `true_return`

其中：

$$
\text{pred\_return}_h = \frac{\hat P_{T+h}}{P_T} - 1,\qquad
\text{true\_return}_h = \frac{P_{T+h}}{P_T} - 1
$$

这里 $P_T$ 是 context 结束日收盘价，$\hat P_{T+h}$ 是模型预测的未来第 $h$ 日收盘价，$P_{T+h}$ 是真实未来第 $h$ 日收盘价。

第三层是横截面评价。按 `context_end_date` 分组，把某一天全部股票的 `pred_return` 和 `true_return` 放在同一个横截面内，计算当天的：
- `IC`
- `RankIC`
- `top-k return`
- `bottom-k return`
- `long-short return`

最后再把所有评估日的日度指标做聚合，得到：
- `mean_rank_ic`
- `mean_ic`
- `rank_ic_ir`
- `ic_ir`
- `topk_mean_return`
- `long_short_topk_mean_return`

其中：
- `rank_ic_ir` 和 `ic_ir` 是基于“日度指标序列”的均值与波动计算得到的稳定性指标
- `MAE` 不按天先算，而是直接在全部样本层面统计预测收益与真实收益的误差

因此，最终结果不是把所有股票和所有日期混在一起只算一次相关性，而是先按交易日做横截面排序评价，再对所有交易日的评价结果做汇总。

## Kronos zero-shot 设置

context = 20 / 32
H = 1 / 5
sample_count = 10
temperature = 0.6
top_p = 0.9

## TimesFM zero-shot 设置

直接使用其原生 zero-shot point forecast 作为主结果：
第一组：
context = 20 / 32
H = 1 / 5
使用默认 point forecast 输出
第二组：
在第一组基础上添加 xreg 辅助

## kronos 实验
```shell
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh

export KRONOS_LOOKBACK_WINDOW=32
export KRONOS_INFERENCE_SAMPLE_COUNT=10
export KRONOS_INFERENCE_T=0.6
export KRONOS_INFERENCE_TOP_P=0.9
export KRONOS_INFERENCE_TOP_K=0
export KRONOS_EVAL_TOPK=20

# 可选：
# split 模式：沿用 train / val / test 划分
# all 模式：把一整段时间视为一个评估集，不再区分 val 和 test
export EVAL_MODE="all"

export STAGE1_ROOT="$PWD/zlab/results/timesfm_kronos_stage1/kronos"
for CTX in 20 32; do
  export KRONOS_LOOKBACK_WINDOW="$CTX"
  export DATA_ROOT="$STAGE1_ROOT/processed_datasets_ctx${CTX}"
  export EVAL_ROOT="$STAGE1_ROOT/evaluations_ctx${CTX}"
  export RUN_NAME_BASE="kronos_zero_shot_ctx${CTX}_sc10_t06_tp09"

  for H in 1 5; do
    export KRONOS_PREDICT_WINDOW="$H"
    export KRONOS_DATASET_PATH="$DATA_ROOT/h${H}"

    if [[ "$EVAL_MODE" == "all" ]]; then
      export KRONOS_VAL_TIME_RANGE="2025-12-01,2025-12-31"
      export KRONOS_TEST_TIME_RANGE="2025-06-01,2026-02-28"
      export RUN_NAME="${RUN_NAME_BASE}_all"
      export SPLITS=(test)
    else
      export KRONOS_VAL_TIME_RANGE="2025-12-01,2025-12-31"
      export KRONOS_TEST_TIME_RANGE="2026-01-01,2026-02-28"
      export RUN_NAME="${RUN_NAME_BASE}"
      export SPLITS=(val test)
    fi

    mkdir -p "$KRONOS_DATASET_PATH" "$EVAL_ROOT/h${H}"

    PYTHONPATH="$PWD" "$PYTHON_BIN" finetune/csv_data_preprocess.py \
      --csv-dir "$KRONOS_LOCAL_CSV_DIR" \
      --output-dir "$KRONOS_DATASET_PATH"

    PYTHONPATH="$PWD" "$PYTHON_BIN" finetune/evaluate_ab.py \
      --device "$DEVICE" \
      --tokenizer-path "$KRONOS_PRETRAINED_TOKENIZER_PATH" \
      --model-path "$KRONOS_PRETRAINED_PREDICTOR_PATH" \
      --data-path "$KRONOS_DATASET_PATH" \
      --result-save-path "$EVAL_ROOT/h${H}" \
      --result-name "$RUN_NAME" \
      --pred-len "$H" \
      --sample-count "$KRONOS_INFERENCE_SAMPLE_COUNT" \
      --batch-size "$KRONOS_EVAL_BATCH_SIZE" \
      --topk "$KRONOS_EVAL_TOPK" \
      --splits "${SPLITS[@]}"
  done
done
```

说明：
- 默认 `EVAL_MODE=split`，输出 `val` 和 `test`
- 若设 `EVAL_MODE=all`，则只推理 `test`，并把 `test` 时间段固定为 `2025-06-01 ~ 2026-02-28`
- 结果仍落在 `test/metrics.json` 下，但语义上表示 `all`
- 这是因为当前 `evaluate_ab.py` 只支持 `val/test`，没有原生 `all` 选项；因此这里用“只保留 test 输出，并把 test 时间段改写为全区间”的方式实现等价 `all` 评估

## timesfm 实验

下面这组命令运行 TimesFM stage1 zero-shot，并把原始 backtest 输出整理成统一目录。默认模式是 `all`：

- 只推理 `2025-06-01 ~ 2026-02-28`
- 整段结果统一记为 `all`

如切到 `protocol` 模式，则默认只推理 `2025-12-01 ~ 2026-02-28`，再按：

- `val: 2025-12-01 ~ 2025-12-31`
- `test: 2026-01-01 ~ 2026-02-28`

切分输出。

默认目录包含：

- `run_config.json`
- `all/predictions.csv`
- `all/daily_metrics.csv`
- `all/metrics.json`

当前只覆盖 plain zero-shot 主链，即：
- `target=close`
- `context=20 / 32`
- `H=1 / 5`
- 默认使用 TimesFM 的 `q50` 作为 point forecast
- 默认关闭 XReg
- 默认 `split_mode=all`
- 默认评估区间：
  - `all`: `2025-06-01 ~ 2026-02-28`
  - `protocol`: `2025-12-01 ~ 2026-02-28`
- 可选用 `TIMESFM_EVAL_START` / `TIMESFM_EVAL_END` 覆盖默认评估时间段

```shell
conda activate timesfm
cd /home/yzh/workspace/timesfm

export TIMESFM_PYTHON=/home/yzh/miniconda3/envs/timesfm/bin/python
export STAGE1_ROOT="$PWD/zlab/results/timesfm_kronos_stage1/timesfm"
export TIMESFM_DATA_DIR="$PWD/zlab/financial_forecasting/data"
export TIMESFM_MODEL_ID="google/timesfm-2.5-200m-pytorch"
export TIMESFM_CONTEXTS="20 32"
export TIMESFM_TOPK=20
export TIMESFM_BATCH_SIZE=32
export RUN_NAME_PREFIX="timesfm_zero_shot"
export TIMESFM_SPLIT_MODE="all"

# 默认：
#   TIMESFM_SPLIT_MODE=all      -> 只推理 2025-06-01 ~ 2026-02-28，并输出 all/
#   TIMESFM_SPLIT_MODE=protocol -> 只推理 2025-12-01 ~ 2026-02-28，并输出 val/ test/

# 可选：手动覆盖默认评估时间段
# export TIMESFM_EVAL_START="2025-06-01"
# export TIMESFM_EVAL_END="2026-02-28"

for LOOKBACK_WINDOW in $TIMESFM_CONTEXTS; do
  export EVAL_ROOT="$STAGE1_ROOT/evaluations_ctx${LOOKBACK_WINDOW}"
  export RUN_NAME="${RUN_NAME_PREFIX}_ctx${LOOKBACK_WINDOW}_q50_${TIMESFM_SPLIT_MODE}"

  for H in 1 5; do
    export RUN_ROOT="$EVAL_ROOT/h${H}/$RUN_NAME"
    mkdir -p "$RUN_ROOT/_artifacts"

    CMD=(
      "$TIMESFM_PYTHON"
      zlab/financial_forecasting/scripts/forecast_ohlcva_csv.py
      "$TIMESFM_DATA_DIR"
      --skip-check
      --mode backtest
      --target close
      --horizon "$H"
      --context-length "$LOOKBACK_WINDOW"
      --min-context "$LOOKBACK_WINDOW"
      --stride 1
      --batch-size "$TIMESFM_BATCH_SIZE"
      --top-k "$TIMESFM_TOPK"
      --model-id "$TIMESFM_MODEL_ID"
      --no-xreg
      --split-mode "$TIMESFM_SPLIT_MODE"
      --output "$RUN_ROOT/_artifacts/predictions_all.csv"
      --daily-metrics-output "$RUN_ROOT/_artifacts/daily_metrics_all.csv"
      --metrics-output "$RUN_ROOT/_artifacts/metrics_all.json"
      --stage1-run-dir "$RUN_ROOT"
      --run-name "$RUN_NAME"
    )

    if [[ -n "${TIMESFM_EVAL_START:-}" ]]; then
      CMD+=(--eval-start "$TIMESFM_EVAL_START")
    fi
    if [[ -n "${TIMESFM_EVAL_END:-}" ]]; then
      CMD+=(--eval-end "$TIMESFM_EVAL_END")
    fi

    "${CMD[@]}"
  done
done
```

如需直接运行仓库内置包装脚本，可使用：

```shell
bash zlab/financial_forecasting/scripts/run_timesfm_stage1.sh
```

## 对比图脚本
当 Kronos 与 TimesFM 两边的输出目录结构一致后，可直接用下面的脚本生成对比图。默认是自动模式，会扫描并兼容：
- `all/metrics.json`
- `test/metrics.json`
- `val/metrics.json`

对同一个 `family + ctx + horizon`，默认优先级为：
- `all`
- `test`
- `val`

因此可以直接把所有适配路径一起导入比较，不需要手工先判断对方是 `all` 还是 `test`。默认输出 6 张图：
- `rank_ic_h1_auto.png`
- `ic_h1_auto.png`
- `mae_h1_auto.png`
- `rank_ic_h5_auto.png`
- `ic_h5_auto.png`
- `mae_h5_auto.png`

图中实验顺序固定为：
- `kronos_ctx20`
- `kronos_ctx32`
- `timesfm_ctx20`
- `timesfm_ctx32`

脚本同时兼容两种 `metrics.json` 格式：
- Kronos 当前使用的“指标直接位于顶层”的格式
- TimesFM 当前使用的“指标位于 `summary` 下”的格式
- 同时兼容 `val`、`test`、`all` 三种 split 目录

```shell
conda activate kronos
cd /home/yzh/workspace/Kronos-0

python zlab/scripts/compare_timesfm_kronos.py \
  /home/yzh/workspace/Kronos-0/zlab/results/timesfm_kronos_stage1/kronos \
  /home/yzh/workspace/timesfm/zlab/results/timesfm_kronos_stage1/timesfm \
  --output-dir /home/yzh/workspace/Kronos-0/zlab/results/timesfm_kronos_stage1/comparisons
```

如果你想强制只比较某一种 split，可以显式指定：

```shell
python zlab/scripts/compare_timesfm_kronos.py \
  /home/yzh/workspace/Kronos-0/zlab/results/timesfm_kronos_stage1/kronos \
  /home/yzh/workspace/timesfm/zlab/results/timesfm_kronos_stage1/timesfm \
  --split all \
  --output-dir /home/yzh/workspace/Kronos-0/zlab/results/timesfm_kronos_stage1/comparisons_all
```
