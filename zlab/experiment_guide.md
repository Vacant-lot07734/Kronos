# A/B 实验指导文档

适用范围：

* A 组：预训练 Kronos 直接推理
* B 组：只做日线微调

对应规划见 [lab.md](/home/yzh/workspace/Kronos-0/zlab/lab.md)。


## 1. 当前实验的固定口径

### 数据

当前实验直接使用本地日线 CSV：

* 目录：`zlab/data/daily/`
* 文件格式：`*_qfq_day.csv`
* 列：`ts_code, trade_date, open, high, low, close, vol, amount ...`

### 时间切分

* Train：`2025-06-01 ~ 2025-11-30`
* Val：`2025-12-01 ~ 2025-12-31`
* Test：`2026-01-01 ~ 2026-02-28`

### 任务

* 历史窗口：`L=20`
* 预测 horizon：`H=1` 和 `H=5`
* 预测目标：未来 `H` 日最后一天的收益率

收益率定义：

```text
pred_return = pred_close_H / last_close - 1
true_return = true_close_H / last_close - 1
```

这里的 `last_close` 是历史窗口最后一天收盘价。

## 2. 代码里已经补好的内容

这轮为了让实验能直接落地，已经补了四类入口。

### 2.1 数据预处理

新增：

* `finetune/csv_data_preprocess.py`

作用：

* 直接读取 `zlab/data/daily/*.csv`
* 转成训练脚本可直接使用的 `train_data.pkl / val_data.pkl / test_data.pkl`
* 自动给 `val/test` 加 lookback buffer
* 只保留 train/val/test 三段都能形成样本的股票
* 额外保存：
  * `metadata.json`
  * `symbol_stats.csv`

### 2.2 B 组训练口径

已补：

* `finetune/train_predictor.py`
* `finetune/dataset.py`

现在 B 组训练支持：

* 跳过 tokenizer 微调，直接用预训练 tokenizer
* 冻结 predictor 前 `2/3`，只训后 `1/3` + `norm` + `dep_layer` + `head`
* `future-only loss`

`future-only loss` 的意思是：

* 输入仍然是整段序列
* 但反向传播只对未来 `H` 个位置算 token CE
* 不再用整段历史重建 loss 作为主训练目标

这和 [lab.md](/home/yzh/workspace/Kronos-0/zlab/lab.md) 里的实验定义是一致的。

### 2.3 A/B 统一评估

新增：

* `finetune/evaluate_ab.py`

作用：

* 不再依赖 Qlib 回测
* 直接从 `val_data.pkl / test_data.pkl` 构造评估窗口
* 统一输出 A/B 两组指标

保存内容包括：

* `predictions.csv`
* `daily_metrics.csv`
* `metrics.json`
* `*_metrics.png`

### 2.4 一键脚本

新增：

* `zlab/scripts/setup_ab_env.sh`
* `zlab/scripts/check_ab_setup.sh`
* `zlab/scripts/prepare_ab_data.sh`
* `zlab/scripts/run_group_a.sh`
* `zlab/scripts/run_group_b.sh`

## 3. 你实际要做的事

顺序固定，不要跳。

### 第一步：准备环境

```bash
conda activate kronos
zlab/scripts/setup_ab_env.sh
cp zlab/ab_env.example zlab/ab_env.sh
source zlab/ab_env.sh
zlab/scripts/check_ab_setup.sh
```

`setup_ab_env.sh` 现在不会自动安装任何包，它只会把建议命令打印出来。  
你当前仓库应直接使用 `AGENTS.md` 指定的训练环境：

* `PYTHON_BIN`
* `TORCHRUN_BIN`

推荐写成：

```bash
export PYTHON_BIN="$HOME/miniconda3/envs/kronos/bin/python"
export TORCHRUN_BIN="$HOME/miniconda3/envs/kronos/bin/torchrun"
```

### 第二步：准备 H=1 和 H=5 数据

```bash
source zlab/ab_env.sh
zlab/scripts/prepare_ab_data.sh 1
zlab/scripts/prepare_ab_data.sh 5
```

输出目录：

```text
zlab/results/processed_datasets/h1/
zlab/results/processed_datasets/h5/
```

你至少要检查：

* `train_data.pkl`
* `val_data.pkl`
* `test_data.pkl`
* `metadata.json`
* `symbol_stats.csv`

### 第三步：跑 A 组

```bash
source zlab/ab_env.sh
zlab/scripts/run_group_a.sh 1
zlab/scripts/run_group_a.sh 5
```

输出目录：

```text
zlab/results/evaluations/h1/group_a/
zlab/results/evaluations/h5/group_a/
```

### 第四步：跑 B 组

```bash
source zlab/ab_env.sh
zlab/scripts/run_group_b.sh 1
zlab/scripts/run_group_b.sh 5
```

输出目录：

```text
zlab/results/models/h1/group_b_predictor/
zlab/results/models/h5/group_b_predictor/
zlab/results/evaluations/h1/group_b/
zlab/results/evaluations/h5/group_b/
```

## 4. 训练时需要看什么

这里只看两层。

### 4.1 训练过程指标

B 组训练日志里重点看：

* `Loss`
* `Validation Loss`

它们现在都是：

* `future-only token CE`

不是价格 MAE，也不是 RankIC。

### 4.2 怎么判断收敛

可以用下面这个简单标准：

* `Validation Loss` 连续 2 到 3 个 epoch 不再明显下降
* 同时没有出现 `nan / inf / OOM`

如果出现下面这种形态，说明开始过拟合：

* 训练 `Loss` 继续下降
* 但 `Validation Loss` 持平甚至回升

当前脚本保存的是：

* 验证集 `Validation Loss` 最低的 checkpoint

也就是：

* `zlab/results/models/.../checkpoints/best_model/`

## 5. 测试时怎么看结果

最终比较 A/B，不看训练 loss，主要看 `metrics.json`。

### 5.1 主指标

优先顺序建议：

* `mean_rank_ic`
* `mean_ic`
* `rank_ic_ir`

解释：

* `mean_rank_ic`
  最重要。它反映模型对横截面排序是否更好。
* `mean_ic`
  看线性相关性。
* `rank_ic_ir`
  看排序能力是否稳定，不只是偶然几天好。

### 5.2 辅指标

还要看：

* `direction_accuracy`
* `mae`
* `rmse`

解释：

* `direction_accuracy`
  看涨跌方向是否判断得更准。
* `mae / rmse`
  看预测收益率偏差有多大。

### 5.3 策略代理指标

为了补充“排序指标是否真能转成选股收益”，还会输出：

* `top10_mean_return`
* `top10_cum_return`
* `long_short_top10_mean_return`
* `long_short_top10_cum_return`

这不是完整交易回测，而是轻量的 Top-K 收益代理。

比较时看法很简单：

* 如果 B 的 `mean_rank_ic` 提升，同时 `top10_mean_return` 也提升
* 可以认为 B 比 A 更有价值

## 6. 结果文件怎么读

以 `zlab/results/evaluations/h1/group_a/test/` 为例：

* `metrics.json`
  最终汇总，先看这个。
* `predictions.csv`
  每个样本的预测结果，适合排查单只股票。
* `daily_metrics.csv`
  每天的 `IC / RankIC / Top-K Return`。
* `test_metrics.png`
  累计收益代理曲线和日度 IC 曲线。

### 最小对比方法

你只要把四个文件拿出来对比就够了：

* `zlab/results/evaluations/h1/group_a/test/metrics.json`
* `zlab/results/evaluations/h1/group_b/test/metrics.json`
* `zlab/results/evaluations/h5/group_a/test/metrics.json`
* `zlab/results/evaluations/h5/group_b/test/metrics.json`

## 7. 你现在最该关心的改动项

如果你不熟训练，优先记住这几个地方。

### 必改配置

通常只需要改 `zlab/ab_env.sh`：

* `PYTHON_BIN`
* `TORCHRUN_BIN`
* `DEVICE`
* `KRONOS_PRETRAINED_TOKENIZER_PATH`
* `KRONOS_PRETRAINED_PREDICTOR_PATH`

### 一般不用改

除非实验不稳定，否则先不要动：

* `KRONOS_LOOKBACK_WINDOW=20`
* `KRONOS_BATCH_SIZE=32`
* `KRONOS_EPOCHS=10`
* `KRONOS_PREDICTOR_LR=5e-5`
* `KRONOS_PREDICTOR_TRAIN_LAST_RATIO=0.333333`
* `KRONOS_FUTURE_ONLY_LOSS=true`

## 8. 推荐执行顺序

不要一上来四个实验一起跑。先这样：

1. `prepare_ab_data.sh 1`
2. `run_group_a.sh 1`
3. `run_group_b.sh 1`
4. 先看 `h1` 的 `metrics.json`
5. 没问题再跑 `h5`

理由很简单：

* `H=1` 更快
* 先确认流程、指标、保存路径都对
* 再跑 `H=5`

## 9. 一句话判断实验结论

最后只需要回答两个问题：

* `H=1` 上，B 的 `test mean_rank_ic` 是否高于 A？
* `H=5` 上，B 的 `test mean_rank_ic` 和 `top10_mean_return` 是否同时高于 A？

如果答案是“是”，那这轮 A/B 实验就完成了，而且结论是清楚的：

* 日线微调值得做
