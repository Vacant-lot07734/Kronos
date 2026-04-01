# c 组训练说明

## 目标

这份说明只覆盖当前仓库里的 C 组实现：日线主流，小时线辅流，训练入口为 `zlab/scripts/run_group_c.sh`。

当前 C 组仍沿用与 B 组一致的选模协议：

* 主损失：`future-only token loss`
* best checkpoint：`val loss`
* 最终对比：`RankIC / IC / DA / MAE / RMSE / top-k / long-short`

因此，当前实验结论应表述为：

* “在相同 `val loss` 选模协议下，C 相比 B 的下游指标变化”

不要写成：

* “C 相比 B 的最优金融指标提升”

## 环境

训练环境：

* Python：`/home/yzh/miniconda3/envs/kronos/bin/python`
* 激活命令：`conda activate kronos`

仓库约定：

* 不自动安装依赖
* 使用共享服务器环境，尽量只改自己 conda 环境，不改系统驱动或系统包

常用准备命令：

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh
```

## 关键环境变量

`zlab/ab_env.sh` 提供了 A/B 的公共环境。跑 C 组时，重点关注下面这些变量：

```bash
export DEVICE="cuda:0"
export NPROC_PER_NODE="1"

export KRONOS_HOURLY_WINDOW="32"
export KRONOS_HOURLY_LR="1e-4"
export KRONOS_HOURLY_ENCODER_LAYERS="2"

export KRONOS_PREDICTOR_LR="5e-5"
export KRONOS_BATCH_SIZE="32"
export KRONOS_EPOCHS="15"

export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_predictor"
export RESULT_NAME="group_c"
export KRONOS_EVAL_ONLY="false"
```

说明：

* `KRONOS_EVAL_ONLY=true` 时，`run_group_c.sh` 只评估，不重新训练
* C 组没有像 B 组那样做跨 horizon 模型复用逻辑；`H=1` 和 `H=5` 默认是两次独立训练/评估
* 每次重新 `source zlab/ab_env.sh`，可避免旧 shell override 串到下一次实验
* `zlab/ab_env.sh` 文件底部已经补了 C 组专用 override 模板，可直接在后半段按需取消注释

## 推荐流程

### 1. 准备 C 组数据

```bash
source zlab/ab_env.sh
zlab/scripts/prepare_c_data.sh 1
zlab/scripts/prepare_c_data.sh 5
```

生成目录默认是：

* `zlab/results/processed_datasets/h1_c`
* `zlab/results/processed_datasets/h5_c`

准备完后，先看：

* `metadata.json`
* `symbol_stats.csv`

必须确认两件事：

* `n_symbols_kept` 是否足够
* C 组 `symbols` 是否与用于比较的 B 组股票池一致

如果股票池不一致，B/C 对比需要明确写成“不同资产池下结果”，或者把 B 评估过滤到相同股票池。

### 2. 跑 H=1

```bash
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="false"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_predictor_h1"
export RESULT_NAME="group_c_h1"
zlab/scripts/run_group_c.sh 1
```

### 3. 跑 H=5

```bash
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="false"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_predictor_h5"
export RESULT_NAME="group_c_h5"
zlab/scripts/run_group_c.sh 5
```

### 4. 只重新评估已有模型

```bash
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="true"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_predictor_h1"
export RESULT_NAME="group_c_h1_eval2"
zlab/scripts/run_group_c.sh 1
```

## 当前保存逻辑

训练输出目录：

* `zlab/results/models/h{H}/{save_folder}/`

重点文件：

* `train.log`
* `summary.json`
* `tensorboard/`
* `checkpoints/best_model/`

当前 `best_model` 的保存规则是：

* 每个 epoch 跑完验证后计算 `avg_val_loss`
* 若 `avg_val_loss` 更低，则覆盖保存 `checkpoints/best_model`

因此：

* 当前保存的是 `val loss` 最优 checkpoint
* 不是 `RankIC` 最优 checkpoint

评估输出目录：

* `zlab/results/evaluations/h{H}/{result_name}/`

重点文件：

* `run_config.json`
* `inference.log`
* `val/metrics.json`
* `test/metrics.json`
* `val/predictions.csv`
* `test/predictions.csv`

## 建议先看什么

训练阶段先看三类信息：

* `train loss` 是否稳定下降
* `val loss` 何时触底
* TensorBoard 中 `s1_loss / s2_loss / grad_norm / lr`

评估阶段先看三类信息：

* `mean_rank_ic`
* `rank_ic_ir`
* `long_short_top10_mean_return`

当前协议下，不建议只看 `val loss` 判断 C 是否成功。`val loss` 只负责当前版本的 checkpoint 选择，不代表下游金融指标一定同步变好。

## 可能遇到的问题

### 1. `torch` 和 CUDA 不兼容

典型现象：

* 训练一启动就报 `driver too old`
* `torch.cuda.is_available()` 异常

建议：

* 不改共享服务器驱动
* 只处理自己 conda 环境里的 `torch` 版本

### 2. 小时线数据覆盖不足

典型现象：

* `prepare_c_data.sh` 后 `n_symbols_kept` 明显偏小
* `run_group_c.sh` 报找不到有效样本

建议优先检查：

* `zlab/data/hourly/` 文件命名
* 小时线时间列格式
* `hourly_window` 是否过大
* `symbol_stats.csv` 中哪些股票在 `hourly_train_rows/hourly_val_rows/hourly_test_rows` 上不足

### 3. B/C 股票池不一致

这是 C 组最容易被忽略的实验问题。

建议比较：

* `zlab/results/processed_datasets/h1/metadata.json`
* `zlab/results/processed_datasets/h1_c/metadata.json`
* `zlab/results/processed_datasets/h5/metadata.json`
* `zlab/results/processed_datasets/h5_c/metadata.json`

如果 `symbols` 不同，就不能直接把结果写成“纯粹由小时线上下文带来的差异”。

### 4. 结果抖动

评估阶段默认有采样：

* `sample_count`
* `temperature`
* `top_p`

因此：

* 单次 `RankIC` 或 `top-k` 结果会有抖动
* 如果两次结果很接近，应优先怀疑采样波动，而不是立刻下结论

### 5. 当前选模和最终指标不一致

这不是代码 bug，而是当前实验协议的限制。

当前：

* 训练时按 `val loss` 选模
* 汇报时看 `RankIC / long-short`

所以可能出现：

* `val loss` 更好
* 但 `RankIC` 更差

如果 C 组初步显示有潜力，后续再考虑把选模切到 `val mean RankIC`。

## 当前更合理的比较方式

在当前版本下，建议按下面顺序汇报：

1. 先说明 B/C 当前都按 `val loss` 选模
2. 再比较 `val/test` 上的 `RankIC / IC / long-short`
3. 最后再讨论小时线是否可能有帮助

如果 C 组确实优于 B，再进入下一轮更严格的实验：

* B/C 同一股票池
* 按 `val RankIC` 选模
* 增加 `shuffled-hourly` 或 `masked-hourly` 消融
