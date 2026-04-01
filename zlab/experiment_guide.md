# 实验执行指南

本文面向当前仓库中的 A/B/C 实验实现，说明环境、参数、执行顺序、调参方式和结果判断方法。

## 1. 当前实现范围

当前仓库已落地的实验链路：

* A 组：`csv_data_preprocess.py -> evaluate_ab.py`
* B 组：`csv_data_preprocess.py -> train_predictor.py -> evaluate_ab.py`
* C 组：`csv_data_preprocess_c.py -> train_predictor_c.py -> evaluate_c.py`

说明：

* B/C 当前都按 `val loss` 选 `best_model`
* A/B/C 的最终对比仍以 `RankIC / IC / long-short` 为主
* C 组保持独立的数据入口和评估入口，不与 B 组预处理脚本合并

## 2. 环境

训练环境固定为：

* Python：`/home/yzh/miniconda3/envs/kronos/bin/python`
* 激活命令：`conda activate kronos`

典型准备命令：

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh
```

`zlab/ab_env.sh` 现在作为 A/B/C 共用环境入口，底部包含：

* B 组 override 模板
* C 组 override 模板

每次重新 `source zlab/ab_env.sh`，都会先清理常用 override，避免旧 shell 状态污染下一次实验。

## 3. 关键参数

### 通用参数

* `KRONOS_LOOKBACK_WINDOW`
  日线历史窗口长度，当前默认 `20`
* `KRONOS_PREDICT_WINDOW`
  预测窗口，由 `run_group_*.sh` 根据 `HORIZON` 自动写入
* `KRONOS_BATCH_SIZE`
  每卡 batch size
* `KRONOS_EPOCHS`
  训练 epoch 数
* `KRONOS_PREDICTOR_LR`
  predictor 可训练部分学习率
* `KRONOS_EVAL_ONLY`
  `true` 时跳过训练，只做评估

### B 组相关

* `KRONOS_PREDICTOR_TRAIN_LAST_RATIO`
  predictor 最后多少比例的层参与训练，当前默认 `0.333333`
* `KRONOS_FREEZE_EMBEDDING`
  是否冻结 embedding，当前默认 `true`

### C 组相关

* `KRONOS_HOURLY_WINDOW`
  小时线窗口长度，默认 `25`；当前按 `5` 个交易日、每天 `5` 根小时线设计
* `KRONOS_HOURLY_LR`
  小时线 encoder / fusion 的学习率，默认 `1e-4`
* `KRONOS_HOURLY_ENCODER_LAYERS`
  小时线 encoder 层数，默认 `2`

## 4. 预测窗口 `H` 的含义

`H` 不只是评估参数，而是会直接改变训练目标：

* 数据集窗口长度依赖 `predict_window`
* `future-only loss` 只对未来 `H` 步 token 计算

因此正式实验中：

* `H=1` 应单独训练
* `H=5` 也应单独训练

探索性评估时，可以复用已训练模型跨 horizon 做 `eval-only`。但这类结果只能作为快速探测，不应作为正式结论。

## 5. 执行顺序

### A 组

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh

zlab/scripts/prepare_ab_data.sh 1
zlab/scripts/prepare_ab_data.sh 5

zlab/scripts/run_group_a.sh 1
zlab/scripts/run_group_a.sh 5
```

### B 组

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh

zlab/scripts/prepare_ab_data.sh 1
zlab/scripts/prepare_ab_data.sh 5

zlab/scripts/run_group_b.sh 1
zlab/scripts/run_group_b.sh 5
```

探索性跨 horizon 评估示例：

```bash
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="true"
zlab/scripts/run_group_b.sh 5
```

说明：

* 这会尝试复用已有模型做 `H=5` 评估
* 该结果不作为正式 `H=5` 训练结论

### C 组

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh

zlab/scripts/prepare_c_data.sh 1
zlab/scripts/prepare_c_data.sh 5

zlab/scripts/run_group_c.sh 1
zlab/scripts/run_group_c.sh 5
```

如果修改了 `KRONOS_HOURLY_WINDOW`，需要重新执行：

* `zlab/scripts/prepare_c_data.sh 1`
* `zlab/scripts/prepare_c_data.sh 5`
* 以及所有依赖该数据集训练出的 C 组模型评估

只重新评估已有 C 模型：

```bash
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="true"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_predictor_h1"
export RESULT_NAME="group_c_h1_eval2"
zlab/scripts/run_group_c.sh 1
```

## 6. 训练时怎么看

### 当前实现下的训练判断

当前 B/C 训练脚本的 checkpoint 选择标准都是：

* `val loss`

所以训练阶段首先看：

* `train loss`
* `val loss`
* TensorBoard 里的 `s1_loss / s2_loss / grad_norm / lr`

一般判断：

* `train loss` 和 `val loss` 同时下降：优化正常
* `train loss` 下降但 `val loss` 持续上升：过拟合
* `val loss` 基本不动：当前学习率太低、解冻范围太小或 epoch 不够

### 对 B 组的调参建议

建议一次只改一个维度。

优先顺序：

1. 学习率
2. epoch
3. batch size
4. 解冻比例

建议起点：

* `lr=5e-5`
* `batch_size=32`
* `epochs=10`

常见搜索：

* `5e-5 -> 3e-5 -> 2e-5 -> 1e-5`
* `epochs: 5 / 10 / 15`

### 对 C 组的调参建议

先固定 C 的结构，再调超参。推荐顺序：

1. 先固定 `hourly_window=25`
2. 先固定 `hourly_encoder_layers=2`
3. 先用 `predictor_lr=5e-5`，`hourly_lr=1e-4`
4. 只有在 B 基线稳定后，再扫 C 的超参

否则很难分辨：

* 提升来自小时线信息
* 还是来自训练策略变化

## 7. 测试时怎么看

测试阶段不再看 token loss，而是看下游金融指标。

优先顺序建议：

1. `mean_rank_ic`
2. `mean_ic`
3. `rank_ic_ir`
4. `long_short_top10_mean_return`
5. `top10_mean_return`

解释：

* `mean_rank_ic` 是主排序指标
* `mean_ic` 与 `rank_ic_ir` 用来补充线性相关性和稳定性
* `long_short` 与 `top-k` 用来判断排序信号是否能转化成收益

### 结果对比时的约束

正式比较前必须确认：

* 股票池一致
* 时间切分一致
* `sample_count / top_p / temperature` 一致
* `H` 一致

如果股票池不同，结论必须显式写成“不同股票池下的结果”，不能直接解释成模型结构差异。

## 8. 当前实现下的结论表述

当前 B/C 的更准确表述是：

* “在相同 `val loss` 选模协议下，比较下游金融指标差异”

而不是：

* “比较各自最优金融指标 checkpoint 的最好结果”

如果后续要升级为正式实验口径，再做两件事：

1. 用 `val mean RankIC` 选模
2. 保证 B/C 在完全相同股票池上比较

## 9. 常见问题

### 1. `torch` 与 CUDA 不兼容

现象：

* 一启动训练就报 CUDA/driver 相关错误

处理方向：

* 不改共享服务器驱动
* 只修自己 conda 环境中的 `torch`

### 2. C 组股票池缩小

现象：

* `h1_c` 或 `h5_c` 的 `n_symbols_kept` 低于 B 组

原因：

* 小时线覆盖不足
* 某些股票在某个 split 中不满足 `hourly_window`

### 3. 结果抖动

现象：

* 同一模型重复评估，`RankIC` 或 `top-k` 有小幅波动

原因：

* 当前评估仍使用采样

处理：

* 比较接近的实验时，不要只凭单次结果下结论

### 4. `H=1` 模型在 `H=5` 上看起来也不错

解释：

* 这说明模型有一定跨 horizon 泛化
* 但不代表可以省掉正式的 `H=5` 训练

正式实验里，`H=5` 仍应单独训练。
