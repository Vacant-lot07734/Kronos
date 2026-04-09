# 实验执行指南

本文面向当前仓库中的 A/B/C 实验实现，说明环境、参数、执行顺序、调参方式和结果判断方法。

## 1. 当前实现范围

当前仓库已落地的实验链路：

* A 组：`csv_data_preprocess.py -> evaluate_ab.py`
* B 组：`csv_data_preprocess.py -> train_predictor.py -> evaluate_ab.py`
* C 组：`csv_data_preprocess_c.py -> train_predictor_c.py -> evaluate_c.py`

说明：

* B/C 当前都保留两类 checkpoint：`best_model_by_loss` 与 `best_model_by_rankic`
* `checkpoints/best_model` 继续保留，并与 `best_model_by_loss` 对齐，兼容旧路径
* A/B/C 的最终对比统一以 `rank_ic / ic / rank_icir / icir / da / mae / rmse` 为主
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

当前 runner 的默认命名：

* A 组结果目录默认是 `group_a`
* B 组模型和结果目录默认是 `group_b_lr{lr}_e{epochs}_bs{batch}`
* C 组模型和结果目录默认是 `group_c_hlr{hourly_lr}_lr{predictor_lr}_e{epochs}_bs{batch}`
* B/C 测试结果目录下固定生成两个子目录：`cross-entropy/` 与 `rankIc/`

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
* `KRONOS_INFERENCE_SAMPLE_COUNT`
  单样本推理时的内部采样条数，当前默认 `10`
* `KRONOS_PREDICTOR_SAVE_FOLDER_NAME`
  模型目录名；默认使用脚本生成的参数化名称
* `RESULT_NAME`
  评估目录名；默认使用脚本生成的参数化名称

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
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_c_hlr1e4_lr5e5_e10_bs64"
export RESULT_NAME="group_c_hlr1e4_lr5e5_e10_bs64"
zlab/scripts/run_group_c.sh 1
```

顺序扫多组 C 组超参数：

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0
source zlab/ab_env.sh

export KRONOS_C_SWEEP_HORIZONS="1"
export KRONOS_C_SWEEP_PRED_LRS="5e-5 3e-5 2e-5"
export KRONOS_C_SWEEP_HOURLY_LRS="1e-4"
export KRONOS_C_SWEEP_BATCH_SIZES="64"
export KRONOS_C_SWEEP_EPOCHS_LIST="10"
export KRONOS_C_SWEEP_HOURLY_WINDOWS="25"
export KRONOS_C_SWEEP_ENCODER_LAYERS="2"
export KRONOS_C_SWEEP_PREPARE_DATA="false"

bash zlab/scripts/run_group_c_sweep.sh
```

说明：

* `run_group_c_sweep.sh` 会顺序执行，不会并行占用同一张 GPU
* 默认沿用 `ab_env.sh` 当前值；只有你显式设置的 sweep 变量会形成组合
* 如果 sweep 中包含多个 `hourly_window`，必须设置 `KRONOS_C_SWEEP_PREPARE_DATA=true`
* sweep 日志和状态表会写到 `zlab/results/sweeps/`

## 6. 训练时怎么看

### 当前实现下的训练判断

当前 B/C 训练脚本会维护两类 checkpoint：

* `best_model_by_loss`
* `best_model_by_rankic`

其中：

* 每个 epoch 只计算 `val loss`
* `val loss` 用于在线保存 `best_model_by_loss`
* 同时保存每个 epoch 的 checkpoint 到 `checkpoints/epochs/`
* 训练结束后，再统一在验证集上扫描这些 epoch checkpoint
* 验证集 `rank_ic` 最优的那个 epoch 会被复制为 `best_model_by_rankic`
* 默认会在选完 `best_model_by_rankic` 后删除 `checkpoints/epochs/`，只保留最终两份 best checkpoint；如需保留全部 epoch 模型，设 `KRONOS_CLEANUP_EPOCH_CHECKPOINTS=false`
* 测试集只在训练完成后推理两次：一次 `cross-entropy`，一次 `rankIc`

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

1. `rank_ic`
2. `ic`
3. `rank_icir`
4. `icir`
5. `da / mae / rmse`

解释：

* `rank_ic` 是主排序指标
* `ic`、`rank_icir` 与 `icir` 用来补充线性相关性和稳定性
* `da / mae / rmse` 用来补充方向与误差表现

### 结果对比时的约束

正式比较前必须确认：

* 股票池一致
* 时间切分一致
* `sample_count / top_p / temperature` 一致
* `H` 一致

如果股票池不同，结论必须显式写成“不同股票池下的结果”，不能直接解释成模型结构差异。

## 8. 当前实现下的结论表述

当前 B/C 的更准确表述是：

* “在统一训练目标下，同时比较 `loss` 选模路径与 `rank_ic` 选模路径的下游金融指标差异”

而不是：

* “只看单一 checkpoint 就代表最终实验结论”

如果后续要升级为正式实验口径，再做两件事：

1. 明确最终报告使用 `best_model_by_rankic` 还是同时报告两条选模路径
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

* 同一模型重复评估，`rank_ic` 或 `ic` 有小幅波动

原因：

* 当前评估仍使用采样

处理：

* 固定 `sample_count / top_p / temperature`
* 使用统一脚本汇总多个实验结果，而不是手工逐个查看 `metrics.json`

## 10. 结果对比脚本

可直接使用：

```bash
conda activate kronos
cd /home/yzh/workspace/Kronos-0

python zlab/scripts/compare_eval_metrics.py \
  --root zlab/results/evaluations \
  --output-dir zlab/results/evaluation_comparisons
```

默认只比较 `pred_len=1`。如果要只生成 `pred_len=5` 的图，直接在命令最后加 `5`：

```bash
python zlab/scripts/compare_eval_metrics.py 5 \
  --root zlab/results/evaluations \
  --output-dir zlab/results/evaluation_comparisons
```

默认会递归读取 `zlab/results/evaluations/**/metrics.json`，导出：

* `metrics_summary_all.csv`
* `metrics_summary_selected.csv`
* `all_rank_ic1.png`
* `all_ic1.png`
* `all_rank_icir1.png`
* `all_icir1.png`
* `all_da1.png`
* `all_mae1.png`
* `all_rmse1.png`
* `selected_rank_ic1.png`
* `selected_ic1.png`
* `selected_rank_icir1.png`
* `selected_icir1.png`
* `selected_da1.png`
* `selected_mae1.png`
* `selected_rmse1.png`

其中：

* `all_*.png` 包含当前筛选条件下的全部实验
* `selected_*.png` 只保留 `group_a` 与“优秀的 B/C 实验”
* “优秀的 B/C 实验”定义为：某个 B/C 结果在任一指标下进入前二名，就会参与全部指标的子图比较
* 所有图都使用同一固定顺序，便于直接对照不同指标
* 图中的 `ic` 直接对应原始 `metrics.json` 里的 `ic` 字段

如果只想比较测试集：

```bash
python zlab/scripts/compare_eval_metrics.py \
  --root zlab/results/evaluations \
  --output-dir zlab/results/evaluation_comparisons_test \
  --splits test
```

* 比较接近的实验时，不要只凭单次结果下结论

### 4. `H=1` 模型在 `H=5` 上看起来也不错

解释：

* 这说明模型有一定跨 horizon 泛化
* 但不代表可以省掉正式的 `H=5` 训练

正式实验里，`H=5` 仍应单独训练。
