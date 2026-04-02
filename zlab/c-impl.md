# c 组实现说明

本文只记录 C 组的实现结构、代码边界和当前实现约束，不承担实验流程和训练指导职责。实验设计见 `zlab/lab.md`，执行方式见 `zlab/experiment_guide.md`。

## 1. 实现范围

当前 C 组链路由以下文件组成：

* `finetune/csv_data_preprocess_c.py`
* `finetune/dataset_c.py`
* `finetune/models/hourly_encoder.py`
* `finetune/models/hourly_fusion.py`
* `finetune/models/kronos_with_hourly.py`
* `finetune/train_predictor_c.py`
* `finetune/evaluate_c.py`
* `zlab/scripts/prepare_c_data.sh`
* `zlab/scripts/run_group_c.sh`

## 2. 设计边界

### 最小侵入

当前实现尽量不改：

* `model/kronos.py`
* `model/module.py`

而是在 `finetune/models/` 下新增小时线分支模块，再通过包装模型完成训练和推理。

### 双入口预处理

当前保留两个数据预处理入口：

* `csv_data_preprocess.py`：A/B 日线单流
* `csv_data_preprocess_c.py`：C 组日线 + 小时线双流

当前**不再抽公共 helper**，也不把两个入口强行合并。原因是：

* A/B 与 C 的输出 schema 不同
* C 组需要额外的小时线对齐与过滤逻辑
* 当前阶段优先保证入口清晰和实验稳定

## 3. 数据流

### 预处理输出

`csv_data_preprocess_c.py` 输出：

* `train_data.pkl`
* `val_data.pkl`
* `test_data.pkl`
* `metadata.json`
* `symbol_stats.csv`

其中每个 symbol 对应：

```python
{"daily": daily_df, "hourly": hourly_df}
```

### 数据集输出

`dataset_c.py` 返回四个张量：

* `x_daily`
* `x_stamp_d`
* `x_hourly`
* `x_stamp_h`

约束：

* 日线窗口与 B 组一致
* 小时线只取到日线 context 结束时点之前
* 不允许未来小时线泄露到当前样本

## 4. 模型结构

### `hourly_encoder.py`

职责：

* 把小时线 OHLCVA 序列编码成隐藏表示

当前实现：

* 线性输入投影
* 独立时间 embedding
* 轻量 Transformer blocks
* RMSNorm

### `hourly_fusion.py`

职责：

* 用 cross-attention 将小时线隐藏状态注入日线主干 hidden states

当前实现：

* pre-norm
* daily hidden 作为 query
* hourly hidden 作为 key/value
* fusion 内部使用普通 cross-attention，不复用主干的 RoPE attention
* residual 输出

原因：

* 日线 query 长度与小时线 memory 长度天然不同
* 主干里带 RoPE 的 cross-attention 当前实现要求 `q_len == k_len`
* 在 C 组 fusion 里去掉 RoPE 更符合“异构序列条件注入”的语义，也避免长度不一致时报 shape 错误

### `kronos_with_hourly.py`

职责：

* 组合预训练 Kronos predictor 与小时线分支

插入位置：

* 在 predictor 冻结层与可训练层的分界处插入 fusion

当前结构：

1. 日线 token embedding + time embedding
2. predictor 下层 frozen transformer
3. hourly encoder
4. fusion
5. predictor 上层 trainable transformer
6. `norm / dep_layer / head`

## 5. 训练逻辑

`train_predictor_c.py` 的核心行为：

* tokenizer 冻结
* predictor 前 `2/3` 冻结
* predictor 后 `1/3` 训练
* `hourly_encoder / fusion` 训练
* 主损失仍为 `future-only token CE`
* 每个 epoch 只计算 `val loss`
* 每个 epoch 额外保存一份 epoch checkpoint
* `best_model_by_loss` 在线按 `val loss` 更新
* `best_model_by_rankic` 在训练结束后统一扫验证集选出

当前仍然与 B 组保持同一训练协议，但把 `RankIC` 选模挪到了训练后统一执行，避免训练期逐轮推理。

## 6. 评估逻辑

`evaluate_c.py` 与 `evaluate_ab.py` 保持同一输出格式，区别只有两点：

* 推理时额外输入小时线窗口
* 小时线 hidden 在自回归推理时先编码一次，再在各步复用

当前没有把 C 组推理折回 `evaluate_ab.py`，而是保留独立入口，后续需要人工确保 A/B/C 指标定义同步。

## 7. 当前实现修复

### 2026-04-01

已修复以下问题：

* `dataset_c.py` 不再错误使用 `pd.os.environ`，改为 `os.getenv(...)`
* `dataset.py` 和 `dataset_c.py` 的验证集采样改为按 `idx` 取样，避免验证阶段随机抽样导致的不稳定
* `train_predictor_c.py` 中原来失效的 `setdefault(...)` 默认配置逻辑已改正，只有在环境变量未显式设置时才回落到 C 组默认值
* `hourly_fusion.py` 不再复用带 RoPE 的 cross-attention，改为普通 cross-attention，修复 `q_len != k_len` 时的 shape mismatch
* `train_predictor_c.py` 现在改为训练期只看 `val loss`，训练结束后再统一扫描 epoch checkpoint 生成 `best_model_by_rankic`

## 8. 当前实现约束

这些不是代码 bug，但属于当前实现边界：

* C 组股票池可能因为小时线覆盖不足而缩小
* 当前 B/C 的训练目标仍是 token CE，而不是直接优化金融指标
* C 组不仅引入了小时线信息，也引入了额外参数容量，因此结构提升不能自动解释成“纯由小时线信息带来的增益”

## 9. 后续实现方向

如果后面继续加强 C 组，而不改变整体路线，优先级建议如下：

1. 保持与 B 组一致的评估口径
2. 先完成 B/C 在同一股票池上的正式比较
3. 再考虑按 `val RankIC` 选模
4. 如需增强实验识别，再增加 `shuffled-hourly` 或 `masked-hourly` 消融
