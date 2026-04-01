# 实验设计

## 目标

本轮实验回答三个递进问题：

* A 组：Kronos 预训练模型在当前中国股票日线任务上，零样本能做到什么程度
* B 组：只做日线任务适配后，是否优于 A 组
* C 组：在 B 组基础上加入小时线辅流后，是否进一步优于 B 组

因此核心比较关系是：

* `A vs B`：日线微调是否有价值
* `B vs C`：小时线作为更细粒度上下文是否有增益

## 数据范围与规划

### 时间范围

当前实验数据范围为：

* `2025-06-01 ~ 2026-02-28`

推荐切分：

* Train：`2025-06-01 ~ 2025-11-30`
* Val：`2025-12-01 ~ 2025-12-31`
* Test：`2026-01-01 ~ 2026-02-28`

### 数据来源

* 日线：`zlab/data/daily/*.csv`
* 小时线：`zlab/data/hourly/*.csv`

### 资产池要求

正式比较必须满足：

* A/B/C 使用同一股票池
* 相同时间切分
* 相同历史窗口、预测窗口和采样设置

对 C 组来说，小时线覆盖不足会导致股票池缩小。因此 B/C 正式比较前，应以 `metadata.json` 中的 `symbols` 为准核对股票池是否一致。

## 任务定义

### 日线主任务

公共任务定义：

* 输入：历史日线 OHLCVA
* 输出：未来日线 OHLCVA 序列
* 最终评估对象：未来收益率，而不是 token loss 本身

默认记号：

* 日线历史窗口：`L_d = 20`
* 小时线历史窗口：`L_h = 25`，对应 `5` 个交易日、每天 `5` 根小时线
* 预测窗口：`H ∈ {1, 5}`

### A 组

* 零样本推理
* 不训练
* 直接在 `val/test` 上评估

### B 组

* 单流日线微调
* tokenizer 冻结
* predictor 前 `2/3` 冻结
* predictor 后 `1/3`、`norm`、`dep_layer`、`head` 训练

### C 组

* 日线主流 + 小时线辅流
* 主任务仍是未来日线 token 预测
* 小时线只作为条件输入，不单独设预测目标
* 除 B 组可训练部分外，新增 `hourly_encoder / fusion` 一并训练

## 样本构造

### A/B 组

每个样本包含：

* `x_daily`：最近 `L_d` 个交易日历史
* `x_timestamp_d`
* `y_timestamp_d`
* 训练阶段额外用真实未来 `H` 日构造目标段

### C 组

每个样本包含：

* `x_daily`
* `x_timestamp_d`
* `y_timestamp_d`
* `x_hourly`
* `x_timestamp_h`

关键约束：

* 小时线只能使用预测起点之前可观测到的部分
* 小时线最后一根 bar 不能越过日线 context 结束时点

## 正式实验口径

### 训练目标

B/C 的主训练目标保持一致：

* 主损失：`future-only token loss`
* 历史段只提供上下文
* loss 只对未来段累计

第一轮实验不引入额外辅助损失，不把 `IC / RankIC / 回测收益` 直接作为反向传播目标。

### 正式选模标准

正式实验建议：

* 主选模指标：验证集 `mean RankIC`
* 次选模指标：验证集 `future-token loss`

原因是最终实验目标是收益排序能力，不是 token 重建误差。

### 当前实现口径

当前仓库中的 B/C 实现仍然采用：

* 训练主损失：`future-only token loss`
* 每个 epoch 同时计算：`val loss` 与 `val mean RankIC`
* checkpoint 同时保存：`best_model_by_loss` 与 `best_model_by_rankic`

因此当前跑出的结果应理解为：

* 在统一训练目标下，同时比较 `loss` 选模路径与 `RankIC` 选模路径的下游金融指标

而不是：

* 只用单一 checkpoint 代表全部实验结论

## 统一评估规范

### 评估对象

评估阶段统一从生成结果中抽取未来收益率：

* `pred_return_h = pred_close_{t+h} / close_t - 1`
* `real_return_h = close_{t+h} / close_t - 1`

其中 `h=1` 或 `5`。

### 主指标

主指标用于回答横截面排序是否有效：

* `mean_rank_ic`
* `mean_ic`
* `rank_ic_ir`

### 辅指标

用于补充方向和幅度合理性：

* `direction_accuracy`
* `mae`
* `rmse`

### 收益代理指标

用于验证排序能力是否能转成选股收益：

* `topk_mean_return`
* `topk_cum_return`
* `long_short_topk_mean_return`
* `long_short_topk_cum_return`

### 统一比较约束

三组模型必须使用：

* 相同股票池
* 相同 `val/test` 时间段
* 相同 `L_d / L_h / H`
* 相同推理采样参数
* 相同 `sample_count`

## 训练与收敛判定

训练阶段先看两层信息：

* 优化是否正常：`train loss`、`val loss`
* 下游是否有意义：`val RankIC / IC / long-short`

一般判断：

* `train loss` 下降、`val loss` 也下降：优化正常
* `train loss` 继续下降，但 `val loss` 恶化：开始过拟合
* `val loss` 改善，但 `RankIC` 不改善：训练目标与最终目标出现错位

## 结果解释原则

正式结论建议遵循下面顺序：

1. 先看 `mean_rank_ic`
2. 再看 `mean_ic / rank_ic_ir`
3. 最后结合 `top-k / long-short` 判断收益可转化性

只有当一个模型同时满足：

* 排序指标不差
* 收益代理指标不差

才应认为整体更优。
