# 1. 总体实验目标

你这轮实验要回答三个递进问题：

### A 组：开源模型直接用，能做到什么程度
回答：
> Kronos 预训练模型在你的中国股票**日线任务**上，零训练直接推理效果如何？
### B 组：只做日线微调，能否比 A 更好
回答：
> 仅在你自己的 2025-06 到 2026-02 数据上做日线任务适配，是否能显著提升效果？

### C 组：在 B 的基础上加小时线辅流，是否进一步提升
回答：
> 对日线预测任务而言，小时线作为更细粒度上下文，是否能带来额外增益？
这个三组逻辑是完整的：
* **A vs B**：看任务适配是否值得做
* **B vs C**：看小时线辅流是否有用

# 2. 可用数据范围与切分建议

你现在给出的可用数据范围大致是：
* **2025-06 到 2026-02**
这段时间并不长，所以不适合做“重训练”，但适合做**轻量微调 + 明确时间切分实验**。

## 2.1 推荐时间切分
### 训练集
* **2025-06-01 ~ 2025-11-30**
### 验证集
* **2025-12-01 ~ 2025-12-31**
### 测试集
* **2026-01-01 ~ 2026-02-28**

# 3. 资产池选择建议

由于时间范围短，**不要只用单一指数**。建议使用：
* **30~50 只沪深300成分股**
    
* 要求：    
    * 2025-06 到 2026-02 有连续日线与小时线数据        
    * 无长期停牌        
    * 流动性较高
# 4. 三组实验的正式定义


## A 组：Kronos Daily-only Direct Inference

### 任务
* 输入：历史日线 OHLCVA
* 输出：未来日线预测
### 数据
* 不参与训练
* 直接在验证集、测试集上推理

### 推荐设置

* 历史窗口 $L_d = 20$
* 预测步长 $H_d = 1$ 和 $H_d = 5$
    
解释：
* $H_d=1$：下一交易日
* $H_d=5$：未来一周

### 推理方式
直接调用开源权重，不更新参数。
### 作用
给出“不开任何适配时”的下界基线。

## B 组：Kronos Daily-only Fine-tuning

### 任务
* 输入：历史日线 OHLCVA
* 输出：未来日线预测

### 数据
* Train：2025-06 ~ 2025-11
* Val：2025-12
* Test：2026-01 ~ 2026-02

### 参数更新建议
Kronos 两阶段里，Tokenizer 学的是 OHLCVA 到离散 token 的映射，而 Predictor 才真正结合时间 embedding 做自回归预测。你自己的代码研究也表明，Tokenizer 训练阶段基本不使用时间，而 Predictor 阶段会使用 `minute/hour/weekday/day/month` 这些时间特征。
4-kronos_multigranularity_times…
因此 B 组建议：
* **Tokenizer：冻结**
* **Predictor 下层：冻结**
* **Predictor 上层若干层：解冻**
* **输出头：训练**

### 推荐解冻范围
如果代码里 predictor 有很多层，优先：
* 解冻最后 **1/3** 的 block
* 冻结前 **2/3**

### 为什么这样做

因为：
* 你的训练数据不大
* 小数据下全量微调容易过拟合
* tokenizer 不该轻易改坏

## C 组：Kronos Daily-main + Hourly-aux Fine-tuning

### 任务
* 主任务：未来日线预测
* 主流输入：历史日线
* 辅流输入：历史小时线

### 参数更新建议
* **日线 tokenizer：冻结**
* **daily predictor 下层：冻结**
* **daily predictor 上层：解冻**
* **新增 hourly encoder：训练**
* **新增 cross-attention / fusion：训练**
* **输出头：训练**    

### 关键定义

这里小时线不是预测目标，而是**条件上下文**。  
这和你自己的开题写法是吻合的：主序列保持 Kronos 的离散生成范式，外部多源/细粒度信息作为连续辅流，通过跨注意力等方式注入。

# 5. 样本构造方式
## 5.1 A/B 组样本构造（日线单流）

每个样本包括：
* 历史日线窗口：最近 $L_d$ 个交易日
* 未来日线目标：接下来的 $H_d$ 个交易日

### 推荐窗口
* $$L_d = 20$$
* $$H_d = 1, 5$$

### 时间戳
Kronos 推理时不仅需要历史 `x_timestamp`，还需要未来 `y_timestamp`，因为未来时间戳会被拆成时间特征并参与生成时的时间 embedding，不只是为了对齐索引。
所以 A/B 组都要构造：
* `x_timestamp_d`
* `y_timestamp_d`
且二者都必须保持**日频、同一交易日历**。你自己已经分析过，虽然代码接口上可以让 `x_timestamp` 和 `y_timestamp` 用不同粒度，但这通常不符合训练假设，不建议这样做。


## 5.2 C 组样本构造（日线主流 + 小时线辅流）

每个样本包括两部分输入：

### 主流（日线）

* 最近 $L_d$ 个已结束交易日的日线 OHLCVA
    
* `x_timestamp_d`
    
* 未来 $H_d$ 个交易日的 `y_timestamp_d`
    

### 辅流（小时线）

* 当前预测起点之前最近 $L_h$ 个小时 bar
    
* `x_timestamp_h`
    

### 重要约束

小时线只能使用**预测起点之前可观测到的部分**，不能把未来信息泄漏进去。

例如：

* 若当前样本的预测起点是 2026-01-15 收盘后
    
* 那么小时线辅流最多用到 2026-01-15 收盘前的小时数据
    
* 不能用 2026-01-16 的任何小时线
    

* * *

## 5.3 小时线窗口长度建议

由于小时线只是辅流，不建议太长。

### 推荐

* $L_h = 32$ 或 64
    

A 股若按 4 个小时 bar / 交易日估算：

* 32 小时 ≈ 8 个交易日
    
* 64 小时 ≈ 16 个交易日
    

### 建议优先跑

* 第一版：`L_h = 32`
    
* 第二版：再试 `L_h = 64`
    

因为小时线越长，cross-attn 成本越高，而且当前数据总量并不大。

* * *

# 6. 训练目标、反向传播指标与收敛标准

这一节要明确区分三件事：

* **训练时真正参与反向传播的目标**
* **验证集上用于判断收敛和选模的指标**
* **测试集上用于比较 A/B/C 三组的最终指标**

这三者不必完全相同。  
对当前这个小样本日线任务，最稳妥的做法是：
* 训练时仍然使用 **Kronos 原生 token 自回归目标**
* 验证/测试时用 **金融任务指标** 进行比较

## 6.1 A 组：不训练

A 组是零训练基线：
* 不更新任何参数
* 直接用开源 tokenizer + predictor 推理
* 只做验证集/测试集评估

因此 A 组没有“反向传播指标”和“收敛问题”。

## 6.2 B/C 组训练时应该优化什么

第一轮实验**不要**直接把：
* IC
* RankIC
* 回测收益

作为主反向传播目标。原因是：
* 这些指标噪声大
* 不够平滑，训练不稳定
* 你的样本期较短，直接优化交易指标很容易过拟合
* Kronos 的预训练范式本身就是 token 级自回归建模

所以 B/C 组训练时，建议保留 Kronos 原始范式：
* 输入：历史日线 `x_daily`
* 条件时间：`x_timestamp_d`
* 真实未来：`y_daily`
* 未来时间：`y_timestamp_d`    
* 目标：未来日线 token 序列的自回归建模
即继续优化：
$$p(b_{T+1:T+H}\mid b_{1:T})$$

### 具体到 loss 的定义

设：
* 历史窗口长度为 $T=L_d$
* 预测长度为 $H=H_d$
* 历史对应 token 为 $b_{1:T}$
* 未来真实 token 为 $b_{T+1:T+H}$

建议训练时采用 **teacher forcing**：
* 用真实 `y_daily` 构造未来 token
* 把历史段和未来段拼接后送入 predictor    
* 但 **loss 只在未来段 $T+1:T+H$ 上累计**

也就是：
* 历史段只是提供上下文
* 反向传播的主损失应该对齐“未来预测”
* 不建议把“历史重建误差”作为主损失的一部分

### 反向传播的主指标
建议用：
* **Future-token CE / NLL**

如果保留 Kronos 的双头离散结构，则主损失就是：
* `L = L_s1_future + L_s2_future`
其中：
* `L_s1_future`：未来段 coarse token 的交叉熵
* `L_s2_future`：未来段 fine token 的交叉熵
    
### 是否加入辅助损失
第一轮实验建议：
* **不加辅助损失**
也就是只用 token loss 跑通 A/B/C 基线。等基线稳定后，再考虑：
* future return 的 MSE / MAE
* direction 的 BCE / focal loss

如果后续要加，也应只作为小权重辅助项，例如：
$$L = L_{token} + \lambda L_{return} + \mu L_{dir}, \quad \lambda,\mu \ll 1$$
但不建议把它作为第一轮主方案。

## 6.3 B 组怎么训练

B 组建议采用“轻量任务适配”，训练策略如下：
* tokenizer：冻结
* predictor 前 2/3：冻结
* predictor 后 1/3：解冻
* output head：训练

训练流程：

1. 用历史日线窗口 `x_daily` 和未来日线 `y_daily` 构造样本  
2. tokenizer 只做前向编码，不更新参数  
3. predictor 用 teacher forcing 预测未来 token  
4. loss 只对未来段 token 计算  
5. 每个 epoch 在验证集做一次生成式评估，记录金融指标  

## 6.4 C 组怎么训练

C 组和 B 组的**主损失保持一致**：
* 主任务仍然是未来日线 token 预测
* 小时线只是条件输入，不是预测目标

因此 C 组训练时建议：
* 日线 future-token loss 仍是唯一主损失
* 不单独给小时线设重建损失
* 小时线 encoder / fusion 的梯度，来自主任务 loss 反传

这能保证：
* B/C 之间可公平比较
* C 组的增益确实来自“小时线条件信息”，而不是来自多加了一个训练目标

## 6.5 如何判断训练收敛

这里建议区分：
* **优化监控指标**
* **选模指标**

### 优化监控指标

用于看训练是否正常进行：
* `train_future_token_loss`
* `val_future_token_loss`

如果这两个指标不下降，说明模型还没学到；  
如果 `train_loss` 持续下降但 `val_loss` 开始恶化，说明开始过拟合。

### 选模指标

验证集上建议额外计算：
* `RankIC@1`
* `IC@1`
* `RankIC@5`
* `IC@5`
* `Direction Accuracy@1/@5`

然后用以下规则选 best checkpoint：

1. **主规则：验证集 mean RankIC 最高**
2. **次规则：若 mean RankIC 很接近，再选 val future-token loss 更低的模型**

推荐定义：
* `mean RankIC = (RankIC@1 + RankIC@5) / 2`

如果你第一轮只跑一个 horizon，那么就直接用对应 horizon 的 `RankIC`。

### Early stopping 建议

* patience：`3~5`
* 至少每个 epoch 在验证集做一次完整评估
* 如果验证集 `mean RankIC` 连续 `3~5` 次无提升，则停
* 如果 `val_future_token_loss` 明显恶化，同时 `RankIC` 也不提升，也应停
    
### 一个很重要的实现口径

当前很多 demo 代码容易默认成：
* 用整段 shifted token loss 训练
* 用 `val_loss` 存 best model

这个做法适合“先跑通”，但不适合作为你这轮 A/B/C 的**正式实验口径**。  
正式实验建议写清楚：
* 训练主损失：**future-only token loss**
* 选模主指标：**validation mean RankIC**

# 7. A/B/C 三组的统一测试评估规范

测试阶段的目标不是再看 token loss，而是看：
* 预测是否能对真实未来收益进行排序
* 预测是否有方向判断能力
* 这些预测能否转化为稳定的选股收益

## 7.1 三组模型如何统一评估

三组都必须在**同一套测试协议**下比较：

* 相同资产池    
* 相同测试时间段
* 相同日线历史窗口 `L_d`
* 相同 horizon 设置 `H_d=1`、`H_d=5`
* 相同未来交易日历构造方式
* 相同采样策略和 `sample_count`

建议：
* 正式主表统一用 `sample_count = 5`
* A 组额外做一次 `N=1/5/10` 的消融

这样最终 A/B/C 的差异主要来自：
* A：不开适配
* B：只做日线适配
* C：日线适配 + 小时线条件增强

## 7.2 测试时先从生成结果中抽取什么目标

对每个测试样本，模型会生成未来 $H_d$ 个交易日的日线。  
从这些生成结果中，统一抽取以下金融目标：

### Horizon = 1

* `pred_return_1 = (pred_close_{t+1} - close_t) / close_t`
* `real_return_1 = (close_{t+1} - close_t) / close_t`

### Horizon = 5

* `pred_return_5 = (pred_close_{t+5} - close_t) / close_t`
* `real_return_5 = (close_{t+5} - close_t) / close_t`

### 方向标签

* `sign(pred_return_h)`
* `sign(real_return_h)`

因此：
* **测试的核心对象应当是 future return**
* 不是原始 close 误差本身    
* 更不是 token loss 本身

## 7.3 主评估指标：横截面预测能力

这一类指标用于回答：  
模型能否在同一天把“未来涨得更好”和“未来涨得更差”的股票区分开。

### 主指标

* **RankIC@1**
* **RankIC@5**
* **IC@1**
* **IC@5**

建议主排序如下：
1. `mean RankIC`
2. `mean IC`

其中：
* `mean RankIC = (RankIC@1 + RankIC@5) / 2`
* `mean IC = (IC@1 + IC@5) / 2`
    

如果篇幅允许，建议同时报告：

* RankIC 的均值
* RankIC 的标准差
* IC 的均值
* IC 的标准差

因为这能反映信号稳定性。

## 7.4 辅指标：点预测和方向判断能力

这类指标用于回答：  
模型生成出来的未来路径，是否至少在方向和幅度上大致合理。
### 辅指标

* `Direction Accuracy@1`
* `Direction Accuracy@5`
* `MAE@1`
* `MAE@5`
* `RMSE@1`
* `RMSE@5`

说明：

* `MAE/RMSE` 都建议对 **future return** 计算
* 不建议把 OHLCVA 全字段逐元素误差作为主实验指标

因为你的任务本质是金融预测，不是图像式重建。

## 7.5 下游指标：统一回测

在统计预测指标之外，再做一层统一的策略回测。  
回测的作用是验证：
* 排序能力是否能转化成实际选股收益
* B 是否真的优于 A
* C 是否真的优于 B
    

### 回测原则

三组必须保持：

* 相同信号构造方式
* 相同调仓频率
* 相同 Top-K 参数
* 相同手续费 / 滑点设置
* 相同 benchmark

### 回测信号建议

建议主回测统一使用：
* `pred_return_1` 作为日频选股 score

理由是：

* `H_d=1` 的信号定义最清楚
* 与日频调仓最自然对应
* A/B/C 更容易公平对比

而 `H_d=5` 更适合：
* 作为统计预测补充指标
* 或额外做一个低频调仓的附录实验

### 回测需要报告的指标

至少报告：

* `excess return with cost`
* `annualized return`
* `information ratio / Sharpe`
* `max drawdown`

如果工程上方便，再补充：
* turnover
* win rate

## 7.6 最终实验表应该怎么比较

建议最终表格分两层：

### 表 1：统计预测指标

对 A/B/C 分别报告：
* RankIC@1
* RankIC@5
* IC@1
* IC@5
* DA@1
* DA@5
* MAE@1 / RMSE@1
* MAE@5 / RMSE@5

### 表 2：回测指标

对 A/B/C 分别报告：
* 年化收益
* 年化超额收益
* Information Ratio / Sharpe
* Max Drawdown
    

### 结论判定建议

只有当一个模型同时满足：

* `mean RankIC` 更高
* 回测超额收益不更差

才认为它“整体更优”。  
不要只凭单次回测曲线或者单个 MAE 指标下结论。

# 8. 参数建议
下面给你一组适合第一轮实验的默认参数。

## 8.1 数据参数

| 参数 | 建议值 |
| --- | --- |
| 资产数 | 30~50 |
| 日线窗口 $L_d$ | 20 |
| 小时线窗口 $L_h$ | 32 |
| 预测 horizon $H_d$ | 1, 5 |
| 日线频率 | 1D |
| 小时线频率 | 1H |

## 8.2 微调参数（B/C）

| 参数 | 建议值 |
| --- | --- |
| Optimizer | AdamW |
| learning rate | 1e-4 ~ 5e-5 |
| weight decay | 1e-2 |
| batch size | 16 或 32 |
| epochs | 10~20 |
| early stopping | patience = 3~5，主监控 val mean RankIC，次监控 val future-token loss |
| scheduler | cosine 或 linear warmup |

### 更细建议

#### B 组

* lr：`5e-5`
* epoch：`10`
* batch size：`32`
* 主反向传播目标：`future-token CE`
* best checkpoint：按验证集 `mean RankIC` 选
    

#### C 组

* daily predictor 上层：`5e-5`
* 新增 hourly encoder / fusion：`1e-4`
* epoch：`15`
* batch size：`16` 或 `32`

因为 C 组新增模块是随机初始化的，通常需要比预训练部分更大的学习率。

* * *

## 8.3 冻结策略建议

### B 组

* tokenizer：冻结
* predictor 前 2/3 层：冻结
* predictor 后 1/3 层：解冻
* lm head / output head：解冻

### C 组

* tokenizer：冻结
* predictor 前 2/3 层：冻结
* predictor 后 1/3 层：解冻
* hourly encoder：训练
* cross-attn / fusion：训练
* output head：训练

* * *

# 9. 代码修改建议
## 9.1 A 组代码修改建议：基本不改模型，只加推理脚本

### 需要做的事

1. 写一个 `run_daily_inference.py`
    
2. 输入：
    
    * 日线 CSV
        
    * `x_timestamp_d`
        
    * `y_timestamp_d`
        
3. 调用开源 Kronos 直接推理
    
4. 保存结果并计算统一指标：`RankIC / IC / DA / MAE / RMSE + backtest`
    

### 关键点

* `y_timestamp` 可以自己构造，不需要真实未来价格；你之前的代码分析已经说明，线上推理需要的是未来时间位置，而不是未来标签值。
    
    4-kronos_multigranularity_times…
    
* 对于 A 股日线，直接按交易日历生成未来交易日 `y_timestamp`
    

* * *

## 9.2 B 组代码修改建议：单流日线微调

### 数据集类

新增或改造：

* `DailyFinetuneDataset`
    

输出：

* `x_daily`
* `x_timestamp_d`
* `y_daily`
* `y_timestamp_d`

### 训练脚本

新增：

* `finetune_daily_predictor.py`
    

功能：
* 加载开源 tokenizer + predictor
* 冻结 tokenizer
* 部分解冻 predictor
* 用 teacher forcing 训练 `future-only next-token loss`
* 每个 epoch 记录验证集 `RankIC / IC`
* 按验证集 `mean RankIC` 保存 best model
    

### 最小改动原则

尽量保持原 predictor 训练流程不变，只改：

* 数据来源为你的 A 股日线
* 参数冻结策略
* 配置文件
    

* * *

## 9.3 C 组代码修改建议：新增 hourly encoder + 条件化 decoder

这是主要改动点。

### 新增模块 1：小时线 encoder

例如：

* `models/hourly_encoder.py`

输入：

* `x_hourly`
* `x_timestamp_h`

输出：

* `z_hourly` 或 hourly hidden states
    

推荐做法：

* 轻量 Transformer encoder
* 或 TSMixer / TemporalConv
* 不要太重

### 新增模块 2：fusion / cross-attn

例如：

* `models/daily_hourly_fusion.py`

推荐结构：

* daily predictor hidden state 作为 query
* hourly encoder outputs 作为 key/value
    
即：

$$H_d' = \text{CrossAttn}(Q=H_d, K=H_h, V=H_h)$$

然后再送入后续 predictor block 或 output head。

### 最小侵入式改法

不要重写整个 decoder。  
更合理的方式是：

* 在 predictor 的后 1~2 个 block 之间插入 cross-attn
* 或在 predictor 输出前做一次 fusion

这样更稳。

* * *

## 9.4 C 组数据集类设计

新增：

* `DailyHourlyFinetuneDataset`
    

输出字段建议：

* `x_daily`
* `x_timestamp_d`
* `y_daily`
* `y_timestamp_d`
* `x_hourly`
* `x_timestamp_h`

注意：

* 小时线没有 `y_hourly`
* 因为它不是预测目标，只是条件
    

* * *

# 10. 推荐的代码目录组织

你可以按下面这样改，最清晰，无侵入：

Kronos-0/zlab/
  data/  
    daily/  
    hourly/  
  dataloader/  
    daily_dataset.py  
    daily_hourly_dataset.py  
  models/  
    kronos_wrapper.py  
    hourly_encoder.py  
    fusion.py  
  scripts/  
    run_daily_inference.py  
    finetune_daily.py  
    finetune_daily_hourly.py  
  configs/  
    daily_infer.yaml  
    daily_finetune.yaml  
    daily_hourly_finetune.yaml
