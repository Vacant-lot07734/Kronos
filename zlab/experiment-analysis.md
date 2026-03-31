# A/B 实验结果分析与后续建议

## 1. 实验总览

### 已完成的实验

| 组别 | Horizon | 配置 | 训练参数 | 说明 |
|------|---------|------|----------|------|
| A | H=1 | zero-shot | — | 预训练模型直接推理 |
| A | H=5 | zero-shot | — | 预训练模型直接推理 |
| B (默认) | H=1 | lr=5e-5, bs=32, ep=10 | 解冻后1/3 + future-only loss | **当前主力配置** |
| B (lr2e5) | H=1 | lr=2e-5, bs=32, ep=10 | 同上 | 降低学习率 |
| B (lr1e5_e5) | H=1 | lr=1e-5, bs=32, ep=5 | 同上 | 更保守调参 |
| B (lr1e5_bs64) | H=1 | lr=1e-5, bs=64, ep=10 | 同上 | 小学习率+大batch |
| B | H=5 | eval-only, 复用H1默认模型 | — | H1训练模型在H5数据上评估 |

### 关键设计口径
- 资产池 ≈ 283 只（沪深300级别）
- 历史窗口 L=20, 冻结 tokenizer, 冻结 predictor 前 2/3
- 评估：36 个 test 交易日（H=1）/ 32 个 test 交易日（H=5）

---

## 2. H=1 测试集指标对比

| 指标 | A (zero-shot) | **B (默认 5e-5)** | B (lr=2e-5) | B (lr=1e-5, ep5) | B (lr=1e-5, bs64) |
|------|:---:|:---:|:---:|:---:|:---:|
| **mean_rank_ic** | 0.0590 | **0.0406** | 0.0252 | 0.0229 | 0.0230 |
| **mean_ic** | 0.0284 | **0.0465** | 0.0247 | 0.0148 | 0.0347 |
| rank_ic_ir | **0.539** | **0.410** | 0.287 | 0.247 | 0.267 |
| ic_ir | 0.193 | **0.369** | 0.199 | 0.124 | 0.271 |
| direction_accuracy | **0.513** | 0.506 | 0.506 | 0.506 | 0.504 |
| mae | **0.0178** | 0.0180 | 0.0181 | 0.0182 | 0.0182 |
| top10_mean_return | 0.0106 | **0.0150** | 0.0107 | 0.0092 | 0.0138 |
| top10_cum_return | 0.433 | **0.679** | 0.435 | 0.366 | 0.603 |
| long_short_mean_ret | 0.0030 | **0.0065** | 0.0021 | -0.0003 | 0.0023 |
| long_short_cum_ret | 0.103 | **0.253** | 0.070 | -0.018 | 0.078 |

### H=1 验证集指标对比

| 指标 | A | **B (默认)** | B (lr=2e-5) | B (lr=1e-5, ep5) | B (lr=1e-5, bs64) |
|------|:---:|:---:|:---:|:---:|:---:|
| mean_rank_ic | 0.0738 | **0.0737** | 0.0538 | -0.0218 | 0.0598 |
| mean_ic | 0.0472 | **0.0976** | 0.0770 | 0.0068 | 0.0750 |
| rank_ic_ir | 0.329 | **0.362** | 0.252 | -0.099 | 0.278 |

---

## 3. H=5 测试集指标对比

| 指标 | A (zero-shot) | B (复用H1默认模型) |
|------|:---:|:---:|
| **mean_rank_ic** | 0.0249 | **0.0333** |
| **mean_ic** | 0.0115 | **0.0235** |
| rank_ic_ir | 0.218 | **0.380** |
| direction_accuracy | 0.525 | **0.533** |
| top10_mean_return | 0.0146 | **0.0193** |
| top10_cum_return | 0.533 | **0.769** |
| long_short_mean_ret | 0.0008 | **0.0116** |
| long_short_cum_ret | -0.019 | **0.403** |

> [!IMPORTANT]
> H=5 的 B 组使用的是 H=1 训练的默认模型（eval-only 模式），没有单独训练 H=5 模型。

---

## 4. 分析与发现

### 4.1 核心结论：B 组微调价值明确，但调参方向需纠正

**B 组默认配置 (lr=5e-5) 是目前最优的微调方案**，在大多数指标上超越了 A 组和其他 B 组变体。但存在一个非常关键的异常：

> [!WARNING]
> **mean_rank_ic 倒退问题**：B 默认在 test 上的 mean_rank_ic (0.0406) 低于 A 组 (0.0590)，而根据 lab.md 的实验判定标准，mean_rank_ic 是最重要的排序指标。

这意味着按照实验方案中**最严格的判定标准**，"B 整体优于 A"这个结论目前**不成立**。

但从交易实用性角度看，B 默认在 ic_ir (0.369 vs 0.193)、top10 累计收益 (0.679 vs 0.433)、多空收益 (0.253 vs 0.103) 等维度大幅领先。说明微调确实有效改善了预测的**线性区分度和收益可转化性**，但在**秩相关（排序一致性）**上反而不如零样本。

这一矛盾的可能原因：

1. **过拟合导致排序压缩**：5e-5 学习率下 val_loss 显著恶化（从预训练水平 ~3.86 升至 4.14），说明模型对训练集过度适配。训练 loss 在后期降到 3.0–3.3（vs val 4.14），gap 达 0.8+，过拟合信号明确。过拟合后模型预测分布可能集中化，削弱了横截面排序的离散度
2. **IC vs RankIC 分离**：IC 提升而 RankIC 下降，暗示微调改善了对"极端股"（头尾）的区分，但对"中间段"的排序能力变差

### 4.2 训练动力学分析

| 配置 | Best Val Loss | 训练末期 Loss | Gap | 过拟合程度 |
|------|:---:|:---:|:---:|:---:|
| B 默认 (5e-5) | ~4.14 (未改善) | 3.0–3.3 | **0.8+** | 严重 |
| B (lr=2e-5) | ~3.92 | 3.5–3.7 | 0.2 | 轻微 |
| B (lr=1e-5, ep5) | ~3.87 | 3.4–3.9 | <0.1 | 最小 |
| B (lr=1e-5, bs64) | ~3.86 | 3.4–3.8 | <0.1 | 最小 |

关键观察：

- **lr=5e-5 过拟合最严重**，val loss 从未低于预训练初始水平，说明训练从一开始就在"错误方向"上走。但它偏偏在下游任务指标上表现最好——这说明 **val token loss 与金融指标之间存在去耦合**
- **lr=1e-5 系列训练最稳定**，val loss 单调下降并收敛。但下游金融指标却最差（lr1e5_e5 甚至 long_short 为负），说明**学习率过低，模型几乎没有有效适配**
- **lr=2e-5 处于中间**，过拟合轻微，金融指标也居中

### 4.3 调参趋势的"U型"特征

从调参结果看，存在一个 U 型关系：

```
学习率 ↗ :  token loss 过拟合 ↗,  但金融指标 ↗
学习率 ↘ :  token loss 稳定 ↗,    但金融指标 ↘（模型没学到东西）
```

这暗示：
- 当前的 future-only token CE 作为唯一训练目标，不够对齐金融排序任务
- 你需要的可能是一个"中等强度学习 + 更好的正则/目标"的组合

### 4.4 H=5 迁移评估

B 默认模型（H=1 训练）在 H=5 eval-only 上全面优于 A 组。特别是 long_short 从 -0.019（A，完全无效）变为 0.403（B，强信号），说明微调模型学到的特征具有跨 horizon 泛化能力。

---

## 5. 后续建议

### 5.1 短期：完善 B 组实验（优先级最高）

当前 B 组调参采样了 4 个点，但落在两个极端（5e-5 过拟合严重、1e-5 几乎没学习），**精细搜索中间区域是当务之急**。

#### 建议新增 B 组配置

| # | lr | epochs | bs | 理由 |
|---|:---:|:---:|:---:|------|
| B5 | **3e-5** | 10 | 32 | 5e-5 与 2e-5 之间的最优搜索点 |
| B6 | **5e-5** | **5** | 32 | 保持强学习率，但通过减少 epoch 控制过拟合 |
| B7 | **3e-5** | 15 | 32 | 稍低学习率 + 更多 epoch，看是否能收敛到更好位置 |

执行命令参考：

```bash
# B5: lr=3e-5, ep=10, bs=32
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="false"
export KRONOS_PREDICTOR_LR="3e-5"
export KRONOS_EPOCHS="10"
export KRONOS_BATCH_SIZE="32"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_b_predictor_lr3e5_e10_bs32"
export RESULT_NAME="group_b_lr3e5_e10_bs32"
zlab/scripts/run_group_b.sh 1

# B6: lr=5e-5, ep=5, bs=32
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="false"
export KRONOS_PREDICTOR_LR="5e-5"
export KRONOS_EPOCHS="5"
export KRONOS_BATCH_SIZE="32"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_b_predictor_lr5e5_e5_bs32"
export RESULT_NAME="group_b_lr5e5_e5_bs32"
zlab/scripts/run_group_b.sh 1

# B7: lr=3e-5, ep=15, bs=32
source zlab/ab_env.sh
export KRONOS_EVAL_ONLY="false"
export KRONOS_PREDICTOR_LR="3e-5"
export KRONOS_EPOCHS="15"
export KRONOS_BATCH_SIZE="32"
export KRONOS_PREDICTOR_SAVE_FOLDER_NAME="group_b_predictor_lr3e5_e15_bs32"
export RESULT_NAME="group_b_lr3e5_e15_bs32"
zlab/scripts/run_group_b.sh 1
```

#### 为什么推荐这三个

- **B5 (3e-5)**：位于默认 5e-5 和 lr2e5 之间。lr2e5 的 val loss gap 仅 0.2，金融指标已比 1e-5 好很多，3e-5 可能正好平衡过拟合和任务适配
- **B6 (5e-5, ep5)**：默认配置的过拟合主要在后半程加剧。5e-5 可能在前 5 个 epoch 就已经达到最佳点，提前停止可避免恶化
- **B7 (3e-5, ep15)**：如果 3e-5 学习较慢，给更多 epoch 让它充分收敛

### 5.2 中期：改进选模标准

当前选模用 val_loss 最低的 checkpoint。但实验数据已经表明 **val token loss 和金融指标之间不一致**。建议：

1. **每个 epoch 结束时同时记录 val RankIC**。现有 `evaluate_ab.py` 已经支持计算这些指标，只需要在训练循环中调用
2. **选模改为 val mean_rank_ic 最高**（与 lab.md 建议一致）
3. 具体改动建议：在 `train_predictor.py` 的每个 epoch 结束后，调用 `evaluate_ab.py` 的评估逻辑计算 val set 的 RankIC，用其代替（或辅助）val loss 作为 checkpoint 保存条件

> [!IMPORTANT]
> 这是当前实验流程中最值得改进的地方。当 token loss 和金融指标脱钩时，用 token loss 选模等于随机选模。建议在跑 B5/B6/B7 之前先完成此改动。

### 5.3 中期：H=5 独立训练

当前 H=5 的 B 组只是 eval-only（复用 H1 模型）。建议：

- 在确认 B 组最优 H=1 配置后，用同样配置**独立训练 H=5 模型**
- 即 `KRONOS_EVAL_ONLY=false` 跑 `run_group_b.sh 5`
- 目的：看专门为 H=5 训练的模型是否优于 H=1 模型跨 horizon 迁移

### 5.4 长期：C 组实验准备事项

在 A/B 结论稳定后再启动 C 组。C 组需要的准备：

1. **小时线数据准备**：`zlab/data/hourly/` 目录，与日线对齐的 A 股小时 bar 数据
2. **新增代码模块**（按 lab.md 9.3/9.4）：
   - `models/hourly_encoder.py`：轻量 Transformer encoder
   - `models/daily_hourly_fusion.py`：cross-attention 融合
   - `DailyHourlyFinetuneDataset`：双流数据集
3. **关键设计决策**：
   - hourly encoder 用新参数，lr 建议设为 daily predictor 的 2 倍（1e-4 vs 5e-5）
   - L_h = 32 优先（≈8 交易日的小时线）
   - 主损失保持 daily future-token CE，不给小时线加重建目标

### 5.5 可选：正则化手段

如果 B5/B6/B7 仍然存在过拟合问题，可以尝试：

| 手段 | 配置建议 | 适用场景 |
|------|----------|----------|
| Dropout | 在解冻的 predictor 层加 0.1–0.2 | 轻度过拟合 |
| Weight decay 调大 | 1e-2 → 5e-2 | 参数膨胀 |
| Label smoothing | 0.05–0.1 | token CE 过拟合 |
| 减少解冻比例 | train_last_ratio 从 0.333 改为 0.25 | 减少可训练参数 |
| Gradient clipping | max_norm=1.0 | 训练不稳定 |

---

## 6. 总结

### 当前结论

- **微调有效**：B 默认配置在 IC、ic_ir、top10 收益等实用维度大幅优于零样本
- **但 mean_rank_ic 倒退**：按最严格标准，"B ≥ A"尚未确立
- **根因是过拟合**：5e-5 训练过拟合严重，1e-5 又学不到东西。当前调参还没找到最优点
- **选模标准与评估标准脱钩**：用 val token loss 选模，但实际金融指标不与之单调相关

### 优先行动

1. **改进选模逻辑**：用 val RankIC 替代 val loss 选 checkpoint
2. **补充 B5/B6/B7 三组实验**：搜索 lr=3e-5 和早停区间
3. 确认最优 B 组配置后，跑 H=5 独立训练
4. 完成 A/B 结论后再启动 C 组

## gpt
Kronos 用交叉熵训练并不意味着交叉熵必须和 IC/RankIC 一致；交叉熵只是其生成式预训练/微调目标，而金融任务真正关心的是横截面排序与投资效用。你的实验恰恰说明，在金融微调中，训练代理目标与下游任务目标存在显著错位，因此“继续用 CE 训练，但用 RankIC 选模”是合理且比“最低 val loss 选模”更符合任务本质的
### Chronos 论文其实已经直接点出了一个关键问题：CE 不是 distance-aware

Chronos 明确说，categorical cross-entropy **不是 distance-aware**，它不会显式区分“预测到相邻 bin”和“预测到更远 bin”的差别；它本质上是 regression-via-classification。
在金融里，很多时候真正决定 IC/RankIC 的，不是 token 是否精确命中，而是：
-   方向对不对
-   强弱顺序对不对
-   极值样本有没有被拉开
但 CE 只在乎“正确 token 的概率有没有变大”。  
它不直接奖励“排序更合理”。
所以你看到：
-   `lr=1e-5`：CE 非常稳定下降，但下游最差
-   `lr=5e-5`：CE 甚至恶化，但金融指标最好


