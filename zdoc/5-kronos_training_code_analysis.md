# Kronos 训练代码完备性与 Loss 计算分析

## 一、训练代码完备性：结论

**该仓库拥有完整的训练脚手架代码，而非仅有模型骨架。** Tokenizer 和 Predictor（Decoder）两部分均有详尽的训练脚本，支持 DDP 分布式训练、梯度累积、学习率调度、 checkpoint 保存、Comet/TensorBoard 日志记录等完整功能。

### 1.1 Tokenizer 训练代码

| 文件 | 用途 |
| :--- | :--- |
| `finetune/train_tokenizer.py` | 主训练脚本，支持 `torchrun` DDP 多卡训练 |
| `finetune_csv/finetune_tokenizer.py` | 备选单卡/CPU 训练路径，使用 CSV 数据源 |
| `finetune_csv/train_sequential.py` | 编排器，顺序执行 Tokenizer → Predictor 训练 |

`train_tokenizer.py` 包含的完整训练设施：

- **DDP 初始化** (`setup_ddp`)：自动检测 rank/world_size/local_rank
- **数据加载** (`create_dataloaders`)：`DistributedSampler` + `DataLoader`，支持 pin_memory
- **优化器**：`AdamW`，学习率由 `config.tokenizer_learning_rate` 控制（默认 2e-4）
- **调度器**：`OneCycleLR`，pct_start=0.03，div_factor=10
- **梯度累积**：支持 `accumulation_steps` 模拟更大 batch size
- **梯度裁剪**：`clip_grad_norm_` max_norm=2.0
- **训练循环**：epoch 级别 + batch 级别双层循环，每个 epoch 设置不同的 sampler seed
- **验证循环**：`torch.no_grad()` 下计算 validation MSE，跨卡 all_reduce 汇总
- **Checkpoint**：验证 loss 最优时保存 best_model（`save_pretrained`）
- **日志**：Comet ML + TensorBoard 双重日志（`experiment_logger.py`）

### 1.2 Predictor（Decoder）训练代码

| 文件 | 用途 |
| :--- | :--- |
| `finetune/train_predictor.py` | 主训练脚本，支持 `torchrun` DDP 多卡训练 |
| `finetune_csv/finetune_base_model.py` | 备选单卡/CPU 训练路径，使用 CSV 数据源 |

`train_predictor.py` 包含的完整训练设施：

- **DDP 初始化**：同 Tokenizer
- **数据加载**：同 Tokenizer，但 DataLoader 同时返回 `batch_x` 和 `batch_x_stamp`
- **On-the-fly Tokenization**：每个 batch 使用冻结的 Tokenizer 实时编码（`torch.no_grad()` 下）
- **自回归错位**：`token_in = [:, :-1]`, `token_out = [:, 1:]` 构造 (input, target) 对
- **前向传播**：`model(token_in[0], token_in[1], batch_x_stamp)` → `(s1_logits, s2_logits)`
- **Loss 计算**：`compute_token_loss()` 支持 full-sequence 和 future-only 两种模式
- **优化器**：`AdamW`，学习率由 `config.predictor_learning_rate` 控制（默认 4e-5）
- **调度器**：`OneCycleLR`，同 Tokenizer
- **梯度裁剪**：`clip_grad_norm_` max_norm=3.0
- **冻结策略**：`apply_predictor_finetune_strategy()` 支持冻结底层 Transformer 层、冻结 Embedding 等 A/B 实验配置
- **验证 + Checkpoint**：同 Tokenizer 模式

### 1.3 下游封装（zlab）

`zlab/kronos/finetune.py` 是对上述训练脚本的**环境变量 + subprocess 封装**：

- 通过环境变量注入所有配置参数（`KRONOS_*` 前缀）
- 调用 `torchrun --standalone --nproc_per_node=N train_tokenizer.py` / `train_predictor.py`
- 自动解析 split metadata 确定 `context_length` 和 `horizon`
- 训练后自动调用 evaluation + backtest 流程

这意味着该仓库的代码是**可直接训练的**，而非仅提供模型定义供参考。

---

## 二、Tokenizer 训练 Loss 计算

### 2.1 Loss 定义位置

核心代码在 `finetune/train_tokenizer.py` L136-143：

```python
zs, bsq_loss, _, _ = model(batch_x)          # KronosTokenizer.forward()
z_pre, z = zs

recon_loss_pre = F.mse_loss(z_pre, batch_x)   # S1-only 重建 loss
recon_loss_all = F.mse_loss(z, batch_x)       # Full codebook 重建 loss
recon_loss = recon_loss_pre + recon_loss_all
loss = (recon_loss + bsq_loss) / 2
```

### 2.2 各分量详解

#### (1) `recon_loss_pre` — S1 粗粒度重建 Loss

```
MSE( z_pre, batch_x )
```

- `z_pre` 是仅使用 S1 比特（前 `s1_bits` 位，即粗粒度 token）解码得到的重建结果
- 度量：粗粒度 token 保留了多少原始信息

#### (2) `recon_loss_all` — Full Codebook 重建 Loss

```
MSE( z, batch_x )
```

- `z` 是使用完整 codebook（`s1_bits + s2_bits` 共 20 位）解码得到的重建结果
- 度量：完整量化重建的精度

#### (3) `bsq_loss` — Binary Spherical Quantization Loss

`bsq_loss` 来自 `model/module.py` 中的 `BinarySphericalQuantizer.forward()` L124-126：

```python
commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))
# ...
bsq_loss = commit_loss + self.zeta * entropy_penalty / self.inv_temperature
```

包含两个子项：

**a) Commit Loss**（承诺损失）：

$$\mathcal{L}_{commit} = \beta \cdot \frac{1}{N} \sum_i \| \text{sg}[z_q] - z \|^2$$

- `sg[·]` 是 stop-gradient 操作（`.detach()`）
- 作用：鼓励编码器输出 $z$ 靠近量化后的值 $z_q$，防止编码器输出无限增长
- 超参 $\beta$：默认 0.05

**b) Entropy Penalty**（熵正则）：

$$\mathcal{L}_{entropy} = \gamma_0 \cdot H_{per\_sample} - \gamma \cdot H_{codebook}$$

- $H_{per\_sample}$：每个样本的编码熵，越大表示单个样本的编码越分散（不趋于坍缩）
- $H_{codebook}$：codebook 整体的使用熵，越大表示 codebook 利用率越高
- 设计意图：**鼓励 codebook 被均匀使用**（最大化 $H_{codebook}$），同时**防止单个样本坍缩到少数 code**（最大化 $H_{per\_sample}$）
- 超参：$\gamma_0=1.0$（persample entropy 权重），$\gamma=1.1$（codebook entropy 权重），$\zeta=0.05$（entropy penalty 总权重）

### 2.3 Tokenizer 总 Loss 公式

$$\boxed{\mathcal{L}_{tokenizer} = \frac{1}{2}\Big( \underbrace{\text{MSE}(z_{pre}, x) + \text{MSE}(z, x)}_{recon\_loss} + \underbrace{\mathcal{L}_{commit} + \zeta \cdot \mathcal{L}_{entropy}}_{bsq\_loss} \Big)}$$

等权平均的设计假设重建质量与量化正则化同等重要。

### 2.4 验证 Loss

验证阶段仅计算重建 MSE（不含 BSQ loss）：

```python
# train_tokenizer.py L182
val_loss_item = F.mse_loss(z, ori_batch_x)
```

### 2.5 Tokenizer 前向传播流程（Loss 相关部分）

```
batch_x [B, T, 6]
  → embed (Linear: 6→256)
  → Encoder Transformer × (n_enc_layers - 1)
  → quant_embed (Linear: 256→20)
  → L2 Normalize
  → BSQ (二值化 + STE + 缩放)
     ├→ bsq_loss (commit + entropy)
     └→ quantized [B, T, 20]
        ├→ quantized[:, :, :s1_bits] → post_quant_embed_pre → Decoder → head → z_pre
        └→ quantized[:, :, :]         → post_quant_embed     → Decoder → head → z
```

---

## 三、Predictor（Decoder）训练 Loss 计算

### 3.1 Loss 定义位置

核心在 `model/module.py` L494-507 的 `DualHead.compute_loss()`：

```python
def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
    if padding_mask is not None:
        valid_mask = (padding_mask == 0)
        s1_logits = s1_logits[valid_mask]
        # ... (同上处理 s2_logits, s1_targets, s2_targets)
        ce_s1 = F.cross_entropy(s1_logits, s1_targets)
        ce_s2 = F.cross_entropy(s2_logits, s2_targets)
    else:
        ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
        ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
    ce_loss = (ce_s1 + ce_s2) / 2
    return ce_loss, ce_s1, ce_s2
```

### 3.2 各分量详解

#### (1) S1 Cross-Entropy Loss

$$\mathcal{L}_{CE}^{(s1)} = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log \frac{e^{z_{target}^{(b,t)}}}{\sum_{j=1}^{V_1} e^{z_j^{(b,t)}}}$$

- $V_1 = 2^{s1\_bits} = 1024$：S1 词表大小
- Logits shape: `[B, T, 1024]` → reshape 为 `[B·T, 1024]`
- Targets shape: `[B, T]` → reshape 为 `[B·T]`
- Reduction: **mean**（对所有位置取平均）

#### (2) S2 Cross-Entropy Loss

$$\mathcal{L}_{CE}^{(s2)} = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log \frac{e^{z_{target}^{(b,t)}}}{\sum_{j=1}^{V_2} e^{z_j^{(b,t)}}}$$

- $V_2 = 2^{s2\_bits} = 1024$：S2 词表大小
- 计算方式与 S1 完全一致，只是 S2 logits 条件化于 S1 信息

### 3.3 Predictor 总 Loss

$$\boxed{\mathcal{L}_{predictor} = \frac{1}{2}\left(\mathcal{L}_{CE}^{(s1)} + \mathcal{L}_{CE}^{(s2)}\right)}$$

S1 和 S2 等权平均，假设粗粒度和细粒度 token 预测同等重要。

### 3.4 两种 Loss 范围模式

在 `finetune/train_predictor.py` L81-98 的 `compute_token_loss()` 中：

#### 模式 1：Full-sequence Loss（默认）

对整个序列所有 100 个位置（`window-1`）计算 next-token prediction loss。

#### 模式 2：Future-only Loss

仅对预测窗口的 10 个位置计算 loss。

```python
start_idx = config['lookback_window'] - 1   # = 89
end_idx = start_idx + config['predict_window']  # = 99
# 仅对 logits/targets[:, 89:99, :] 计算 loss
```

启用方式：`--future-only-loss` 参数或 `KRONOS_FUTURE_ONLY_LOSS=true` 环境变量。

### 3.5 S2 条件化与梯度隔离

在 `model/kronos.py` L268-276：

```python
# 非 teacher-forcing 模式
s1_probs = F.softmax(s1_logits.detach(), dim=-1)   # .detach() 切断梯度
sample_s1_ids = torch.multinomial(s1_probs, ...)
sibling_embed = self.embedding.emb_s1(sample_s1_ids)

x2 = self.dep_layer(x, sibling_embed)    # Cross-Attention 融合
s2_logits = self.head.cond_forward(x2)    # Linear: d_model → vocab_s2
```

关键设计：
- **S1 logits 在传入 S2 路径前 `.detach()`**，因此 S2 loss 的梯度不会回传到 S1 head
- S1 head 仅通过 $\mathcal{L}_{CE}^{(s1)}$ 更新
- S2 head 仅通过 $\mathcal{L}_{CE}^{(s2)}$ 更新
- 两个 head 的训练信号**相互独立**，互不干扰

### 3.6 Predictor 完整数据流

```
batch_x [B, 101, 6]
  → Tokenizer.encode(half=True)    ← 冻结，no_grad
     ├→ token_seq_0 (S1) [B, 101]
     └→ token_seq_1 (S2) [B, 101]

自回归错位:
  token_in  = [S1[:, :-1], S2[:, :-1]]   → [B, 100]
  token_out = [S1[:, 1:],  S2[:, 1:]]    → [B, 100]

Kronos.forward(token_in, stamp):
  → HierarchicalEmbedding + TemporalEmbedding
  → Transformer × n_layers (causal mask)
  → RMSNorm
  → Head.s1_proj: s1_logits [B, 100, 1024]
  → s1_logits.detach() → sample → sibling_embed
  → DependencyAwareLayer (CrossAttn)
  → Head.s2_proj: s2_logits [B, 100, 1024]

Loss:
  CE_s1 = CrossEntropy(s1_logits, s1_targets)
  CE_s2 = CrossEntropy(s2_logits, s2_targets)
  Loss   = (CE_s1 + CE_s2) / 2
```

---

## 四、两份 Loss 对比总结

| 维度 | Tokenizer Loss | Predictor Loss |
| :--- | :--- | :--- |
| **任务类型** | 连续值重建（回归） | 离散 Token 分类 |
| **Loss 类型** | MSE + BSQ 量化正则 | Cross-Entropy |
| **目标** | 学习编码/解码映射 | 学习 P(next_token \| history) |
| **S1/S2 权重** | 不适用（仅重建两个版本） | 等权平均 (0.5/0.5) |
| **额外正则** | Commit Loss + Entropy Penalty | 无（仅 Token Dropout） |
| **输入** | 原始连续值 `[B, T, 6]` | S1 Token IDs + S2 Token IDs |
| **Target** | 原始连续值 `[B, T, 6]` | 后移一位的 Token IDs |
| **训练范围** | 全窗口 | 全序列 / 仅未来窗口（可切换） |
| **梯度关系** | 无层级依赖 | S2 依赖 S1，但梯度通过 `.detach()` 隔离 |
| **验证指标** | MSE(z, x) | CE Loss（full-sequence 或 future-only） |

---

## 五、训练脚手架完备性清单

| 功能 | Tokenizer | Predictor |
| :--- | :---: | :---: |
| DDP 分布式训练 | ✓ | ✓ |
| 梯度累积 | ✓ | ✗（单步更新） |
| OneCycleLR 调度 | ✓ | ✓ |
| 梯度裁剪 | ✓ (max_norm=2.0) | ✓ (max_norm=3.0) |
| 分布式 Sampler | ✓ | ✓ |
| Reproducible 采样 (epoch seed) | ✓ | ✓ |
| Comet ML 日志 | ✓ | ✓ |
| TensorBoard 日志 | ✓ | ✓ |
| Best-model Checkpoint | ✓ | ✓ |
| 训练/验证 Split | ✓ | ✓ |
| 环境变量配置 | ✓ | ✓ |
| 命令行参数覆盖 | ✗ | ✓ (`argparse`) |
| 冻结策略 (A/B 实验) | ✗ | ✓ (freeze layers/embed) |
| Future-only Loss | ✗ | ✓ |
| 单卡/CPU 备选路径 | ✓ (`finetune_csv/`) | ✓ (`finetune_csv/`) |
| 顺序训练编排 | ✓ (`train_sequential.py`) | ✓ (`train_sequential.py`) |
| Subprocess 封装 (zlab) | ✓ | ✓ |
