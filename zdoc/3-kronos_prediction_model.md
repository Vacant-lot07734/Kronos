# Kronos Predictor 逐层深度解析

本文档逐步拆解 `Kronos` 预测模型的每一层操作，对照代码解释训练和推理的数据流差异。

代码位置：[kronos.py](file:///home/yzh/workspace/Kronos/model/kronos.py) `Kronos` 类，[module.py](file:///home/yzh/workspace/Kronos/model/module.py) 中的 `HierarchicalEmbedding`, `DependencyAwareLayer`, `DualHead` 等。

## 一、架构全貌

```
输入: (s1_ids, s2_ids, stamp) ∈ (B, T)

① HierarchicalEmbedding
   Emb_s1(s1) × √D  ─┐
                      ├─ concat → Linear(2D → D) → x ∈ (B, T, D)
   Emb_s2(s2) × √D  ─┘

② TemporalEmbedding
   x = x + Σ(minute_emb + hour_emb + weekday_emb + day_emb + month_emb)

③ TokenDropout
   x = Dropout(x)

④ Transformer Backbone
   for layer in transformer:
       x = TransformerBlock(x)          # Pre-LN, RoPE, SwiGLU, causal mask
   x = RMSNorm(x)                      # context ∈ (B, T, D)

⑤ Head 1 → s1_logits
   s1_logits = Linear(D → V₁)          # V₁ = 1024

⑥ DependencyAwareLayer
   sibling_embed = Emb_s1(s1_condition) # 训练: ground-truth s1; 推理: 采样 s1
   attn = CrossAttn(Q=sibling_embed, KV=context)
   x2 = RMSNorm(context + attn)

⑦ Head 2 → s2_logits
   s2_logits = Linear(D → V₂)          # V₂ = 1024
```

---

## 二、各层详解

### ① HierarchicalEmbedding

代码: [module.py HierarchicalEmbedding](file:///home/yzh/workspace/Kronos/model/module.py#L400-L443)

```python
s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)   # (B,T) → (B,T,D)
s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)   # (B,T) → (B,T,D)
x = self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1)) # (B,T,2D) → (B,T,D)
```

- `emb_s1` 和 `emb_s2` 各是独立的 `nn.Embedding(1024, D)`
- 乘以 `√D` 是标准 Transformer 的 embedding scaling，防止 embedding 值过小被位置编码淹没
- concat 后通过一个线性层降维融合，把 s1 和 s2 的信息合并到同一个 D 维空间

**设计意图**：s1 (coarse) 和 s2 (fine) 编码了不同粒度的信息。分别嵌入再融合，比直接拼接 20-bit 做一个大 Embedding 更灵活。

### ② TemporalEmbedding

代码: [module.py TemporalEmbedding](file:///home/yzh/workspace/Kronos/model/module.py#L536-L562)

```python
# stamp ∈ (B, T, 5)，5个通道分别是 minute, hour, weekday, day, month
time_emb = minute_embed(stamp[:,:,0]) + hour_embed(stamp[:,:,1])
         + weekday_embed(stamp[:,:,2]) + day_embed(stamp[:,:,3])
         + month_embed(stamp[:,:,4])
x = x + time_emb
```

每个时间维度有独立的 Embedding 表（可选 Fixed 正弦或 Learnable），相加后叠加到 token embedding 上。

**作用**：注入 K 线的绝对时间信息。模型可以学到"周一开盘"、"14:30 尾盘"等时间模式。

### ③ Transformer Backbone

代码: [kronos.py L261-L264](file:///home/yzh/workspace/Kronos/model/kronos.py#L261-L264)

```python
for layer in self.transformer:
    x = layer(x, key_padding_mask=padding_mask)
x = self.norm(x)    # RMSNorm
```

- N 层 `TransformerBlock`，每层结构同 Tokenizer（Pre-LN + RoPE Self-Attention + SwiGLU FFN）
- **Causal mask**：`is_causal=True`，位置 t 只能看到 ≤ t 的 token，确保自回归性质
- 最终通过 RMSNorm 稳定输出分布

输出 `x` 即为 context 向量 `H_ctx ∈ (B, T, D)`，包含了每个位置截至当前的全部历史信息。

### ⑤ Head 1: 预测 s1

代码: [module.py DualHead.forward()](file:///home/yzh/workspace/Kronos/model/module.py#L509-L510)

```python
def forward(self, x):
    return self.proj_s1(x)   # Linear(D → 1024)
```

直接将 context 投影到 s1 的词表空间。输出 `s1_logits ∈ (B, T, 1024)`，每个位置一个 1024 类的概率分布。

### ⑥ DependencyAwareLayer: s1 → s2 的条件注入

代码: [module.py DependencyAwareLayer](file:///home/yzh/workspace/Kronos/model/module.py#L446-L462)

这是 Predictor 中最核心的设计。s2 不是独立预测的，它**以 s1 为条件**。

```python
def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
    attn_out = self.cross_attn(
        query=sibling_embed,       # s1 的 embedding 作为 Query
        key=hidden_states,         # Transformer 输出的 context 作为 Key
        value=hidden_states,       # Transformer 输出的 context 作为 Value
    )
    return self.norm(hidden_states + attn_out)   # 残差 + RMSNorm
```

**直觉**：s1 是粗粒度信息（大致走势），s2 是细粒度信息（精确数值）。预测细节时，模型用"我已经知道的粗粒度走势"去查询"历史上下文中相关的细节模式"。

Cross-Attention 的 Q/K/V 角色：
- **Query = s1 embedding**：用当前预测的 s1（走势方向）发起查询
- **Key/Value = context**：从整个历史序列的上下文中检索相关信息

### ⑦ Head 2: 预测 s2

代码: [module.py DualHead.cond_forward()](file:///home/yzh/workspace/Kronos/model/module.py#L512-L513)

```python
def cond_forward(self, x2):
    return self.proj_s2(x2)   # Linear(D → 1024)
```

结构与 Head 1 对称，但输入是经过 DependencyAwareLayer 增强的向量。

---

## 三、训练 vs 推理的关键差异

### 训练阶段 (`Kronos.forward()`)

代码: [kronos.py L240-L277](file:///home/yzh/workspace/Kronos/model/kronos.py#L240-L277)

```
完整序列 (s1_ids, s2_ids) ∈ (B, T)
  ↓
token_in  = ids[:, :-1]    # T-1 个输入
token_out = ids[:, 1:]     # T-1 个目标 (next token)
  ↓
Embedding → Transformer → context
  ↓
Head1 → s1_logits ∈ (B, T-1, 1024)
  ↓
DependencyLayer(context, Emb(s1_condition))
  ↓
Head2 → s2_logits ∈ (B, T-1, 1024)
  ↓
Loss = (CE(s1_logits, s1_targets) + CE(s2_logits, s2_targets)) / 2
```

**s1_condition 的来源**（训练时）：

```python
if use_teacher_forcing:
    sibling_embed = self.embedding.emb_s1(s1_targets)   # 用真实的下一时刻 s1
else:
    s1_probs = F.softmax(s1_logits.detach(), dim=-1)
    sample_s1_ids = torch.multinomial(s1_probs.view(-1, 1024), 1).view(...)
    sibling_embed = self.embedding.emb_s1(sample_s1_ids) # 从 s1 分布中采样
```

训练代码中默认 `use_teacher_forcing=False`（见 `train_predictor.py` L190 调用），即使用采样 s1 而非 ground-truth。

**并行计算**：因为有 causal mask，所有 T-1 个位置的预测在一次前向中并行完成。

### 推理阶段 (`decode_s1` + `decode_s2` + 自回归循环)

代码: [kronos.py auto_regressive_inference()](file:///home/yzh/workspace/Kronos/model/kronos.py#L390-L471)

```python
for i in range(pred_len):
    # 步骤 1: 编码全部历史 → 预测 s1
    s1_logits, context = model.decode_s1(buffer_s1, buffer_s2, stamp)
    s1_logits = s1_logits[:, -1, :]        # 只取最后一个时间步
    sample_pre = sample(s1_logits, T, top_k, top_p)

    # 步骤 2: 用刚采样的 s1 → 条件预测 s2
    s2_logits = model.decode_s2(context, sample_pre)
    s2_logits = s2_logits[:, -1, :]
    sample_post = sample(s2_logits, T, top_k, top_p)

    # 步骤 3: 追加到 buffer（滑动窗口）
    buffer.append(sample_pre, sample_post)
```

**串行瓶颈**：每一步都要重新跑完整的 Transformer（因为没有 KV cache）。这是主要的推理开销。

### 差异对比表

| 维度 | 训练 | 推理 |
|:---|:---|:---|
| 计算模式 | 并行（所有 T-1 位置同时计算） | 串行（逐步自回归） |
| s1 来源 | 从 logits 采样（或 Teacher Forcing） | 从 logits 采样 |
| s2 条件 | 采样/GT 的 s1 的 embedding | 刚采样出的 s1 的 embedding |
| 取 logits | 全序列都参与 loss | 只取 `[:, -1, :]`（最后时间步） |
| Dropout | 启用 | 禁用 (`model.eval()`) |
| 输入长度 | 固定窗口 (lookback + predict) | 滑动窗口 (max_context) |

---

## 四、采样策略

代码: [kronos.py sample_from_logits()](file:///home/yzh/workspace/Kronos/model/kronos.py#L374-L387)

```python
logits = logits / temperature          # 温度缩放
logits = top_k_top_p_filtering(logits) # 截断低概率 token
probs = F.softmax(logits, dim=-1)
x = torch.multinomial(probs, num_samples=1)
```

三个控制参数：

| 参数 | 作用 | 典型值 |
|:---|:---|:---|
| T (temperature) | < 1.0 更确定性，> 1.0 更随机 | 0.6 ~ 1.0 |
| top_k | 只保留概率最高的 k 个候选 | 0 (不启用) |
| top_p | 累积概率达到 p 时截断 (nucleus sampling) | 0.9 |

**多样本采样**：推理时将 batch 复制 `sample_count` 份并行生成，最终取均值：

```python
x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, T, C)
# ... 自回归生成 ...
z = z.reshape(-1, sample_count, seq_len, 6)
preds = np.mean(z, axis=1)   # 对 sample_count 维取均值
```

这相当于 Monte Carlo 采样的近似期望。

---

## 五、从 Token 回到 K 线

推理的最后一步，Predictor 输出的 Token ID 需要经过 Tokenizer 解码和反归一化：

```python
# 1. Token ID → 归一化 K 线
z = tokenizer.decode(input_tokens, half=True)   # → (B*S, T, 6) 归一化空间

# 2. 反归一化
preds = preds * (x_std + 1e-5) + x_mean          # 还原原始量纲
```

`decode(half=True)` 内部流程：
1. `indices_to_bits`: s1_ids → 10-bit binary → ±1, s2_ids → 10-bit binary → ±1, concat → 20D → ×(1/√20)
2. `post_quant_embed`: Linear(20 → 256)
3. Decoder Transformer stack
4. `head`: Linear(256 → 6)

---

## 六、完整推理流程图

```
                     KronosPredictor.predict()
                              │
            ┌─────────────────┴──────────────────┐
            │  预处理                              │
            │  • df → numpy                       │
            │  • z-score: (x-μ)/σ                 │
            │  • clip [-5, 5]                     │
            │  • 时间戳 → (minute,hour,wd,day,mo) │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │  Tokenizer.encode(half=True)        │
            │  → (s1_ids, s2_ids) 历史 Token      │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │  自回归循环 × pred_len               │
            │  ┌──────────────────────────┐      │
            │  │ decode_s1 → sample → ŝ1  │      │
            │  │ decode_s2 → sample → ŝ2  │      │
            │  │ buffer ← append(ŝ1, ŝ2)  │      │
            │  └──────────────────────────┘      │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │  Tokenizer.decode(half=True)        │
            │  → 归一化空间的 K 线                  │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │  反归一化                            │
            │  pred × σ + μ → 原始价格量纲         │
            └─────────────────┬──────────────────┘
                              │
                         pred_df (DataFrame)
```

---

## 七、常见疑问

### Q1: Predictor 训练时需要调用 Tokenizer 的 decoder 吗？

**不需要。** Predictor 是一个语言模型，损失函数是 Cross-Entropy（分类），不是 MSE（回归）。

看 [train_predictor.py L182-L191](file:///home/yzh/workspace/Kronos/finetune/train_predictor.py#L182-L191)：

```python
with torch.no_grad():
    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)  # 只调 encode

token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]   # 输入 (T-1)
token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]    # 目标 (next-token)

logits = model(token_in[0], token_in[1], ...)
loss = CrossEntropy(logits, token_out)                     # 分类损失
```

Predictor 只关心"下一个 token ID 是几号"，不需要回到连续空间。只有**推理的最后一步**才需要 `decode()` 把 token 序列转回 K 线给用户。

### Q2: 论文中的"coarse-to-fine chain rule 分解"是否有包装成分？

论文原文：
> p(bt|b<t) = p(bc_t|b<t) · p(bf_t|b<t, bc_t)  
> "This formulation allows the model to first predict the coarse-grained subtoken, which serves as a scaffold..."

**数学上**：链式法则 `p(a,b) = p(a)·p(b|a)` 对任何联合分布都成立。把一个 token 随便拆成两半，这个式子都对。它是概率公理，不是设计洞见。

**代码上**：这个分解确实被实现了——Head1 独立预测 s1，DependencyAwareLayer 以 s1 为条件增强 context，Head2 基于增强 context 预测 s2。所以代码和公式对得上。

**夸大的部分**："coarse-to-fine" 叙事暗示 s1 天然就是粗粒度特征、s2 是残差细节。但实际上：

- s1 就是 BSQ 量化后 20 维向量的**前 10 维**，s2 是**后 10 维**
- s1 "承载主要信息"这个性质不是天然的，而是靠 Tokenizer 训练时 Path A（仅 s1 解码）的 MSE loss **强制塑造**出来的
- 去掉 Path A 的约束，encoder 没有理由把更重要的信息放在前 10 维，叙事就不成立了

**公平的评价**：条件预测确实比独立预测好——如果 s1 表示"今天涨了"，那 s2 的细节分布应该和"今天跌了"不同。DependencyAwareLayer 通过 cross-attention 让 s2 看到 s1 的结果，是有实际意义的。但这个好处来自"条件分解优于独立分解"这个朴素事实，不需要 "coarse-to-fine hierarchy" 的叙事包装。

