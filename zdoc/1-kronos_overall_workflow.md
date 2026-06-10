# Kronos 端到端工作流

## 一、系统架构总览

Kronos 由两个独立训练、协同推理的模型组成：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KronosTokenizer (VQ-VAE)                                               │
│  职责: 连续 K 线 ↔ 离散 Token 的双向转换                                 │
│                                                                         │
│  编码路径 (连续→离散):                                                   │
│  OHLCVA (6D)                                                            │
│    → z-score normalize + clip [-5, 5]     # 数据预处理 (外部)            │
│    → Linear(6 → 256)                      # 输入投影                    │
│    → Transformer encoder (3 layers)       # 上下文建模                  │
│    → Linear(256 → 20)                     # 量化前投影                  │
│    → L2 normalize                         # 映射到单位超球面             │
│    → BSQ: sign(z) → ±1 → ±1/√20          # 二值化 + 缩放               │
│    → split: s1 (10 bit) + s2 (10 bit)     # 拆分为两个 Token ID         │
│                                                                         │
│  解码路径 A (coarse):                                                   │
│    s1 (10D) → Linear(10 → 256) → Transformer decoder (3 layers)        │
│            → Linear(256 → 6)                                            │
│                                                                         │
│  解码路径 B (fine):                                                     │
│    full (20D) → Linear(20 → 256) → Transformer decoder (3 layers)      │
│              → Linear(256 → 6)                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Kronos Predictor (Autoregressive Transformer)                          │
│  职责: 在离散 Token 空间预测下一时刻的 (s1, s2)                          │
│                                                                         │
│  (s1_id, s2_id) 序列                                                    │
│    → HierarchicalEmbedding: Emb(s1)·√D ⊕ Emb(s2)·√D → Linear → D     │
│    → + TemporalEmbedding (minute + hour + weekday + day + month)        │
│    → Transformer encoder (N layers, causal mask)                        │
│    → RMSNorm                                                            │
│    → Head1: Linear(D → 1024) → s1 logits                               │
│    → DependencyLayer: CrossAttn(Q=Emb(s1), KV=context) + residual      │
│    → Head2: Linear(D → 1024) → s2 logits                               │
└─────────────────────────────────────────────────────────────────────────┘
```

参数约定（默认配置）：

| 符号 | 含义 | 值 |
|:---|:---|:---|
| D_in | 输入特征维度 (Open, High, Low, Close, Volume, Amount) | 6 |
| D | 隐藏层维度 (d_model) | 256 |
| K | 总量化比特数 (s1_bits + s2_bits) | 20 |
| s1 / s2 | coarse / fine 比特数 | 各 10 |
| V | 词表大小 (2^10) | 1024 |

---

## 二、三阶段工作流

### 第一阶段：Tokenizer 训练

**目标**：学会把 6D 连续 K 线压缩为 20-bit 离散码，并能高质量还原。

```
输入 X ∈ (B, T, 6)
  ↓ Encoder
  ↓ BSQ 量化
  ↓ 双路 Decoder 重建
输出 (X̂_s1, X̂_full) ∈ (B, T, 6)

Loss = (MSE(X, X̂_s1) + MSE(X, X̂_full) + BSQ_loss) / 2
```

- 训练脚本: `finetune/train_tokenizer.py`
- Optimizer: AdamW, OneCycleLR
- BSQ loss 包含 commitment loss 和 entropy regularization
- 双路解码的设计意图：**强制 s1 承载主要信息**，s2 补充细节

### 第二阶段：Predictor 训练

**目标**：在离散 Token 空间学习序列演化规律，预测下一时刻的 Token。

```
输入 X ∈ (B, T, 6)
  ↓ Tokenizer.encode (冻结, 不更新梯度)
  ↓ → (s1_ids, s2_ids) ∈ (B, T), 值域 [0, 1023]
  ↓
token_in  = (s1[:, :-1], s2[:, :-1])   # 输入序列 (T-1)
token_out = (s1[:, 1:],  s2[:, 1:])    # 目标序列 (next-token)
  ↓
Predictor.forward(token_in, stamp)
  ↓ → (s1_logits, s2_logits)
  ↓
Loss = (CE(s1_logits, s1_targets) + CE(s2_logits, s2_targets)) / 2
```

- 训练脚本: `finetune/train_predictor.py`
- **Tokenizer 完全冻结**，只更新 Predictor 参数
- s2 预测依赖 s1：训练时使用 Teacher Forcing（ground-truth s1）或采样 s1

### 第三阶段：推理（预测）

**目标**：给定历史 K 线，自回归生成未来 P 步的 K 线。

```
历史 K 线 X_hist ∈ (1, T_hist, 6)
  ↓ z-score normalize + clip
  ↓ Tokenizer.encode(half=True) → (s1_hist, s2_hist)

for t = 1 to P:
    ┌ Predictor.decode_s1(buffer) → s1_logits[:, -1, :]
    │   ↓ sample(top-k/top-p) → ŝ1
    │
    │ Predictor.decode_s2(context, ŝ1) → s2_logits[:, -1, :]
    │   ↓ sample(top-k/top-p) → ŝ2
    │
    └ buffer ← append(ŝ1, ŝ2)  (滑动窗口, 最大 max_context)

全序列 (s1_all, s2_all)
  ↓ Tokenizer.decode(half=True)
  ↓ denormalize
  ↓ → 预测 DataFrame
```

- 推理入口: `KronosPredictor.predict()` → `auto_regressive_inference()`
- 支持多样本采样 (sample_count)，最终取均值
- 滑动窗口推理：超过 max_context 时，buffer 滚动丢弃最早的 token

---

## 三、Tokenizer 与 Predictor 的协作关系

两个模型在训练阶段完全解耦，在推理阶段串联。

```
                  训练                              推理
           ┌──────────────┐                  ┌──────────────┐
 阶段一    │  Tokenizer   │                  │              │
           │  (独立训练)   │                  │              │
           └──────────────┘                  │              │
                                             │  Tokenizer   │
           ┌──────────────┐                  │  .encode()   │
 阶段二    │  Predictor   │  Tokenizer       │      ↓       │
           │  训练时调用   │ ← .encode()     │  Predictor   │
           │  Tokenizer    │   (冻结)        │  自回归生成   │
           └──────────────┘                  │      ↓       │
                                             │  Tokenizer   │
                                             │  .decode()   │
                                             └──────────────┘
```

| 阶段 | Tokenizer 状态 | Predictor 状态 | 数据流向 |
|:---|:---|:---|:---|
| Tokenizer 训练 | 训练中 | 不存在 | 连续 → 离散 → 连续 (自编码) |
| Predictor 训练 | 冻结 (仅 encode) | 训练中 | 连续 → Token → logits → CE loss |
| 推理 | 冻结 (encode + decode) | 冻结 (自回归) | 连续 → Token → 自回归 Token → 连续 |

关键点：
- Tokenizer 的 `encode(half=True)` 返回 `(s1_ids, s2_ids)` 两个独立的 10-bit Token ID
- Tokenizer 的 `decode(half=True)` 接受 `[s1_ids, s2_ids]`，内部将它们拼回 20-bit 向量后通过 full-path 解码器还原
- Predictor 不接触原始数值，只在 Token ID 空间做序列建模
