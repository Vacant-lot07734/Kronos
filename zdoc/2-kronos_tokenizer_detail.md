# KronosTokenizer 逐层深度解析

本文档逐步拆解 `KronosTokenizer` 的每一层操作，对照代码解释数据如何从 6D 连续向量变成 20-bit 离散码，再变回 6D。

代码位置：[kronos.py](file:///home/yzh/workspace/Kronos/model/kronos.py) `KronosTokenizer` 类，[module.py](file:///home/yzh/workspace/Kronos/model/module.py) `BSQuantizer` / `BinarySphericalQuantizer` 类。

## 一、编码路径 (Encoder)

### 数据流总览

```
x ∈ (B, T, 6)                        # 归一化后的 OHLCVA
  ↓ ① Linear(6 → 256)                # embed
z ∈ (B, T, 256)
  ↓ ② TransformerBlock × (enc_layers-1)   # encoder stack
z ∈ (B, T, 256)
  ↓ ③ Linear(256 → 20)               # quant_embed
z ∈ (B, T, 20)
  ↓ ④ L2 normalize                    # F.normalize(z, dim=-1)
z ∈ (B, T, 20)    ‖z‖₂ = 1，位于 20D 单位超球面
  ↓ ⑤ sign → ±1 → ±1/√20             # BSQ quantize
zq ∈ (B, T, 20)   每个分量 = ±1/√20
  ↓ ⑥ split + bits_to_indices
s1_ids ∈ (B, T)   值域 [0, 1023]
s2_ids ∈ (B, T)   值域 [0, 1023]
```

### ① 输入投影 `self.embed`

```python
self.embed = nn.Linear(self.d_in, self.d_model)  # Linear(6, 256)
```

把 6 维原始特征映射到 256 维隐空间。无激活函数，纯仿射变换。

### ② Encoder Transformer Stack `self.encoder`

```python
self.encoder = nn.ModuleList([
    TransformerBlock(d_model, n_heads, ff_dim, ...)
    for _ in range(self.enc_layers - 1)   # 默认 3-1=2 层
])
```

每个 `TransformerBlock` 结构（Pre-LN）：

```
x → RMSNorm → MultiHeadAttention(RoPE, causal) → + residual
  → RMSNorm → SwiGLU FFN                        → + residual
```

- **RoPE**: 旋转位置编码，注入序列位置信息
- **SwiGLU FFN**: `w2(silu(w1(x)) * w3(x))`，比标准 ReLU FFN 表达力更强
- **Causal mask**: 即使是编码器，注意力也是因果的（`is_causal=True`）

编码器的作用：让每个时间步的 256D 表示融合了过去序列的上下文信息，而不是孤立地看单根 K 线。

### ③ 量化前投影 `self.quant_embed`

```python
self.quant_embed = nn.Linear(self.d_model, self.codebook_dim)  # Linear(256, 20)
```

从 256 维压缩到 20 维。这个 20 维向量即将被二值化。维度极低是有意设计——迫使模型用最少的比特捕获最重要的信息。

### ④ L2 归一化

```python
z = F.normalize(z, dim=-1)   # BSQuantizer.forward() 第一行
```

将 20 维向量归一化为单位向量（欧氏范数 = 1），映射到 20D 单位超球面。

**为什么要 L2 归一化？** BSQ（Binary Spherical Quantization）的理论基础是在超球面上做二值量化。归一化后，向量的方向携带信息，幅度被丢弃。

### ⑤ BSQ 二值化 `BinarySphericalQuantizer.quantize()`

```python
def quantize(self, z):
    zhat = torch.where(z > 0, 1.0, -1.0)       # sign 函数
    return z + (zhat - z).detach()               # STE (Straight-Through Estimator)
```

**逐元素取符号**：20 维单位向量的每个分量 → `+1` 或 `-1`。结果是 20D 超立方体的一个顶点。

**STE 技巧**：`sign()` 不可导，训练时梯度直接穿过量化层，等效于 `grad(zhat) = grad(z)`。前向取离散值，反向传连续梯度。

**缩放**：

```python
q_scale = 1.0 / (self.embed_dim ** 0.5)  # = 1/√20
zq = zhat * q_scale                       # ±1 → ±1/√20
```

缩放后 `‖zq‖₂ = 1`（20 个分量，每个 `(1/√20)²`，求和 = 1），保持和归一化前一致的范数。

### ⑥ 拆分 + 转 Token ID `BSQuantizer.bits_to_indices()`

```python
# 分割
q_pre  = quantized[:, :, :10]    # 前 10 维 → s1
q_post = quantized[:, :, 10:]    # 后 10 维 → s2

# 每组 10 个 ±1/√20 → 0/1 → 十进制
bits = (bits >= 0).to(torch.long)                       # ±1/√20 → 0 或 1
indices = 2 ** torch.arange(0, 10, device=bits.device)  # [1,2,4,...,512]
token_id = (bits * indices).sum(-1)                     # 二进制 → 十进制
```

10 个二进制位 → 一个 [0, 1023] 的整数 ID。注意这里是 **LSB first**（最低位在前）。

最终输出 `(s1_ids, s2_ids)`，每个都是 `(B, T)` 形状的整数张量。

---

## 二、解码路径 (Decoder)

### 为什么是双路？

Tokenizer 有两条解码路径，训练时**同时**计算两条路径的重建损失：

| 路径 | 输入 | 目的 |
|:---|:---|:---|
| Path A (coarse) | 仅 s1 的 10 bit (量化后前 10 维) | 迫使 s1 承载大部分信息 |
| Path B (fine) | 全部 20 bit | 利用 s1+s2 还原全部细节 |

### Path A: Coarse 解码（仅 s1）

```python
quantized_pre = quantized[:, :, :self.s1_bits]        # (B, T, 10)
z_pre = self.post_quant_embed_pre(quantized_pre)       # Linear(10 → 256)
for layer in self.decoder:
    z_pre = layer(z_pre)                                # TransformerBlock × (dec_layers-1)
z_pre = self.head(z_pre)                                # Linear(256 → 6)
```

### Path B: Fine 解码（完整 20 bit）

```python
z = self.post_quant_embed(quantized)                    # Linear(20 → 256)
for layer in self.decoder:
    z = layer(z)                                         # 同一套 decoder 参数（权重共享）
z = self.head(z)                                         # 同一个 head（权重共享）
```

**关键细节**：两条路径共享 decoder 层和 output head 的权重。这意味着 decoder 必须同时适应 "只有 s1" 和 "s1+s2" 两种输入条件。

### 推理时的解码 `KronosTokenizer.decode()`

推理时只走 Path B（完整解码），因为我们同时拥有 s1 和 s2：

```python
def decode(self, x, half=False):
    quantized = self.indices_to_bits(x, half)    # Token ID → ±1/√20 向量
    z = self.post_quant_embed(quantized)          # Linear(20 → 256)
    for layer in self.decoder:
        z = layer(z)
    z = self.head(z)                              # Linear(256 → 6)
    return z
```

`indices_to_bits` 是编码的逆操作：

```python
# 十进制 → 二进制位
mask = 2 ** torch.arange(codebook_dim//2)  # [1,2,4,...,512]
x = (x.unsqueeze(-1) & mask) != 0          # 按位与 → bool
x = x.float() * 2 - 1                      # bool → ±1
x = x / sqrt(20)                            # ±1 → ±1/√20
```

当 `half=True` 时，输入是 `(s1_ids, s2_ids)` 的元组，分别转换后在最后一维拼接成 20D 向量。

---

## 三、训练损失

```python
# train_tokenizer.py 中的损失计算
zs, bsq_loss, _, _ = model(batch_x)
z_pre, z = zs

recon_loss_pre = F.mse_loss(z_pre, batch_x)    # Path A 重建损失
recon_loss_all = F.mse_loss(z, batch_x)         # Path B 重建损失
recon_loss = recon_loss_pre + recon_loss_all
loss = (recon_loss + bsq_loss) / 2
```

`bsq_loss` 由 `BinarySphericalQuantizer` 内部计算，包含：

| 分量 | 公式 | 作用 |
|:---|:---|:---|
| Commitment loss | `β · mean(‖zq.detach() - z‖²)` | 拉近编码器输出与量化点 |
| Per-sample entropy | `γ₀ · H_per_sample` | 惩罚单样本的编码太确定（鼓励使用接近决策边界的值） |
| Codebook entropy | `-γ · H_codebook` | 鼓励整体使用所有码字（负号 = 最大化熵） |

最终：`bsq_loss = commitment_loss + ζ · (γ₀ · H_per_sample - γ · H_codebook)`

设计意图：commitment loss 保证量化稳定，entropy 项避免 codebook collapse（大量码字不被使用）。

---

## 四、编码-解码的数值对称性

完整的编解码路径中，数值经历以下变换：

```
编码 (连续 → 离散):
  6D float → 256D → 20D → L2 norm → sign → ±1 → ×(1/√20) → ±1/√20
  → split → 10-bit → decimal → Token ID ∈ [0, 1023]

解码 (离散 → 连续):
  Token ID → binary → ±1 → ×(1/√20) → ±1/√20
  → concat(s1, s2) → 20D → Linear(20 → 256) → Transformer decoder → Linear(256 → 6)
```

注意：编码路径中的 ①②③ 步（Input Proj → Encoder → Quant Proj）在解码时**不存在逆过程**。解码器不是编码器的精确逆映射，而是一个独立学习的生成模型——它从 20-bit 离散码重新"想象"出原始信号。

---

## 五、常见疑问

### Q1: 两条解码路径是级联关系（先 Coarse 再 Fine）吗？

**不是。** 两条路径完全独立、并行执行，Path B 不依赖 Path A 的输出。看 [forward() L99-112](file:///home/yzh/workspace/Kronos/model/kronos.py#L99-L112)：Path A 和 Path B 各自从 `quantized` 张量中取不同切片，独立跑完 decoder 到 head，互不影响。

但关键是：**两条路径共享 `self.decoder` 和 `self.head` 的权重**。这个设计的本质是一个多任务约束，同时施压两个对象：

| 被约束的对象 | 约束来自 | 效果 |
|:---|:---|:---|
| **Encoder** | Path A 的 loss 要求仅用前 10 维就能重建 | 被迫把最重要的信息编码到 s1 |
| **Decoder** | 两条路径共享权重，都要重建好 | 被迫适应"只有 s1"和"完整 s1+s2"两种输入 |

**Path A 不是一个推理路径，而是一个训练技巧**——通过额外的重建约束，塑造 encoder 的信息分配策略。训练完成后，Path A 的使命就结束了。

### Q2: 推理时为什么只走 Path B？

Tokenizer 有两个独立的推理接口：

- **`encode()`**：被 Predictor 调用，只跑 encoder → BSQ → 输出 token ID。**完全不走 decoder。**
- **`decode()`**：被推理最后一步调用，token ID → Path B 解码器 → 输出 K 线。

推理时 s1 和 s2 都已经有了（Predictor 已经预测出来了），拼成完整 20D 走 Path B 即可。Path A 的投影层 `post_quant_embed_pre` 在推理时根本不会被调用。

### Q3: MSE 重建损失具体怎么算？

```python
recon_loss_pre = F.mse_loss(z_pre, batch_x)    # Path A
recon_loss_all = F.mse_loss(z, batch_x)         # Path B
```

`F.mse_loss` 默认 `reduction='mean'`，即：

```
MSE = mean of all (重建值 - 原始值)²
    = Σᵢ (z_pre[i] - batch_x[i])² / (B × T × 6)
```

对 batch 内所有样本、所有时间步、所有 6 个通道（OHLCVA）的逐元素平方差取全局平均，得到一个标量。两条路径的 MSE 权重相等（都是 1.0），encoder 自行通过梯度学会在 s1 和 s2 之间分配信息。
