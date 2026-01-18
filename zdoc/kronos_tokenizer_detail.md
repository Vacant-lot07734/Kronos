# KronosTokenizer 深度解析

本文档依据 `zdoc/llm-transformerblock.md` 的分析逻辑，对 `model/kronos.py` 中的 `KronosTokenizer` 类进行逐层拆解。

## 一、符号与维度约定

为保持统一，采用以下符号：

*   **B**：Batch size
*   **T**：Sequence length
*   **$D_{in}$**：Input dimension (`d_in`, 原始特征维度)
*   **D**：Model dimension (`d_model`, 隐藏层维度)
*   **N_enc**：Encoder layers (`n_enc_layers`)
*   **N_dec**：Decoder layers (`n_dec_layers`)
*   **$S_1, S_2$**：量化比特数 (`s1_bits`, `s2_bits`)
*   **K**：Total Codebook dimension ($K = S_1 + S_2$)

输入张量：
$$X \in \mathbb{R}^{B \times T \times D_{in}}$$

---

## 二、整体架构 (VQ-VAE Encoder-Decoder)

`KronosTokenizer` 实现了一个基于 Transformer 的变分自编码器，结构如下：

```
X (Input)
│
├─ 1. Input Projection (Linear)
│
├─ 2. Encoder Stack (Pre-LN Transformer Blocks)
│
├─ 3. Quantization Head (Linear + BSQ)
│      └─ [ Latent Code Z ] (Binary/Quantized)
│
├─ 4. Dual-Path Decoder Stack
│      ├─ Path A: Coarse (S1 bits only)
│      └─ Path B: Fine (All bits)
│
└─ Output (Reconstruction)
```

---

## 三、Encoder 侧详解

### 1️⃣ 输入线性投影 (Input Embedding)

代码对应：`self.embed = nn.Linear(self.d_in, self.d_model)`

计算：
$$H_0 = X W_{in} + b_{in}$$

形状变化：
$$X \in \mathbb{R}^{B \times T \times D_{in}} \longrightarrow H_0 \in \mathbb{R}^{B \times T \times D}$$

### 2️⃣ Encoder Transformer Stack

代码对应：
```python
self.encoder = nn.ModuleList([
    TransformerBlock(...) for _ in range(self.enc_layers - 1)
])
```

*   **重复次数**：$N_{enc} - 1$ 次。
*   **Block 结构**：遵循标准 Pre-LN 结构 (详见 `llm-transformerblock.md`)。
    *   Self-Attention (RoPE)
    *   SwiGLU FFN
    *   Residual Connections

计算：
$$H_{enc} = \text{Stack}(H_0)$$

### 3️⃣ 量化前投影 (Pre-Quant Projection)

代码对应：`self.quant_embed = nn.Linear(..., out_features=self.codebook_dim)`

计算：
$$Z_{logits} = H_{enc} W_{quant}$$

形状：
$$Z_{logits} \in \mathbb{R}^{B \times T \times K}$$

### 4️⃣ BSQ 量化 (Bottleneck)

代码对应：`self.tokenizer = BSQuantizer(...)`

此步骤将连续向量压缩为离散/二值编码：
$$Z_{q}, \text{Loss}_{bsq}, \text{Indices} = \text{BSQ}(Z_{logits})$$

*   **Training**: 使用 Straight-Through Estimator (STE) 允许梯度反向传播。
*   **Inference**: 输出离散 `Indices` (Int64)。

---

## 四、Decoder 侧详解 (双路重建)

解码器采用 **分层重建 (Hierarchical Reconstruction)** 策略，强制模型将主要信息压缩在 $S_1$ 位中。

### 5️⃣ 路径一：S1 Coarse Reconstruction

仅使用前 $S_1$ 个比特 $Z_q[:, :, :S_1]$。

1.  **S1 解码投影**:
    $$H_{dec}^{(s1)} = Z_q^{(s1)} W_{proj1}$$
    对应 `self.post_quant_embed_pre`。

2.  **Decoder Stack (S1)**:
    堆叠 $N_{dec}-1$ 层 `TransformerBlock`。
    $$H_{out}^{(s1)} = \text{Stack}_{dec}(H_{dec}^{(s1)})$$

3.  **Output Head (S1)**:
    线性映射回输入空间。
    $$\hat{X}_{s1} = H_{out}^{(s1)} W_{out}$$
    对应 `self.head`。

### 6️⃣ 路径二：Full Fine Reconstruction

使用完整 $K$ 个比特 $Z_q$。

1.  **Full 解码投影**:
    $$H_{dec}^{(full)} = Z_q W_{proj2}$$
    对应 `self.post_quant_embed`。

2.  **Decoder Stack (Full)**:
    堆叠 $N_{dec}-1$ 层 `TransformerBlock`。
    $$H_{out}^{(full)} = \text{Stack}_{dec}(H_{dec}^{(full)})$$

3.  **Output Head (Full)**:
    线性映射回输入空间。
    $$\hat{X}_{full} = H_{out}^{(full)} W_{out}$$

---

## 五、Training Objective (损失函数)

总损失包含三部分：

$$L = \underbrace{\text{MSE}(X, \hat{X}_{s1}) + \text{MSE}(X, \hat{X}_{full})}_{\text{Reconstruction}} + \underbrace{\beta \|Z_{logits} - \text{sg}[Z_q]\|^2}_{\text{Commitment}} + \underbrace{L_{entropy}}_{\text{Regularization}}$$

该设计迫使 Tokenizer 既能通过 $S_1$ 恢复大概轮廓，又能通过 $S_1+S_2$ 恢复细节。
