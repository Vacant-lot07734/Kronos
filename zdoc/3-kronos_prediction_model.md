# Kronos 预测模型 (L180-L331) 深度解析

本文档基于 `model/kronos.py` (L180-L331) 代码，对 `Kronos` 类（时序预测主模型）的工作流、核心架构、数据形状变换及关键数学公式进行深度解析。

## 一、模型概述

`Kronos` 是一个用于处理量化后 Token 序列的自回归预测模型。它接收由 `KronosTokenizer` 生成的 $S_1$ (Coarse) 和 $S_2$ (Fine) 两组 Token 序列，结合时间戳信息，通过 Transformer 架构预测未来的 Token。

核心特点是采用了 **双头依赖解码 (Dual-Head Dependency Decoding)** 机制：
1.  首先基于历史上下文预测粗粒度的 $S_1$ Token。
2.  然后利用 **Dependency Aware Layer**，将生成的 $S_1$ Embedding 作为条件，预测对应的细粒度 $S_2$ Token。

---

## 二、符号与维度约定

### 1. 核心超参数

*   **B**: Batch size
*   **T**: Sequence length (Input history length)
*   **$D_{model}$** (`d_model`): 模型隐藏层维度。
*   **L**: Transformer Layers (`n_layers`)。
*   **H**: Attention Heads (`n_heads`)。
*   **$S_1, S_2$**: Token 的量化比特数。
*   **$V_1, V_2$**: 词表大小，其中 $V_1 = 2^{S_1}, V_2 = 2^{S_2}$。

### 2. 输入张量

*   **$Ids_{s1}, Ids_{s2} \in \mathbb{R}^{B \times T}$**: 输入的 Token ID 序列。
*   **$Stamp \in \mathbb{R}^{B \times T \times 5}$**: 时间戳特征（分钟、小时、星期、日、月）。

---

## 三、前向传播工作流详解 (Forward & Inference)

`Kronos` 的工作流可以分为 **Context Encoding** 和 **Dual-Stage Decoding** 两个主要阶段。

### 阶段 1: 层次化嵌入与上下文编码 (Context Encoding)

1.  **分层嵌入融合 (Hierarchical Embedding)**:
    *   代码: `self.embedding`
    *   将双路 Token $S_1, S_2$ 映射为向量并融合。
    *   $$E_{s1} = \text{Embed}_{s1}(Ids_{s1}) \cdot \sqrt{D_{model}}$$
    *   $$E_{s2} = \text{Embed}_{s2}(Ids_{s2}) \cdot \sqrt{D_{model}}$$
    *   $$X_{emb} = \text{Linear}_{fusion}([E_{s1}; E_{s2}])$$
    *   **形状**: $(B, T, D_{model}) \leftarrow \text{cat}((B, T, D_{model}), (B, T, D_{model}))$

2.  **时间嵌入叠加 (Temporal Embedding)**:
    *   代码: `self.time_emb(stamp)`
    *   $$X_{in} = X_{emb} + \sum_{feat} \text{Embed}_{time}(Stamp_{feat})$$
    *   **形状**: $(B, T, D_{model})$ 保持不变。

3.  **Transformer 主干提取**:
    *   代码: `self.transformer`
    *   $$H_{ctx} = \text{TransformerStack}(X_{in})$$
    *   **形状**: $(B, T, D_{model})$

4.  **归一化**:
    *   代码: `self.norm`
    *   $$H_{ctx} = \text{RMSNorm}(H_{ctx})$$

### 阶段 2: 第一级预测 (Predict S1)

直接利用 Transformer 的输出 $H_{ctx}$ 预测粗粒度 Token $S_1$。

1.  **S1 Logits 计算**:
    *   代码: `self.head(x)`
    *   $$Logits_{s1} = H_{ctx} W_{head\_s1}$$
    *   **形状**: $(B, T, D_{model}) \rightarrow (B, T, V_1)$

2.  **S1 采样 (Inference 时)**:
    *   $$P(s1) = \text{Softmax}(Logits_{s1})$$
    *   $$\hat{s1} \sim \text{Multinomial}(P(s1))$$ or $$\text{Argmax}$$

### 阶段 3: 第二级依赖预测 (Predict S2 Conditioned on S1)

利用 **Dependency Aware Layer**，使得 $S_2$ 的预测依赖于 **当前步已知的 S1 信息**。

1.  **S1 条件嵌入**:
    *   代码: `self.embedding.emb_s1(s1_targets 或 \hat{s1})`
    *   $$E_{cond} = \text{Embed}_{s1}(\hat{s1})$$
    *   **形状**: $(B, T, D_{model})$

2.  **依赖感知层融合 (Dependency Aware Layer)**:
    *   代码: `self.dep_layer(x, sibling_embed)`
    *   这是一个 Cross-Attention 变体，Query 为 $E_{cond}$ (sibling)，Key/Value 为 $H_{ctx}$ (context)。
    *   **注意**: 代码实现中 `self.dep_layer` 的 `forward` 参数是 `(hidden_states, sibling_embed)`。
    *   仔细查阅 `module.py` 的 `DependencyAwareLayer`:
        ```python
        def forward(self, hidden_states, sibling_embed ...):
            attn_out = self.cross_attn(query=sibling_embed, key=hidden_states, value=hidden_states ...)
            return self.norm(hidden_states + attn_out)
        ```
    *   **修正理解**: 这里 Query 是 `sibling_embed` ($S_1信息$)，Key/Value 是 `hidden_states` ($H_{ctx}$)。这意味着实际上是用 $S_1$ 的 Embedding 去 "查询" 上下文信息，来增强 $S_2$ 的预测。
    *   $$H_{s2} = \text{RMSNorm}(H_{ctx} + \text{CrossAttn}(Q=E_{cond}, K=H_{ctx}, V=H_{ctx}))$$
    *   **形状**: $(B, T, D_{model})$

3.  **S2 Logits 计算**:
    *   代码: `self.head.cond_forward(x2)`
    *   $$Logits_{s2} = H_{s2} W_{head\_s2}$$
    *   **形状**: $(B, T, D_{model}) \rightarrow (B, T, V_2)$

---

## 四、数据形状变化全览表

假设 Batch Size = $B$, Sequence Length = $T$。

| 步骤 | 变量名 | 张量形状 (Shape) | 说明 |
| :--- | :--- | :--- | :--- |
| **Input** | `s1_ids`, `s2_ids` | $(B, T)$ | 输入 Token 序列 |
| Embedding | `x` | $(B, T, D_{model})$ | Hierarchical Fusion 之后 |
| Time Emb | `x` | $(B, T, D_{model})$ | 加上时间特征后 |
| **Transformer** | `x` | $(B, T, D_{model})$ | 经过 $L$ 层 Transformer Block |
| RMSNorm | `x` | $(B, T, D_{model})$ | 归一化后的上下文 $H_{ctx}$ |
| **Head S1** | `s1_logits` | $(B, T, V_1)$ | S1 预测分布 |
| S1 Embed | `sibling_embed` | $(B, T, D_{model})$ | 采样的 S1 再次 Embedding |
| **Dep Layer** | `x2` | $(B, T, D_{model})$ | Cross-Attn 融合 S1信息与上下文 |
| **Head S2** | `s2_logits` | $(B, T, V_2)$ | S2 预测分布 |

---

## 五、关键矩阵公式

### 1. 层次化嵌入融合 (Hierarchical Embedding)
$$X_{emb} = ([E_{s1}; E_{s2}]) W_{fusion} + b_{fusion}$$
其中 $W_{fusion} \in \mathbb{R}^{2D_{model} \times D_{model}}$。这是一个降维融合操作。

### 2. 依赖感知层 (Dependency Aware Layer)
这是一个特殊的层，用于在给定 $S_1$ 的情况下细化上下文以预测 $S_2$。

$$Q = E_{s1} W_Q$$
$$K = H_{ctx} W_K, \quad V = H_{ctx} W_V$$
$$Attn = \text{Softmax}(\frac{(Q + \text{RoPE}) (K + \text{RoPE})^T}{\sqrt{d}}) V$$
$$H_{s2} = \text{RMSNorm}(H_{ctx} + Attn W_O)$$

*注*: 此处的 Cross Attention 使用了 `RoPE`，确保位置信息在 Query 和 Key 交互时的相对性。

### 3. 双头输出 (Dual Head)
虽然封装在 `DualHead` 类中，但实际上是两个独立的线性层：
*   **S1 Projection**: $Logits_{s1} = H_{ctx} W_{s1} + b_{s1}$
*   **S2 Projection**: $Logits_{s2} = H_{s2} W_{s2} + b_{s2}$

---

## 六、自回归推理 (Autoregressive Inference)

在 `auto_regressive_inference` 函数中 (L390+)，模型进行逐步生成。对于每一个时间步 $t$：

1.  **Decode S1**:
    *   输入历史 $(X_{0:t-1})$。
    *   运行 Transformer 得到 $H_{ctx}^{(t)}$。
    *   预测并采样得到 $\hat{s1}_t$。
    *   输出: `s1_logits`, `context` ($H_{ctx}$)

2.  **Decode S2**:
    *   利用 $H_{ctx}^{(t)}$ 和 刚刚生成的 $\hat{s1}_t$。
    *   通过 Dependency Layer 融合。
    *   预测并采样得到 $\hat{s2}_t$。

3.  **Update**:
    *   将 $(\hat{s1}_t, \hat{s2}_t)$ 加入历史序列，进入 $t+1$ 步。

这种机制确保了 $S_2$ 的生成不仅依赖历史，还强依赖于同构的 $S_1$，保证了粗细粒度的一致性。
