# Kronos 整体训练与推理流程梳理

本文档梳理了 `Kronos` 项目的端到端工作流，涵盖 **Tokenizer 训练**、**Predictor 训练** 以及 **推理（预测）** 三个核心阶段。

## 零、参数设定
*   **输入特征 ($D_{in}$)**: 6维，`[Open, High, Low, Close, Volume, Amount]`。
*   **隐藏层维度 ($D_{model}$)**: 512。
*   **量化编码长度 ($K$)**: 20 Bit。
    *   **Coarse ($S_1$)**: 前 10 Bit。
    *   **Fine ($S_2$)**: 后 10 Bit。

---

## 一、第一阶段：Tokenizer 训练流程 (VQ-VAE)

目标：训练一个能将 6维连续时序数据压缩为 20位离散编码，并能高质量还原的模型。

### 1. 数据流 (Data Flow)
*   **Input**: Normalized Batch Tensor $X \in \mathbb{R}^{B \times T \times 6}$。
*   **Forward**:
    1.  **Encode**: $X \xrightarrow{\text{Encoder}} Z \xrightarrow{\text{Quant}} Z_{logits}$
    2.  **Quantize**: $Z_{logits} \xrightarrow{\text{BSQ}} (Z_q, \text{Indices}, Loss_{bsq})$
        *   $Z_q$ 是量化后的连续向量，用于重建。
        *   Indices 是 0/1 比特流（训练中仅用于统计，不参与梯度）。
    3.  **Decode (Dual Path)**:
        *   **Path 1 (S1-only)**: $Z_q[:, :10] \xrightarrow{\text{Decoder}} \hat{X}_{s1}$
        *   **Path 2 (Full)**: $Z_q[:, :20] \xrightarrow{\text{Decoder}} \hat{X}_{full}$
*   **Loss Calculation**:
    *   $L_{recon\_s1} = \text{MSE}(X, \hat{X}_{s1})$
    *   $L_{recon\_full} = \text{MSE}(X, \hat{X}_{full})$
    *   $L_{total} = \frac{1}{2}(L_{recon\_s1} + L_{recon\_full}) + L_{bsq}$
*   **Optimization**: 更新 Encoder, Decoder, 和 BSQ 参数。

---

## 二、第二阶段：Predictor 训练流程 (Autoregressive Transformer)

目标：在离散的 Token 空间上，学习序列的演化规律，预测未来 Token。

### 1. 数据准备
*   **Input**: Normalized Batch Tensor $X \in \mathbb{R}^{B \times T \times 6}$ 以及时间戳 $Stamp$。
*   **Tokenization (On-the-fly)**:
    *   冻结训练好的 Tokenizer。
    *   $X \xrightarrow{\text{Tokenizer.encode}} (IDs_{s1}, IDs_{s2})$
    *   $IDs \in [0, 2^{10}-1]^{B \times T}$ (整数索引)。

### 2. 训练流 (Training Flow)
*   **Input Seq**: $Tokens_{in} = IDs[:, :-1]$ (T-1 长度)。
*   **Target Seq**: $Tokens_{out} = IDs[:, 1:]$ (Next Token)。
*   **Forward**:
    1.  **Embedding**: 将 $(S_1, S_2)$ IDs 映射并融合为 $D_{model}=512$ 的向量。
    2.  **Add Time**: 叠加时间戳 Embedding。
    3.  **Transformer**: 处理序列信息，输出上下文 $H_{ctx}$。
    4.  **Prediction Head**:
        *   $H_{ctx} \xrightarrow{\text{Head 1}} Logits_{s1} \xrightarrow{\text{Loss}} L_{s1}$
        *   $(H_{ctx}, \text{GroundTruth}_{s1}) \xrightarrow{\text{DepLayer}} H_{s2} \xrightarrow{\text{Head 2}} Logits_{s2} \xrightarrow{\text{Loss}} L_{s2}$
*   **Loss**: $L_{total} = L_{s1} + L_{s2}$ (Cross Entropy)。
*   **Optimization**: 仅更新 Predictor 参数。

---

## 三、第三阶段：推理（预测）流程

目标：基于历史数据，自回归地生成未来的价格序列。

### 1. 初始化与编码 (Initialization & Encoding)
*   **Input**: 历史窗口长度的数据 $X_{hist}$ (e.g. 400 steps)。
*   **Tokenize**: $X_{hist} \xrightarrow{\text{Tokenizer}} (Hist_{s1}, Hist_{s2})$。

### 2. 自回归生成循环 (Autoregressive Generation Loop)
设需要预测 $P$ 步 (e.g. 120 steps)。
For $t = 1$ to $P$:
1.  **Context**: 输入当前的 $(Hist_{s1}, Hist_{s2})$ 到 Predictor。
2.  **Predict S1**:
    *   Transformer 输出 $H_{ctx}$。
    *   $Logits_{s1} = \text{Head}_1(H_{ctx}[-1])$。
    *   采样: $\hat{s}_{1, next} \sim \text{Sample}(Logits_{s1})$ (Top-k/Top-p)。
3.  **Predict S2 (Conditional)**:
    *   利用 $H_{ctx}[-1]$ 和 $\hat{s}_{1, next}$ 通过 Dependency Layer。
    *   $Logits_{s2} = \text{Head}_2(\text{DepOut})$。
    *   采样: $\hat{s}_{2, next} \sim \text{Sample}(Logits_{s2})$。
4.  **Update**: 将 $(\hat{s}_{1, next}, \hat{s}_{2, next})$ 追加到历史序列中。

### 3. 解码与后处理 (Decoding & Post-processing)
*   **Decode**:
    *   获取生成的预测 Token 序列。
    *   Encode 过程的逆变换: $(Pred_{s1}, Pred_{s2}) \xrightarrow{\text{Tokenizer.decode}} \hat{X}_{pred}$。
    *   $\hat{X}_{pred}$ 是 Normalized 的连续空间数值。
*   **Denormalize**: 反归一化，恢复到原始股价范围 (Open, Close 等)。
*   **Output**: 预测的 DataFrame。

---

## 四、核心变换总结表

| 阶段 | 输入形态 | 核心操作 | 输出形态 | 维度备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Tokenizer Train** | $B \times T \times 6$ (Float) | VQ-VAE (Encoder-Quant-Decoder) | $B \times T \times 6$ (Recon) | $D_{in}=6 \to D_{latent}=512 \to K=20 \to D_{in}=6$ |
| **Predictor Train** | $B \times T \times 6$ (Float) | Tokenize -> Transformer -> CE Loss | Loss Scalar | Token ID $\in [0, 1023]$ (10 bit) |
| **Inference** | $T_{hist} \times 6$ (Float) | Tokenize -> Autoregressive Gen -> Decode | $T_{pred} \times 6$ (Float) | 序列长度 $T_{hist} \to T_{hist} + P$ |
