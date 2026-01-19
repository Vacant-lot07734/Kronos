# Morden LLM TransformerBlock 详解

## 一、符号与维度约定（统一口径）

* **B**：batch size
    
* **T**：sequence length
    
* **D**：model dimension（hidden size）
    
* **H**：attention heads 数
    
* **d = D / H**：每个 head 的维度
    

输入张量：

$$X \in \mathbb{R}^{B \times T \times D}$$

* * *

## 二、整体结构（Pre-LN）

```
X
│
├─ LN
│   └─ Self-Attention
│       └─ Attn Dropout
│
├─ Residual + Dropout
│
├─ LN
│   └─ FFN
│       └─ FFN Dropout
│
└─ Residual
```

LayerNorm **位于子层之前**（Pre-LN），这是关键。

* * *

## 三、Self-Attention 子层（详细）

### 1️⃣ LayerNorm（Pre-LN）

$$\tilde{X} = \text{LN}(X)$$

* 形状不变：
    

$$\tilde{X} \in \mathbb{R}^{B \times T \times D}$$

* * *

### 2️⃣ QKV 线性投影

参数：

$$W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$$

计算：

$$Q = \tilde{X} W_Q,\quad  
K = \tilde{X} W_K,\quad  
V = \tilde{X} W_V$$

形状：

$$Q,K,V \in \mathbb{R}^{B \times T \times D}$$

* * *

### 3️⃣ 分头（reshape）

$$Q \rightarrow Q^{(h)} \in \mathbb{R}^{B \times H \times T \times d}$$

同理：

$$K^{(h)}, V^{(h)} \in \mathbb{R}^{B \times H \times T \times d}$$

* * *

### 4️⃣ RoPE（Rotary Positional Embedding）

**仅作用在 Q 和 K 上**：

$$(Q^{(h)}, K^{(h)}) \leftarrow \text{RoPE}(Q^{(h)}, K^{(h)})$$

* 形状不变
    
* RoPE 本质：对向量偶/奇维做二维旋转
    

* * *

### 5️⃣ Scaled Dot-Product Attention

#### Attention score：

$$S = \frac{Q^{(h)} {K^{(h)}}^\top}{\sqrt{d}}  
\quad  
S \in \mathbb{R}^{B \times H \times T \times T}$$

（decoder-only 模型中再加 causal mask）

* * *

#### Softmax（沿最后一维）

$$P = \text{softmax}(S)$$

#### Attention Dropout

$$\tilde{P} = \text{Dropout}(P,\; p=\text{attn\_dropout})$$

* * *

#### Attention 输出

$$A^{(h)} = \tilde{P} \cdot V^{(h)}  
\quad  
A^{(h)} \in \mathbb{R}^{B \times H \times T \times d}$$

* * *

### 6️⃣ 合并头 + 输出投影

合并 heads：

$$A = \text{Concat}(A^{(1)},\dots,A^{(H)})  
\in \mathbb{R}^{B \times T \times D}$$

输出投影：

$$O = A W_O  
\quad  
W_O \in \mathbb{R}^{D \times D}$$

* * *

### 7️⃣ Residual + Dropout

$$X' = X + \text{Dropout}(O,\; p=\text{resid\_dropout})$$

* * *

## 四、FFN 子层（详细）

### 8️⃣ LayerNorm（Pre-LN）

$$\tilde{X}' = \text{LN}(X')$$

* * *

### 9️⃣ 前馈网络（以 SwiGLU 为例）

设扩展倍率为 **r**（通常 r=4）：

$$W_1, W_2 \in \mathbb{R}^{D \times rD},\quad  
W_3 \in \mathbb{R}^{rD \times D}$$

#### 升维 + 门控

$$U = \text{SiLU}(\tilde{X}' W_1) \odot (\tilde{X}' W_2)  
\quad  
U \in \mathbb{R}^{B \times T \times rD}$$

* * *

#### 降维

$$F = U W_3  
\quad  
F \in \mathbb{R}^{B \times T \times D}$$

* * *

### 🔟 FFN Dropout + Residual

$$X_{\text{out}} = X' + \text{Dropout}(F,\; p=\text{ffn\_dropout})$$

* * *

## 五、完整 Block 的数学表达（压缩版）

$$\begin{aligned}  
X' &= X + \text{Dropout}\big(  
\text{Attn}(\text{LN}(X))  
\big) \\  
X_{\text{out}} &= X' + \text{Dropout}\big(  
\text{FFN}(\text{LN}(X'))  
\big)  
\end{aligned}$$