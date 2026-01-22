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

* 具体计算（逐 token: 对每一个 token，自身的特征维 $D$ 做平均）：
    
$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
    
其中 $\mu,\sigma^2$ 为最后一维（特征维）均值与方差，$\gamma,\beta \in \mathbb{R}^D$ 为可训练缩放与平移参数，$\epsilon$ 为数值稳定常数。
    
* Pre-LN 好处：
    
- 残差路径更短，梯度在深层网络里更稳（相比 Post-LN）。
- 训练初期更容易收敛，深层更不易爆/消。
- 若激活与残差方差分布差异大，$\gamma$ 会学习到合适缩放，避免 residual 被覆盖或失效。

* 常见变体：RMSNorm（去掉均值，只归一化方差）在部分模型中能减小计算开销但表达力略低。

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

#### Attention Dropout（作用在注意力权重上）
Attention Dropout 作用于 Self-Attention 或 Cross-Attention 中 Softmax 之后的注意力权重，用于随机屏蔽部分 token-to-token 的关注关系，防止模型过度依赖某些固定的注意力模式
$$\tilde{P} = \text{Dropout}(P,\; p=\text{attn\_dropout})$$

* 仅在训练时随机置零部分权重，期望值保持不变；推理时关闭 Dropout。
* 放在 softmax 之后、与 V 相乘之前，可抑制注意力过度集中，提升泛化。

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

* Residual Dropout 作用在子层输出 $O$ 上，先丢弃再加回残差 $X$；训练时防过拟合，推理时关闭。
* 与预激活残差配合，保持梯度稳定同时提供正则化。

* * *

## 四、FFN 子层（详细）
FFN 是 token-wise 的非线性映射模块，其作用是对每个 token 的特征维进行高阶变换，与 Attention 在 token 维上的交互形成正交分工

### 8️⃣ LayerNorm（Pre-LN）

$$\tilde{X}' = \text{LN}(X')$$

* * *

### 9️⃣ 前馈网络（以 SwiGLU 为例）

设扩展倍率为 **r**（通常 r=4）：

$$W_{gate}, W_{up} \in \mathbb{R}^{D \times rD},\quad  
W_{down} \in \mathbb{R}^{rD \times D}$$

* 参数量：升维两路各 $D \times rD$，回投 $rD \times D$，总参数 $\approx 2rD^2 + rD^2 = 3rD^2$（不计偏置）。
* r 取值：常见 4，也有 8（大模型）或低至 2（轻量模型）；越大表示容量越强但计算/显存线性上升。
* SwiGLU 相比 GeLU/SiLU 单路 FFN：引入门控 $\text{SiLU}(xW_1)\odot(xW_2)$，能在相近 FLOPs 或更优参数–性能比下提升建模非线性能力。
* 归一化与 Dropout 顺序：在 Pre-LN 架构中，FFN 前有 LN；Dropout 作用于 FFN 输出，随后走残差。

#### 升维 + 门控

$$U = \text{SiLU}(\tilde{X}' W_{gate}) \odot (\tilde{X}' W_{up})  
\quad  
U \in \mathbb{R}^{B \times T \times rD}$$

* * *

#### 降维

$$F = U W_{down}  
\quad  
F \in \mathbb{R}^{B \times T \times D}$$

* * *

### 🔟 FFN Dropout + Residual

$$X_{\text{out}} = X' + \text{Dropout}(F,\; p=\text{ffn\_dropout})$$

* FFN Dropout 作用于前馈输出 $F$，降低 co-adaptation，提升泛化；推理阶段关闭。
* 与 Residual Dropout 位置一致：先 Dropout，再做残差相加。

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
## 附录

### Dropout 函数定义

在深度学习框架（如 PyTorch/TensorFlow）中，使用的是 **Inverted Dropout（反向 Dropout）**。

#### 数学定义

假设输入张量为 $X \in \mathbb{R}^{B \times T \times D}$，设定的丢弃概率为 $p$（即 $p$ 的概率变为 0，保留概率为 $1-p$）。

**函数逻辑：**

$$\text{Dropout}(X) = \frac{X \odot M}{1-p}$$

其中：

* **$M$ (Mask)**：一个与 $X$ 形状完全相同的二值矩阵。
    
    $$M_{ij} \sim \text{Bernoulli}(1-p)$$
    
    即：有 $1-p$ 的概率是 1，有 $p$ 的概率是 0。

* **$\odot$**：元素级乘法（Element-wise multiplication）。

* **$\frac{1}{1-p}$ (Rescaling)**：缩放因子。因为训练时随机扔掉了一部分数据，导致数值的总期望变小了，所以必须把保留下来的数值放大，以保证训练和推理（推理时 Dropout 不启用）时的数据分布期望一致。

#### 训练与推理模式

* **训练模式（training=True）**：随机生成 mask $M$，按上述公式计算。
* **推理模式（training=False）**：Dropout 不生效，直接返回 $X$（等价于 $p=0$）。

