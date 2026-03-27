# Kronos 预测与评估公式全解

## 1. 记号约定
- 第 $i$ 只股票、时点 $t$ 真实数据：收盘价 $C_{i,t}$, 收益率 $r_{i,t}$, 未来 $H$ 步 OHLCVA $\mathbf{Y}_{i,t:t+H}$。预测值以 $\hat{}$ 表示。

## 2. 训练目标 (Loss)
- **Cross-Entropy (CE)**: $\mathcal{L}_{CE} = - \sum_{t=1}^{T} \log p_\theta(z_t \mid z_{<t})$。保留 Kronos 原始离散 Token 自回归模式的主损失。
- **均方误差 (MSE)**: $\mathcal{L}_{MSE} = \frac{1}{B \cdot H \cdot 6} \sum_{b,h,d} (\hat Y_{b,h,d} - Y_{b,h,d})^2$。若改为连续值回归（如 Direct-AR）的主损失，对异常大误差惩罚重。
- **平均绝对误差 (MAE)**: $\mathcal{L}_{MAE} = \frac{1}{B \cdot H \cdot 6} \sum_{b,h,d} |\hat Y_{b,h,d} - Y_{b,h,d}|$。对异常值鲁棒，多作为评估参考或辅助损失。

## 3. 核心评估指标
### 3.1 截面排序因子与回测 (选股核心)
量化选股通常不关心绝对预测数值，而关注股票间的相对强弱排序（Rank）以及模拟交易构建出的组合收益。
- **预期收益信号**: $\hat r_{i,t+H} = (\hat C_{i,t+H} - C_{i,t}) / C_{i,t}$，由多步预测价格反推预期收益率，从而转为选股因子。
- **IC (Information Coefficient)**: $IC_t = \mathrm{corr}(\hat s_{1:N_t,t},\ r_{1:N_t,t}) = \frac{\sum_{i=1}^{N_t}(\hat s_{i,t}-\bar{\hat s}_t)(r_{i,t}-\bar r_t)}{\sqrt{\sum_{i=1}^{N_t}(\hat s_{i,t}-\bar{\hat s}_t)^2}\sqrt{\sum_{i=1}^{N_t}(r_{i,t}-\bar r_t)^2}}$。预测收益 $\hat s$ 与真实收益 $r$ 在横截面上的 Pearson 线性相关系数。
- **RankIC**: $RankIC_t = \mathrm{corr}_{rank}(\hat s_{1:N_t,t},\ r_{1:N_t,t})$。Spearman 秩相关系数（即将预测与真实值分别转换为秩次 $\mathrm{rank}$ 后代入同上 Pearson 公式）。即使绝对数值不准，只要相对强弱排对，RankIC 即高；它是考察最核心模型选股能力的指标。
- **超额年化 (AER)**: $AER = AR_{strategy} - AR_{benchmark}$。策略减基准的超额回报。
- **信息比率 (IR)**: $IR = \mathbb{E}[r^{excess}] / \sigma(r^{excess})$。每承担一单位主动风险（波动率）所获取的超额收益，评估策略稳定性。

### 3.2 数值准确度指标
主要回答价格预测偏差及拟合程度差异。
- **RMSE**: $\sqrt{\frac{1}{N}\sum(\hat y - y)^2}$。数值预测常用的均方根误差。
- **$R^2$ (决定系数)**: $R^2 = 1 - \frac{\sum(y - \hat y)^2}{\sum(y - \bar y)^2}$。衡量模型可解释方差，若 <0 说明模型性能不及简单均值替代。
- **方向准确率 (DA)**: $DA = \frac{1}{N} \sum \mathbf{1}(\mathrm{sign}(\hat r)=\mathrm{sign}(r))$。预测涨跌方向正确的比例。

## 4. 实验全流程应用建议
- **Train**: 维持 Token 化生成范式用 CE；改写为连续值预测模型用 MSE。
- **Val**: 以 `val_rankic` 指标作为挑选最优 Checkpoint 的金标准。
- **Test**: 重点报告 IC、RankIC 及衍生多空组合回测的 AER/IR；同时附列 MAE、$R^2$ 等基础数学误差指标对齐论文格式。