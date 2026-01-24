# EmoMarkets 金融情绪预测系统

## 📖 项目介绍

EmoMarkets 是一个专为金融市场设计的先进预测模型，能够理解和预测市场行为。该项目结合了情绪分析和K线数据处理，利用先进的Transformer架构和MoE(Mixture of Experts)技术，实现对金融市场的深度理解和预测。

主要特性：
- 利用情绪数据增强市场预测准确性
- 支持多种市场类型（沪市、深市、创业板、北交所等）
- 采用MoE结构为不同市场提供专门的处理逻辑
- 包含完整的训练、预测和回测流程

## 📁 项目结构

```
.
├── examples                  # 示例脚本
│   ├── new_prediction_example.py
│   ├── prediction_example.py
│   └── backtest_example.py   # 新增：回测示例
├── finetune                  # 微调相关代码
│   ├── utils                 # 工具函数
│   ├── config.py             # 配置文件
│   ├── dataset.py            # 数据集处理
│   ├── new_data_processor.py # 新数据处理器
│   ├── new_dataset.py        # 新数据集
│   ├── new_train_predictor.py# 新预测器训练
│   ├── new_train_tokenizer.py# 新分词器训练
│   ├── train_predictor.py    # 预测器训练
│   └── train_tokenizer.py    # 分词器训练
├── model                     # 原始模型实现
│   ├── __init__.py
│   ├── emoMarkets.py
│   └── module.py
├── new_model                 # 新版模型实现（推荐使用）
│   ├── emoMarkets.py
│   └── module.py
├── README.md
├── introduction.md
├── limit_board_stats.py      # 涨跌停板统计工具
└── requirements.txt

```


## 🔧 核心模块

### 1. 模型模块 ([model](new_model/emoMarkets.py ))
- 包含基础的 `EmoMarkets` 模型实现
- 提供情绪增强版本的预测模型 ([EmotionEnhancedWithMoE](new_model/emoMarkets.py))
- 实现了分层编码和Transformer架构

### 2. 微调模块 ([finetune](finetune))
- 数据处理和下载工具
- 模型训练脚本
- 分词器训练和数据集处理

### 3. 示例代码 ([examples](examples/new_prediction_example.py))
- 提供预测示例和使用指南
- 展示如何使用情绪数据增强预测效果

## ⭐ 核心特性

### 1. 情绪增强预测
EmoMarkets 不仅基于历史价格数据进行预测，还整合了市场情绪数据，包括：
- 指数收益率（上证、深证、创业板等）
- 涨跌停板统计

### 2. 分层编码技术
采用20比特分层编码方案：
- 粗粒度编码：10比特
- 细粒度编码：10比特

### 3. Transformer架构
模型基于Transformer架构，具有：
- 多头注意力机制
- 残差连接和层归一化
- 可配置的网络深度和宽度

### 4. Mixture of Experts (MoE)
为不同市场类型提供专门的处理逻辑：
- 沪市专家网络
- 深市专家网络
- 创业板专家网络
- 北交所专家网络

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```


### 数据准备

1. 准备K线数据和情绪数据
2. 使用 [limit_board_stats.py](limit_board_stats.py) 获取训练数据

### 模型训练

1. 训练分词器：
```bash
python finetune/new_train_tokenizer.py
```


2. 训练预测模型：
```bash
python finetune/new_train_predictor.py
```


### 运行预测示例

```bash
python examples/new_prediction_example.py
```


## 📊 使用方法

### 1. 数据准备(使用[new_data_processor.py](finetune/new_data_processor.py))
- 将K线数据整理为标准格式（CSV）
- 确保包含以下列：datetime, open, high, low, close, volume, amount
- 准备相应的情绪数据

### 2. 模型训练
- 配置训练参数在 `finetune/config.py`
- 运行分词器训练脚本 `finetune/new_train_tokenizer.py`
- 运行预测器训练脚本 `finetune/new_train_predictor.py`

### 3. 预测执行
- 参考 `examples/new_prediction_example.py`
- 可视化预测结果

## 📈 技术细节

### 模型架构
EmoMarkets 模型主要包括以下组件：
- [MarketSpecificExpert](new_model/emoMarkets.py): 针对不同市场的专家网络
- [MoEEmotionEnhancedTokenizer](new_model/emoMarkets.py): 情绪增强的分词器
- [EmotionEnhancedWithMoE](new_model/emoMarkets.py): 基于MoE的情绪增强预测模型

### 训练流程
1. 数据预处理和标准化
2. 分词器训练（无监督）
3. 预测模型微调（有监督）
4. 模型评估和验证

### 特征工程
- 时间特征提取（分钟、小时、星期几、日、月）
- 价格变化率计算
- 成交量变化率计算
- 情绪指标聚合

## 🛠️ 配置说明

主要配置项位于 `finetune/config.py`：
- `batch_size`: 批处理大小
- `learning_rate`: 学习率
- `epochs`: 训练轮数
- `d_model`: 模型维度
- `n_heads`: 注意力头数
- `n_layers`: Transformer层数

## 📜 许可证

本项目采用 [MIT License](./LICENSE)。

---


## Kronos 与 ARIMA、LSTM、Informer 股票预测对比实验

这是一个基于时间序列模型的股票预测对比实验项目，旨在比较 Kronos、ARIMA、LSTM 和 Informer 四种模型在股票预测任务上的性能表现。

### 项目结构

- **[preprocess.py](file://finetune\qlib_data_preprocess.py)**: 数据预处理模块，包含数据加载和时间范围过滤功能
- **[arima_model.py](file://comparison_experiment\arima_model.py)**: ARIMA 模型实现
- **`lstm_model_pytorch.py`**: PyTorch 版本的 LSTM 模型实现
- **[informer_model.py](file://comparison_experiment\informer_model.py)**: Informer 模型实现
- **[experiment_runner.py](file://comparison_experiment\experiment_runner.py)**: 主实验运行器
- **[metrics.py](file://comparison_experiment\metrics.py)**: 评估指标计算模块

### 功能特性

- **统一数据集**: 所有模型使用相同的 6 维 K 线数据（开、高、低、收、量、额）
- **配置文件同步**: 数据集划分与 `finetune/config.py` 中的时间范围保持一致
- **滑动窗口**: 20 天历史数据预测 7 天未来数据
- **公平比较**: 相同的数据集、相同的评估标准

### 数据集划分

- **训练时间范围**: ["2021-09-23", "2024-12-31"]
- **验证时间范围**: ["2025-01-01", "2025-08-31"]
- **测试时间范围**: ["2025-08-01", "2025-10-01"]

### 性能指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **IC**: 信息系数（皮尔逊相关系数）
- **RankIC**: 排序信息系数
- **方向准确率**: 预测方向准确率

### 使用方法

```bash
python experiment_runner.py
```


### 依赖项

- Python 3.8+
- PyTorch
- TensorFlow (可选)
- scikit-learn
- pandas
- numpy
- statsmodels
- pmdarima

*注：此项目仅为演示目的，不构成投资建议。实际量化交易需要更复杂的风险管理和投资组合优化技术。*