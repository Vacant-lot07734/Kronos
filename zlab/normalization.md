# 微调数据归一化

[`Config.normalize_with_context_only`](../finetune/config.py) 默认启用。[`QlibDataset`](../finetune/dataset.py) 在训练和验证中，逐特征仅用前 `lookback_window` 个历史时间步计算均值和标准差，再对整个样本窗口执行 z-score 归一化和裁剪。未来数据不参与统计量估计。

历史结果复现可显式设置环境变量 `KRONOS_NORMALIZE_WITH_CONTEXT_ONLY=0`，恢复使用完整窗口计算统计量的旧行为；此时归一化后的历史输入会依赖未来数据。

此配置只控制微调数据加载，不改变 `KronosPredictor` 或主仓库 `zlab.kronos` 的 zero-shot 推理。
