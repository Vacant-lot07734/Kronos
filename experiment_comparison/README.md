
---
# KLine Forecast Project (LSTM / Informer / ARIMA)

本项目用于 **日线 K 线序列预测**，数据集为 **几千支股票，每只股票一个 CSV 文件**。  
任务目标：用 **过去 20 天的 6 维 K 线特征** 预测 **目标 7 天的 6 维 K 线特征**。

支持模型：
- ✅ LSTM（全局训练，GPU）
- ✅ Informer（全局训练，GPU）
- ✅ ARIMA（逐股票逐特征拟合，无需全局训练）

支持功能：
- ✅ 单股票预测（预测文件最后 7 天，方便与真实值对比）
- ✅ 多股票批量预测（输出每股对比 CSV / close 对比图）
- ✅ 汇总指标（close）：MSE / MAE
- ✅ RankIC（横截面 Spearman）统计与导出

---

## 1. 数据格式要求

每只股票一个 CSV 文件，至少包含以下列（默认）：

- `datetime`（日期列，日线）
- `open, high, low, close, vol, amount`（6 维特征）

示例：

| datetime | open | high | low | close | vol | amount |
|---|---:|---:|---:|---:|---:|---:|
| 2021-11-11 | 17.35 | 18.43 | 17.32 | 18.35 | 2084729 | 3752413 |


## 2. 项目结构

```

kline_lstm/
├─ requirements.txt
├─ README.md
├─ configs/
│  └─ default.yaml
└─ src/
├─ config.py
├─ utils.py
├─ data/
│  ├─ io.py
│  └─ dataset.py
├─ models/
│  ├─ lstm.py
│  ├─ informer.py
│  └─ arima.py
├─ train_lstm.py                  # LSTM 训练
├─ predict_lstm.py                # LSTM 单股票预测（最后7天对比+画图）
├─ train_informer.py         # Informer 训练
├─ predict_informer.py       # Informer 单股票预测（最后7天对比+画图）
├─ predict_arima.py          # ARIMA 单股票预测（最后7天对比+画图）
└─ batch_predict_last7.py    # ✅ 多股票批量预测（输出CSV/图/汇总/RankIC）

````

---

## 3. 安装环境

建议使用 conda：



ARIMA 依赖：

```bash
pip install statsmodels
```

---

## 4. 配置文件说明（configs/default.yaml）

你需要重点修改：

```yaml
csv_dir: "../cache_data"        # CSV目录（每只股票一个csv）
pattern: "*.csv"
max_files: null             # 可限制只读前N只股票

date_col: "datetime"
features: ["open","high","low","close","vol","amount"]

lookback: 20
pred_len: 7

train_end: "2022-12-31"
val_end:   "2023-06-30"

batch_size: 256
epochs: 30
lr: 0.001
patience: 6
num_workers: 2

hidden_dim: 128
num_layers: 2
dropout: 0.1

ckpt_dir: "checkpoints"
ckpt_name: "lstm_kline.pth"
```

Informer 参数（推荐 full attention，更稳定）：

```yaml
informer:
  d_model: 128
  n_heads: 4
  e_layers: 2
  d_layers: 1
  d_ff: 256
  dropout: 0.1
  attn: "full"
  factor: 5
  label_len: 10
```

---

## 5. 预测逻辑说明（重要）

为了方便与真实值对比，本项目 **预测的是每个 CSV 文件最后 7 天**，而不是“最后一天之后的未来 7 天”。

即：

* 输入：倒数第 7 天之前的 20 天（共 20 天）
* 输出：预测该文件最后 7 天（共 7 天）
* 对比：预测结果 vs CSV 中真实最后 7 天

---

## 6. LSTM 模型（全局训练 + 单股票预测）

### 6.1 训练

```bash
python -m src.train_lstm --config configs/default.yaml
```

会输出 checkpoint：

```
checkpoints/lstm_kline.pth
```

### 6.2 单股票预测（最后7天对比 + close 对比图）

```bash
python -m src.predict_lstm --config configs/default.yaml --stock_csv ../cache_data/000001.SZ.csv
```

输出包括：

* 最后 7 天日期
* Pred（7×6）
* True（7×6）
* close 真实 vs 预测对比图（保存到 outputs/）

---

## 7. Informer 模型（全局训练 + 单股票预测）

> 注意：你当前 lookback=20、decoder_len=17 序列很短，建议使用 `attn: full` 更稳定。

### 7.1 训练

```bash
python -m src.train_informer --config configs/default.yaml
```

会输出 checkpoint：

```
checkpoints/informer_lstm_kline.pth
```

### 7.2 单股票预测（最后7天对比 + close 对比图）

```bash
python -m src.predict_informer --config configs/default.yaml --stock_csv ../cache_data/000001.SZ.csv
```

---

## 8. ARIMA 模型（无需全局训练）

ARIMA 的训练方式是 **每次预测时调用 fit() 在当前历史序列上拟合参数**。
本项目默认做法：对 6 维分别拟合 ARIMA，然后输出 7×6。

### 8.1 单股票预测（最后7天对比 + close 对比图）

```bash
python -m src.arima_predict --config configs/default.yaml --stock_csv ../cache_data/000001.SZ.csv --p 1 --d 1 --q 0
```

只预测 close（速度更快）：

```bash
python -m src.arima_predict --config configs/default.yaml --stock_csv ../cache_data/000001.SZ.csv --only_close --p 1 --d 1 --q 0
```

> 提示：ARIMA 在短序列（20天）上可能出现收敛警告或预测“接近水平线”，属于常见现象。

---

# ✅ 9. 多股票批量预测（最后7天对比 + CSV/图 + 汇总指标 + RankIC）

该功能由脚本 `src/batch_predict_last7.py` 提供，支持三种模型统一调用：

* `--model lstm`
* `--model informer`
* `--model arima`

批量预测的任务是：**对每个股票文件，预测该文件最后 7 天，并与真实值对齐对比。**

---

## 9.1 输出内容说明

假设输出目录为 `outputs_batch_lstm/`，将生成：

1）每只股票的对比 CSV：

* `outputs_batch_lstm/<stock_id>_lstm_last7_compare.csv`

包含列：

* `datetime`
* `true_open ... true_amount`
* `pred_open ... pred_amount`

2）每只股票的 close 对比图（可选）：

* `outputs_batch_lstm/<stock_id>_lstm_close_last7.png`

3）汇总指标（每股一行）：

* `outputs_batch_lstm/summary_lstm.csv`

字段包含：

* `stock_id`
* `mse_close`
* `mae_close`
* `error`（若预测失败）

4）RankIC 按日期导出（若启用 RankIC 功能）：

* `outputs_batch_lstm/rankic_by_date_lstm.csv`

字段包含：

* `datetime`
* `horizon`（1~7）
* `rankic`
* `n_stocks`

5）整体指标记录：

* `outputs_batch_lstm/metrics_lstm.txt`

包含：

* 全部股票平均 MSE(close) / MAE(close)
* RankIC(h=1) mean / IR

---

## 9.2 批量预测 LSTM

```bash
python -m src.batch_predict_last7 --config configs/default.yaml --model lstm --out_dir outputs_batch_lstm --plot
```

只跑前 200 支股票：

```bash
python -m src.batch_predict_last7 --config configs/default.yaml --model lstm --out_dir outputs_batch_lstm --max_files 200 --plot
```

---

## 9.3 批量预测 Informer

```bash
python -m src.batch_predict_last7 --config configs/default.yaml --model informer --out_dir outputs_batch_informer --plot
```

---

## 9.4 批量预测 ARIMA（无需 ckpt）

```bash
python -m src.batch_predict_last7 --config configs/default.yaml --model arima --out_dir outputs_batch_arima --plot --p 1 --d 0 --q 0
```

> 建议短序列优先尝试 `(1,0,0)` 或 `(0,1,1)`，避免拟合不稳定或水平线预测。

---

## 9.5 画图说明

* `--plot`：保存 close 对比图到 out_dir
* `--show`：弹窗显示（不建议批量时开启）

示例：

```bash
python -m src.batch_predict_last7 --config configs/default.yaml --model lstm --out_dir outputs_batch_lstm --plot
```

---

## 9.6 RankIC 说明

RankIC 为横截面指标：
对同一天（最后7天的每一天）所有股票的 **预测收益排序 vs 真实收益排序** 计算 Spearman 相关。

本项目默认使用：

* 因子：预测累计收益（pred_close 相对预测窗口起点的收益）
* 标签：真实累计收益（true_close 相对同一起点的收益）
* 按 (datetime, horizon) 输出 RankIC，并汇总 horizon=1 的均值与 IR。

---

