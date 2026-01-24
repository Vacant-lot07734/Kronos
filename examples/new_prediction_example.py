#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 从新模型模块导入情绪增强的Kronos相关类
from new_model.emoMarkets import MoEEmotionEnhancedTokenizer, EmotionEnhancedWithMoE, EmoMarketsPredictorWithMoE


def load_kline_data(kline_file):
    """加载K线数据"""
    df = pd.read_csv(kline_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def load_emotion_data(emotion_file):
    """加载情绪数据（从CSV文件读取）"""
    df = pd.read_csv(emotion_file)
    # 确保日期列存在并转为datetime格式
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    elif not isinstance(df.index, pd.DatetimeIndex):
        # 若索引不是datetime，尝试转为datetime
        df.index = pd.to_datetime(df.index)
    return df


def prepare_data_for_prediction(kline_df, emotion_df=None, lookback_window=20, pred_len=7, pred_start_idx=None):
    """准备预测数据，使用指定位置的历史数据预测未来几天"""

    # 确保datetime列是datetime类型
    kline_df['datetime'] = pd.to_datetime(kline_df['datetime'])

    # 确定预测起始位置
    if pred_start_idx is None:
        if len(kline_df) < lookback_window + pred_len:
            print("Warning: 数据不足，无法进行预测")
            return None, None, None, None, None, None, None
        pred_start_idx = len(kline_df) - lookback_window - pred_len
    else:
        if pred_start_idx + lookback_window + pred_len > len(kline_df):
            print("Warning: 指定位置数据不足，无法进行预测")
            return None, None, None, None, None, None, None

    # 历史数据索引范围
    hist_start_idx = pred_start_idx
    hist_end_idx = pred_start_idx + lookback_window
    print("历史数据索引范围:", hist_start_idx, "到", hist_end_idx)
    x_df = kline_df.iloc[hist_start_idx:hist_end_idx][['open', 'high', 'low', 'close', 'vol', 'amount']].copy()
    x_timestamp = kline_df.iloc[hist_start_idx:hist_end_idx]['datetime'].copy()

    # 实际值用于对比
    actual_start_idx = hist_end_idx
    actual_end_idx = actual_start_idx + pred_len
    actual_y_df = kline_df.iloc[actual_start_idx:actual_end_idx][
        ['open', 'high', 'low', 'close', 'vol', 'amount']].copy()
    actual_y_timestamp = kline_df.iloc[actual_start_idx:actual_end_idx]['datetime'].copy()

    # 从data集中获取预测时间戳
    y_timestamp = kline_df.iloc[actual_start_idx:actual_end_idx]['datetime'].copy()
    y_timestamp = pd.DatetimeIndex(y_timestamp)

    # 处理历史情绪数据
    emotion_data = None
    # 处理未来情绪数据（用于预测）
    y_emotion_data = None

    if emotion_df is not None:
        try:
            # 获取历史情绪data - 使用日期对齐
            emotion_dates = x_timestamp.dt.date
            emotion_dates_dt = pd.to_datetime(emotion_dates)

            # 使用更安全的方式处理日期匹配
            try:
                # 直接使用日期对齐，确保数据类型一致
                if isinstance(emotion_df.index, pd.DatetimeIndex):
                    # 如果情绪数据的索引是DatetimeIndex，直接reindex
                    emotion_data = emotion_df.reindex(emotion_dates_dt, method='nearest', fill_value=0.0).copy()
                else:
                    # 如果情绪数据有datetime列，使用该列进行合并
                    if 'datetime' in emotion_df.columns:
                        emotion_df_temp = emotion_df.set_index('datetime')
                        emotion_data = emotion_df_temp.reindex(emotion_dates_dt, method='nearest', fill_value=0.0).copy()
                    else:
                        # 如果都没有datetime信息，按位置索引处理
                        if len(emotion_df) >= len(emotion_dates_dt):
                            emotion_data = emotion_df.iloc[hist_start_idx:hist_end_idx].copy()
                        else:
                            print("Warning: Emotion data length is less than required, using zeros")
                            emotion_data = pd.DataFrame(0.0, index=emotion_dates_dt,
                                                      columns=emotion_df.columns if hasattr(emotion_df, 'columns') else ['value'])

            except Exception as e:
                print(f"Reindex failed: {e}")
                # 如果精确匹配失败，使用近似匹配
                emotion_data = emotion_df.reindex(emotion_dates_dt, method='nearest', fill_value=0.0).copy()

            # 选数值型列并确保形状正确
            emotion_data = emotion_data.select_dtypes(include=[np.number])
            if emotion_data.empty:
                print("Warning: No numeric emotion data available")
                emotion_data = None
            else:
                pass
                # 保持DataFrame格式，不转换为tensor
                # emotion_data = torch.tensor(emotion_data.values, dtype=torch.long)

            # 获取未来情绪数据 - 使用日期对齐
            y_emotion_dates = actual_y_timestamp.dt.date
            y_emotion_dates_dt = pd.to_datetime(y_emotion_dates)

            try:
                if isinstance(emotion_df.index, pd.DatetimeIndex):
                    y_emotion_data = emotion_df.reindex(y_emotion_dates_dt, method='nearest', fill_value=0.0).copy()
                else:
                    if 'datetime' in emotion_df.columns:
                        emotion_df_temp = emotion_df.set_index('datetime')
                        y_emotion_data = emotion_df_temp.reindex(y_emotion_dates_dt, method='nearest', fill_value=0.0).copy()
                    else:
                        # 如果都没有datetime信息，按位置索引处理
                        future_start_idx = actual_start_idx
                        future_end_idx = actual_end_idx
                        if len(emotion_df) >= future_end_idx:
                            y_emotion_data = emotion_df.iloc[future_start_idx:future_end_idx].copy()
                        else:
                            print("Warning: Emotion data length is less than required for future, using zeros")
                            y_emotion_data = pd.DataFrame(0.0, index=y_emotion_dates_dt,
                                                        columns=emotion_df.columns if hasattr(emotion_df, 'columns') else ['value'])
            except Exception as e:
                print(f"Future reindex failed: {e}")
                y_emotion_data = emotion_df.reindex(y_emotion_dates_dt, method='nearest', fill_value=0.0).copy()

            y_emotion_data = y_emotion_data.select_dtypes(include=[np.number])
            if y_emotion_data.empty:
                y_emotion_data = None
            else:
                pass

        except Exception as e:
            print(f"Warning: Error processing emotion data: {e}")
            y_emotion_data = None

    return x_df, x_timestamp, y_timestamp, emotion_data, y_emotion_data, actual_y_df, actual_y_timestamp



def calculate_performance_metrics(actual_df, pred_df):
    """
    计算预测性能指标

    Args:
        actual_df: 实际值DataFrame
        pred_df: 预测值DataFrame

    Returns:
        dict: 包含各种性能指标的字典
    """
    # 检查输入数据是否为空
    if actual_df.empty or pred_df.empty:
        return {
            'MSE': float('nan'),
            'MAE': float('nan'),
            'IC': float('nan'),
            'RankIC': float('nan')
        }

    # 确保两个DataFrame有相同的索引
    common_index = actual_df.index.intersection(pred_df.index)

    # 如果没有共同索引，尝试通过位置对齐
    if len(common_index) == 0:
        # 使用相同长度的数据进行比较
        min_len = min(len(actual_df), len(pred_df))
        if min_len == 0:
            return {
                'MSE': float('nan'),
                'MAE': float('nan'),
                'IC': float('nan'),
                'RankIC': float('nan')
            }

        # 取前min_len行进行比较
        actual = actual_df.iloc[:min_len]
        pred = pred_df.iloc[:min_len]
    else:
        actual = actual_df.loc[common_index]
        pred = pred_df.loc[common_index]

    # 只计算收盘价指标
    actual_close = actual['close'].values
    pred_close = pred['close'].values

    # 再次检查数据是否为空
    if len(actual_close) == 0 or len(pred_close) == 0:
        return {
            'MSE': float('nan'),
            'MAE': float('nan'),
            'IC': float('nan'),
            'RankIC': float('nan')
        }

    # 计算基本指标
    try:
        mse = mean_squared_error(actual_close, pred_close)
        mae = mean_absolute_error(actual_close, pred_close)


            # 计算IC (Information Coefficient) - 皮尔逊相关系数
        try:
            # 添加一个小的epsilon值避免除零错误
            epsilon = 1e-8
            if len(actual_close) > 1:
                # 检查是否存在常数序列
                actual_std = np.std(actual_close)
                pred_std = np.std(pred_close)

                if actual_std < epsilon or pred_std < epsilon:
                    ic = 0.0  # 如果任一序列是常数，则相关系数为0
                else:
                    ic = np.corrcoef(actual_close, pred_close)[0, 1]
            else:
                ic = float('nan')
        except Exception as e:
            print(f"Warning: Error calculating IC: {e}")
            ic = float('nan')

        # 计算RankIC - 基于排序的相关系数
        rank_actual = pd.Series(actual_close).rank(pct=True).values
        rank_pred = pd.Series(pred_close).rank(pct=True).values
        rankic = np.corrcoef(rank_actual, rank_pred)[0, 1] if len(actual_close) > 1 else float('nan')

    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        return {
            'MSE': float('nan'),
            'MAE': float('nan'),
            'IC': float('nan'),
            'RankIC': float('nan')
        }

    return {
        'MSE': mse,
        'MAE': mae,
        'IC': ic,
        'RankIC': rankic
    }


def plot_prediction_results(kline_df, pred_df, actual_y_df, actual_y_timestamp, stock_name, output_dir, metrics=None):
    """绘制预测结果，显示预测的7天和真实值对比"""

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取预测开始日期
    pred_start_date = pred_df.index[0] if len(pred_df) > 0 else None

    # 显示最近60天的历史数据和预测值
    recent_kline = kline_df.tail(60)

    plt.figure(figsize=(14, 7))

    # 绘制历史收盘价（蓝色）
    plt.plot(recent_kline['datetime'], recent_kline['close'],
             label='Historical Close Price', color='blue', linewidth=2)

    # 绘制预测值（红色）
    if len(pred_df) > 0:
        plt.plot(pred_df.index, pred_df['close'],
                 label='Predicted Close Price', color='red', linewidth=2, marker='o', markersize=8)

    # 绘制真实值（绿色）
    if not actual_y_df.empty:
        plt.plot(actual_y_timestamp, actual_y_df['close'],
                 label='Actual Close Price', color='green', linewidth=2, marker='s', markersize=8)

    # 添加预测开始标记
    if pred_start_date is not None:
        plt.axvline(x=pred_start_date, color='gray', linestyle='--', alpha=0.7,
                    label='Prediction Point')

    # 添加性能指标到标题
    title = f'{stock_name} - Close Price Prediction (Kronos+情绪)'
    if metrics:
        valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        if valid_metrics:
            metric_text = " | ".join([f"{k}: {v:.4f}" for k, v in valid_metrics.items() if k in ['MSE', 'MAE',  'IC', 'RankIC']])
            title += f'\n{metric_text}'

    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, f"{stock_name}_prediction_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to {plot_file}")


def save_metrics_summary(metrics_dict, output_dir):
    """
    保存所有股票的性能指标汇总

    Args:
        metrics_dict: 包含各股票性能指标的字典
        output_dir: 输出目录
    """
    # 过滤掉全为NaN的指标
    filtered_metrics = {}
    for stock, metrics in metrics_dict.items():
        valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        if valid_metrics:  # 只保存有有效指标的股票
            filtered_metrics[stock] = metrics

    if not filtered_metrics:
        print("No valid metrics to save")
        return

    # 转换为DataFrame
    metrics_df = pd.DataFrame.from_dict(filtered_metrics, orient='index')

    # 计算平均值（只对数值型列）
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        avg_metrics = metrics_df[numeric_columns].mean()

        # 保存详细指标
        metrics_file = os.path.join(output_dir, "performance_metrics.csv")
        metrics_df.to_csv(metrics_file)
        print(f"Performance metrics saved to {metrics_file}")

        # 打印汇总信息
        print("\n=== Performance Metrics Summary ===")
        print(f"Number of stocks processed: {len(metrics_df)}")
        print(f"Number of stocks with valid metrics: {len(filtered_metrics)}")
        print("\nAverage Performance Metrics:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("No numeric metrics to calculate averages")


def plot_overall_performance(metrics_dict, output_dir):
    """
    绘制所有股票的综合性能指标柱状图

    Args:
        metrics_dict: 包含各股票性能指标的字典
        output_dir: 输出目录
    """
    # 过滤掉全为NaN的指标
    filtered_metrics = {}
    for stock, metrics in metrics_dict.items():
        valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        if valid_metrics:  # 只保存有有效指标的股票
            filtered_metrics[stock] = metrics

    if not filtered_metrics:
        print("No valid metrics to plot")
        return

    # 转换为DataFrame
    metrics_df = pd.DataFrame.from_dict(filtered_metrics, orient='index')

    # 计算平均值（只对数值型列）
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        avg_metrics = metrics_df[numeric_columns].mean()

        # 创建柱状图
        plt.figure(figsize=(12, 8))

        # 绘制平均性能指标
        bars = plt.bar(avg_metrics.index, avg_metrics.values,
                      color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])

        # 在柱子上添加数值标签
        for bar, value in zip(bars, avg_metrics.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(avg_metrics.values),
                     f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        plt.title('Overall Performance Metrics (Average Across All Stocks)', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # 保存图像
        plot_file = os.path.join(output_dir, "overall_performance_metrics.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overall performance metrics plot saved to {plot_file}")

        # 创建每个指标的分布图
        plot_individual_metrics_distribution(metrics_df, output_dir)


def plot_individual_metrics_distribution(metrics_df, output_dir):
    """
    绘制各个性能指标的分布图

    Args:
        metrics_df: 包含所有股票性能指标的DataFrame
        output_dir: 输出目录
    """
    # 获取数值型列
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) == 0:
        return

    # 创建子图
    n_metrics = len(numeric_columns)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    # 绘制每个指标的分布
    for i, metric in enumerate(numeric_columns):
        ax = axes[i]
        data = metrics_df[metric].dropna()
        ax.hist(data, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution of {metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # 添加平均值线
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
        ax.legend()

    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "individual_metrics_distribution.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual metrics distribution plot saved to {plot_file}")


def get_top_stocks(cache_data_path, num_stocks=50):
    """获取前N支股票文件"""
    csv_files = glob(os.path.join(cache_data_path, "*.csv"))
    stock_files = [f for f in csv_files if not os.path.basename(f).startswith('process')]
    stock_files.sort()
    return stock_files[:num_stocks]


def main():
    # 配置参数
    cache_data_path = "../cache_data"
    emotion_file = "../finetune/data/processed_datasets/emotion_data.csv"  # 修改为CSV文件
    model_path = "../finetune/outputs/full_new_models1/finetune_predictor_demo/checkpoints/best_model"
    tokenizer_path = "../finetune/outputs/full_new_models1/finetune_tokenizer_demo/checkpoints/best_tokenizer"
    output_dir = "./emoMarkets_prediction_results"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 预测参数
    lookback_window = 20  # 使用20天历史数据
    pred_len = 5         # 预测7天
    pred_start_idx = None # 预测起始索引，None表示使用最新data
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading emotion data from CSV file...")  # 更新提示信息
    emotion_df = load_emotion_data(emotion_file)

    print("Loading model and tokenizer...")
    tokenizer = MoEEmotionEnhancedTokenizer.from_pretrained(tokenizer_path)
    model = EmotionEnhancedWithMoE.from_pretrained(model_path).to(device)  # 模型移到指定设备

    # 创建预测器时设置合适的参数
    predictor = EmoMarketsPredictorWithMoE(
        model, tokenizer, device=device, max_context=512, clip=5
    )

    stock_files = get_top_stocks(cache_data_path, 100)
    print(f"Found {len(stock_files)} stock files for prediction")

    # 存储所有股票的性能指标
    all_metrics = {}

    for i, kline_file in enumerate(stock_files):
        stock_name = os.path.basename(kline_file).replace('.csv', '')
        print(f"\n[{i + 1}/{len(stock_files)}] Processing {stock_name}...")
        # 添加市场类型信息（根据股票代码判断市场）
        if stock_name.endswith('.SH'):
            market_type =  0  # 上证
        elif stock_name.endswith('.SZ'):
            market_type =  1   # 深证
        elif '300' in stock_name or '688' in stock_name:  # 创业板/科创板
            market_type =  2   # 创业板
        else:
            market_type =  3   # 默认为北交所或其他
        try:
            kline_df = load_kline_data(kline_file)
            if len(kline_df) < lookback_window + pred_len:
                print(f"Warning: {stock_name} has less than {lookback_window + pred_len} days of data, skipping...")
                continue

            # 准备数据（保持情绪数据为DataFrame）
            x_df, x_timestamp, y_timestamp, emotion_data, y_emotion_data, actual_y_df, actual_y_timestamp = prepare_data_for_prediction(
                kline_df, emotion_df, lookback_window, pred_len, pred_start_idx,
            )

            if x_df is None:
                print(f"Warning: Failed to prepare data for {stock_name}, skipping...")
                continue

            print("Performing prediction...")
            # 传递情绪数据给predict方法
            pred_result = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                emotion_data=emotion_data,      # 历史情绪数据
                y_emotion_data=None,  # 未来情绪数据
                market_type=market_type,
                T=0.6,
                top_p=0.9,
                sample_count=10,
                verbose=True
            )

            # 处理预测结果（确保CPU转换）
            if isinstance(pred_result, torch.Tensor):
                # 先移到CPU -> 脱离计算图 -> 转为numpy -> 构建DataFrame
                if pred_result.is_cuda:
                    pred_np = pred_result.cpu().detach().numpy()
                else:
                    pred_np = pred_result.detach().numpy()
                pred_df = pd.DataFrame(
                    pred_np,
                    index=y_timestamp,
                    columns=['open', 'high', 'low', 'close', 'vol', 'amount']
                )
            elif isinstance(pred_result, pd.DataFrame):
                # 若已为DataFrame，确保索引是时间戳
                pred_df = pred_result.set_index(y_timestamp)
            else:
                raise TypeError(f"Unexpected prediction result type: {type(pred_result)}")

            # 计算性能指标
            if not actual_y_df.empty:
                metrics = calculate_performance_metrics(actual_y_df, pred_df)
                # 检查是否有有效指标
                valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
                if valid_metrics:
                    all_metrics[stock_name] = metrics
                    print(f"Performance metrics for {stock_name}:")
                    for metric, value in metrics.items():
                        if not np.isnan(value):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: NaN")
                else:
                    print(f"Warning: All metrics are NaN for {stock_name}")
            else:
                metrics = None
                print(f"Warning: No actual data for {stock_name}, skipping metrics calculation")

            # 保存预测结果
            output_file = os.path.join(output_dir, f"{stock_name}_prediction.csv")
            pred_df.to_csv(output_file)
            print(f"Prediction completed for {stock_name}. Results saved to {output_file}")

            # 绘制图表
            plot_prediction_results(kline_df, pred_df, actual_y_df, actual_y_timestamp, stock_name, plot_dir, metrics)

        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误栈
            continue

    print(f"\nAll predictions completed. Results saved to {output_dir}")
    print(f"Prediction plots saved to {plot_dir}")

    # 保存性能指标汇总
    if all_metrics:
        save_metrics_summary(all_metrics, output_dir)
        # 绘制综合性能指标柱状图
        plot_overall_performance(all_metrics, output_dir)


if __name__ == "__main__":
    main()
