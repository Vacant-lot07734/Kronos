#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据预测结果选择每日前5名股票

功能：
1. 读取 output_dir 中所有预测结果 CSV 文件
2. 计算 pred_return = close_d2 - last_close
3. 每天按 pred_return 从大到小排序，选出前5名
4. 输出格式：trade_date, ts_code, name, pred_return
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')


def load_stock_name_mapping(stock_code_file):
    """
    加载股票代码到名称的映射
    
    Args:
        stock_code_file: 股票代码文件路径（包含ts_code和name列）
    
    Returns:
        dict: {ts_code: name} 映射字典
    """
    if not os.path.exists(stock_code_file):
        print(f"Warning: Stock code file not found: {stock_code_file}")
        return {}
    
    try:
        df = pd.read_csv(stock_code_file)
        if 'ts_code' in df.columns and 'name' in df.columns:
            mapping = dict(zip(df['ts_code'], df['name']))
            print(f"Loaded {len(mapping)} stock name mappings")
            return mapping
        else:
            print(f"Warning: Required columns 'ts_code' and 'name' not found in {stock_code_file}")
            return {}
    except Exception as e:
        print(f"Error loading stock code file: {e}")
        return {}


def load_prediction_files(output_dir):
    """
    加载所有预测结果文件
    
    Args:
        output_dir: 预测结果输出目录
    
    Returns:
        list: 包含所有预测结果的DataFrame列表
    """
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        return []
    
    # 查找所有预测结果文件（格式：*_predictions.csv）
    prediction_files = glob(os.path.join(output_dir, "*_predictions.csv"))
    
    if not prediction_files:
        print(f"Warning: No prediction files found in {output_dir}")
        return []
    
    print(f"Found {len(prediction_files)} prediction files")
    
    all_predictions = []
    
    for file_path in prediction_files:
        try:
            df = pd.read_csv(file_path)
            
            # 检查必需的列
            required_cols = ['trade_date', 'ts_code', 'last_close', 'close_d2']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns in {os.path.basename(file_path)}: {missing_cols}")
                continue
            
            # 确保 trade_date 是日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            all_predictions.append(df)
            
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
            continue
    
    if all_predictions:
        # 合并所有预测结果
        combined_df = pd.concat(all_predictions, ignore_index=True)
        print(f"Loaded {len(combined_df)} prediction records from {len(all_predictions)} files")
        return combined_df
    else:
        return pd.DataFrame()


def calculate_pred_return(df):
    """
    计算 pred_return = close_d2 - last_close
    
    Args:
        df: 包含 last_close 和 close_d2 的 DataFrame
    
    Returns:
        DataFrame: 添加了 pred_return 列的 DataFrame
    """
    df = df.copy()
    
    # 计算 pred_return
    df['pred_return'] = df['close_d2'] - df['last_close']
    
    # 移除无效值
    df = df.dropna(subset=['pred_return', 'trade_date', 'ts_code'])
    
    return df


def select_top_stocks_by_date(df, top_n=5, stock_name_mapping=None):
    """
    按日期分组，每天选择 pred_return 最高的前 top_n 只股票
    
    Args:
        df: 包含 trade_date, ts_code, pred_return 的 DataFrame
        top_n: 每天选择的股票数量
        stock_name_mapping: 股票代码到名称的映射字典
    
    Returns:
        DataFrame: 包含 trade_date, ts_code, name, pred_return 的 DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    results = []
    
    # 按日期分组
    for trade_date, group in df.groupby('trade_date'):
        # 按 pred_return 从大到小排序
        sorted_group = group.sort_values('pred_return', ascending=False)
        
        # 选择前 top_n 名
        top_stocks = sorted_group.head(top_n)
        
        for _, row in top_stocks.iterrows():
            ts_code = row['ts_code']
            pred_return = row['pred_return']
            
            # 获取股票名称
            name = stock_name_mapping.get(ts_code, '') if stock_name_mapping else ''
            
            results.append({
                'trade_date': trade_date,
                'ts_code': ts_code,
                'name': name,
                'pred_return': pred_return
            })
    
    result_df = pd.DataFrame(results)
    
    # 按日期和 pred_return 排序
    if not result_df.empty:
        result_df = result_df.sort_values(['trade_date', 'pred_return'], ascending=[True, False])
        result_df = result_df.reset_index(drop=True)
    
    return result_df


def format_output_date(date):
    """
    格式化日期为 YYYYMMDD 格式（与参考文件一致）
    
    Args:
        date: pandas Timestamp 或 datetime
    
    Returns:
        str: 格式化的日期字符串
    """
    if isinstance(date, pd.Timestamp):
        return date.strftime('%Y%m%d')
    elif isinstance(date, str):
        # 尝试解析
        try:
            dt = pd.to_datetime(date)
            return dt.strftime('%Y%m%d')
        except:
            return date
    else:
        return str(date)


def main():
    """
    主函数
    """
    # 设置参数
    output_dir = "output/predictions"  # 预测结果输出目录
    stock_code_file = "extra/all_stock_code_more.csv"  # 股票代码文件（用于获取股票名称）
    output_file = "output/select_top_stocks_by_EmoMarkets.csv"  # 输出文件路径
    top_n = 5  # 每天选择前N名
    
    print("=" * 60)
    print("根据预测结果选择每日前5名股票")
    print("=" * 60)
    
    # 1. 加载股票名称映射
    print("\n[1/4] 加载股票名称映射...")
    stock_name_mapping = load_stock_name_mapping(stock_code_file)
    
    # 2. 加载所有预测结果文件
    print(f"\n[2/4] 加载预测结果文件 from {output_dir}...")
    predictions_df = load_prediction_files(output_dir)
    print(type(predictions_df))
    if len(predictions_df)==0:
        print("Error: No valid prediction data found")
        return
    
    # 3. 计算 pred_return
    print("\n[3/4] 计算 pred_return (close_d2 - last_close)...")
    predictions_df = calculate_pred_return(predictions_df)
    
    print(f"Valid predictions: {len(predictions_df)}")
    print(f"Date range: {predictions_df['trade_date'].min()} to {predictions_df['trade_date'].max()}")
    print(f"Unique stocks: {predictions_df['ts_code'].nunique()}")
    print(f"Unique dates: {predictions_df['trade_date'].nunique()}")
    
    # 4. 按日期选择前 top_n 名股票
    print(f"\n[4/4] 按日期选择前 {top_n} 名股票...")
    top_stocks_df = select_top_stocks_by_date(
        predictions_df, 
        top_n=top_n, 
        stock_name_mapping=stock_name_mapping
    )
    
    if top_stocks_df.empty:
        print("Error: No top stocks selected")
        return
    
    # 格式化日期为 YYYYMMDD 格式（与参考文件一致）
    top_stocks_df['trade_date'] = top_stocks_df['trade_date'].apply(format_output_date)
    
    # 5. 保存结果
    print(f"\n保存结果到 {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    top_stocks_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 完成！")
    print(f"   总记录数: {len(top_stocks_df)}")
    print(f"   日期范围: {top_stocks_df['trade_date'].min()} 到 {top_stocks_df['trade_date'].max()}")
    print(f"   唯一日期数: {top_stocks_df['trade_date'].nunique()}")
    print(f"   平均每天股票数: {len(top_stocks_df) / top_stocks_df['trade_date'].nunique():.1f}")
    print(f"\n结果已保存到: {output_file}")
    
    # 显示前几行示例
    print("\n前10行示例:")
    print(top_stocks_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


