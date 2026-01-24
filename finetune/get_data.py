# -*- coding: utf-8 -*-
import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

def get_all_stock_kline_data(stock_list_file: str = "./data/akshare_data",
                            save_dir: str = "./data/akshare_data_pre",
                            n_days: int = 300):
    """
    获取akshare_data目录中全部股票的K线数据并保存到akshare_data_pre目录

    Args:
        stock_list_file: 包含股票数据的目录路径
        save_dir: 保存K线数据的目录
        n_days: 获取最近n天的K线数据
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取akshare_data目录中所有CSV文件
    if not os.path.exists(stock_list_file):
        print(f"目录 {stock_list_file} 不存在")
        return

    csv_files = [f for f in os.listdir(stock_list_file) if f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个股票文件")

    # 遍历所有股票文件
    for i, csv_file in enumerate(csv_files):
        try:
            # 从文件名提取股票代码
            stock_symbol = csv_file.replace('.csv', '')
            print(f"[{i+1}/{len(csv_files)}] 正在处理 {stock_symbol}...")

            # 获取股票K线数据
            df = get_single_stock_kline(stock_symbol, n_days)

            if not df.empty:
                # 保存数据
                save_path = os.path.join(save_dir, csv_file)
                df.to_csv(save_path, index=False, encoding="utf-8-sig")
                print(f"  ✓ 已保存 {stock_symbol} 的K线数据，共 {len(df)} 条记录")
            else:
                print(f"  ✗ 未能获取到 {stock_symbol} 的K线数据")

        except Exception as e:
            print(f"  ✗ 处理 {stock_symbol} 时出错: {e}")
            continue

    print(f"所有股票数据处理完成，保存在 {save_dir} 目录中")

def get_single_stock_kline(stock_symbol: str, n_days: int = 300) -> pd.DataFrame:
    """
    获取单个股票的K线数据

    Args:
        stock_symbol: 股票代码
        n_days: 获取最近n天的K线数据

    Returns:
        pd.DataFrame: K线数据
    """
    try:
        # 处理股票代码格式
        if stock_symbol.startswith("sh") or stock_symbol.startswith("sz"):
            # 已经是akshare格式
            symbol = stock_symbol
        else:
            # 转换为akshare格式
            if stock_symbol.startswith(("6", "9")):
                symbol = f"sh{stock_symbol}"  # 上交所股票
            else:
                symbol = f"sz{stock_symbol}"  # 深交所股票

        # 计算日期范围
        end_date = datetime.today()
        start_date = end_date - timedelta(days=n_days * 2)  # 多获取一些天数以防节假日

        # 格式化日期
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        # 获取个股历史数据
        df = ak.stock_zh_a_hist(symbol=symbol,
                               period="daily",
                               start_date=start_date_str,
                               end_date=end_date_str,
                               adjust="qfq")  # 使用前复权数据

        if df.empty:
            return pd.DataFrame()

        # 重命名列
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount"
        }

        # 检查并重命名存在的列
        existing_columns = {col: column_mapping[col] for col in column_mapping if col in df.columns}
        df = df.rename(columns=existing_columns)

        # 只保留最近n_days的数据
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=False).head(n_days)

        return df

    except Exception as e:
        print(f"获取个股 {stock_symbol} K线数据失败: {e}")
        return pd.DataFrame()

def main():
    """主函数"""
    print("开始获取akshare_data中全部股票的K线数据...")
    get_all_stock_kline_data(
        stock_list_file="./data/akshare_data",
        save_dir="./data/akshare_data_pre",
        n_days=300
    )

if __name__ == "__main__":
    main()
