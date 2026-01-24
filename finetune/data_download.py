import akshare as ak
import pandas as pd
import os
from datetime import datetime

# 创建数据目录
os.makedirs("./data/akshare_data_pre", exist_ok=True)

# 从 akshare_data 目录获取已有股票代码
akshare_data_path = "./data/akshare_data"
if not os.path.exists(akshare_data_path):
    raise FileNotFoundError("请先确保 akshare_data 目录存在且包含数据文件")

# 获取目录中所有 CSV 文件的股票代码
csv_files = [f for f in os.listdir(akshare_data_path) if f.endswith('.csv')]
symbols = [f.replace('.csv', '') for f in csv_files]  # 提取股票代码

print(f"从 akshare_data 目录找到 {len(symbols)} 只股票: {symbols[:10]}...")  # 显示前10个

# 限定时间范围
start_date = "20200101"
end_date = datetime.today().strftime("%Y%m%d")

# 下载历史数据
all_data = {}
for i, symbol in enumerate(symbols):
    try:
        print(f"[{i+1}/{len(symbols)}] 正在下载 {symbol} 数据...")
        # 获取个股历史数据（限定时间范围）
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if not df.empty:
            df = df.rename(columns={
                '日期': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            df = df.sort_index()

            if len(df) > 0:
                df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
                all_data[symbol] = df
                print(f"已下载 {symbol} 数据，共 {len(df)} 条记录 ({df.index.min().date()} 到 {df.index.max().date()})")
        else:
            print(f"警告: {symbol} 返回空数据")
    except Exception as e:
        print(f"下载 {symbol} 数据失败: {e}")

# 保存数据
print(f"\n开始保存 {len(all_data)} 只股票数据...")
for symbol, df in all_data.items():
    try:
        df.to_csv(f"./data/akshare_data_pre/{symbol}.csv")
        print(f"已保存 {symbol} 数据到 ./data/akshare_data_pre/{symbol}.csv")
    except Exception as e:
        print(f"保存 {symbol} 数据失败: {e}")

print(f"数据下载完成，共处理 {len(all_data)} 只股票")
