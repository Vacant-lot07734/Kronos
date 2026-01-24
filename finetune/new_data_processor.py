# new_data_processor.py
import os
import pickle
import pandas as pd
import numpy as np
from config import Config

def preprocess_cache_data():
    config = Config()
    raw_data_path = config.cache_data_path  # K线数据目录
    emotion_data_path = config.emotion_data_path  # 情绪数据文件路径
    processed_data_path = "./data/processed_datasets"  # 输出目录
    os.makedirs(processed_data_path, exist_ok=True)

    # 定义时间范围 (与config.py中保持一致)
    train_start, train_end = pd.to_datetime(config.train_time_range[0]), pd.to_datetime(config.train_time_range[1])
    val_start, val_end = pd.to_datetime(config.val_time_range[0]), pd.to_datetime(config.val_time_range[1])
    test_start, test_end = pd.to_datetime(config.test_time_range[0]), pd.to_datetime(config.test_time_range[1])

    # 修改 new_data_processor.py 中的情绪数据处理部分
    print("Loading emotion data...")
    emotion_df = pd.read_csv(emotion_data_path)

    # 确保第一列是日期列并转换为datetime类型
    # 使用更明确的日期解析方式
    emotion_df['date'] = pd.to_datetime(emotion_df.iloc[:, 0], format='%Y-%m-%d')  # 明确指定日期格式
    emotion_df.set_index('date', inplace=True)

    # 确保索引名称与股票数据一致
    emotion_df.index.name = 'datetime'  # 保持一致性

    # 情绪特征列定义
    emotion_feature_list = [
        'index_return_sh', 'index_return_sz', 'index_return_cyb', 'index_return_bse',
        'limit_up_1st', 'limit_up_2nd', 'limit_up_3rd', 'limit_up_4th',
        'limit_up_5th', 'limit_up_6th', 'limit_down', 'limit_up_20cm_1st',
        'limit_up_20cm', 'limit_up_30cm_1st', 'limit_up_30cm'
    ]

    # 确保列名匹配
    if list(emotion_df.columns[:15]) != emotion_feature_list:
        emotion_df.columns = emotion_feature_list + list(emotion_df.columns[15:])

    # 只保留需要的特征列
    emotion_df = emotion_df[emotion_feature_list]

    # 确保索引数据类型正确
    emotion_df.index = pd.to_datetime(emotion_df.index)

    # 修改股票数据加载部分，添加股票数量限制
    # 在 "加载所有股票数据" 部分替换为以下代码：

    # 加载所有股票数据（限制前300支）
    all_data = {}
    print("Loading stock data...")
    csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    # 限制只处理前300支股票
    csv_files = csv_files

    for filename in csv_files:
        if filename.endswith('.csv'):
            symbol = filename[:-4]  # 移除 .csv 后缀
            filepath = os.path.join(raw_data_path, filename)
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)

                # 确保索引是日期时间类型
                df.index = pd.to_datetime(df.index)

                # 统一列名处理
                column_mapping = {
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'vol',  # 统一使用 vol 而不是 volume
                    '成交额': 'amount'
                }

                # 应用列名映射
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df.rename(columns={old_col: new_col}, inplace=True)

                # 确保只保留需要的特征列（6列）
                required_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
                df = df[required_columns]
                df = df.dropna()

                # 确保索引是日期格式（保持日期作为索引）
                df.index.name = 'datetime'

                # 添加市场类型信息（根据股票代码判断市场）
                if symbol.endswith('.SH'):
                    df['market_type'] = 0  # 上证
                elif symbol.endswith('.SZ'):
                    df['market_type'] = 1  # 深证
                elif '300' in symbol or '688' in symbol:  # 创业板/科创板
                    df['market_type'] = 2  # 创业板
                else:
                    df['market_type'] = 3  # 默认为北交所或其他

                if len(df) > config.lookback_window:
                    all_data[symbol] = df

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # 分割数据集
    train_data, val_data, test_data = {}, {}, {}
    print("Splitting datasets...")
    for symbol, df in all_data.items():
        # 确保股票数据和情绪数据有共同的时间范围
        common_dates = df.index.intersection(emotion_df.index).sort_values()
        df = df.loc[common_dates]

        train_mask = (df.index >= train_start) & (df.index <= train_end)
        val_mask = (df.index >= val_start) & (df.index <= val_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        # 保持索引格式，不转换为列
        train_data[symbol] = df[train_mask]
        val_data[symbol] = df[val_mask]
        test_data[symbol] = df[test_mask]

    # 保存数据集
    with open(os.path.join(processed_data_path, "train_data.pkl"), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(processed_data_path, "val_data.pkl"), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(processed_data_path, "test_data.pkl"), 'wb') as f:
        pickle.dump(test_data, f)

    # 在保存情绪数据之前添加标准化处理
    print("Normalizing emotion data...")

    # 3.1 指数情绪数据处理 (z-score归一化)
    index_cols = ['index_return_sh', 'index_return_sz', 'index_return_cyb', 'index_return_bse']
    # 计算均值和标准差
    index_mean = emotion_df[index_cols].mean()
    index_std = emotion_df[index_cols].std()
    # 避免除以零
    index_std = index_std.replace(0, 1)
    # 应用z-score归一化
    emotion_df[index_cols] = (emotion_df[index_cols] - index_mean) / index_std
    # 异常值过滤 (3σ法则)
    emotion_df[index_cols] = emotion_df[index_cols].clip(-3, 3)

    # 3.2 涨跌停板情绪数据处理 (归一化到[0,1])
    limit_cols = [col for col in emotion_feature_list if col not in index_cols]
    # 计算最小值和最大值
    limit_min_values = emotion_df[limit_cols].min()
    limit_max_values = emotion_df[limit_cols].max()

    # 处理当最大值等于最小值的情况（防止除以零）
    limit_range = limit_max_values - limit_min_values
    # 对于全为相同值的列，将其范围设置为1，并将这些值映射到0.5
    zero_range_mask = (limit_range == 0)
    limit_range = limit_range.mask(zero_range_mask, 1)

    # 归一化到[0,1]
    emotion_df[limit_cols] = (emotion_df[limit_cols] - limit_min_values) / limit_range

    # 对于原本全为相同值的列，将其设置为0（而不是0.5，保持原始相对关系）
    for col in limit_cols:
        if zero_range_mask[col]:
            emotion_df[col] = emotion_df[col] * 0  # 全置为0，保持一致性

    # 异常值处理 (超过99分位数替换为99分位数)
    for col in limit_cols:
        p99 = emotion_df[col].quantile(0.99)
        emotion_df[col] = emotion_df[col].clip(upper=p99)
    # 确保在[0,1]区间内
    emotion_df[limit_cols] = emotion_df[limit_cols].clip(0, 1)

    # 特别处理可能仍然存在的NaN值
    emotion_df[limit_cols] = emotion_df[limit_cols].fillna(0)

    # 保存情绪数据（保持索引格式）
    with open(os.path.join(processed_data_path, "emotion_data.pkl"), 'wb') as f:
        pickle.dump(emotion_df, f)
    csv_file_path = os.path.join(processed_data_path, "emotion_data.csv")
    emotion_df.to_csv(csv_file_path)
    print("Data preprocessing finished.")

if __name__ == '__main__':
    preprocess_cache_data()
