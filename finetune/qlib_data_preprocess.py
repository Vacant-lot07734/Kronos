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

    # 加载情绪数据
    print("Loading emotion data...")
    emotion_df = pd.read_csv(emotion_data_path)

    # 处理日期列
    emotion_df['date'] = pd.to_datetime(emotion_df.iloc[:, 0])  # 假设第一列是日期
    emotion_df.set_index('date', inplace=True)

    # 情绪特征列定义 (与new_model/kronos.py中的EmotionEnhancedTokenizer.emotion_dim一致)
    emotion_feature_list = [
        'index_return_sh', 'index_return_sz', 'index_return_cyb', 'index_return_bse',
        'limit_up_1st', 'limit_up_2nd', 'limit_up_3rd', 'limit_up_4th',
        'limit_up_5th', 'limit_up_6th', 'limit_down', 'limit_up_20cm_1st',
        'limit_up_20cm', 'limit_up_30cm_1st', 'limit_up_30cm'
    ]

    # 列名映射 - 将CSV中的中文列名映射为模型需要的英文列名
    column_mapping = {
        '上证指数_涨跌幅(%)': 'index_return_sh',
        '深证成指_涨跌幅(%)': 'index_return_sz',
        '创业板指_涨跌幅(%)': 'index_return_cyb',
        '北证50_涨跌幅(%)': 'index_return_bse',
        '首板': 'limit_up_1st',
        '2连板': 'limit_up_2nd',
        '3连板': 'limit_up_3rd',
        '4连板': 'limit_up_4th',
        '5连板': 'limit_up_5th',
        '6连板及以上': 'limit_up_6th',
        '跌停板': 'limit_down',
        '20cm首板': 'limit_up_20cm_1st',
        '20cm连板': 'limit_up_20cm',
        '30cm首板': 'limit_up_30cm_1st',
        '30cm连板': 'limit_up_30cm'
    }

    # 重命名列
    emotion_df.rename(columns=column_mapping, inplace=True)

    # 确保列顺序与模型期望一致
    emotion_df = emotion_df[emotion_feature_list]

    # 加载所有股票数据
    all_data = {}
    print("Loading stock data...")
    for filename in os.listdir(raw_data_path):
        if filename.endswith('.csv'):
            symbol = filename[:-4]  # 移除 .csv 后缀
            filepath = os.path.join(raw_data_path, filename)
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                # 确保列名正确，选择需要的特征
                df = df[config.feature_list]
                df = df.dropna()
                if len(df) > config.lookback_window:
                    all_data[symbol] = df
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # 在训练集上计算归一化参数
    print("Calculating normalization parameters...")
    # 获取训练集时间范围内的所有数据用于计算归一化参数
    train_kline_data = []
    for symbol, df in all_data.items():
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        train_df = df[train_mask]
        if not train_df.empty:
            train_kline_data.append(train_df[config.feature_list].values)

    if train_kline_data:
        # 计算K线数据的全局均值和标准差
        all_train_data = np.concatenate(train_kline_data, axis=0)
        kline_mean = np.mean(all_train_data, axis=0)
        kline_std = np.std(all_train_data, axis=0)
        kline_std = np.where(kline_std == 0, 1, kline_std)  # 避免除以零
    else:
        # 默认值
        kline_mean = np.zeros(len(config.feature_list))
        kline_std = np.ones(len(config.feature_list))

    # 计算情绪数据的归一化参数
    train_emotion_mask = (emotion_df.index >= train_start) & (emotion_df.index <= train_end)
    train_emotion_df = emotion_df[train_emotion_mask]

    # 指数情绪数据处理 (z-score归一化)
    index_cols = ['index_return_sh', 'index_return_sz', 'index_return_cyb', 'index_return_bse']
    emotion_index_mean = train_emotion_df[index_cols].mean()
    emotion_index_std = train_emotion_df[index_cols].std()
    emotion_index_std = emotion_index_std.replace(0, 1)  # 避免除以零

    # 涨跌停板情绪数据处理 (归一化到[0,1])
    limit_cols = [col for col in emotion_feature_list if col not in index_cols]
    emotion_limit_min = train_emotion_df[limit_cols].min()
    emotion_limit_max = train_emotion_df[limit_cols].max()
    emotion_limit_max = emotion_limit_max.replace(0, 1)  # 避免除以零

    # 保存归一化参数
    normalization_params = {
        'kline_mean': kline_mean,
        'kline_std': kline_std,
        'emotion_index_mean': emotion_index_mean,
        'emotion_index_std': emotion_index_std,
        'emotion_limit_min': emotion_limit_min,
        'emotion_limit_max': emotion_limit_max
    }

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

    # 保存情绪数据
    with open(os.path.join(processed_data_path, "emotion_data.pkl"), 'wb') as f:
        pickle.dump(emotion_df, f)

    # 保存归一化参数
    with open(os.path.join(processed_data_path, "normalization_params.pkl"), 'wb') as f:
        pickle.dump(normalization_params, f)

    print("Data preprocessing finished.")

if __name__ == '__main__':
    preprocess_cache_data()
