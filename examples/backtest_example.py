# file:F:\Kronos\examples\backtest_example.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
import torch
from pathlib import Path
import sys
from glob import glob

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))


class StockPredictorWithRealModel:
    def __init__(self, model_path, tokenizer_path, lookback=30, pred_len=2):
        """
        初始化股票预测器（使用真实模型）

        Args:
            model_path: 模型路径
            tokenizer_path: 分词器路径
            lookback: 历史回看天数
            pred_len: 预测天数
        """
        self.lookback = lookback
        self.pred_len = pred_len

        # 加载模型和分词器
        from new_model.emoMarkets import MoEEmotionEnhancedTokenizer, EmotionEnhancedWithMoE, EmoMarketsPredictorWithMoE

        self.tokenizer = MoEEmotionEnhancedTokenizer.from_pretrained(tokenizer_path)
        self.model = EmotionEnhancedWithMoE.from_pretrained(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.predictor = EmoMarketsPredictorWithMoE(
            self.model, self.tokenizer, device=self.device, max_context=512, clip=5
        )

    def load_kline_data(self, kline_file):
        """
        加载K线数据（与参考文件相同的方法）

        Args:
            kline_file: K线数据文件路径

        Returns:
            DataFrame: 加载的数据
        """
        df = pd.read_csv(kline_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # 添加ts_code列，从文件名获取
        if 'ts_code' not in df.columns:
            ts_code = os.path.basename(kline_file).replace('.csv', '')
            df['ts_code'] = ts_code
        df = df.sort_values(['ts_code', 'datetime']).reset_index(drop=True)
        return df

    def get_stock_list(self, cache_data_path, stock_code_file=None):
        """
        获取股票列表

        Args:
            cache_data_path: 缓存数据路径
            stock_code_file: 股票代码列表文件路径（可选），如果提供则只返回该文件中存在的股票

        Returns:
            list: 股票文件列表
        """
        csv_files = glob(os.path.join(cache_data_path, "*.csv"))
        # 过滤掉非股票数据文件
        stock_files = [f for f in csv_files if not os.path.basename(f).startswith('process')]

        # 如果提供了股票代码文件，则只返回该文件中存在的股票
        if stock_code_file and os.path.exists(stock_code_file):
            try:
                stock_code_df = pd.read_csv(stock_code_file)
                # 获取股票代码列表（去除.csv扩展名后的文件名应该匹配ts_code）
                if 'ts_code' in stock_code_df.columns:
                    valid_codes = set(stock_code_df['ts_code'].values)
                    # 过滤股票文件，只保留在代码列表中的
                    filtered_files = []
                    for f in stock_files:
                        stock_code = os.path.basename(f).replace('.csv', '')
                        if stock_code in valid_codes:
                            filtered_files.append(f)
                    stock_files = filtered_files
                    print(f"Filtered to {len(stock_files)} stocks from stock code file")
                else:
                    print(f"Warning: 'ts_code' column not found in {stock_code_file}")
            except Exception as e:
                print(f"Warning: Error loading stock code file {stock_code_file}: {e}")

        stock_files.sort()
        return stock_files

    def prepare_data_for_prediction(self, kline_df, emotion_df=None, lookback_window=14, pred_len=7,
                                    pred_start_idx=None):
        """
        准备预测数据，使用指定位置的历史数据预测未来几天（与参考文件相同的方法）

        Args:
            kline_df: K线数据
            emotion_df: 情绪数据
            lookback_window: 回看窗口
            pred_len: 预测长度
            pred_start_idx: 预测起始索引

        Returns:
            tuple: 包含预测数据的元组
        """
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
        # print("历史数据索引范围:", hist_start_idx, "到", hist_end_idx)

        # 包含所有字段（包括vol和amount）用于模型输入
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
                # 检查情绪数据的索引类型
                # print(f"Emotion dataframe index type: {emotion_df.index.dtype}")

                # 获取历史情绪data - 使用日期对齐
                emotion_dates = x_timestamp.dt.date
                emotion_dates_dt = pd.to_datetime(emotion_dates)

                if pd.api.types.is_datetime64_any_dtype(emotion_df.index):
                    # 如果情绪数据索引已经是datetime类型
                    try:
                        emotion_data = emotion_df.loc[emotion_dates_dt].copy()
                    except KeyError:
                        # 如果精确匹配失败，使用近似匹配
                        emotion_data = emotion_df.reindex(emotion_dates_dt, method='nearest', fill_value=0.0).copy()
                else:
                    # 如果情绪数据索引不是datetime类型，尝试转换或重新构建索引
                    # 假设情绪数据有一个日期列
                    if 'date' in emotion_df.columns or 'datetime' in emotion_df.columns:
                        date_col = 'date' if 'date' in emotion_df.columns else 'datetime'
                        emotion_df_with_date_index = emotion_df.set_index(pd.to_datetime(emotion_df[date_col]))

                        try:
                            emotion_data = emotion_df_with_date_index.loc[emotion_dates_dt].copy()
                        except KeyError:
                            emotion_data = emotion_df_with_date_index.reindex(emotion_dates_dt, method='nearest',
                                                                              fill_value=0.0).copy()
                    else:
                        print("Warning: Emotion data doesn't have a recognizable date column")
                        emotion_data = None

                if emotion_data is not None and not emotion_data.empty:
                    # 选数值型列并确保形状正确
                    emotion_data = emotion_data.select_dtypes(include=[np.number])
                    if emotion_data.empty:
                        print("Warning: No numeric emotion data available")
                        emotion_data = None

                # 获取未来情绪数据 - 使用日期对齐
                y_emotion_dates = actual_y_timestamp.dt.date
                y_emotion_dates_dt = pd.to_datetime(y_emotion_dates)

                if pd.api.types.is_datetime64_any_dtype(emotion_df.index):
                    try:
                        y_emotion_data = emotion_df.loc[y_emotion_dates_dt].copy()
                    except KeyError:
                        y_emotion_data = emotion_df.reindex(y_emotion_dates_dt, method='nearest', fill_value=0.0).copy()
                else:
                    if 'date' in emotion_df.columns or 'datetime' in emotion_df.columns:
                        date_col = 'date' if 'date' in emotion_df.columns else 'datetime'
                        emotion_df_with_date_index = emotion_df.set_index(pd.to_datetime(emotion_df[date_col]))

                        try:
                            y_emotion_data = emotion_df_with_date_index.loc[y_emotion_dates_dt].copy()
                        except KeyError:
                            y_emotion_data = emotion_df_with_date_index.reindex(y_emotion_dates_dt, method='nearest',
                                                                                fill_value=0.0).copy()
                    else:
                        y_emotion_data = None

                if y_emotion_data is not None and not y_emotion_data.empty:
                    y_emotion_data = y_emotion_data.select_dtypes(include=[np.number])
                    if y_emotion_data.empty:
                        y_emotion_data = None

            except Exception as e:
                print(f"Warning: Error processing emotion data: {e}")
                emotion_data = None
                y_emotion_data = None

        return x_df, x_timestamp, y_timestamp, emotion_data, y_emotion_data, actual_y_df, actual_y_timestamp

    def predict_single_stock(self, stock_data, start_date, end_date, emotion_df=None):
        """
        预测单个股票在指定时间区间内的数据

        Args:
            stock_data: 单个股票的历史数据
            start_date: 预测开始日期
            end_date: 预测结束日期
            emotion_df: 情绪数据

        Returns:
            DataFrame: 预测结果
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 获取股票在指定时间范围内的交易日
        stock_trading_dates = stock_data[
            (stock_data['datetime'] >= start_date) &
            (stock_data['datetime'] <= end_date)
            ]['datetime'].unique()

        results = []

        for current_date in stock_trading_dates:
            # 找到当前日期在数据中的索引
            date_idx = stock_data[stock_data['datetime'] == current_date].index
            if len(date_idx) == 0:
                continue  # 当前日期不在数据中，跳过

            date_idx = date_idx[0]

            # 计算预测起始索引
            pred_start_idx = date_idx - self.lookback

            # 检查是否有足够的历史数据
            if pred_start_idx < 0:
                continue  # 数据不足，跳过

            # 准备预测数据（包含vol和amount用于模型输入）
            x_df, x_timestamp, y_timestamp, emotion_data, y_emotion_data, actual_y_df, actual_y_timestamp = self.prepare_data_for_prediction(
                stock_data, emotion_df, self.lookback, self.pred_len, pred_start_idx
            )

            if x_df is None:
                continue  # 数据准备失败，跳过

            # 使用真实模型进行预测
            prediction_result = self._real_prediction(
                x_df, x_timestamp, y_timestamp, emotion_data, current_date, stock_data
            )

            if prediction_result is not None:
                results.append(prediction_result)

        # 创建结果DataFrame
        result_df = pd.DataFrame(results) if results else pd.DataFrame()

        # 在输出结果中过滤掉vol和amount相关的扩展字段
        if not result_df.empty:
            filtered_columns = []
            for col in result_df.columns:
                # 保留trade_date和ts_code，以及非vol和amount的扩展字段
                if col in ['trade_date', 'ts_code', 'last_close'] or not any(
                        col.startswith(field + '_d') for field in ['vol', 'amount']
                ):
                    filtered_columns.append(col)
            result_df = result_df[filtered_columns]

        return result_df

    def _real_prediction(self, x_df, x_timestamp, y_timestamp, emotion_data, current_date, stock_data):
        """
        使用真实模型进行预测

        Args:
            x_df: 输入数据（包含vol和amount）
            x_timestamp: 输入时间戳
            y_timestamp: 输出时间戳
            emotion_data: 情绪数据
            current_date: 当前日期
            stock_data: 股票数据

        Returns:
            dict: 预测结果字典
        """
        try:
            # 获取市场类型
            stock_code = stock_data['ts_code'].iloc[0]
            if stock_code.endswith('.SH'):
                market_type = 0  # 上证
            elif stock_code.endswith('.SZ'):
                market_type = 1  # 深证
            elif '300' in stock_code or '688' in stock_code:  # 创业板/科创板
                market_type = 2  # 创业板
            else:
                market_type = 3  # 默认为北交所或其他

            # 获取上一个交易日的收盘价
            last_close = x_df['close'].iloc[-1]  # 使用输入数据的最后一个收盘价作为上一个交易日的收盘价

            # 进行预测（使用包含vol和amount的数据）
            pred_result = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=self.pred_len,
                emotion_data=emotion_data,  # 历史情绪数据
                y_emotion_data=None,  # 未来情绪数据
                market_type=market_type,
                T=0.6,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )

            # 处理预测结果
            if isinstance(pred_result, torch.Tensor):
                if pred_result.is_cuda:
                    pred_np = pred_result.cpu().detach().numpy()
                else:
                    pred_np = pred_result.detach().numpy()

                # 包含所有字段（包括vol和amount）用于创建预测结果
                pred_df = pd.DataFrame(
                    pred_np,
                    index=y_timestamp,
                    columns=['open', 'high', 'low', 'close', 'vol', 'amount']
                )
            elif isinstance(pred_result, pd.DataFrame):
                pred_df = pred_result.set_index(y_timestamp)
            else:
                raise TypeError(f"Unexpected prediction result type: {type(pred_result)}")

            # 创建预测结果字典（包含所有字段）
            result = {
                'trade_date': current_date,
                'ts_code': stock_data['ts_code'].iloc[0],
                'last_close': last_close  # 添加上一个交易日的收盘价
            }

            # 为每个预测字段创建pred_len个预测值（包括vol和amount）
            for field in pred_df.columns:
                for i in range(1, self.pred_len + 1):
                    result[f'{field}_d{i}'] = pred_df.iloc[i - 1][field]

            return result

        except Exception as e:
            print(f"Prediction error for date {current_date}: {str(e)}")
            return None

    def run_prediction_pipeline(self, cache_data_path, start_date, end_date, output_dir, emotion_file=None,
                                stock_code_file=None):
        """
        运行完整的预测流程

        Args:
            cache_data_path: 缓存数据路径
            start_date: 预测开始日期
            end_date: 预测结束日期
            output_dir: 输出目录
            emotion_file: 情绪数据文件路径
            stock_code_file: 股票代码列表文件路径（可选），如果提供则只处理该文件中的股票
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载情绪数据（如果提供）
        emotion_df = None
        if emotion_file and os.path.exists(emotion_file):
            from new_prediction_example import load_emotion_data
            emotion_df = load_emotion_data(emotion_file)
            print("Loaded emotion data")

        # 获取股票文件列表
        print("Loading stock data...")
        stock_files = self.get_stock_list(cache_data_path, stock_code_file)
        print(f"Found {len(stock_files)} stock files to process")

        # 对每只股票进行预测
        for i, stock_file in enumerate(stock_files):
            stock_name = os.path.basename(stock_file).replace('.csv', '')
            print(f"Processing stock {i + 1}/{len(stock_files)}: {stock_name}")

            # 加载单只股票的数据（包含所有字段）
            stock_data = self.load_kline_data(stock_file)

            # 预测该股票在指定时间区间内的数据
            predictions = self.predict_single_stock(stock_data, start_date, end_date, emotion_df)

            if not predictions.empty:
                # 保存预测结果到文件（已过滤vol和amount字段）
                output_file = os.path.join(output_dir, f"{stock_name}_predictions.csv")
                predictions.to_csv(output_file, index=False)
                print(f"Saved predictions for {stock_name} to {output_file}")
            else:
                print(f"No predictions generated for {stock_name}")

        print("Prediction pipeline completed!")


def main():
    """
    主函数 - 执行完整的预测流程（使用真实模型）
    """
    # 设置参数
    cache_data_path = "../cache_data"  # 缓存数据路径
    start_date = "2025-10-01"  # 预测开始日期
    end_date = "2025-12-26"  # 预测结束日期
    output_dir = "output/predictions"  # 输出目录
    emotion_file = "../finetune/data/processed_datasets/emotion_data.csv"  # 情绪数据文件
    stock_code_file = "../extra/all_stock_code_more.csv"  # 股票代码列表文件
    model_path = "../finetune/outputs/full_new_models1/finetune_predictor_demo/checkpoints/best_model"
    tokenizer_path = "../finetune/outputs/full_new_models1/finetune_tokenizer_demo/checkpoints/best_tokenizer"
    lookback = 20  # 回看天数
    pred_len = 2  # 预测天数

    # 创建预测器实例
    predictor = StockPredictorWithRealModel(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        lookback=lookback,
        pred_len=pred_len
    )

    # 运行预测流程
    predictor.run_prediction_pipeline(
        cache_data_path=cache_data_path,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        emotion_file=emotion_file,
        stock_code_file=stock_code_file
    )


if __name__ == "__main__":
    main()
