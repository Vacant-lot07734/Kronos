import os
import pickle
import pandas as pd
import numpy as np
from config import Config

class AkshareDataPreprocessor:
   def __init__(self):
       self.config = Config()
       self.data = {}

   def load_akshare_data(self):
       """加载akshare下载的数据"""
       data_path = "../cache_data/000001.SZ.csv"  # 直接指定CSV文件路径
       if not os.path.exists(data_path):
           raise FileNotFoundError("请先下载akshare数据")

       # 直接读取指定的CSV文件
       df = pd.read_csv(data_path)
       df['datetime'] = pd.to_datetime(df['datetime'])
       df = df.set_index('datetime')

       # 计算amt字段
       df['vol'] = df['vol']
       df['amt'] = df['amount']
       df = df[self.config.feature_list]
       df = df.dropna()

       # 从文件名中提取股票代码作为symbol
       symbol = "000001.SZ"

       if len(df) >= self.config.lookback_window + self.config.predict_window + 1:
           self.data[symbol] = df

   def prepare_dataset(self):
       """准备训练数据集"""
       print("Splitting data into train, validation, and test sets...")
       train_data, val_data, test_data = {}, {}, {}

       symbol_list = list(self.data.keys())
       for symbol in symbol_list:
           symbol_df = self.data[symbol]

           # Define time ranges from config.
           train_start, train_end = self.config.train_time_range
           val_start, val_end = self.config.val_time_range
           test_start, test_end = self.config.test_time_range

           # Create boolean masks for each dataset split.
           train_mask = (symbol_df.index >= train_start) & (symbol_df.index <= train_end)
           val_mask = (symbol_df.index >= val_start) & (symbol_df.index <= val_end)
           test_mask = (symbol_df.index >= test_start) & (symbol_df.index <= test_end)

           # Apply masks to create the final datasets.
           train_data[symbol] = symbol_df[train_mask]
           val_data[symbol] = symbol_df[val_mask]
           test_data[symbol] = symbol_df[test_mask]

       # Save the datasets using pickle.
       os.makedirs(self.config.dataset_path, exist_ok=True)
       with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
           pickle.dump(train_data, f)
       with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
           pickle.dump(val_data, f)
       with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
           pickle.dump(test_data, f)

       print("Datasets prepared and saved successfully.")

if __name__ == '__main__':
   preprocessor = AkshareDataPreprocessor()
   preprocessor.load_akshare_data()
   preprocessor.prepare_dataset()
