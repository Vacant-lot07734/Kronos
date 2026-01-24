# new_dataset.py
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data with emotion data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation. It also
    incorporates emotion data aligned with the time series.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If [data_type](file://F:\Kronos\finetune\dataset.py#L26-L26) is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.emotion_data_path = "F:/Kronos/finetune/data/processed_datasets/emotion_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.emotion_data_path = "F:/Kronos/finetune/data/processed_datasets/emotion_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        # Load emotion data
        with open(self.emotion_data_path, 'rb') as f:
            self.emotion_data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())

        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list
        self.emotion_feature_list = self.config.emotion_feature_list
        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list + ['datetime', 'market_type']]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset with emotion data.

        Note: The [idx](file://F:\Kronos\deepspeed-0.18.2\csrc\includes\reduction_utils.h#L792-L792) argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
                - x_stamp_tensor (torch.Tensor): The time feature tensor.
                - emotion_tensor (torch.Tensor): The emotion data tensor.
                - market_type_tensor (torch.Tensor): The market type tensor.
        """
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # 获取市场类型
        market_type = win_df['market_type'].iloc[0]  # 假设整个窗口期内市场类型不变
        market_type_tensor = torch.tensor(market_type, dtype=torch.long)

        # Get corresponding emotion data (independent of symbol)
        emotion_features = np.zeros((self.window, len(self.emotion_feature_list)), dtype=np.float32)

        # 读取时间区间内部的情绪数据
        if hasattr(self, 'emotion_data') and self.emotion_data is not None:
            emotion_df = self.emotion_data
            # 检查datetime是否为索引
            if hasattr(emotion_df.index, 'name') and emotion_df.index.name in ['datetime', 'date', 'time', 'timestamp']:
                time_column = emotion_df.index.name
                # 获取窗口的起始和结束时间
                start_datetime = win_df.iloc[0]['datetime']
                end_datetime = win_df.iloc[-1]['datetime']

                # 使用索引进行时间区间筛选
                mask = (emotion_df.index >= start_datetime) & (emotion_df.index <= end_datetime)
                emotion_window = emotion_df.loc[mask]

                if len(emotion_window) > 0:
                    # 按时间排序（索引通常已经排序）
                    emotion_window = emotion_window.sort_index()
                    # 取情绪特征数据
                    emotion_values = emotion_window[self.emotion_feature_list].values.astype(np.float32)
                    # 填充到emotion_features中
                    min_len = min(len(emotion_values), self.window)
                    emotion_features[:min_len] = emotion_values[:min_len]
            else:
                # 尝试查找列中的时间字段
                time_column = None
                for col in ['datetime', 'date', 'time', 'timestamp']:
                    if col in emotion_df.columns:
                        time_column = col
                        break

                if time_column:
                    # 原有的列处理逻辑
                    start_datetime = win_df.iloc[0]['datetime']
                    end_datetime = win_df.iloc[-1]['datetime']

                    mask = (emotion_df[time_column] >= start_datetime) & (emotion_df[time_column] <= end_datetime)
                    emotion_window = emotion_df[mask]

                    if len(emotion_window) > 0:
                        emotion_window = emotion_window.sort_values(time_column)
                        emotion_values = emotion_window[self.emotion_feature_list].values.astype(np.float32)
                        min_len = min(len(emotion_values), self.window)
                        emotion_features[:min_len] = emotion_values[:min_len]
                else:
                    print(
                        f"Warning: 未找到时间列或时间索引. 可用列: {emotion_df.columns.tolist() if hasattr(emotion_df, 'columns') else 'N/A'}")
                    if hasattr(emotion_df.index, 'name'):
                        print(f"索引名: {emotion_df.index.name}")

        # Perform instance-level normalization.
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        emotion_tensor = torch.from_numpy(emotion_features)
        return x_tensor, x_stamp_tensor, emotion_tensor, market_type_tensor



if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='val')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp, try_emotion, try_market = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
        print(f"Sample emotion feature shape: {try_emotion.shape}")
        print(f"Sample market type: {try_market}")

    else:
        print("Dataset is empty.")
