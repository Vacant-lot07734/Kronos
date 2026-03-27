import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import Config


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
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
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.metadata = self._load_metadata()
        self.prediction_start, self.prediction_end = self._resolve_prediction_range()

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

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
                self.data[symbol] = df[['datetime'] + self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    if self._sample_in_prediction_range(df, i):
                        self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def _load_metadata(self) -> dict:
        metadata_path = f"{self.config.dataset_path}/metadata.json"
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _resolve_prediction_range(self):
        split_meta = self.metadata.get('splits', {}).get(self.data_type, {})
        prediction_start = split_meta.get('prediction_start', split_meta.get('score_start'))
        prediction_end = split_meta.get('prediction_end', split_meta.get('score_end'))
        if not prediction_start or not prediction_end:
            return None, None
        return pd.Timestamp(prediction_start), pd.Timestamp(prediction_end)

    def _sample_in_prediction_range(self, df, start_idx: int) -> bool:
        if self.prediction_start is None or self.prediction_end is None:
            return True

        prediction_start_idx = start_idx + self.config.lookback_window
        prediction_end_idx = prediction_start_idx + self.config.predict_window - 1
        if prediction_start_idx >= len(df) or prediction_end_idx >= len(df):
            return False

        prediction_start_time = df.iloc[prediction_start_idx]['datetime']
        prediction_end_time = df.iloc[prediction_end_idx]['datetime']
        return (
            self.prediction_start <= prediction_start_time
            and prediction_end_time <= self.prediction_end
        )

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

        Args:
            idx (int): Ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
                - x_stamp_tensor (torch.Tensor): The time feature tensor.
        """
        # Keep training stochastic, but make validation index-driven so the
        # DistributedSampler can deterministically shard samples across ranks.
        if self.data_type == 'train':
            sample_idx = self.py_rng.randint(0, len(self.indices) - 1)
        else:
            sample_idx = idx % len(self.indices)
        symbol, start_idx = self.indices[sample_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Match inference-time scaling for A/B experiments by using only the
        # historical lookback slice to estimate normalization statistics.
        if self.config.normalize_with_context_only:
            norm_source = x[:self.config.lookback_window]
        else:
            norm_source = x

        x_mean, x_std = np.mean(norm_source, axis=0), np.std(norm_source, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
