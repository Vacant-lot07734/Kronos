"""
Dual-stream dataset for C-group experiments.

Each sample returns four tensors:
    x_daily     (L_d + H + 1, 6)   -- daily OHLCVA normalised
    x_stamp_d   (L_d + H + 1, 5)   -- daily time features
    x_hourly    (L_h, 6)            -- hourly OHLCVA normalised (history only)
    x_stamp_h   (L_h, 5)           -- hourly time features

The daily part is identical to B-group's QlibDataset output.
The hourly part is the most recent L_h hourly bars ending at or before
the daily prediction start point (no future leakage).
"""

import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import Config


class DailyHourlyDataset(Dataset):
    """
    Dataset for C-group: daily main stream + hourly auxiliary stream.

    The pickle file should contain per-symbol dicts of the form::

        {symbol: {"daily": pd.DataFrame, "hourly": pd.DataFrame}}

    where both DataFrames are datetime-indexed.
    """

    def __init__(self, data_type: str = "train"):
        self.config = Config()
        if data_type not in ["train", "val"]:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type
        self.py_rng = random.Random(self.config.seed)

        self.hourly_window = int(self.config.__dict__.get(
            "hourly_window",
            int(os.getenv("KRONOS_HOURLY_WINDOW", "25"))
        ))

        if data_type == "train":
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)

        self.daily_window = self.config.lookback_window + self.config.predict_window + 1
        self.metadata = self._load_metadata()
        self.score_start, self.score_end = self._resolve_score_range()

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute valid (symbol, daily_start_idx) pairs
        self.indices = []
        print(f"[{data_type.upper()} C] Pre-computing sample indices...")
        for symbol in self.symbols:
            entry = self.data[symbol]
            daily_df = entry["daily"].reset_index()
            hourly_df = entry["hourly"].reset_index()

            n_daily = len(daily_df)
            n_possible = n_daily - self.daily_window + 1

            if n_possible <= 0:
                continue

            # Generate time features for daily
            daily_df["minute"] = daily_df["datetime"].dt.minute
            daily_df["hour"] = daily_df["datetime"].dt.hour
            daily_df["weekday"] = daily_df["datetime"].dt.weekday
            daily_df["day"] = daily_df["datetime"].dt.day
            daily_df["month"] = daily_df["datetime"].dt.month
            entry["daily"] = daily_df[["datetime"] + self.feature_list + self.time_feature_list]

            # Generate time features for hourly
            hourly_df["minute"] = hourly_df["datetime"].dt.minute
            hourly_df["hour"] = hourly_df["datetime"].dt.hour
            hourly_df["weekday"] = hourly_df["datetime"].dt.weekday
            hourly_df["day"] = hourly_df["datetime"].dt.day
            hourly_df["month"] = hourly_df["datetime"].dt.month
            entry["hourly"] = hourly_df[["datetime"] + self.feature_list + self.time_feature_list]

            for i in range(n_possible):
                if not self._sample_in_score_range(daily_df, i):
                    continue
                # Check hourly availability for this sample
                context_end_date = daily_df.iloc[i + self.config.lookback_window - 1]["datetime"]
                hourly_end_mask = hourly_df["datetime"] <= context_end_date + pd.Timedelta(hours=15)
                if hourly_end_mask.sum() >= self.hourly_window:
                    self.indices.append((symbol, i))

        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()} C] Found {len(self.indices)} valid samples. Using {self.n_samples} per epoch.")

    def _load_metadata(self) -> dict:
        path = f"{self.config.dataset_path}/metadata.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _resolve_score_range(self):
        split_meta = self.metadata.get("splits", {}).get(self.data_type, {})
        s, e = split_meta.get("score_start"), split_meta.get("score_end")
        if not s or not e:
            return None, None
        return pd.Timestamp(s), pd.Timestamp(e)

    def _sample_in_score_range(self, df, start_idx: int) -> bool:
        if self.score_start is None or self.score_end is None:
            return True
        target_end_idx = start_idx + self.config.lookback_window + self.config.predict_window - 1
        if target_end_idx >= len(df):
            return False
        t = df.iloc[target_end_idx]["datetime"]
        return self.score_start <= t <= self.score_end

    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.config.seed + epoch)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.data_type == "train":
            sample_idx = self.py_rng.randint(0, len(self.indices) - 1)
        else:
            sample_idx = idx % len(self.indices)
        symbol, start_idx = self.indices[sample_idx]
        entry = self.data[symbol]

        # --- Daily stream ---
        daily_df = entry["daily"]
        end_idx = start_idx + self.daily_window
        win_df = daily_df.iloc[start_idx:end_idx]

        x_d = win_df[self.feature_list].values.astype(np.float32)
        x_stamp_d = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalise daily features using context-only statistics
        if self.config.normalize_with_context_only:
            norm_source = x_d[: self.config.lookback_window]
        else:
            norm_source = x_d
        d_mean, d_std = np.mean(norm_source, axis=0), np.std(norm_source, axis=0)
        x_d = (x_d - d_mean) / (d_std + 1e-5)
        x_d = np.clip(x_d, -self.config.clip, self.config.clip)

        # --- Hourly stream ---
        hourly_df = entry["hourly"]
        context_end_date = daily_df.iloc[start_idx + self.config.lookback_window - 1]["datetime"]
        # Hourly bars up to context_end_date's close (15:00)
        hourly_cutoff = context_end_date + pd.Timedelta(hours=15)
        valid_hourly = hourly_df.loc[hourly_df["datetime"] <= hourly_cutoff]

        # Take last L_h bars
        if len(valid_hourly) >= self.hourly_window:
            hourly_win = valid_hourly.iloc[-self.hourly_window:]
        else:
            # Pad with zeros if insufficient (should rarely happen given index filtering)
            hourly_win = valid_hourly.iloc[-self.hourly_window:] if len(valid_hourly) > 0 else valid_hourly
            pad_len = self.hourly_window - len(hourly_win)
            if pad_len > 0:
                pad = pd.DataFrame(
                    np.zeros((pad_len, len(self.feature_list) + len(self.time_feature_list) + 1)),
                    columns=["datetime"] + self.feature_list + self.time_feature_list,
                )
                hourly_win = pd.concat([pad, hourly_win], ignore_index=True)

        x_h = hourly_win[self.feature_list].values.astype(np.float32)
        x_stamp_h = hourly_win[self.time_feature_list].values.astype(np.float32)

        # Normalise hourly features independently
        h_mean, h_std = np.mean(x_h, axis=0), np.std(x_h, axis=0)
        x_h = (x_h - h_mean) / (h_std + 1e-5)
        x_h = np.clip(x_h, -self.config.clip, self.config.clip)

        return (
            torch.from_numpy(x_d),
            torch.from_numpy(x_stamp_d),
            torch.from_numpy(x_h),
            torch.from_numpy(x_stamp_h),
        )


if __name__ == "__main__":
    print("Creating C-group training dataset...")
    ds = DailyHourlyDataset("train")
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        xd, xsd, xh, xsh = ds[0]
        print(f"x_daily shape: {xd.shape}, x_stamp_d shape: {xsd.shape}")
        print(f"x_hourly shape: {xh.shape}, x_stamp_h shape: {xsh.shape}")
