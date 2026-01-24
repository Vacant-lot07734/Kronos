import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from .io import split_by_date

class GlobalMinMax:
    """
    用所有股票的训练段数据fit一个全局MinMaxScaler，避免泄露：只fit train。
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, stock_dfs: List[Tuple[str, pd.DataFrame]], date_col: str, features: List[str],
            train_end: str, val_end: str):
        chunks = []
        for _, df in stock_dfs:
            train_df, _, _ = split_by_date(df, date_col, train_end, val_end)
            if len(train_df) > 0:
                chunks.append(train_df[features].values.astype(np.float32))
        if not chunks:
            raise ValueError("没有可用于fit scaler 的训练数据")
        all_train = np.concatenate(chunks, axis=0)
        self.scaler.fit(all_train)
        return self

    def transform_df(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        out = df.copy()
        out[features] = self.scaler.transform(out[features].values.astype(np.float32)).astype(np.float32)
        return out

    def inverse_transform(self, arr_2d: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(arr_2d)

class MultiStockWindowDataset(Dataset):
    """
    每个样本来自某股票:
      X: [lookback, 6], y: [pred_len, 6]
    split:
      train: 只用 train段
      val:   只用 val段
      test:  用 val尾部lookback做上下文 + test段（更贴近真实）
    """
    def __init__(
        self,
        stock_dfs: List[Tuple[str, pd.DataFrame]],
        split: str,
        date_col: str,
        features: List[str],
        lookback: int,
        pred_len: int,
        train_end: str,
        val_end: str,
        scaler: Optional[GlobalMinMax] = None,
        use_test_context_from_val: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.samples = []
        self.date_col = date_col
        self.features = features
        self.lookback = lookback
        self.pred_len = pred_len

        for stock_id, df in stock_dfs:
            tr, va, te = split_by_date(df, date_col, train_end, val_end)

            if split == "train":
                cur = tr
            elif split == "val":
                cur = va
            else:
                if use_test_context_from_val:
                    if len(va) < lookback:
                        continue
                    ctx = pd.concat([va.iloc[-lookback:].copy(), te], axis=0).reset_index(drop=True)
                    cur = ctx
                else:
                    cur = te

            if len(cur) < lookback + pred_len:
                continue

            if scaler is not None:
                cur = scaler.transform_df(cur, features)

            values = cur[features].values.astype(np.float32)
            times = cur[date_col].to_numpy()

            n = len(cur)
            for i in range(n - lookback - pred_len + 1):
                x = values[i:i+lookback]
                y = values[i+lookback:i+lookback+pred_len]
                xt = times[i:i+lookback]
                yt = times[i+lookback:i+lookback+pred_len]

                # 若split=test且用了val上下文，确保标签落在 test_start 之后（即 > val_end）
                if split == "test" and use_test_context_from_val:
                    if pd.to_datetime(yt[0]) <= pd.to_datetime(val_end):
                        continue

                self.samples.append((stock_id, x, y, xt, yt))

        print(f"{split} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stock_id, x, y, xt, yt = self.samples[idx]
        # 训练不需要时间戳，避免 datetime64 被 collate
        return (
            torch.from_numpy(x),  # [lookback,6]
            torch.from_numpy(y),  # [pred_len,6]
            stock_id
        )
