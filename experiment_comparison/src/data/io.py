import os
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def read_one_csv(path: str, date_col: str, features: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{path} 缺少列 {date_col}")

    for c in features:
        if c not in df.columns:
            raise ValueError(f"{path} 缺少特征列 {c}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    keep_cols = [date_col] + features
    df = df[keep_cols].dropna().drop_duplicates(subset=[date_col])
    df[features] = df[features].astype(np.float32)
    return df

def load_all_stocks(
    csv_dir: str,
    pattern: str,
    date_col: str,
    features: List[str],
    min_len: int,
    max_files: Optional[int] = None
) -> List[Tuple[str, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
    if max_files is not None:
        paths = paths[:max_files]

    out = []
    for p in paths:
        try:
            df = read_one_csv(p, date_col, features)
            if len(df) >= min_len:
                stock_id = os.path.splitext(os.path.basename(p))[0]
                out.append((stock_id, df))
        except Exception as e:
            print(f"[跳过] {p} 读取失败: {e}")

    print(f"Loaded stocks: {len(out)}")
    return out

def split_by_date(df: pd.DataFrame, date_col: str, train_end: str, val_end: str):
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    train_df = df[df[date_col] <= train_end].copy()
    val_df   = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test_df  = df[df[date_col] > val_end].copy()
    return train_df, val_df, test_df
