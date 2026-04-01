"""
C-group data preprocessing: builds train/val/test pickle datasets from
local daily + hourly CSV files.

Each symbol's value in the output pickle is a dict::

    {"daily": pd.DataFrame, "hourly": pd.DataFrame}

Only symbols that have sufficient data in **both** daily and hourly
across all three splits are retained.
"""

import argparse
import json
import os
import pickle
from glob import glob

import pandas as pd
from tqdm import tqdm

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build C-group train/val/test pickles from daily + hourly CSV."
    )
    parser.add_argument("--daily-csv-dir", type=str, help="Directory with *_qfq_day.csv files.")
    parser.add_argument("--hourly-csv-dir", type=str, help="Directory with *_qfq_60min.csv files.")
    parser.add_argument("--output-dir", type=str, help="Output directory for pickles.")
    parser.add_argument("--hourly-window", type=int, default=25, help="Hourly lookback window L_h.")
    parser.add_argument("--min-symbols", type=int, default=10)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_daily_df(csv_path: str, feature_list: list[str]) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    symbol = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else os.path.basename(csv_path).split("_")[0]
    rename_map = {"trade_date": "datetime", "amount": "amt", "volume": "vol"}
    df = df.rename(columns=rename_map)
    if "amt" not in df.columns:
        df["amt"] = df["vol"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d")
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    df = df[["datetime"] + feature_list].dropna().set_index("datetime")
    return symbol, df


def _load_hourly_df(csv_path: str, feature_list: list[str]) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    symbol = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else os.path.basename(csv_path).split("_")[0]
    rename_map = {"trade_date": "datetime", "trade_time": "datetime", "amount": "amt", "volume": "vol"}
    df = df.rename(columns=rename_map)
    if "amt" not in df.columns:
        if "vol" in df.columns:
            df["amt"] = df["vol"] * df[["open", "high", "low", "close"]].mean(axis=1)
        else:
            df["amt"] = 0.0
    if "vol" not in df.columns:
        df["vol"] = 0.0

    # Parse datetime: try multiple formats
    dt_col = df["datetime"].astype(str)
    try:
        df["datetime"] = pd.to_datetime(dt_col, format="%Y%m%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            df["datetime"] = pd.to_datetime(dt_col, format="%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            df["datetime"] = pd.to_datetime(dt_col, format="mixed", dayfirst=False)

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    df = df[["datetime"] + feature_list].dropna().set_index("datetime")
    return symbol, df


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def _slice_daily_with_buffer(df: pd.DataFrame, score_start: pd.Timestamp, score_end: pd.Timestamp, lookback_window: int):
    if df.empty:
        return df
    split_df = df.loc[df.index <= score_end].copy()
    if split_df.empty:
        return split_df
    pos = split_df.index.searchsorted(score_start)
    buffered_pos = max(0, pos - lookback_window)
    return split_df.iloc[buffered_pos:]


def _slice_hourly(hourly_df: pd.DataFrame, daily_split_df: pd.DataFrame, hourly_window: int):
    """Slice hourly data to cover the daily split period + hourly lookback buffer."""
    if daily_split_df.empty or hourly_df.empty:
        return pd.DataFrame()

    # Earliest daily date in split (which already includes lookback buffer)
    earliest_daily = daily_split_df.index.min()
    latest_daily = daily_split_df.index.max()

    # hourly data up to end of latest_daily (just use <= latest_daily + 1 day to be safe)
    end_ts = latest_daily + pd.Timedelta(days=1)
    hourly_in_range = hourly_df.loc[hourly_df.index <= end_ts]

    if hourly_in_range.empty:
        return pd.DataFrame()

    # A trading day currently has 5 hourly bars: 09:30, 10:30, 11:30, 14:00, 15:00.
    # Estimate the daily buffer from bar count, then leave a few extra days for safety.
    earliest_hourly_needed = earliest_daily - pd.Timedelta(days=hourly_window // 5 + 5)
    hourly_slice = hourly_in_range.loc[hourly_in_range.index >= earliest_hourly_needed]
    return hourly_slice


# ---------------------------------------------------------------------------
# Window counting
# ---------------------------------------------------------------------------

def _count_daily_windows(df, lookback_window, predict_window, score_start, score_end):
    window = lookback_window + predict_window + 1
    if len(df) < window:
        return 0
    count = 0
    for start_idx in range(len(df) - window + 1):
        target_end_idx = start_idx + lookback_window + predict_window - 1
        target_end_time = df.index[target_end_idx]
        if score_start <= target_end_time <= score_end:
            count += 1
    return count


def _count_eval_windows(df, lookback_window, predict_window, score_start, score_end):
    min_target_end_idx = lookback_window + predict_window - 1
    if len(df) <= min_target_end_idx:
        return 0
    count = 0
    for target_end_idx in range(min_target_end_idx, len(df)):
        target_end_time = df.index[target_end_idx]
        if score_start <= target_end_time <= score_end:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = Config()

    daily_csv_dir = args.daily_csv_dir or config.local_csv_dir
    hourly_csv_dir = args.hourly_csv_dir or os.path.join(os.path.dirname(config.local_csv_dir), "hourly")
    output_dir = args.output_dir or config.dataset_path
    hourly_window = args.hourly_window
    feature_list = config.feature_list

    train_start, train_end = map(pd.Timestamp, config.train_time_range)
    val_start, val_end = map(pd.Timestamp, config.val_time_range)
    test_start, test_end = map(pd.Timestamp, config.test_time_range)

    # Load all daily CSVs
    daily_csvs = sorted(glob(os.path.join(daily_csv_dir, "*_qfq_day.csv")))
    if not daily_csvs:
        raise FileNotFoundError(f"No *_qfq_day.csv files in {daily_csv_dir}")

    daily_all = {}
    for path in tqdm(daily_csvs, desc="Loading daily CSV"):
        sym, df = _load_daily_df(path, feature_list)
        daily_all[sym] = df

    # Load all hourly CSVs
    hourly_csvs = sorted(glob(os.path.join(hourly_csv_dir, "*_qfq_60min.csv")))
    if not hourly_csvs:
        raise FileNotFoundError(f"No *_qfq_60min.csv files in {hourly_csv_dir}")

    hourly_all = {}
    for path in tqdm(hourly_csvs, desc="Loading hourly CSV"):
        sym, df = _load_hourly_df(path, feature_list)
        hourly_all[sym] = df

    # Intersect symbols
    common_symbols = sorted(set(daily_all.keys()) & set(hourly_all.keys()))
    print(f"Daily symbols: {len(daily_all)}, Hourly symbols: {len(hourly_all)}, Common: {len(common_symbols)}")

    train_data, val_data, test_data = {}, {}, {}
    kept_symbols = []
    symbol_stats = []

    for sym in tqdm(common_symbols, desc="Building splits"):
        d_full = daily_all[sym]
        h_full = hourly_all[sym]

        # Daily splits (same logic as B group)
        d_train = d_full.loc[(d_full.index >= train_start) & (d_full.index <= train_end)].copy()
        d_val = _slice_daily_with_buffer(d_full, val_start, val_end, config.lookback_window)
        d_test = _slice_daily_with_buffer(d_full, test_start, test_end, config.lookback_window)

        # Hourly splits (aligned to daily)
        h_train = _slice_hourly(h_full, d_train, hourly_window)
        h_val = _slice_hourly(h_full, d_val, hourly_window)
        h_test = _slice_hourly(h_full, d_test, hourly_window)

        # Check minimum windows
        n_train_win = _count_daily_windows(d_train, config.lookback_window, config.predict_window, train_start, train_end)
        n_val_win = _count_daily_windows(d_val, config.lookback_window, config.predict_window, val_start, val_end)
        n_test_win = _count_eval_windows(d_test, config.lookback_window, config.predict_window, test_start, test_end)

        has_hourly = len(h_train) >= hourly_window and len(h_val) >= hourly_window and len(h_test) >= hourly_window

        symbol_stats.append({
            "symbol": sym,
            "daily_train_rows": len(d_train),
            "daily_val_rows": len(d_val),
            "daily_test_rows": len(d_test),
            "hourly_train_rows": len(h_train),
            "hourly_val_rows": len(h_val),
            "hourly_test_rows": len(h_test),
            "train_windows": n_train_win,
            "val_windows": n_val_win,
            "test_windows": n_test_win,
            "has_hourly": has_hourly,
        })

        if n_train_win <= 0 or n_val_win <= 0 or n_test_win <= 0 or not has_hourly:
            continue

        kept_symbols.append(sym)
        train_data[sym] = {"daily": d_train, "hourly": h_train}
        val_data[sym] = {"daily": d_val, "hourly": h_val}
        test_data[sym] = {"daily": d_test, "hourly": h_test}

    if len(kept_symbols) < args.min_symbols:
        raise RuntimeError(f"Only {len(kept_symbols)} symbols retained (min={args.min_symbols}).")

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train_data.pkl", train_data), ("val_data.pkl", val_data), ("test_data.pkl", test_data)]:
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(data, f)

    stats_df = pd.DataFrame(symbol_stats).sort_values("symbol")
    stats_df.to_csv(os.path.join(output_dir, "symbol_stats.csv"), index=False)

    metadata = {
        "source": "local_daily_hourly_csv",
        "daily_csv_dir": os.path.abspath(daily_csv_dir),
        "hourly_csv_dir": os.path.abspath(hourly_csv_dir),
        "output_dir": os.path.abspath(output_dir),
        "lookback_window": config.lookback_window,
        "predict_window": config.predict_window,
        "hourly_window": hourly_window,
        "feature_list": feature_list,
        "n_daily_files": len(daily_csvs),
        "n_hourly_files": len(hourly_csvs),
        "n_common_symbols": len(common_symbols),
        "n_symbols_kept": len(kept_symbols),
        "symbols": kept_symbols,
        "splits": {
            "train": {
                "score_start": train_start.strftime("%Y-%m-%d"),
                "score_end": train_end.strftime("%Y-%m-%d"),
                "daily_rows_total": int(sum(len(d["daily"]) for d in train_data.values())),
                "hourly_rows_total": int(sum(len(d["hourly"]) for d in train_data.values())),
            },
            "val": {
                "score_start": val_start.strftime("%Y-%m-%d"),
                "score_end": val_end.strftime("%Y-%m-%d"),
                "daily_rows_total": int(sum(len(d["daily"]) for d in val_data.values())),
                "hourly_rows_total": int(sum(len(d["hourly"]) for d in val_data.values())),
            },
            "test": {
                "score_start": test_start.strftime("%Y-%m-%d"),
                "score_end": test_end.strftime("%Y-%m-%d"),
                "daily_rows_total": int(sum(len(d["daily"]) for d in test_data.values())),
                "hourly_rows_total": int(sum(len(d["hourly"]) for d in test_data.values())),
            },
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved C-group datasets to {output_dir}")
    print(f"Retained symbols: {len(kept_symbols)} / {len(common_symbols)} (common)")
    for split_name in ["train", "val", "test"]:
        m = metadata["splits"][split_name]
        print(f"  {split_name}: daily={m['daily_rows_total']}, hourly={m['hourly_rows_total']}")


if __name__ == "__main__":
    main()
