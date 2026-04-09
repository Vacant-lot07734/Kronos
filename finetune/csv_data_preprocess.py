import argparse
import json
import os
import pickle
from glob import glob

import pandas as pd
from tqdm import tqdm

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Build Kronos train/val/test pickle datasets from local daily CSV files.")
    parser.add_argument("--csv-dir", type=str, help="Directory containing per-symbol daily CSV files.")
    parser.add_argument("--output-dir", type=str, help="Directory to save train/val/test pickle files.")
    parser.add_argument("--min-symbols", type=int, default=10, help="Fail if fewer than this many symbols are retained.")
    return parser.parse_args()


def _load_symbol_df(csv_path: str, feature_list: list[str]) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")

    symbol = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else os.path.basename(csv_path).split("_")[0]
    rename_map = {"trade_date": "datetime", "amount": "amt", "volume": "vol"}
    df = df.rename(columns=rename_map)

    required_cols = {"datetime", "open", "high", "low", "close", "vol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if "amt" not in df.columns:
        df["amt"] = df["vol"] * df[["open", "high", "low", "close"]].mean(axis=1)

    df["datetime"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d")
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    df = df[["datetime"] + feature_list].dropna()
    df = df.set_index("datetime")
    return symbol, df


def _slice_with_prediction_buffer(
    df: pd.DataFrame,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
    lookback_window: int,
    predict_window: int,
):
    if df.empty:
        return df

    start_pos = df.index.searchsorted(prediction_start, side="left")
    if start_pos >= len(df):
        return df.iloc[0:0].copy()

    end_pos = df.index.searchsorted(prediction_end, side="right") - 1
    if end_pos < start_pos:
        return df.iloc[0:0].copy()

    buffered_start_pos = max(0, start_pos - lookback_window)
    buffered_end_pos = min(len(df) - 1, end_pos + predict_window - 1)
    return df.iloc[buffered_start_pos:buffered_end_pos + 1].copy()


def _count_prediction_windows(
    df: pd.DataFrame,
    lookback_window: int,
    predict_window: int,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> int:
    window = lookback_window + predict_window + 1
    if len(df) < window:
        return 0

    count = 0
    for start_idx in range(len(df) - window + 1):
        prediction_start_idx = start_idx + lookback_window
        prediction_start_time = df.index[prediction_start_idx]
        if prediction_start <= prediction_start_time <= prediction_end:
            count += 1
    return count


def main():
    args = parse_args()
    config = Config()

    csv_dir = args.csv_dir or config.local_csv_dir
    output_dir = args.output_dir or config.dataset_path

    train_start, train_end = map(pd.Timestamp, config.train_time_range)
    val_start, val_end = map(pd.Timestamp, config.val_time_range)
    test_start, test_end = map(pd.Timestamp, config.test_time_range)

    csv_paths = sorted(glob(os.path.join(csv_dir, "*_qfq_day.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No *_qfq_day.csv files found in {csv_dir}")

    train_data = {}
    val_data = {}
    test_data = {}
    kept_symbols = []
    symbol_stats = []

    for csv_path in tqdm(csv_paths, desc="Loading CSV data"):
        symbol, full_df = _load_symbol_df(csv_path, config.feature_list)

        train_df = _slice_with_prediction_buffer(
            full_df,
            train_start,
            train_end,
            config.lookback_window,
            config.predict_window,
        )
        val_df = _slice_with_prediction_buffer(
            full_df,
            val_start,
            val_end,
            config.lookback_window,
            config.predict_window,
        )
        test_df = _slice_with_prediction_buffer(
            full_df,
            test_start,
            test_end,
            config.lookback_window,
            config.predict_window,
        )

        n_train_windows = _count_prediction_windows(train_df, config.lookback_window, config.predict_window, train_start, train_end)
        n_val_windows = _count_prediction_windows(val_df, config.lookback_window, config.predict_window, val_start, val_end)
        n_test_windows = _count_prediction_windows(test_df, config.lookback_window, config.predict_window, test_start, test_end)

        symbol_stats.append({
            "symbol": symbol,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_windows": n_train_windows,
            "val_windows": n_val_windows,
            "test_windows": n_test_windows,
            "first_date": full_df.index.min().strftime("%Y-%m-%d"),
            "last_date": full_df.index.max().strftime("%Y-%m-%d"),
        })

        if n_train_windows <= 0 or n_val_windows <= 0 or n_test_windows <= 0:
            continue

        kept_symbols.append(symbol)
        train_data[symbol] = train_df
        val_data[symbol] = val_df
        test_data[symbol] = test_df

    if len(kept_symbols) < args.min_symbols:
        raise RuntimeError(
            f"Only retained {len(kept_symbols)} symbols, below min-symbols={args.min_symbols}. "
            f"Check data coverage and split settings."
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, "val_data.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)

    stats_df = pd.DataFrame(symbol_stats).sort_values("symbol")
    stats_df.to_csv(os.path.join(output_dir, "symbol_stats.csv"), index=False)

    metadata = {
        "source": "local_daily_csv",
        "csv_dir": os.path.abspath(csv_dir),
        "output_dir": os.path.abspath(output_dir),
        "lookback_window": config.lookback_window,
        "predict_window": config.predict_window,
        "feature_list": config.feature_list,
        "n_input_files": len(csv_paths),
        "n_symbols_kept": len(kept_symbols),
        "symbols": kept_symbols,
        "splits": {
            "train": {
                "prediction_start": train_start.strftime("%Y-%m-%d"),
                "prediction_end": train_end.strftime("%Y-%m-%d"),
                "rows_total": int(sum(len(df) for df in train_data.values())),
            },
            "val": {
                "prediction_start": val_start.strftime("%Y-%m-%d"),
                "prediction_end": val_end.strftime("%Y-%m-%d"),
                "rows_total": int(sum(len(df) for df in val_data.values())),
            },
            "test": {
                "prediction_start": test_start.strftime("%Y-%m-%d"),
                "prediction_end": test_end.strftime("%Y-%m-%d"),
                "rows_total": int(sum(len(df) for df in test_data.values())),
            },
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved processed datasets to {output_dir}")
    print(f"Retained symbols: {len(kept_symbols)} / {len(csv_paths)}")
    print(f"Train rows: {metadata['splits']['train']['rows_total']}")
    print(f"Val rows: {metadata['splits']['val']['rows_total']}")
    print(f"Test rows: {metadata['splits']['test']['rows_total']}")


if __name__ == "__main__":
    main()
