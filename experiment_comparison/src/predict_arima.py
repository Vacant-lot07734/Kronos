import argparse
import numpy as np
import pandas as pd

from .config import Config
from .data.io import read_one_csv
from .models.arima import ARIMAForecaster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stock_csv", type=str, required=True)

    # ARIMA(p,d,q)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=0)

    # 是否只预测 close
    parser.add_argument("--only_close", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    df = read_one_csv(args.stock_csv, cfg.date_col, cfg.features)[:100]

    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    lookback = cfg.lookback
    pred_len = cfg.pred_len

    if len(df) < lookback:
        raise ValueError(f"数据不足{lookback}天，无法预测。")

    forecaster = ARIMAForecaster(order=(args.p, args.d, args.q))

    if args.only_close:
        # 只预测 close：返回 [7,1]
        if "close" not in cfg.features:
            raise ValueError("features 中不包含 close，无法 --only_close")
        close_idx = cfg.features.index("close")
        hist = df[cfg.features].values.astype(np.float64)[-lookback:, close_idx:close_idx+1]  # [20,1]
        pred = forecaster.fit_predict_one_window(hist, pred_len)  # [7,1]
        print("Predict next 7 days (close):")
        print(pred.reshape(-1))
    else:
        need_len = lookback + pred_len
        if len(df) < need_len:
            raise ValueError(f"数据不足：需要至少 {need_len} 行")

        x_hist = df[cfg.features].values.astype(np.float64)[-need_len:-pred_len]  # [20,6]
        y_true = df[cfg.features].values.astype(np.float64)[-pred_len:]  # [7,6]
        y_time = df[cfg.date_col].to_numpy()[-pred_len:]

        pred = forecaster.fit_predict_one_window(x_hist, pred_len)  # [7,6]

        print("=== Compare on LAST 7 DAYS in file (ARIMA) ===")
        print("Dates:", y_time)
        print("\nPred:")
        print(pred)
        print("\nTrue:")
        print(y_true)

        from .utils import plot_close_compare
        import os

        close_idx = cfg.features.index("close") if "close" in cfg.features else 3

        pred_close = pred[:, close_idx]
        true_close = y_true[:, close_idx]

        stock_id = os.path.splitext(os.path.basename(args.stock_csv))[0]
        save_path = os.path.join("outputs", f"{stock_id}_arima_close_last7.png")

        plot_close_compare(
            dates=y_time,
            true_close=true_close,
            pred_close=pred_close,
            title=f"ARIMA Close Compare (Last 7 Days) - {stock_id}",
            save_path=save_path,
            show=True
        )
        print("Saved plot:", save_path)


if __name__ == "__main__":
    main()
