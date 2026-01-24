import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import Config
from .utils import get_device, ensure_dir, plot_close_compare
from .data.io import read_one_csv
from .models.lstm import LSTMForecast
from .models.informer import Informer
from .models.arima import ARIMAForecaster


def save_compare_csv(save_path, dates, pred, true, features):
    df_true = pd.DataFrame(true, columns=[f"true_{c}" for c in features])
    df_pred = pd.DataFrame(pred, columns=[f"pred_{c}" for c in features])
    df_out = pd.concat([df_true, df_pred], axis=1)
    df_out.insert(0, "datetime", dates)
    ensure_dir(os.path.dirname(save_path))
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")


@torch.no_grad()
def lstm_predict_last7_infile(model, scaler, df, date_col, features, lookback, pred_len, device):
    df = df.sort_values(date_col).reset_index(drop=True)
    need_len = lookback + pred_len
    if len(df) < need_len:
        raise ValueError(f"len(df)={len(df)} < need_len={need_len}")

    x_hist_df = df.iloc[-need_len:-pred_len]
    y_true_df = df.iloc[-pred_len:]

    x_raw = x_hist_df[features].values.astype(np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    x = torch.from_numpy(x_scaled).unsqueeze(0).to(device, non_blocking=True)
    pred_scaled = model(x).squeeze(0).cpu().numpy()

    pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, len(features))).reshape(pred_len, len(features))
    y_true = y_true_df[features].values.astype(np.float32)
    y_time = y_true_df[date_col].to_numpy()
    return pred_real, y_true, y_time


def make_decoder_input(x_enc, label_len, pred_len):
    x_label = x_enc[:, -label_len:, :]
    zeros = torch.zeros((x_enc.size(0), pred_len, x_enc.size(-1)),
                        device=x_enc.device, dtype=x_enc.dtype)
    return torch.cat([x_label, zeros], dim=1)


@torch.no_grad()
def informer_predict_last7_infile(model, scaler, df, date_col, features, lookback, pred_len, label_len, device):
    df = df.sort_values(date_col).reset_index(drop=True)
    need_len = lookback + pred_len
    if len(df) < need_len:
        raise ValueError(f"len(df)={len(df)} < need_len={need_len}")

    x_hist_df = df.iloc[-need_len:-pred_len]
    y_true_df = df.iloc[-pred_len:]

    x_raw = x_hist_df[features].values.astype(np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    x_enc = torch.from_numpy(x_scaled).unsqueeze(0).to(device, non_blocking=True)
    x_dec = make_decoder_input(x_enc, label_len, pred_len)

    pred_scaled = model(x_enc, x_dec).squeeze(0).cpu().numpy()
    pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, len(features))).reshape(pred_len, len(features))

    y_true = y_true_df[features].values.astype(np.float32)
    y_time = y_true_df[date_col].to_numpy()
    return pred_real, y_true, y_time


def arima_predict_last7_infile(forecaster, df, date_col, features, lookback, pred_len):
    df = df.sort_values(date_col).reset_index(drop=True)
    need_len = lookback + pred_len
    if len(df) < need_len:
        raise ValueError(f"len(df)={len(df)} < need_len={need_len}")

    x_hist = df[features].values.astype(np.float64)[-need_len:-pred_len]  # [20,6]
    y_true = df[features].values.astype(np.float64)[-pred_len:]           # [7,6]
    y_time = df[date_col].to_numpy()[-pred_len:]

    pred = forecaster.fit_predict_one_window(x_hist, pred_len)            # [7,6]
    return pred, y_true, y_time


def load_scaler_from_ckpt(ckpt, n_features: int) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.min_ = ckpt["scaler_min"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = n_features
    return scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # model: lstm / informer / arima
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "informer", "arima"])

    # 输入目录（默认用 config.csv_dir）
    parser.add_argument("--csv_dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=None)

    # 输出目录
    parser.add_argument("--out_dir", type=str, default="outputs_batch")
    parser.add_argument("--plot", action="store_true")  # 是否画 close 对比图
    parser.add_argument("--show", action="store_true")  # 画图时是否弹窗

    # ckpt（lstm/informer需要）
    parser.add_argument("--ckpt_path", type=str, default=None)

    # arima 参数
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=0)

    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)

    csv_dir = args.csv_dir or cfg.csv_dir
    pattern = args.pattern or cfg.pattern
    paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
    if args.max_files is not None:
        paths = paths[: args.max_files]

    if len(paths) == 0:
        raise RuntimeError(f"No csv files found in: {csv_dir} with pattern={pattern}")

    ensure_dir(args.out_dir)
    device = get_device()

    close_idx = cfg.features.index("close") if "close" in cfg.features else 3

    # ---- Load model/scaler ----
    model = None
    scaler = None
    label_len = None

    if args.model == "lstm":
        ckpt_path = args.ckpt_path or os.path.join(cfg.ckpt_dir, cfg.ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        scaler = load_scaler_from_ckpt(ckpt, len(cfg.features))

        model = LSTMForecast(
            input_dim=len(cfg.features),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pred_len=cfg.pred_len,
            output_dim=len(cfg.features),
        )
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt["model"])
        model.to(device)
        model.eval()

    elif args.model == "informer":
        ckpt_path = args.ckpt_path or os.path.join(cfg.ckpt_dir, "informer_" + cfg.ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        scaler = load_scaler_from_ckpt(ckpt, len(cfg.features))
        inf_cfg = ckpt["informer_cfg"]

        label_len = int(inf_cfg.get("label_len", 10))
        model = Informer(
            input_dim=len(cfg.features),
            output_dim=len(cfg.features),
            pred_len=int(ckpt["pred_len"]),
            label_len=label_len,
            d_model=int(inf_cfg.get("d_model", 128)),
            n_heads=int(inf_cfg.get("n_heads", 4)),
            e_layers=int(inf_cfg.get("e_layers", 2)),
            d_layers=int(inf_cfg.get("d_layers", 1)),
            d_ff=int(inf_cfg.get("d_ff", 256)),
            dropout=float(inf_cfg.get("dropout", 0.1)),
            attn=str(inf_cfg.get("attn", "full")),
            factor=int(inf_cfg.get("factor", 5)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

    else:
        # ARIMA
        model = ARIMAForecaster(order=(args.p, args.d, args.q))

    # ---- Batch predict ----
    summary_rows = []
    for p in tqdm(paths, desc=f"batch-{args.model}"):
        stock_id = os.path.splitext(os.path.basename(p))[0]
        try:
            df = read_one_csv(p, cfg.date_col, cfg.features)

            if args.model == "lstm":
                pred, true, dates = lstm_predict_last7_infile(
                    model, scaler, df, cfg.date_col, cfg.features,
                    cfg.lookback, cfg.pred_len, device
                )
            elif args.model == "informer":
                pred, true, dates = informer_predict_last7_infile(
                    model, scaler, df, cfg.date_col, cfg.features,
                    cfg.lookback, cfg.pred_len, label_len, device
                )
            else:
                pred, true, dates = arima_predict_last7_infile(
                    model, df, cfg.date_col, cfg.features,
                    cfg.lookback, cfg.pred_len
                )

            # 指标（close）
            pred_close = np.asarray(pred)[:, close_idx]
            true_close = np.asarray(true)[:, close_idx]
            mse = mean_squared_error(true_close, pred_close)
            mae = mean_absolute_error(true_close, pred_close)

            # 保存 per-stock 对比 CSV
            out_csv = os.path.join(args.out_dir, f"{stock_id}_{args.model}_last7_compare.csv")
            save_compare_csv(out_csv, dates, pred, true, cfg.features)

            # 画图（close）
            if args.plot:
                out_png = os.path.join(args.out_dir, f"{stock_id}_{args.model}_close_last7.png")
                plot_close_compare(
                    dates=dates,
                    true_close=true_close,
                    pred_close=pred_close,
                    title=f"{args.model.upper()} Close Compare (Last 7 Days) - {stock_id}",
                    save_path=out_png,
                    show=args.show
                )

            summary_rows.append({
                "stock_id": stock_id,
                "model": args.model,
                "mse_close": float(mse),
                "mae_close": float(mae),
                "n_days": int(len(true_close))
            })

        except Exception as e:
            summary_rows.append({
                "stock_id": stock_id,
                "model": args.model,
                "mse_close": np.nan,
                "mae_close": np.nan,
                "n_days": 0,
                "error": str(e)
            })

    # ---- Save summary ----
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, f"summary_{args.model}.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("Saved summary:", summary_path)

    # 打印总体均值（跳过失败样本）
    ok = summary_df["n_days"] > 0
    if ok.any():
        print("AVG MSE(close):", float(summary_df.loc[ok, "mse_close"].mean()))
        print("AVG MAE(close):", float(summary_df.loc[ok, "mae_close"].mean()))
        print("OK stocks:", int(ok.sum()), "/", len(summary_df))


if __name__ == "__main__":
    main()
