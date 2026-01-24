import argparse
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .utils import get_device
from .data.io import read_one_csv
from .models.informer import Informer


@torch.no_grad()
def make_decoder_input(x_enc, label_len, pred_len):
    x_label = x_enc[:, -label_len:, :]
    zeros = torch.zeros((x_enc.size(0), pred_len, x_enc.size(-1)),
                        device=x_enc.device, dtype=x_enc.dtype)
    return torch.cat([x_label, zeros], dim=1)


@torch.no_grad()
def predict_last7_infile(model, scaler, df, date_col, features, lookback, pred_len, label_len, device):
    """
    用倒数第pred_len天之前的lookback天预测文件最后pred_len天（Informer版）
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    need_len = lookback + pred_len
    if len(df) < need_len:
        raise ValueError(f"数据长度不足：需要至少 {need_len} 行")

    x_hist_df = df.iloc[-need_len:-pred_len]  # [lookback]
    y_true_df = df.iloc[-pred_len:]           # [pred_len]

    x_raw = x_hist_df[features].values.astype(np.float32)   # [20,6]
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    x_enc = torch.from_numpy(x_scaled).unsqueeze(0).to(device)  # [1,20,6]
    x_dec = make_decoder_input(x_enc, label_len, pred_len)       # [1,label+pred,6]

    pred_scaled = model(x_enc, x_dec).squeeze(0).cpu().numpy()   # [7,6]
    pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, len(features))).reshape(pred_len, len(features))

    y_true = y_true_df[features].values.astype(np.float32)
    y_time = y_true_df[date_col].to_numpy()
    return pred_real, y_true, y_time



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stock_csv", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()

    ckpt_path = args.ckpt_path or f"{cfg.ckpt_dir}/informer_{cfg.ckpt_name}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    inf_cfg = ckpt["informer_cfg"]
    label_len = int(inf_cfg.get("label_len", 10))
    pred_len = int(ckpt["pred_len"])
    lookback = int(ckpt["lookback"])

    # restore scaler
    scaler = MinMaxScaler()
    scaler.min_ = ckpt["scaler_min"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = len(cfg.features)

    model = Informer(
        input_dim=len(cfg.features),
        output_dim=len(cfg.features),
        pred_len=pred_len,
        label_len=label_len,
        d_model=int(inf_cfg.get("d_model", 128)),
        n_heads=int(inf_cfg.get("n_heads", 4)),
        e_layers=int(inf_cfg.get("e_layers", 2)),
        d_layers=int(inf_cfg.get("d_layers", 1)),
        d_ff=int(inf_cfg.get("d_ff", 256)),
        dropout=float(inf_cfg.get("dropout", 0.1)),
        attn=str(inf_cfg.get("attn", "prob")),
        factor=int(inf_cfg.get("factor", 5)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    df = read_one_csv(args.stock_csv, cfg.date_col, cfg.features)
    pred, y_true, y_time = predict_last7_infile(
        model, scaler, df, cfg.date_col, cfg.features,
        lookback, pred_len, label_len, device
    )

    print("=== Compare on LAST 7 DAYS in file (Informer) ===")
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
    save_path = os.path.join("outputs", f"{stock_id}_informer_close_last7.png")

    plot_close_compare(
        dates=y_time,
        true_close=true_close,
        pred_close=pred_close,
        title=f"Informer Close Compare (Last 7 Days) - {stock_id}",
        save_path=save_path,
        show=True
    )
    print("Saved plot:", save_path)


if __name__ == "__main__":
    main()
