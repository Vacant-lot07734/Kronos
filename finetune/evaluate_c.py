"""
C-group evaluation script.

Uses C-group's trained KronosWithHourly model for auto-regressive inference
on val/test splits.  Output format is identical to evaluate_ab.py so that
A/B/C metrics can be directly compared.

The main difference from evaluate_ab.py:
    - Loads KronosWithHourly instead of plain Kronos
    - Each inference batch also receives hourly context
    - Hourly context is encoded once per sample, then reused across AR steps
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import Config
from model.kronos import KronosTokenizer, Kronos, sample_from_logits
from models.kronos_with_hourly import KronosWithHourly
from models.hourly_encoder import HourlyEncoder
from models.hourly_fusion import HourlyFusionLayer


def parse_args():
    parser = argparse.ArgumentParser(description="C-group evaluation.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to C-group checkpoint dir (containing kronos_predictor/, hourly_encoder.pt, etc.)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to C-group processed dataset dir.")
    parser.add_argument("--result-save-path", type=str, required=True)
    parser.add_argument("--result-name", type=str, required=True)
    parser.add_argument("--pred-len", type=int)
    parser.add_argument("--hourly-window", type=int, default=25)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (may need to be smaller than AB due to hourly context).")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    return parser.parse_args()


def _resolve_device(req: str) -> str:
    req = (req or "auto").strip().lower()
    if req == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return req


def _load_metadata(data_path: str) -> dict:
    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        return json.load(f)


def _load_split_data(data_path: str, split: str) -> dict:
    with open(os.path.join(data_path, f"{split}_data.pkl"), "rb") as f:
        return pickle.load(f)


def _resolve_prediction_range(split_meta: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    prediction_start = split_meta.get("prediction_start", split_meta.get("score_start"))
    prediction_end = split_meta.get("prediction_end", split_meta.get("score_end"))
    if prediction_start is None or prediction_end is None:
        raise KeyError("Split metadata must define prediction_start/prediction_end.")
    return pd.Timestamp(prediction_start), pd.Timestamp(prediction_end)


# ---------------------------------------------------------------------------
# Build evaluation windows (daily + hourly context)
# ---------------------------------------------------------------------------

def _build_eval_windows(
    split_data: dict,
    split_meta: dict,
    lookback_window: int,
    pred_len: int,
    hourly_window: int,
    feature_list: list[str],
) -> list[dict]:
    """Build evaluation windows identical to evaluate_ab.py but with hourly context added."""
    prediction_start, prediction_end = _resolve_prediction_range(split_meta)

    windows = []
    for symbol, entry in split_data.items():
        daily_df = entry["daily"].sort_index()
        hourly_df = entry["hourly"].sort_index()

        if len(daily_df) < lookback_window + pred_len:
            continue

        for prediction_start_idx in range(lookback_window, len(daily_df) - pred_len + 1):
            prediction_start_dt = daily_df.index[prediction_start_idx]
            prediction_end_idx = prediction_start_idx + pred_len - 1
            if prediction_end_idx >= len(daily_df):
                continue
            prediction_end_dt = daily_df.index[prediction_end_idx]
            if prediction_start_dt < prediction_start or prediction_end_dt > prediction_end:
                continue

            context_end_idx = prediction_start_idx - 1
            context_start_idx = prediction_start_idx - lookback_window
            if context_start_idx < 0:
                continue

            context_df = daily_df.iloc[context_start_idx:context_end_idx + 1]
            future_df = daily_df.iloc[prediction_start_idx:prediction_start_idx + pred_len]
            if len(context_df) != lookback_window or len(future_df) != pred_len:
                continue

            # Hourly context: bars up to context_end_date close
            context_end_date = context_df.index[-1]
            hourly_cutoff = context_end_date + pd.Timedelta(hours=15)
            valid_hourly = hourly_df.loc[hourly_df.index <= hourly_cutoff]
            if len(valid_hourly) < hourly_window:
                continue
            hourly_win = valid_hourly.iloc[-hourly_window:]

            model_context_df = context_df.rename(columns={"vol": "volume", "amt": "amount"})
            last_close = float(context_df["close"].iloc[-1])
            true_close = float(future_df["close"].iloc[-1])
            true_return = (true_close / last_close) - 1.0

            windows.append({
                "instrument": symbol,
                "context_end_date": context_df.index[-1].strftime("%Y-%m-%d"),
                "prediction_start_date": future_df.index[0].strftime("%Y-%m-%d"),
                "prediction_end_date": future_df.index[-1].strftime("%Y-%m-%d"),
                "context_df": model_context_df,
                "x_timestamp": pd.Series(context_df.index),
                "y_timestamp": pd.Series(future_df.index),
                "hourly_df": hourly_win,
                "last_close": last_close,
                "true_close": true_close,
                "true_return": true_return,
            })

    return windows


# ---------------------------------------------------------------------------
# C-group auto-regressive inference
# ---------------------------------------------------------------------------

def _calc_time_stamps(ts):
    """Compute time features from pandas timestamps."""
    time_df = pd.DataFrame()
    time_df["minute"] = ts.dt.minute
    time_df["hour"] = ts.dt.hour
    time_df["weekday"] = ts.dt.weekday
    time_df["day"] = ts.dt.day
    time_df["month"] = ts.dt.month
    return time_df


def _c_auto_regressive_inference(
    tokenizer, model, x, x_stamp, y_stamp,
    x_hourly, x_stamp_h,
    max_context, pred_len, clip=5,
    T=1.0, top_k=0, top_p=0.99, sample_count=10,
):
    """
    Auto-regressive inference for C-group model.
    Same logic as auto_regressive_inference in kronos.py but uses
    model.decode_s1 with pre-computed hourly_hidden.
    """
    with torch.no_grad():
        device = x.device
        x = torch.clip(x, -clip, clip)

        # Repeat for sample_count
        B = x.size(0)
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2))
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2))
        x_hourly = x_hourly.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_hourly.size(1), x_hourly.size(2))
        x_stamp_h = x_stamp_h.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp_h.size(1), x_stamp_h.size(2))

        # Encode daily tokens
        x_token = tokenizer.encode(x, half=True)

        # Encode hourly context ONCE
        hourly_hidden = model.encode_hourly(x_hourly, x_stamp_h)  # (B*S, L_h, d)

        initial_len = x.size(1)
        batch_size = x_token[0].size(0)
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buf = x_token[0].new_zeros(batch_size, max_context)
        post_buf = x_token[1].new_zeros(batch_size, max_context)
        buf_len = min(initial_len, max_context)
        if buf_len > 0:
            s = max(0, initial_len - max_context)
            pre_buf[:, :buf_len] = x_token[0][:, s:s + buf_len]
            post_buf[:, :buf_len] = x_token[1][:, s:s + buf_len]

        for i in range(pred_len):
            cur_len = initial_len + i
            win_len = min(cur_len, max_context)

            if cur_len <= max_context:
                inp = [pre_buf[:, :win_len], post_buf[:, :win_len]]
            else:
                inp = [pre_buf, post_buf]

            ctx_end = cur_len
            ctx_start = max(0, ctx_end - max_context)
            cur_stamp = full_stamp[:, ctx_start:ctx_end, :].contiguous()

            s1_logits, context = model.decode_s1(
                inp[0], inp[1], cur_stamp, hourly_hidden
            )
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if cur_len < max_context:
                pre_buf[:, cur_len] = sample_pre.squeeze(-1)
                post_buf[:, cur_len] = sample_post.squeeze(-1)
            else:
                pre_buf.copy_(torch.roll(pre_buf, shifts=-1, dims=1))
                post_buf.copy_(torch.roll(post_buf, shifts=-1, dims=1))
                pre_buf[:, -1] = sample_pre.squeeze(-1)
                post_buf[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        total_len = initial_len + pred_len
        cs = max(0, total_len - max_context)
        inp_final = [full_pre[:, cs:total_len].contiguous(), full_post[:, cs:total_len].contiguous()]
        z = tokenizer.decode(inp_final, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)
        return preds


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: KronosWithHourly,
    tokenizer: KronosTokenizer,
    windows: list[dict],
    pred_len: int,
    hourly_window: int,
    batch_size: int,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_context: int,
    clip: float,
    device: str,
    feature_list: list[str],
) -> pd.DataFrame:

    price_cols = ["open", "high", "low", "close"]
    vol_col, amt_col = "volume", "amount"

    records = []
    for start in tqdm(range(0, len(windows), batch_size), desc="Inference batches"):
        batch = windows[start:start + batch_size]
        bs = len(batch)

        # Prepare daily data
        x_list, x_stamp_list, y_stamp_list, means, stds = [], [], [], [], []
        x_h_list, x_stamp_h_list = [], []

        for item in batch:
            df = item["context_df"].copy()
            if vol_col not in df.columns: df[vol_col] = 0.0
            if amt_col not in df.columns: df[amt_col] = 0.0
            x = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -clip, clip)
            x_list.append(x_norm)
            means.append(x_mean)
            stds.append(x_std)

            x_stamp_list.append(_calc_time_stamps(item["x_timestamp"]).values.astype(np.float32))
            y_stamp_list.append(_calc_time_stamps(item["y_timestamp"]).values.astype(np.float32))

            # Hourly data
            h_df = item["hourly_df"]
            h_vals = h_df[feature_list].values.astype(np.float32)
            h_mean, h_std = np.mean(h_vals, axis=0), np.std(h_vals, axis=0)
            h_norm = np.clip((h_vals - h_mean) / (h_std + 1e-5), -clip, clip)
            x_h_list.append(h_norm)

            h_time = pd.DataFrame()
            h_time["minute"] = h_df.index.minute if hasattr(h_df.index, "minute") else h_df.reset_index()["datetime"].dt.minute
            h_time["hour"] = h_df.index.hour if hasattr(h_df.index, "hour") else h_df.reset_index()["datetime"].dt.hour
            h_time["weekday"] = h_df.index.weekday if hasattr(h_df.index, "weekday") else h_df.reset_index()["datetime"].dt.weekday
            h_time["day"] = h_df.index.day if hasattr(h_df.index, "day") else h_df.reset_index()["datetime"].dt.day
            h_time["month"] = h_df.index.month if hasattr(h_df.index, "month") else h_df.reset_index()["datetime"].dt.month
            x_stamp_h_list.append(h_time.values.astype(np.float32))

        x_batch = torch.from_numpy(np.stack(x_list)).to(device)
        x_stamp_batch = torch.from_numpy(np.stack(x_stamp_list)).to(device)
        y_stamp_batch = torch.from_numpy(np.stack(y_stamp_list)).to(device)
        x_h_batch = torch.from_numpy(np.stack(x_h_list)).to(device)
        x_stamp_h_batch = torch.from_numpy(np.stack(x_stamp_h_list)).to(device)

        preds = _c_auto_regressive_inference(
            tokenizer, model,
            x_batch, x_stamp_batch, y_stamp_batch,
            x_h_batch, x_stamp_h_batch,
            max_context, pred_len, clip,
            temperature, top_k, top_p, sample_count,
        )
        preds = preds[:, -pred_len:, :]

        for idx, item in enumerate(batch):
            pred_vals = preds[idx] * (stds[idx] + 1e-5) + means[idx]
            pred_close = float(pred_vals[-1, 3])  # close column
            pred_return = (pred_close / item["last_close"]) - 1.0
            records.append({
                "instrument": item["instrument"],
                "context_end_date": item["context_end_date"],
                "prediction_start_date": item["prediction_start_date"],
                "prediction_end_date": item["prediction_end_date"],
                "last_close": item["last_close"],
                "pred_close": pred_close,
                "true_close": item["true_close"],
                "pred_return": pred_return,
                "true_return": item["true_return"],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics (reused from evaluate_ab logic)
# ---------------------------------------------------------------------------

def _safe_corr(frame, method):
    if len(frame) < 2: return np.nan
    if frame["pred_return"].nunique() < 2 or frame["true_return"].nunique() < 2: return np.nan
    return float(frame["pred_return"].corr(frame["true_return"], method=method))


def _build_daily_metrics(predictions):
    rows = []
    for date, grp in predictions.groupby("prediction_start_date", sort=True):
        rows.append({
            "prediction_start_date": date, "n_symbols": len(grp),
            "ic": _safe_corr(grp, "pearson"), "rank_ic": _safe_corr(grp, "spearman"),
        })
    return pd.DataFrame(rows).sort_values("prediction_start_date")


def _compute_summary(predictions, daily_df):
    ic_s = daily_df["ic"].dropna()
    ric_s = daily_df["rank_ic"].dropna()

    def _ir(s):
        if len(s) < 2: return None
        std = float(s.std(ddof=0))
        return float(s.mean() / std) if std > 0 else None

    return {
        "n_predictions": int(len(predictions)),
        "n_eval_dates": int(len(daily_df)),
        "n_instruments_mean": float(daily_df["n_symbols"].mean()) if not daily_df.empty else 0,
        "ic": float(ic_s.mean()) if not ic_s.empty else None,
        "rank_ic": float(ric_s.mean()) if not ric_s.empty else None,
        "icir": _ir(ic_s), "rank_icir": _ir(ric_s),
        "da": float((np.sign(predictions["pred_return"]) == np.sign(predictions["true_return"])).mean()),
        "mae": float(np.mean(np.abs(predictions["pred_return"] - predictions["true_return"]))),
        "rmse": float(np.sqrt(np.mean((predictions["pred_return"] - predictions["true_return"]) ** 2))),
        "pred_return_mean": float(predictions["pred_return"].mean()),
        "true_return_mean": float(predictions["true_return"].mean()),
    }


def _save_plot(daily_df, save_dir, split):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    daily_df.plot(x="prediction_start_date", y=["ic", "rank_ic"], ax=ax, grid=True,
                  title=f"{split.upper()} daily IC / RankIC")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Prediction Start Date")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split}_metrics.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load C-group model
# ---------------------------------------------------------------------------

def _load_c_model(device, tokenizer_path, model_path, max_context, clip):
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).eval()

    c_config_path = os.path.join(model_path, "c_config.json")
    with open(c_config_path, "r") as f:
        c_cfg = json.load(f)

    model = KronosWithHourly.load_for_inference(
        kronos_predictor_path=os.path.join(model_path, "kronos_predictor"),
        c_modules_dir=model_path,
        hourly_encoder_kwargs={
            "d_in": c_cfg.get("d_in", 6),
            "d_model": c_cfg["d_model"],
            "n_heads": c_cfg.get("hourly_n_heads", 8),
            "ff_dim": c_cfg.get("hourly_ff_dim", 1024),
            "n_layers": c_cfg.get("hourly_encoder_layers", 2),
            "ffn_dropout_p": c_cfg.get("hourly_dropout", 0.1),
            "resid_dropout_p": c_cfg.get("hourly_dropout", 0.1),
        },
        fusion_kwargs={
            "d_model": c_cfg["d_model"],
            "n_heads": c_cfg.get("fusion_n_heads", 8),
            "resid_dropout_p": c_cfg.get("fusion_dropout", 0.1),
        },
        split_point=c_cfg["split_point"],
        device=device,
    )

    tokenizer = tokenizer.to(device)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_split(tokenizer, model, split, split_data, split_meta, args, config, run_dir):
    split_dir = os.path.join(run_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    windows = _build_eval_windows(
        split_data, split_meta,
        config.lookback_window, args.pred_len, args.hourly_window,
        config.feature_list,
    )
    if not windows:
        raise RuntimeError(f"No evaluation windows for split={split}.")

    predictions = _run_inference(
        model=model, tokenizer=tokenizer,
        windows=windows, pred_len=args.pred_len,
        hourly_window=args.hourly_window,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        temperature=config.inference_T,
        top_k=config.inference_top_k,
        top_p=config.inference_top_p,
        max_context=config.max_context,
        clip=config.clip,
        device=str(model.kronos.embedding.emb_s1.weight.device),
        feature_list=config.feature_list,
    )
    predictions.to_csv(os.path.join(split_dir, "predictions.csv"), index=False)

    daily_df = _build_daily_metrics(predictions)
    daily_df.to_csv(os.path.join(split_dir, "daily_metrics.csv"), index=False)

    summary = _compute_summary(predictions, daily_df)
    with open(os.path.join(split_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _save_plot(daily_df, split_dir, split)
    print(f"[{split}] metrics saved to {split_dir}")
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    config = Config()
    if args.pred_len is None:
        args.pred_len = config.predict_window

    device = _resolve_device(args.device)
    metadata = _load_metadata(args.data_path)

    run_dir = os.path.join(args.result_save_path, args.result_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": device,
                "requested_device": args.device,
                "tokenizer_path": args.tokenizer_path,
                "model_path": args.model_path,
                "data_path": args.data_path,
                "pred_len": args.pred_len,
                "hourly_window": args.hourly_window,
                "sample_count": args.sample_count,
                "batch_size": args.batch_size,
                "splits": args.splits,
                "protocol": {
                    "split_assignment_rule": "strict_full_horizon_within_split",
                    "non_trading_boundary_policy": (
                        "use the first trading day on or after the configured "
                        "range start as prediction_start_date, and require "
                        "prediction_end_date to stay inside the configured "
                        "range end"
                    ),
                    "split_ranges": metadata.get("splits", {}),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    tokenizer, model = _load_c_model(
        device, args.tokenizer_path, args.model_path,
        config.max_context, config.clip,
    )

    for split in args.splits:
        split_data = _load_split_data(args.data_path, split)
        split_meta = metadata["splits"][split]
        evaluate_split(tokenizer, model, split, split_data, split_meta, args, config, run_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
