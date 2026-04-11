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
from model.kronos import Kronos, KronosPredictor, KronosTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate A/B experiments on local daily CSV-derived datasets.")
    parser.add_argument("--device", type=str, default="auto", help="Device for inference: auto, cpu, or cuda:N.")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer path or Hugging Face model id.")
    parser.add_argument("--model-path", type=str, required=True, help="Predictor path or Hugging Face model id.")
    parser.add_argument("--data-path", type=str, required=True, help="Directory containing train/val/test pickle files and metadata.json.")
    parser.add_argument("--result-save-path", type=str, required=True, help="Root directory where evaluation outputs are written.")
    parser.add_argument("--result-name", type=str, required=True, help="Subdirectory name for this evaluation run.")
    parser.add_argument("--pred-len", type=int, help="Override predict window during evaluation.")
    parser.add_argument("--sample-count", type=int, default=10, help="Sampling count averaged inside Kronos predictor.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size in window units.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"], help="Dataset splits to evaluate.")
    return parser.parse_args()


def _resolve_device(requested_device: str) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "CUDA was requested for evaluation but is unavailable under the current torch build "
            f"(torch={torch.__version__}, torch_cuda={torch.version.cuda}). Falling back to cpu."
        )
        return "cpu"
    return requested_device


def _load_metadata(data_path: str) -> dict:
    metadata_path = os.path.join(data_path, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_split_data(data_path: str, split: str) -> dict:
    file_path = os.path.join(data_path, f"{split}_data.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _resolve_prediction_range(split_meta: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    prediction_start = split_meta.get("prediction_start", split_meta.get("score_start"))
    prediction_end = split_meta.get("prediction_end", split_meta.get("score_end"))
    if prediction_start is None or prediction_end is None:
        raise KeyError("Split metadata must define prediction_start/prediction_end.")
    return pd.Timestamp(prediction_start), pd.Timestamp(prediction_end)


def _build_eval_windows(split_data: dict, split_meta: dict, lookback_window: int, pred_len: int) -> list[dict]:
    prediction_start, prediction_end = _resolve_prediction_range(split_meta)

    windows = []
    for symbol, df in split_data.items():
        df = df.sort_index()
        if len(df) < lookback_window + pred_len:
            continue

        for prediction_start_idx in range(lookback_window, len(df) - pred_len + 1):
            prediction_start_dt = df.index[prediction_start_idx]
            prediction_end_idx = prediction_start_idx + pred_len - 1
            if prediction_end_idx >= len(df):
                continue
            prediction_end_dt = df.index[prediction_end_idx]
            if prediction_start_dt < prediction_start or prediction_end_dt > prediction_end:
                continue

            context_end_idx = prediction_start_idx - 1
            context_start_idx = prediction_start_idx - lookback_window
            if context_start_idx < 0:
                continue

            context_df = df.iloc[context_start_idx:context_end_idx + 1]
            future_df = df.iloc[prediction_start_idx:prediction_start_idx + pred_len]
            if len(context_df) != lookback_window or len(future_df) != pred_len:
                continue

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
                "last_close": last_close,
                "true_close": true_close,
                "true_return": true_return,
            })

    return windows


def _load_predictor(device: str, tokenizer_path: str, model_path: str, max_context: int, clip: float) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).eval()
    model = Kronos.from_pretrained(model_path).eval()
    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=max_context, clip=clip)


def _run_inference(
    predictor: KronosPredictor,
    windows: list[dict],
    pred_len: int,
    batch_size: int,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> pd.DataFrame:
    records = []
    for start in tqdm(range(0, len(windows), batch_size), desc="Inference batches"):
        batch = windows[start:start + batch_size]
        pred_dfs = predictor.predict_batch(
            [item["context_df"] for item in batch],
            [item["x_timestamp"] for item in batch],
            [item["y_timestamp"] for item in batch],
            pred_len=pred_len,
            T=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

        for item, pred_df in zip(batch, pred_dfs):
            pred_close = float(pred_df["close"].iloc[-1])
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


def _safe_corr(frame: pd.DataFrame, method: str) -> float:
    if len(frame) < 2:
        return np.nan
    if frame["pred_return"].nunique() < 2 or frame["true_return"].nunique() < 2:
        return np.nan
    return float(frame["pred_return"].corr(frame["true_return"], method=method))


def _build_daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in predictions.groupby("prediction_start_date", sort=True):
        rows.append({
            "prediction_start_date": trade_date,
            "n_symbols": int(len(group)),
            "ic": _safe_corr(group, method="pearson"),
            "rank_ic": _safe_corr(group, method="spearman"),
        })

    return pd.DataFrame(rows).sort_values("prediction_start_date")


def _compute_summary(predictions: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
    ic_series = daily_df["ic"].dropna()
    rank_ic_series = daily_df["rank_ic"].dropna()

    da = float((np.sign(predictions["pred_return"]) == np.sign(predictions["true_return"])).mean())
    mae = float(np.mean(np.abs(predictions["pred_return"] - predictions["true_return"])))
    rmse = float(np.sqrt(np.mean((predictions["pred_return"] - predictions["true_return"]) ** 2)))

    def _ir(series: pd.Series) -> float | None:
        if len(series) < 2:
            return None
        std = float(series.std(ddof=0))
        if std == 0:
            return None
        return float(series.mean() / std)

    return {
        "n_predictions": int(len(predictions)),
        "n_eval_dates": int(len(daily_df)),
        "n_instruments_mean": float(daily_df["n_symbols"].mean()) if not daily_df.empty else 0.0,
        "ic": float(ic_series.mean()) if not ic_series.empty else None,
        "rank_ic": float(rank_ic_series.mean()) if not rank_ic_series.empty else None,
        "icir": _ir(ic_series),
        "rank_icir": _ir(rank_ic_series),
        "da": da,
        "mae": mae,
        "rmse": rmse,
        "pred_return_mean": float(predictions["pred_return"].mean()),
        "true_return_mean": float(predictions["true_return"].mean()),
    }


def _save_plot(daily_df: pd.DataFrame, save_dir: str, split: str):
    fig, ax = plt.subplots(figsize=(12, 4.5))

    daily_df.plot(
        x="prediction_start_date",
        y=["ic", "rank_ic"],
        ax=ax,
        grid=True,
        title=f"{split.upper()} daily IC / RankIC",
    )
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Prediction Start Date")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split}_metrics.png"), dpi=200)
    plt.close(fig)


def evaluate_split(predictor: KronosPredictor, split: str, split_data: dict, split_meta: dict, args, config: Config, run_dir: str):
    split_dir = os.path.join(run_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    windows = _build_eval_windows(split_data, split_meta, config.lookback_window, args.pred_len)
    if not windows:
        raise RuntimeError(f"No evaluation windows found for split={split}.")

    predictions = _run_inference(
        predictor=predictor,
        windows=windows,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        temperature=config.inference_T,
        top_k=config.inference_top_k,
        top_p=config.inference_top_p,
    )
    predictions.to_csv(os.path.join(split_dir, "predictions.csv"), index=False)

    daily_df = _build_daily_metrics(predictions)
    daily_df.to_csv(os.path.join(split_dir, "daily_metrics.csv"), index=False)

    summary = _compute_summary(predictions, daily_df)
    with open(os.path.join(split_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _save_plot(daily_df, split_dir, split)
    print(f"[{split}] metrics saved to {split_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    config = Config()
    if args.pred_len is None:
        args.pred_len = config.predict_window
    resolved_device = _resolve_device(args.device)

    metadata = _load_metadata(args.data_path)
    run_dir = os.path.join(args.result_save_path, args.result_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "device": resolved_device,
            "requested_device": args.device,
            "tokenizer_path": args.tokenizer_path,
            "model_path": args.model_path,
            "data_path": args.data_path,
            "pred_len": args.pred_len,
            "sample_count": args.sample_count,
            "batch_size": args.batch_size,
            "splits": args.splits,
            "protocol": {
                "split_assignment_rule": "strict_full_horizon_within_split",
                "non_trading_boundary_policy": (
                    "use the first trading day on or after the configured range "
                    "start as prediction_start_date, and require "
                    "prediction_end_date to stay inside the configured range end"
                ),
                "split_ranges": metadata.get("splits", {}),
            },
        }, f, indent=2, ensure_ascii=False)

    predictor = _load_predictor(
        device=resolved_device,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
        max_context=config.max_context,
        clip=config.clip,
    )

    for split in args.splits:
        split_data = _load_split_data(args.data_path, split)
        split_meta = metadata["splits"][split]
        evaluate_split(predictor, split, split_data, split_meta, args, config, run_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
