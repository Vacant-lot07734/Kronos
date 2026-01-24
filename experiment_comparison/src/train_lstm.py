import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .utils import set_seed, ensure_dir, get_device
from .data.io import load_all_stocks
from .data.dataset import GlobalMinMax, MultiStockWindowDataset
from .models.lstm import LSTMForecast

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    losses = []
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device()

    min_len = cfg.lookback + cfg.pred_len
    stock_dfs = load_all_stocks(
        csv_dir=cfg.csv_dir,
        pattern=cfg.pattern,
        date_col=cfg.date_col,
        features=cfg.features,
        min_len=min_len,
        max_files=cfg.max_files
    )
    if len(stock_dfs) == 0:
        raise RuntimeError("没有可用股票数据（检查csv_dir/列名/数据长度）")

    # scaler: 只用训练段fit
    scaler = GlobalMinMax().fit(stock_dfs, cfg.date_col, cfg.features, cfg.train_end, cfg.val_end)

    train_ds = MultiStockWindowDataset(
        stock_dfs, "train", cfg.date_col, cfg.features, cfg.lookback, cfg.pred_len, cfg.train_end, cfg.val_end, scaler=scaler
    )
    val_ds = MultiStockWindowDataset(
        stock_dfs, "val", cfg.date_col, cfg.features, cfg.lookback, cfg.pred_len, cfg.train_end, cfg.val_end, scaler=scaler
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

    model = LSTMForecast(
        input_dim=len(cfg.features),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pred_len=cfg.pred_len,
        output_dim=len(cfg.features)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va = eval_one_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {ep:03d} | train={tr:.6f} | val={va:.6f}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stop. Best val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ensure_dir(cfg.ckpt_dir)
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_min": scaler.scaler.min_,
            "scaler_scale": scaler.scaler.scale_,
            "features": cfg.features,
            "date_col": cfg.date_col,
            "lookback": cfg.lookback,
            "pred_len": cfg.pred_len,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout
        },
        ckpt_path
    )
    print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
