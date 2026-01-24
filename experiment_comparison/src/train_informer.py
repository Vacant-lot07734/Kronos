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
from .models.informer import Informer


def make_decoder_input(x_enc, label_len, pred_len):
    # x_enc: [B, lookback, 6]
    x_label = x_enc[:, -label_len:, :]                   # [B,label_len,6]
    zeros = torch.zeros((x_enc.size(0), pred_len, x_enc.size(-1)),
                        device=x_enc.device, dtype=x_enc.dtype)
    x_dec = torch.cat([x_label, zeros], dim=1)           # [B,label_len+pred_len,6]
    return x_dec


def train_one_epoch(model, loader, opt, loss_fn, device, label_len, pred_len):
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_dec = make_decoder_input(x, label_len, pred_len)

        pred = model(x, x_dec)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device, label_len, pred_len):
    model.eval()
    losses = []
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_dec = make_decoder_input(x, label_len, pred_len)

        pred = model(x, x_dec)
        losses.append(loss_fn(pred, y).item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device()

    inf_cfg = cfg.__dict__.get("informer", None)
    if inf_cfg is None:
        raise ValueError("configs/default.yaml 缺少 informer 配置段")

    lookback = cfg.lookback
    pred_len = cfg.pred_len
    label_len = int(inf_cfg.get("label_len", 10))
    if label_len > lookback:
        raise ValueError("informer.label_len 必须 <= lookback")

    min_len = lookback + pred_len
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

    scaler = GlobalMinMax().fit(stock_dfs, cfg.date_col, cfg.features, cfg.train_end, cfg.val_end)

    train_ds = MultiStockWindowDataset(
        stock_dfs, "train", cfg.date_col, cfg.features, lookback, pred_len, cfg.train_end, cfg.val_end, scaler=scaler
    )
    val_ds = MultiStockWindowDataset(
        stock_dfs, "val", cfg.date_col, cfg.features, lookback, pred_len, cfg.train_end, cfg.val_end, scaler=scaler
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None
    )

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
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device, label_len, pred_len)
        va = eval_one_epoch(model, val_loader, loss_fn, device, label_len, pred_len)
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
    ckpt_path = os.path.join(cfg.ckpt_dir, "informer_" + cfg.ckpt_name)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_min": scaler.scaler.min_,
            "scaler_scale": scaler.scaler.scale_,
            "features": cfg.features,
            "date_col": cfg.date_col,
            "lookback": lookback,
            "pred_len": pred_len,
            "informer_cfg": inf_cfg,
        },
        ckpt_path
    )
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
