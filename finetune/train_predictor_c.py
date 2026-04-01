"""
C-group training script: finetune Kronos daily predictor with hourly
auxiliary context via cross-attention fusion.

Training flow:
    1. Load pretrained tokenizer (frozen) and predictor
    2. Build KronosWithHourly (encoder + fusion + split predictor)
    3. Freeze lower predictor layers; train upper layers + encoder + fusion
    4. Use DailyHourlyDataset for dual-stream data
    5. Loss = future-only token CE (same as B group)

Usage:
    torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor_c.py [options]
"""

import os
import sys
import json
import time
import argparse
from time import gmtime, strftime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../")
from config import Config
from dataset_c import DailyHourlyDataset
from model.kronos import KronosTokenizer, Kronos
from models.hourly_encoder import HourlyEncoder
from models.hourly_fusion import HourlyFusionLayer
from models.kronos_with_hourly import KronosWithHourly
from evaluate_c import (
    _load_metadata as load_eval_metadata,
    _load_split_data as load_eval_split_data,
    _build_eval_windows as build_eval_windows,
    _run_inference as run_eval_inference,
    _build_daily_metrics as build_daily_metrics,
    _compute_summary as compute_eval_summary,
)
from utils.training_utils import setup_ddp, cleanup_ddp, set_seed, get_model_size, format_time
from utils.experiment_logger import build_experiment_logger


def resolve_tokenizer_path(config: dict) -> str:
    if config.get("predictor_tokenizer_path"):
        return config["predictor_tokenizer_path"]
    if config.get("skip_tokenizer_finetune"):
        return config["pretrained_tokenizer_path"]
    return config["finetuned_tokenizer_path"]


def build_model(config: dict, device, rank: int):
    """Build KronosWithHourly with proper freeze strategy."""
    # Kronos predictor
    kronos = Kronos.from_pretrained(config["pretrained_predictor_path"])
    total_layers = len(kronos.transformer)
    train_last_ratio = float(config.get("predictor_train_last_ratio", 0.333333))
    split_point = max(0, int(total_layers * (1.0 - train_last_ratio)))

    # Freeze everything in kronos first
    for p in kronos.parameters():
        p.requires_grad = False

    # Unfreeze upper transformer layers
    for i in range(split_point, total_layers):
        for p in kronos.transformer[i].parameters():
            p.requires_grad = True
    for module in (kronos.norm, kronos.dep_layer, kronos.head):
        for p in module.parameters():
            p.requires_grad = True

    if not config.get("freeze_embedding", True):
        for p in kronos.embedding.parameters():
            p.requires_grad = True
    if config.get("train_time_embedding", False):
        for p in kronos.time_emb.parameters():
            p.requires_grad = True

    # Hourly encoder (all trainable, random init)
    d_model = kronos.d_model
    hourly_encoder = HourlyEncoder(
        d_in=len(config.get("feature_list", ["open", "high", "low", "close", "vol", "amt"])),
        d_model=d_model,
        n_heads=config.get("hourly_n_heads", 8),
        ff_dim=config.get("hourly_ff_dim", 1024),
        n_layers=config.get("hourly_encoder_layers", 2),
        ffn_dropout_p=config.get("hourly_dropout", 0.1),
        attn_dropout_p=0.0,
        resid_dropout_p=config.get("hourly_dropout", 0.1),
    )

    # Fusion layer (all trainable, random init)
    fusion = HourlyFusionLayer(
        d_model=d_model,
        n_heads=config.get("fusion_n_heads", 8),
        attn_dropout_p=0.0,
        resid_dropout_p=config.get("fusion_dropout", 0.1),
    )

    model = KronosWithHourly(kronos, hourly_encoder, fusion, split_point)

    if rank == 0:
        print(f"Kronos layers: {total_layers}, split_point: {split_point}")
        print(f"  Frozen layers: [0, {split_point}), Trainable layers: [{split_point}, {total_layers})")
        print(f"  HourlyEncoder layers: {config.get('hourly_encoder_layers', 2)}, d_model: {d_model}")
        print(f"  freeze_embedding={config.get('freeze_embedding', True)}, "
              f"train_time_embedding={config.get('train_time_embedding', False)}")

    return model, split_point


def build_optimizer(model: KronosWithHourly, config: dict, rank: int):
    """Build optimizer with separate learning rates for predictor vs hourly modules."""
    predictor_lr = config["predictor_learning_rate"]
    hourly_lr = config.get("hourly_learning_rate", predictor_lr * 2)

    predictor_params = []
    hourly_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "hourly_encoder" in name or "fusion_layer" in name:
            hourly_params.append(p)
        else:
            predictor_params.append(p)

    param_groups = [
        {"params": predictor_params, "lr": predictor_lr},
        {"params": hourly_params, "lr": hourly_lr},
    ]

    if rank == 0:
        print(f"Optimizer: predictor_params={len(predictor_params)} (lr={predictor_lr}), "
              f"hourly_params={len(hourly_params)} (lr={hourly_lr})")

    return torch.optim.AdamW(
        param_groups,
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"],
    )


def compute_token_loss(model, logits, token_out, config: dict):
    """Compute future-only or full-sequence token CE loss."""
    # Access kronos head through the DDP wrapper
    head = model.module.kronos.head
    if not config.get("future_only_loss", True):
        return head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
    start = config["lookback_window"] - 1
    end = start + config["predict_window"]
    return head.compute_loss(
        logits[0][:, start:end, :], logits[1][:, start:end, :],
        token_out[0][:, start:end], token_out[1][:, start:end],
    )


def create_dataloaders(config: dict, rank: int, world_size: int):
    print(f"[Rank {rank}] Creating C-group dataloaders...")
    train_ds = DailyHourlyDataset("train")
    val_ds = DailyHourlyDataset("val")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=config.get("num_workers", 2), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], sampler=val_sampler,
        num_workers=config.get("num_workers", 2), pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds


def build_val_rankic_context(config: dict) -> dict:
    metadata = load_eval_metadata(config["dataset_path"])
    split_data = load_eval_split_data(config["dataset_path"], "val")
    split_meta = metadata["splits"]["val"]
    windows = build_eval_windows(
        split_data,
        split_meta,
        config["lookback_window"],
        config["predict_window"],
        config["hourly_window"],
        config["feature_list"],
    )
    if not windows:
        raise RuntimeError("No validation windows available for C-group RankIC evaluation.")
    return {"windows": windows}


def evaluate_val_rankic(model: KronosWithHourly, tokenizer: KronosTokenizer, device, config: dict, eval_context: dict) -> dict:
    predictions = run_eval_inference(
        model=model,
        tokenizer=tokenizer,
        windows=eval_context["windows"],
        pred_len=config["predict_window"],
        hourly_window=config["hourly_window"],
        batch_size=config.get("eval_batch_size", 128),
        sample_count=config.get("inference_sample_count", 5),
        temperature=config.get("inference_T", 0.6),
        top_k=config.get("inference_top_k", 0),
        top_p=config.get("inference_top_p", 0.9),
        max_context=config["max_context"],
        clip=config["clip"],
        device=str(device),
        feature_list=config["feature_list"],
    )
    daily_df = build_daily_metrics(predictions, config.get("eval_topk", 10))
    return compute_eval_summary(predictions, daily_df, config.get("eval_topk", 10))


def _metric_is_valid(value) -> bool:
    return value is not None and np.isfinite(value)


def _write_c_config(model: DDP, config: dict, ckpt_dir: str):
    with open(f"{ckpt_dir}/c_config.json", "w") as f:
        json.dump({
            "split_point": model.module.split_point,
            "hourly_encoder_layers": config.get("hourly_encoder_layers", 2),
            "hourly_n_heads": config.get("hourly_n_heads", 8),
            "hourly_ff_dim": config.get("hourly_ff_dim", 1024),
            "hourly_dropout": config.get("hourly_dropout", 0.1),
            "fusion_n_heads": config.get("fusion_n_heads", 8),
            "fusion_dropout": config.get("fusion_dropout", 0.1),
            "d_model": model.module.kronos.d_model,
            "d_in": len(config.get("feature_list", ["open", "high", "low", "close", "vol", "amt"])),
        }, f, indent=2)


def save_loss_checkpoint(model: DDP, config: dict, save_dir: str):
    save_paths = [
        f"{save_dir}/checkpoints/best_model",
        f"{save_dir}/checkpoints/best_model_by_loss",
    ]
    for ckpt_dir in save_paths:
        model.module.save_all(ckpt_dir)
        _write_c_config(model, config, ckpt_dir)


def save_rankic_checkpoint(model: DDP, config: dict, save_dir: str):
    ckpt_dir = f"{save_dir}/checkpoints/best_model_by_rankic"
    model.module.save_all(ckpt_dir)
    _write_c_config(model, config, ckpt_dir)


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size, val_rankic_context=None):
    start_time = time.time()
    if rank == 0:
        eff_bs = config["batch_size"] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {eff_bs}")
        print(f"Loss mode: {'future-only' if config.get('future_only_loss', True) else 'full-sequence'}")

    train_loader, val_loader, train_ds, val_ds = create_dataloaders(config, rank, world_size)

    optimizer = build_optimizer(model.module, config, rank)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[config["predictor_learning_rate"],
                           config.get("hourly_learning_rate", config["predictor_learning_rate"] * 2)],
        steps_per_epoch=len(train_loader), epochs=config["epochs"],
        pct_start=0.03, div_factor=10,
    )

    best_val_loss = float("inf")
    best_val_loss_epoch = None
    best_val_rank_ic = float("-inf")
    best_val_rank_ic_epoch = None
    batch_idx_global = 0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch)
        train_ds.set_epoch_seed(epoch * 10000 + rank)
        val_ds.set_epoch_seed(0)

        for i, (batch_x_d, batch_stamp_d, batch_x_h, batch_stamp_h) in enumerate(train_loader):
            batch_x_d = batch_x_d.to(device, non_blocking=True)
            batch_stamp_d = batch_stamp_d.to(device, non_blocking=True)
            batch_x_h = batch_x_h.to(device, non_blocking=True)
            batch_stamp_h = batch_stamp_h.to(device, non_blocking=True)

            # Tokenize daily data
            with torch.no_grad():
                tok_s1, tok_s2 = tokenizer.encode(batch_x_d, half=True)

            token_in = [tok_s1[:, :-1], tok_s2[:, :-1]]
            token_out = [tok_s1[:, 1:], tok_s2[:, 1:]]

            # Forward pass with hourly context
            logits = model(
                token_in[0], token_in[1],
                batch_stamp_d[:, :-1, :],
                batch_x_h, batch_stamp_h,
            )
            loss, s1_loss, s2_loss = compute_token_loss(model, logits, token_out, config)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            if rank == 0 and (batch_idx_global + 1) % config["log_interval"] == 0:
                lr_pred = optimizer.param_groups[0]["lr"]
                lr_hourly = optimizer.param_groups[1]["lr"]
                print(
                    f"[Epoch {epoch+1}/{config['epochs']}, Step {i+1}/{len(train_loader)}] "
                    f"LR_pred {lr_pred:.6f}, LR_hourly {lr_hourly:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]["lr"]
                logger.log_metric("train_loss_batch", loss.item(), step=batch_idx_global, tb_tag="c/train/loss")
                logger.log_metric("train_S1_loss", s1_loss.item(), step=batch_idx_global, tb_tag="c/train/s1_loss")
                logger.log_metric("train_S2_loss", s2_loss.item(), step=batch_idx_global, tb_tag="c/train/s2_loss")
                logger.log_metric("pred_lr", lr, step=batch_idx_global, tb_tag="c/train/lr_pred")
                logger.log_metric("hourly_lr", optimizer.param_groups[1]["lr"], step=batch_idx_global, tb_tag="c/train/lr_hourly")
                logger.log_metric("grad_norm", grad_norm, step=batch_idx_global, tb_tag="c/train/grad_norm")

            batch_idx_global += 1

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_x_d, batch_stamp_d, batch_x_h, batch_stamp_h in val_loader:
                batch_x_d = batch_x_d.to(device, non_blocking=True)
                batch_stamp_d = batch_stamp_d.to(device, non_blocking=True)
                batch_x_h = batch_x_h.to(device, non_blocking=True)
                batch_stamp_h = batch_stamp_h.to(device, non_blocking=True)

                tok_s1, tok_s2 = tokenizer.encode(batch_x_d, half=True)
                token_in = [tok_s1[:, :-1], tok_s2[:, :-1]]
                token_out = [tok_s1[:, 1:], tok_s2[:, 1:]]

                logits = model(
                    token_in[0], token_in[1],
                    batch_stamp_d[:, :-1, :],
                    batch_x_h, batch_stamp_h,
                )
                vl, _, _ = compute_token_loss(model, logits, token_out, config)
                val_loss_sum += vl.item()
                val_n += 1

        # All-reduce validation
        vl_tensor = torch.tensor(val_loss_sum, device=device)
        vn_tensor = torch.tensor(val_n, device=device)
        dist.all_reduce(vl_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(vn_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = vl_tensor.item() / vn_tensor.item() if vn_tensor.item() > 0 else 0

        val_rankic_summary = None
        val_mean_rank_ic = None
        if rank == 0 and val_rankic_context is not None:
            val_rankic_summary = evaluate_val_rankic(model.module, tokenizer, device, config, val_rankic_context)
            val_mean_rank_ic = val_rankic_summary.get("mean_rank_ic")

        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            if _metric_is_valid(val_mean_rank_ic):
                print(f"Validation mean RankIC: {val_mean_rank_ic:.4f}")
            else:
                print("Validation mean RankIC: None")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric("val_loss_epoch", avg_val_loss, epoch=epoch, tb_tag="c/val/loss")
                if val_rankic_summary is not None:
                    for key, tb_tag in (
                        ("mean_rank_ic", "c/val/mean_rank_ic"),
                        ("mean_ic", "c/val/mean_ic"),
                        (f"long_short_top{config.get('eval_topk', 10)}_mean_return", "c/val/long_short_mean_return"),
                    ):
                        value = val_rankic_summary.get(key)
                        if _metric_is_valid(value):
                            logger.log_metric(key, value, epoch=epoch, tb_tag=tb_tag)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_loss_epoch = epoch + 1
                save_loss_checkpoint(model, config, save_dir)
                print(
                    f"Best loss model saved to {save_dir}/checkpoints/best_model_by_loss "
                    f"(Val Loss: {best_val_loss:.4f})"
                )
                if logger:
                    logger.log_model("best_model_by_loss", f"{save_dir}/checkpoints/best_model_by_loss")

            if _metric_is_valid(val_mean_rank_ic) and val_mean_rank_ic > best_val_rank_ic:
                best_val_rank_ic = float(val_mean_rank_ic)
                best_val_rank_ic_epoch = epoch + 1
                save_rankic_checkpoint(model, config, save_dir)
                print(
                    f"Best RankIC model saved to {save_dir}/checkpoints/best_model_by_rankic "
                    f"(Val mean RankIC: {best_val_rank_ic:.4f})"
                )
                if logger:
                    logger.log_model("best_model_by_rankic", f"{save_dir}/checkpoints/best_model_by_rankic")

        dist.barrier()

    return {
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_rank_ic": best_val_rank_ic if _metric_is_valid(best_val_rank_ic) else None,
        "best_val_rank_ic_epoch": best_val_rank_ic_epoch,
    }


def main(config: dict):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config["seed"], rank)

    save_dir = os.path.join(config["save_path"], config["predictor_save_folder_name"])

    logger, summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        summary = {
            "start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            "save_directory": save_dir,
            "world_size": world_size,
            "mode": "c_group",
        }
        logger = build_experiment_logger(config, save_dir)
        if getattr(logger, "has_tensorboard", False):
            print(f"TensorBoard log dir: {logger.tensorboard_log_dir}")

    dist.barrier()

    # Tokenizer (frozen)
    tok_path = resolve_tokenizer_path(config)
    tokenizer = KronosTokenizer.from_pretrained(tok_path).eval().to(device)

    # Build C-group model
    model, split_point = build_model(config, device, rank)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    val_rankic_context = None
    if rank == 0:
        val_rankic_context = build_val_rankic_context(config)
        print(f"Validation RankIC windows: {len(val_rankic_context['windows'])}")

    dist.barrier()

    result = train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size, val_rankic_context)

    if rank == 0:
        summary["result"] = result
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        print("Training finished. Summary saved.")
        if logger:
            logger.log_metric("best_val_loss", result["best_val_loss"], step=0, tb_tag="c/val/best_loss")
            if _metric_is_valid(result.get("best_val_rank_ic")):
                logger.log_metric("best_val_rank_ic", result["best_val_rank_ic"], step=0, tb_tag="c/val/best_rank_ic")
            logger.close()

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description="C-group: finetune Kronos with hourly auxiliary data.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--predictor-learning-rate", type=float)
    parser.add_argument("--hourly-learning-rate", type=float)
    parser.add_argument("--hourly-encoder-layers", type=int)
    parser.add_argument("--hourly-window", type=int)
    parser.add_argument("--save-folder-name", type=str)
    parser.add_argument("--train-last-ratio", type=float)
    parser.add_argument("--disable-comet", action="store_true")
    parser.add_argument("--disable-tensorboard", action="store_true")
    return parser.parse_args()


def _env_present(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value != ""


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with `torchrun`.")

    args = parse_args()
    config = Config().__dict__

    # Ensure C-group defaults unless the user already overrode them via env.
    if not _env_present("KRONOS_FUTURE_ONLY_LOSS"):
        config["future_only_loss"] = True
    if not _env_present("KRONOS_FREEZE_EMBEDDING"):
        config["freeze_embedding"] = True
    if not _env_present("KRONOS_TRAIN_TIME_EMBEDDING"):
        config["train_time_embedding"] = False
    if not _env_present("KRONOS_SKIP_TOKENIZER_FINETUNE"):
        config["skip_tokenizer_finetune"] = True
    if not _env_present("KRONOS_PREDICTOR_TRAIN_LAST_RATIO"):
        config["predictor_train_last_ratio"] = 0.333333

    # CLI overrides
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.predictor_learning_rate is not None:
        config["predictor_learning_rate"] = args.predictor_learning_rate
    if args.hourly_learning_rate is not None:
        config["hourly_learning_rate"] = args.hourly_learning_rate
    if args.hourly_encoder_layers is not None:
        config["hourly_encoder_layers"] = args.hourly_encoder_layers
    if args.hourly_window is not None:
        config["hourly_window"] = args.hourly_window
        os.environ["KRONOS_HOURLY_WINDOW"] = str(args.hourly_window)
    if args.save_folder_name:
        config["predictor_save_folder_name"] = args.save_folder_name
    if args.train_last_ratio is not None:
        config["predictor_train_last_ratio"] = args.train_last_ratio
    if args.disable_comet:
        config["use_comet"] = False
    if args.disable_tensorboard:
        config["use_tensorboard"] = False

    main(config)
