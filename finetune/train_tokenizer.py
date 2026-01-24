# train_tokenizer.py
import os

os.environ["USE_LIBUV"] = "0"
import sys
import json
import time
from time import gmtime, strftime
import argparse
import datetime
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import comet_ml

# Ensure project root is in path
sys.path.append("../")
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')

    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=False,  # Shuffle is handled by the sampler
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False
    )
    print(f"[Rank {rank}] Dataloaders created. Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the tokenizer.

    Args:
        model (DDP): The DDP-wrapped model to train.
        device (torch.device): The device for the current process.
        config (dict): Configuration dictionary.
        save_dir (str): Directory to save checkpoints.
        logger (comet_ml.Experiment): Comet logger instance.
        rank (int): Global rank of the process.
        world_size (int): Total number of processes.

    Returns:
        tuple: A tuple containing the trained model and a dictionary of results.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size * config['accumulation_steps']
        print(f"[Rank {rank}] BATCHSIZE (per GPU): {config['batch_size']}")
        print(f"[Rank {rank}] Effective total batch size: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['tokenizer_learning_rate'],
        weight_decay=config['adam_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['tokenizer_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global_train = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)

        # Set dataset seeds for reproducible sampling
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)  # Keep validation sampling consistent

        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)

            # --- Gradient Accumulation Loop ---
            current_batch_total_loss = 0.0
            for j in range(config['accumulation_steps']):
                start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
                batch_x = ori_batch_x[start_idx:end_idx]

                # Forward pass
                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs

                # Loss calculation
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2  # Assuming w_1=w_2=1

                loss_scaled = loss / config['accumulation_steps']
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            # --- Optimizer Step after Accumulation ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # --- Logging (Master Process Only) ---
            if rank == 0 and (batch_idx_global_train + 1) % config['log_interval'] == 0:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )
            if rank == 0 and logger:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                logger.log_metric('train_tokenizer_loss_batch', avg_loss, step=batch_idx_global_train)
                logger.log_metric('tokenizer_learning_rate', optimizer.param_groups[0]["lr"], step=batch_idx_global_train)

            batch_idx_global_train += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)

                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)

        # Reduce validation losses from all processes
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_count_tensor = torch.tensor(val_sample_count_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_count_tensor.item() if val_count_tensor.item() > 0 else 0

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_tokenizer_loss_epoch', avg_val_loss, epoch=epoch_idx)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                if logger:
                    logger.log_model("best_model", save_path)

        dist.barrier()  # Ensure all processes finish the epoch before starting the next one.

    dt_result['best_val_loss'] = best_val_loss
    return model, dt_result


def main(config: dict):
    """
    Main function to orchestrate the DDP training process.
    """
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['tokenizer_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_comet']:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    dist.barrier()  # Ensure save directory is created before proceeding

    # Model Initialization
    # model = KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
    model = KronosTokenizer(
        # 输入与核心维度
        d_in=6,  # 保持6维输入（K线数据：开盘/最高/最低/收盘/成交量/成交额）
        d_model=256,  # 模型维度提升至768（24GB显存可支持）
        n_heads=4,  # 注意力头数与d_model匹配（768/12=64，符合多头注意力设计）
        ff_dim=512,  # 前馈网络维度（通常为d_model的4倍，768*4=3072）

        # 网络深度
        n_enc_layers=4,  # 编码器层数提升至8层（增强特征提取能力）
        n_dec_layers=4,  # 解码器与编码器对称，提升重建精度

        # Dropout参数（防止过拟合）
        ffn_dropout_p=0.1,  # 前馈网络dropout（适度降低，提升特征学习稳定性）
        attn_dropout_p=0.1,  # 注意力dropout（保持常规设置）
        resid_dropout_p=0.1,  # 残差连接dropout（保持常规设置）

        # 量化参数（提升精度，GPU可承载更高计算量）
        s1_bits=10,  # 预量化比特数提升至10（减少信息损失）
        s2_bits=10,  # 后量化比特数提升至10（总码本维度20）
        group_size=5,  # 分组大小增大至16（降低量化计算复杂度）

        # BSQuantizer其他参数（保持与量化策略匹配）
        beta=0.05,  # 承诺损失权重（默认值，平衡量化误差）
        gamma0=1.0,  # 熵惩罚系数（控制量化分布）
        gamma=1.1,  # 熵惩罚系数（维持量化稳定性）
        zeta=0.05  # 总熵惩罚权重（默认值）
    )
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Model Size: {get_model_size(model.module)}")

    # Start Training
    _, dt_result = train_model(
        model, device, config, save_dir, comet_logger, rank, world_size
    )

    # Finalize and save summary (master process only)
    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger:
            comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    config_instance = Config()
    main(config_instance.__dict__)

