# new_train_tokenizer.py
import os
os.environ["USE_LIBUV"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import sys
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import comet_ml

sys.path.append("../")
from config import Config
from new_dataset import QlibDataset
# 使用新的带有MoE和市场类型支持的分词器
from new_model.emoMarkets import MoEEmotionEnhancedTokenizer
from utils.training_utils import setup_ddp, cleanup_ddp, set_seed, get_model_size, format_time


def create_dataloaders(config: dict, rank: int, world_size: int):
    print(f"[Rank {rank}] Creating dataloaders with emotion features and market types...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train size: {len(train_dataset)}, Val size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=False,
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
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir, logger, rank, world_size):
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size * config['accumulation_steps']
        print(f"[Rank {rank}] Effective batch size: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    # 优化器设置
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
        div_factor=10,
        final_div_factor=100
    )

    best_val_loss = float('inf')
    batch_idx_global_train = 0
    patience_counter = 0
    patience = 5

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)

        for i, (ori_batch_x, ori_batch_x_stamp, ori_batch_emotion, ori_batch_market) in enumerate(train_loader):
            # 数据预处理
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)  # (B, T, 6)
            ori_batch_emotion = ori_batch_emotion.to(device, non_blocking=True)  # (B, T, 15)
            ori_batch_market = ori_batch_market.to(device, non_blocking=True).long()  # (B, T) - 市场类型标记
            current_batch_total_loss = 0.0

            for j in range(config['accumulation_steps']):
                # 梯度累积：按批次切片
                start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
                batch_x = ori_batch_x[start_idx:end_idx]
                batch_emotion = ori_batch_emotion[start_idx:end_idx]
                batch_market = ori_batch_market[start_idx:end_idx]  # (B, T) - 市场类型标记

                # 模型前向传播，传入市场类型标记
                forward_result = model(batch_x, emotion=batch_emotion)
                z_pre, z_full = forward_result[0]  # K线重构结果
                kline_bsq_loss = forward_result[1]  # K线量化损失
                quantized_kline = forward_result[2]
                z_indices = forward_result[3]
                z_emotion_pre , z_emotion_full = forward_result[4]
                bsq_loss_emotion = forward_result[5]
                quantized_emotion = forward_result[6]
                emotion_tokens = forward_result[7]

                # 1. K线损失（独立计算）
                l_kline_recon_pre = F.mse_loss(z_pre, batch_x)
                l_kline_recon_full = F.mse_loss(z_full, batch_x)
                l_kline_recon = l_kline_recon_pre + l_kline_recon_full
                l_kline = (kline_bsq_loss + l_kline_recon)/2

                # 2. 情绪重构损失

                l_emotion_recon_pre = F.mse_loss(z_emotion_pre, batch_emotion)

                l_emotion_recon_full = F.mse_loss(z_emotion_full, batch_emotion)
                l_emotion_recon = l_emotion_recon_pre + l_emotion_recon_full
                l_emotion = (l_emotion_recon+bsq_loss_emotion)/2



                # 6. 总损失（包含情绪相关损失）
                loss = l_kline + l_emotion

                # 梯度累积
                loss_scaled = loss / config['accumulation_steps']
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            # 优化器更新
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
                logger.log_metric('tokenizer_learning_rate', optimizer.param_groups[0]["lr"],
                                  step=batch_idx_global_train)

            batch_idx_global_train += 1

        # 验证阶段
        model.eval()
        tot_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for ori_batch_x, ori_batch_x_stamp, ori_batch_emotion, ori_batch_market in val_loader:
                # 数据预处理
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)  # (B, T, 6)
                ori_batch_emotion = ori_batch_emotion.to(device, non_blocking=True)  # (B, T, 15)
                ori_batch_market = ori_batch_market.to(device, non_blocking=True).long()  # (B, T) - 市场类型标记

                for j in range(config['accumulation_steps']):
                    # 梯度累积：按批次切片
                    start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
                    end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
                    batch_x = ori_batch_x[start_idx:end_idx]
                    batch_emotion = ori_batch_emotion[start_idx:end_idx]
                    batch_market = ori_batch_market[start_idx:end_idx]  # (B, T) - 市场类型标记

                    # 模型前向传播，传入市场类型标记
                    forward_result = model(batch_x, emotion=batch_emotion)
                    z_pre, z_full = forward_result[0]  # K线重构结果
                    kline_bsq_loss = forward_result[1]  # K线量化损失
                    quantized_kline = forward_result[2]
                    z_indices = forward_result[3]
                    z_emotion_pre, z_emotion_full = forward_result[4]
                    bsq_loss_emotion = forward_result[5]
                    quantized_emotion = forward_result[6]
                    emotion_tokens = forward_result[7]

                    # 1. K线损失（独立计算）
                    l_kline_recon_pre = F.mse_loss(z_pre, batch_x)
                    l_kline_recon_full = F.mse_loss(z_full, batch_x)
                    l_kline_recon = l_kline_recon_pre + l_kline_recon_full
                    l_kline = (kline_bsq_loss + l_kline_recon) / 2

                    # 2. 情绪重构损失
                    l_emotion_recon_pre = F.mse_loss(z_emotion_pre, batch_emotion)
                    l_emotion_recon_full = F.mse_loss(z_emotion_full, batch_emotion)
                    l_emotion_recon = l_emotion_recon_pre + l_emotion_recon_full
                    l_emotion = (l_emotion_recon + bsq_loss_emotion) / 2

                    # 6. 总损失（包含情绪相关损失）
                    loss = l_kline + l_emotion
                    tot_val_loss += loss.item()  # 累积损失值
                    val_count += 1

        # 分布式损失聚合
        val_loss_tensor = torch.tensor(tot_val_loss, device=device)
        val_count_tensor = torch.tensor(val_count, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / val_count_tensor.item() if val_count > 0 else 0.0

        # 验证日志与早停逻辑
        if rank == 0:
            epoch_time = format_time(time.time() - epoch_start_time)
            print(f"\nEpoch {epoch_idx + 1} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time}")

            # 日志记录
            if logger:
                logger.log_metric('val_total_loss', avg_val_loss, epoch=epoch_idx)
            # 保存最优模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_tokenizer"
                model.module.save_pretrained(save_path)
                print(f"Saved best tokenizer to {save_path} (Val Loss: {best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered (validation loss not improving)")
                    break
        dist.barrier()

    # 训练总时间
    total_time = format_time(time.time() - start_time)
    if rank == 0:
        print(f"\nTraining Finished | Best Val Loss: {best_val_loss:.4f} | Total Time: {total_time}")
    return model, {'best_val_loss': best_val_loss}


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
    model = MoEEmotionEnhancedTokenizer(
        # 输入与核心维度
        d_in=6,  # 保持6维输入（K线数据：开盘/最高/最低/收盘/成交量/成交额）
        d_model=256,  # 模型维度
        n_heads=4,  # 注意力头数
        ff_dim=512,  # 前馈网络维度

        # 网络深度
        n_enc_layers=4,  # 编码器层数
        n_dec_layers=4,  # 解码器层数

        # Dropout参数（防止过拟合）
        ffn_dropout_p=0.1,  # 前馈网络dropout
        attn_dropout_p=0.1,  # 注意力dropout
        resid_dropout_p=0.1,  # 残差连接dropout

        # K线量化参数（完全独立）
        kline_s1_bits=10,  # K线预量化比特数
        kline_s2_bits=10,  # K线后量化比特数
        kline_beta=0.05,  # K线承诺损失权重
        kline_gamma0=1.0,  # K线熵惩罚系数
        kline_gamma=1.1,  # K线熵惩罚系数
        kline_zeta=0.05,  # K线总熵惩罚权重
        kline_group_size=5,  # K线分组大小

        # 情绪量化参数（完全独立）
        emotion_s1_bits=10,  # 情绪预量化比特数
        emotion_s2_bits=10,  # 情绪后量化比特数
        emotion_beta=0.05,  # 情绪承诺损失权重
        emotion_gamma0=1.0,  # 情绪熵惩罚系数
        emotion_gamma=1.1,  # 情绪熵惩罚系数
        emotion_zeta=0.05,  # 情绪总熵惩罚权重
        emotion_group_size=5,  # 情绪分组大小

        # 情绪维度
        emotion_dim=15  # 情绪特征维度
    )
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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
    config = Config()
    # 补充必要的配置默认值
    config_dict = config.__dict__
    config_dict.setdefault('accumulation_steps', 1)
    config_dict.setdefault('log_interval', 10)
    config_dict.setdefault('tokenizer_learning_rate', 1e-4)
    config_dict.setdefault('adam_weight_decay', 1e-5)
    config_dict.setdefault('epochs', 50)
    config_dict.setdefault('seed', 42)
    main(config_dict)
