# train_predictor.py
import os
import sys
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import comet_ml

# 确保项目根目录在Python路径中
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# 导入共享工具函数
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    创建并返回用于训练和验证的分布式数据加载器

    Args:
        config (dict): 配置参数字典
        rank (int): 当前进程的全局排名
        world_size (int): 总进程数

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset)
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    # 创建训练和验证数据集
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    """
    Predictor模型的主训练和验证循环

    Args:
        model: 要训练的模型
        tokenizer: 分词器
        device: 训练设备
        config: 配置字典
        save_dir: 模型保存目录
        logger: 日志记录器
        rank: 当前进程排名
        world_size: 总进程数
    """
    start_time = time.time()
    # 主进程打印有效batch size信息
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    # 创建数据加载器
    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    # 使用更小的学习率初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'] ,  # 降低学习率
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    # 训练控制变量
    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = 0
    patience_counter = 0
    patience = 5  # 早停耐心值

    # 主训练循环
    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        # 设置采样器的epoch以确保不同epoch的数据shuffle不同
        train_loader.sampler.set_epoch(epoch_idx)

        # 为数据集设置随机种子
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        # 训练循环
        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            # 将数据移动到指定设备
            batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # 日志记录（仅主进程）
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- 验证循环 ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        # 聚合验证指标（分布式训练）
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0

        # --- Epoch结束总结和检查点保存（仅主进程） ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

        # 同步所有进程
        dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    # 设置保存目录
    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

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

    dist.barrier()

    # 模型初始化
    # 加载预训练的tokenizer
    tokenizer = KronosTokenizer.from_pretrained("outputs/models/finetune_tokenizer_demo/checkpoints/best_tokenizer")
    tokenizer.eval().to(device)

    # 创建Kronos预测模型
    model = Kronos(
        # 量化相关参数（Kronos类核心参数，与Tokenizer对应）
        s1_bits=10,  # 粗粒度子token位数（原8bit合理，总码本维度16bit）
        s2_bits=10,  # 细粒度子token位数（与s1_bits匹配，保持16bit总长度）

        # 网络结构参数（需符合Kronos类定义）
        n_layers=12,  # Transformer块数量（8层合理，平衡能力与效率）
        d_model=832,  # 模型隐藏层维度（768可被12整除，符合多头注意力要求）
        n_heads=16,  # 注意力头数（768/12=64，每个头维度为64，合理）
        ff_dim=2048,  # 前馈网络维度（768*4=3072，符合常规设计）

        # Dropout参数（防止过拟合，需与任务复杂度匹配）
        ffn_dropout_p=0.2,  # 前馈网络dropout（0.1适合中等复杂度任务）
        attn_dropout_p=0.0,  # 注意力dropout（与ffn_dropout保持一致）
        resid_dropout_p=0.2,  # 残差连接dropout（控制信息流保留比例）
        token_dropout_p=0.0,  # Token嵌入dropout（K线数据时序连续，建议0）

        # 时间嵌入参数（A股K线有明确时序属性）
        learn_te=True  # 学习可训练的时间嵌入（提升时序建模能力）
    )
    model.to(device)
    # 使用DDP包装模型
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Predictor Model Size: {get_model_size(model.module)}")

    # 开始训练
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    # 保存训练总结（仅主进程）
    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger:
            comet_logger.end()

    # 清理DDP环境
    cleanup_ddp()


if __name__ == '__main__':

    # if "WORLD_SIZE" not in os.environ:
    #     raise RuntimeError("This script must be launched with `torchrun`.")

    config_instance = Config()
    main(config_instance.__dict__)
