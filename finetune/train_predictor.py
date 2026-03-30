import os
import sys
import json
import time
import argparse
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)


def create_comet_logger(config: dict):
    if not config.get('use_comet'):
        return None

    try:
        from comet_ml import Experiment
    except ImportError as exc:
        raise ImportError(
            "comet_ml is not installed, but use_comet=True. "
            "Install comet_ml or run train_predictor.py with --disable-comet."
        ) from exc

    logger = Experiment(
        api_key=config['comet_config']['api_key'],
        project_name=config['comet_config']['project_name'],
        workspace=config['comet_config']['workspace'],
    )
    logger.add_tag(config['comet_tag'])
    logger.set_name(config['comet_name'])
    logger.log_parameters(config)
    return logger


def resolve_predictor_tokenizer_path(config: dict) -> str:
    """Returns the tokenizer path to use for predictor finetuning."""
    if config.get('predictor_tokenizer_path'):
        return config['predictor_tokenizer_path']
    if config.get('skip_tokenizer_finetune'):
        return config['pretrained_tokenizer_path']
    return config['finetuned_tokenizer_path']


def apply_predictor_finetune_strategy(model: Kronos, config: dict, rank: int) -> None:
    """
    Optionally freezes most predictor layers for A/B experiments.

    When `freeze_predictor_for_ab` is False, the original full-finetune behavior is preserved.
    """
    if not config.get('freeze_predictor_for_ab', False):
        return

    train_last_ratio = float(config.get('predictor_train_last_ratio', 1.0))
    train_last_ratio = max(0.0, min(1.0, train_last_ratio))

    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.transformer)
    train_from = max(0, int(total_layers * (1.0 - train_last_ratio)))

    for layer_idx in range(train_from, total_layers):
        for param in model.transformer[layer_idx].parameters():
            param.requires_grad = True

    for module in (model.norm, model.dep_layer, model.head):
        for param in module.parameters():
            param.requires_grad = True

    if not config.get('freeze_embedding', False):
        for param in model.embedding.parameters():
            param.requires_grad = True

    if config.get('train_time_embedding', False):
        for param in model.time_emb.parameters():
            param.requires_grad = True

    if rank == 0:
        print(
            f"Applied predictor freeze strategy: training transformer layers "
            f"[{train_from}, {total_layers - 1}], "
            f"freeze_embedding={config.get('freeze_embedding', False)}, "
            f"train_time_embedding={config.get('train_time_embedding', False)}"
        )


def compute_token_loss(model, logits, token_out, config: dict):
    """
    Computes either full-sequence loss or future-only loss.

    The future-only slice aligns with the H-step prediction target:
    token_out position `lookback_window - 1` corresponds to the first future bar.
    """
    if not config.get('future_only_loss', False):
        return model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

    start_idx = config['lookback_window'] - 1
    end_idx = start_idx + config['predict_window']
    return model.module.head.compute_loss(
        logits[0][:, start_idx:end_idx, :],
        logits[1][:, start_idx:end_idx, :],
        token_out[0][:, start_idx:end_idx],
        token_out[1][:, start_idx:end_idx],
    )


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

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
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")
        print(f"Loss mode: {'future-only' if config.get('future_only_loss', False) else 'full-sequence'}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable predictor parameters remain after applying the freeze strategy.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
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
            loss, s1_loss, s2_loss = compute_token_loss(model, logits, token_out, config)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging (Master Process Only)
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

        # --- Validation Loop ---
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
                val_loss, _, _ = compute_token_loss(model, logits, token_out, config)

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        # Reduce validation metrics
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

        dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

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
            comet_logger = create_comet_logger(config)
            print("Comet Logger Initialized.")

    dist.barrier()

    # Model Initialization
    tokenizer_path = resolve_predictor_tokenizer_path(config)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    apply_predictor_finetune_strategy(model, config, rank)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Predictor tokenizer path: {tokenizer_path}")
        print(f"Predictor Model Size: {get_model_size(model.module)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune the Kronos predictor model.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size per GPU.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--predictor-learning-rate", type=float, help="Override predictor learning rate.")
    parser.add_argument("--tokenizer-path", type=str, help="Tokenizer path/model id used during predictor finetuning.")
    parser.add_argument("--save-folder-name", type=str, help="Checkpoint folder name under save_path.")
    parser.add_argument("--use-pretrained-tokenizer", action="store_true", help="Skip tokenizer finetuning and use the pretrained tokenizer directly.")
    parser.add_argument("--freeze-for-ab", action="store_true", help="Freeze lower predictor layers and only finetune the upper layers.")
    parser.add_argument("--train-last-ratio", type=float, help="Fraction of transformer blocks to keep trainable from the top.")
    parser.add_argument("--freeze-embedding", action="store_true", help="Freeze token embedding when using --freeze-for-ab.")
    parser.add_argument("--train-time-embedding", action="store_true", help="Keep time embedding trainable when using --freeze-for-ab.")
    parser.add_argument("--future-only-loss", action="store_true", help="Train and validate only on the future horizon positions.")
    parser.add_argument("--disable-comet", action="store_true", help="Disable Comet logging.")
    return parser.parse_args()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with `torchrun`.")

    args = parse_args()
    config = Config().__dict__

    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.predictor_learning_rate is not None:
        config['predictor_learning_rate'] = args.predictor_learning_rate
    if args.tokenizer_path:
        config['predictor_tokenizer_path'] = args.tokenizer_path
    if args.save_folder_name:
        config['predictor_save_folder_name'] = args.save_folder_name
    if args.use_pretrained_tokenizer:
        config['skip_tokenizer_finetune'] = True
    if args.freeze_for_ab:
        config['freeze_predictor_for_ab'] = True
    if args.train_last_ratio is not None:
        config['predictor_train_last_ratio'] = args.train_last_ratio
    if args.freeze_embedding:
        config['freeze_embedding'] = True
    if args.train_time_embedding:
        config['train_time_embedding'] = True
    if args.future_only_loss:
        config['future_only_loss'] = True
    if args.disable_comet:
        config['use_comet'] = False

    main(config)
