#!/usr/bin/env python3
"""
Distributed Data Parallel Training Script

Train GPT across multiple GPUs using data parallelism.
Each GPU holds a complete model copy.

Usage:
    torchrun --nproc_per_node=4 scripts/train_distributed.py \\
        --model_name gpt2-small \\
        --batch_size 8 \\
        --num_epochs 3
"""

import argparse
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinytrain.model.gpt import GPT
from tinytrain.model.config import GPTConfig
from tinytrain.distributed.comm import init_distributed, get_rank, get_world_size
from tinytrain.distributed.data_parallel import DataParallel
from tinytrain.training.trainer import Trainer
from tinytrain.training.optimizer import AdamW
from tinytrain.training.scheduler import CosineAnnealingWarmup
from tinytrain.training.data_loader import DataLoaderFactory, create_dummy_data
from tinytrain.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main training function."""
    # Initialize distributed training
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    parser = argparse.ArgumentParser(description="Distributed GPT training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-small",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large"],
        help="Model size",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples")
    parser.add_argument(
        "--enable_mixed_precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--enable_gradient_checkpoint",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    logger.info(f"Rank {rank}/{world_size} on {device}")

    # Create model
    logger.info(f"Creating model: {args.model_name}")
    config = GPTConfig.from_name(args.model_name)
    model = GPT(config)

    # Wrap with data parallelism
    model = DataParallel(module=model, sync_gradients=True)

    logger.info(
        f"Model parameters: {model.module.get_num_params() / 1e6:.1f}M"
    )

    # Create data
    data = create_dummy_data(num_examples=args.num_examples)

    # Create distributed data loader
    train_dataloader = DataLoaderFactory.create_dataloader(
        data=data,
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        tokenizer_name="gpt2",
        distributed=True,
        shuffle=True,
    )

    eval_dataloader = DataLoaderFactory.create_dataloader(
        data=data[:100],
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        tokenizer_name="gpt2",
        distributed=True,
        shuffle=False,
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )

    # Create scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = CosineAnnealingWarmup(
        optimizer=optimizer,
        warmup_steps=len(train_dataloader),
        total_steps=total_steps,
        peak_lr=args.learning_rate,
        min_lr=1e-5,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        enable_mixed_precision=args.enable_mixed_precision,
        enable_gradient_checkpoint=args.enable_gradient_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    logger.info("Starting distributed training...")
    results = trainer.train()

    # Print results
    logger.info("Training complete!")
    logger.info(f"Results: {results}")

    # Save final model (only rank 0)
    if rank == 0:
        save_path = os.path.join(args.checkpoint_dir, "final_model.pt")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(model.module.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
