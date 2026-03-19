#!/usr/bin/env python3
"""
Single GPU Training Script

Simple example showing how to train a GPT model on a single GPU.
No distributed setup required.

Usage:
    python scripts/train_single_gpu.py \\
        --model_name gpt2-small \\
        --batch_size 32 \\
        --num_epochs 3
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinytrain.model.gpt import GPT
from tinytrain.model.config import GPTConfig
from tinytrain.training.trainer import Trainer
from tinytrain.training.optimizer import AdamW
from tinytrain.training.scheduler import CosineAnnealingWarmup
from tinytrain.training.data_loader import DataLoaderFactory, create_dummy_data
from tinytrain.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GPT on single GPU")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-small",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large"],
        help="Model size",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating model: {args.model_name}")
    config = GPTConfig.from_name(args.model_name)
    model = GPT(config)
    logger.info(
        f"Model parameters: {model.get_num_params() / 1e6:.1f}M"
    )

    # Create data
    logger.info(f"Creating dataset with {args.num_examples} examples")
    data = create_dummy_data(num_examples=args.num_examples)

    # Create data loader
    train_dataloader = DataLoaderFactory.create_dataloader(
        data=data,
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        tokenizer_name="gpt2",
        distributed=False,
        shuffle=True,
    )

    eval_dataloader = DataLoaderFactory.create_dataloader(
        data=data[:100],  # Use subset for validation
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        tokenizer_name="gpt2",
        distributed=False,
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
    logger.info("Starting training...")
    results = trainer.train()

    # Print results
    logger.info("Training complete!")
    logger.info(f"Results: {results}")

    # Save final model
    save_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
