#!/usr/bin/env python3
"""
Pipeline Parallel Training Script

Train GPT using pipeline parallelism where model layers are split across GPUs.

Usage:
    torchrun --nproc_per_node=4 scripts/train_pipeline.py \\
        --model_name gpt2-small \\
        --num_pipeline_stages 2 \\
        --batch_size 16
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
from tinytrain.distributed.pipeline_parallel import PipelineStage, PipelineParallel
from tinytrain.training.optimizer import AdamW
from tinytrain.training.scheduler import CosineAnnealingWarmup
from tinytrain.training.data_loader import DataLoaderFactory, create_dummy_data
from tinytrain.utils.logging import get_logger

logger = get_logger(__name__)


def split_gpt_into_stages(
    model: GPT,
    num_stages: int,
) -> list:
    """
    Split GPT model into pipeline stages.

    Args:
        model: Full GPT model
        num_stages: Number of pipeline stages

    Returns:
        List of PipelineStage modules
    """
    if num_stages <= 1:
        return [PipelineStage([model], 0, 1)]

    stages = []
    layers_per_stage = (model.config.n_layers + num_stages - 1) // num_stages

    # First stage: embeddings + initial blocks
    stage_layers = [model.embeddings]
    start_idx = 0
    end_idx = min(layers_per_stage, model.config.n_layers)
    stage_layers.extend(model.blocks[start_idx:end_idx])

    stages.append(
        PipelineStage(stage_layers, 0, num_stages)
    )

    # Middle stages
    for stage_id in range(1, num_stages - 1):
        stage_layers = []
        start_idx = stage_id * layers_per_stage
        end_idx = min(start_idx + layers_per_stage, model.config.n_layers)
        stage_layers.extend(model.blocks[start_idx:end_idx])
        stages.append(
            PipelineStage(stage_layers, stage_id, num_stages)
        )

    # Last stage: remaining blocks + normalization + head
    stage_layers = []
    start_idx = (num_stages - 1) * layers_per_stage
    stage_layers.extend(model.blocks[start_idx:])
    stage_layers.extend([model.norm, model.lm_head])
    stages.append(
        PipelineStage(stage_layers, num_stages - 1, num_stages)
    )

    return stages


def main():
    """Main training function."""
    # Initialize distributed training
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    parser = argparse.ArgumentParser(description="Pipeline parallel GPT training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-small",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large"],
        help="Model size",
    )
    parser.add_argument(
        "--num_pipeline_stages",
        type=int,
        default=2,
        help="Number of pipeline stages",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples")
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
    logger.info(
        f"Model parameters: {model.get_num_params() / 1e6:.1f}M"
    )

    # Split into pipeline stages (simplified version)
    # In production, would distribute stages across ranks
    stages = split_gpt_into_stages(model, args.num_pipeline_stages)
    logger.info(f"Created {len(stages)} pipeline stages")

    # Create pipeline model
    pipeline_model = PipelineParallel(stages, num_microbatches=4)
    pipeline_model = pipeline_model.to(device)

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

    # Create optimizer (optimize original model parameters)
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

    # Training loop
    logger.info("Starting pipeline parallel training...")

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass through pipeline
            logits, loss = pipeline_model(input_ids, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0 and rank == 0:
                logger.info(
                    f"  Batch {batch_idx + 1} - Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

    # Save final model (only rank 0)
    if rank == 0:
        save_path = os.path.join(args.checkpoint_dir, "final_model_pipeline.pt")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
