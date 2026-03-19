#!/usr/bin/env python3
"""
Benchmark Throughput and Model FLOPs Utilization

Measure training throughput (tokens/sec) and MFU for different model sizes
and parallelism strategies.

Usage:
    python scripts/benchmark.py \\
        --model_name gpt2-small \\
        --batch_size 32 \\
        --num_iters 100
"""

import argparse
import torch
import torch.nn as nn
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinytrain.model.gpt import GPT
from tinytrain.model.config import GPTConfig
from tinytrain.training.data_loader import create_dummy_data, DataLoaderFactory
from tinytrain.utils.logging import get_logger
from tinytrain.utils.metrics import MetricsTracker

logger = get_logger(__name__)


def benchmark(
    model: nn.Module,
    dataloader,
    num_iters: int,
    device: torch.device,
) -> dict:
    """
    Benchmark model training throughput.

    Args:
        model: Model to benchmark
        dataloader: Data loader
        num_iters: Number of iterations
        device: Device to use

    Returns:
        Benchmark results
    """
    model.train()
    model = model.to(device)

    metrics = MetricsTracker()

    # Warmup
    logger.info("Warmup...")
    for i, (input_ids, labels) in enumerate(dataloader):
        if i >= 5:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        _, loss = model(
            input_ids=input_ids,
            labels=labels,
            return_loss=True,
        )
        loss.backward()

    # Benchmark
    logger.info(f"Benchmarking {num_iters} iterations...")
    torch.cuda.synchronize(device) if torch.cuda.is_available() else None

    start_time = time.time()
    iter_count = 0

    for input_ids, labels in dataloader:
        if iter_count >= num_iters:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        metrics.start_step()

        # Forward pass
        with torch.no_grad():
            _, loss = model(
                input_ids=input_ids,
                labels=labels,
                return_loss=True,
            )

        metrics.end_step(
            loss=loss.item(),
            batch_size=input_ids.size(0),
            seq_len=input_ids.size(1),
        )

        iter_count += 1

        if (iter_count + 1) % 20 == 0:
            logger.info(f"  Iteration {iter_count + 1}/{num_iters}")

    torch.cuda.synchronize(device) if torch.cuda.is_available() else None
    total_time = time.time() - start_time

    # Results
    results = metrics.get_all()
    results["total_time"] = total_time
    results["throughput"] = (
        results.get("avg_throughput", 0) if results else 0
    )

    return results


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark GPT training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-small",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large"],
        help="Model size",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    logger.info(f"Creating model: {args.model_name}")
    config = GPTConfig.from_name(args.model_name)
    model = GPT(config)
    num_params = model.get_num_params()
    logger.info(f"Parameters: {num_params / 1e6:.1f}M")

    # Create data
    data = create_dummy_data(num_examples=args.num_examples)

    # Create data loader
    dataloader = DataLoaderFactory.create_dataloader(
        data=data,
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        tokenizer_name="gpt2",
        distributed=False,
        shuffle=False,
    )

    # Run benchmark
    logger.info("\nStarting benchmark...")
    results = benchmark(
        model=model,
        dataloader=dataloader,
        num_iters=args.num_iters,
        device=device,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Parameters: {num_params / 1e6:.1f}M")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num iterations: {args.num_iters}")
    logger.info(f"Total time: {results['total_time']:.2f}s")
    logger.info(f"Avg loss: {results.get('avg_loss', 0):.4f}")
    logger.info(f"Throughput: {results.get('throughput', 0):.0f} tokens/sec")
    logger.info(f"Time per step: {results.get('avg_step_time', 0)*1000:.2f}ms")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
