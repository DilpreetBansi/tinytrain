"""
Training Metrics Tracking

Tracks loss, throughput, Model FLOPs Utilization (MFU), etc.
"""

import time
from typing import Dict, Any, Optional
import math


class MetricsTracker:
    """
    Tracks training metrics across steps/epochs.

    Metrics tracked:
    - Loss (training and validation)
    - Throughput (tokens/sec)
    - Model FLOPs Utilization (MFU)
    - Time per step
    """

    def __init__(self) -> None:
        self.losses = []
        self.throughputs = []
        self.mfu_scores = []
        self.step_times = []

        self.start_time = None
        self.step_start_time = None

    def start_step(self) -> None:
        """Mark start of a training step."""
        self.step_start_time = time.time()

    def end_step(
        self,
        loss: float,
        batch_size: int = 1,
        seq_len: int = 1,
        num_tokens: Optional[int] = None,
    ) -> None:
        """
        Mark end of a training step and record metrics.

        Args:
            loss: Loss value
            batch_size: Batch size
            seq_len: Sequence length
            num_tokens: Total number of tokens in batch (auto-computed if None)
        """
        if self.step_start_time is None:
            return

        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        self.losses.append(loss)

        if num_tokens is None:
            num_tokens = batch_size * seq_len

        # Throughput (tokens/sec)
        throughput = num_tokens / step_time
        self.throughputs.append(throughput)

    def compute_mfu(
        self,
        batch_size: int,
        seq_len: int,
        num_params: int,
        num_layers: int,
        peak_flops: float = 312e12,  # A100 FP32 TFLOPs
    ) -> float:
        """
        Compute Model FLOPs Utilization.

        MFU = (Actual FLOPs) / (Theoretical Peak FLOPs)

        For forward + backward + optimizer:
        FLOPs ≈ 6 * seq_len * batch_size * hidden_dim * num_layers

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_params: Total model parameters
            num_layers: Number of transformer layers
            peak_flops: Theoretical peak FLOPs of device

        Returns:
            MFU as percentage (0-100)
        """
        # Rough estimate: 6 FLOPs per param per sequence position
        # (forward + backward + optimizer)
        actual_flops = 6.0 * batch_size * seq_len * num_params

        # Get average step time
        if not self.step_times:
            return 0.0

        avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])
        flops_per_sec = actual_flops / avg_step_time

        mfu = (flops_per_sec / peak_flops) * 100

        self.mfu_scores.append(mfu)
        return mfu

    def get_all(self) -> Dict[str, Any]:
        """Get all tracked metrics."""
        if not self.losses:
            return {}

        return {
            "avg_loss": sum(self.losses) / len(self.losses),
            "final_loss": self.losses[-1],
            "avg_throughput": sum(self.throughputs) / len(self.throughputs)
            if self.throughputs
            else 0.0,
            "avg_step_time": sum(self.step_times) / len(self.step_times)
            if self.step_times
            else 0.0,
            "avg_mfu": sum(self.mfu_scores) / len(self.mfu_scores)
            if self.mfu_scores
            else 0.0,
        }
