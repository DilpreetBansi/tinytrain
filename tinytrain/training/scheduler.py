"""
Learning Rate Schedulers

Implements cosine annealing with linear warmup for stable training.
"""

import math
from typing import Callable, Optional


class CosineAnnealingWarmup:
    """
    Cosine annealing schedule with linear warmup.

    Learning rate starts at 0, linearly increases to peak_lr over warmup_steps,
    then decreases following a cosine curve to min_lr over total_steps.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        peak_lr: Peak learning rate (after warmup)
        min_lr: Minimum learning rate (at end of training)
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> float:
        """
        Update learning rate and return current LR.

        Returns:
            Current learning rate
        """
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1
        return lr

    def _get_lr(self) -> float:
        """Compute current learning rate based on schedule."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def get_last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self._get_lr()
