"""
Mixed Precision Training

Automatic mixed precision (AMP) with dynamic loss scaling for stable FP16/BF16 training.
Reduces memory usage and increases training speed while maintaining accuracy.
"""

import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
from typing import Optional


class GradScaler:
    """
    Loss scaling for mixed precision training.

    Dynamically scales loss to prevent gradient underflow in FP16.
    If overflow detected (NaN/Inf gradients), reduces scale and skips update.
    Otherwise, gradually increases scale over time.

    Args:
        init_scale: Initial loss scale (2^16 = 65536 typical)
        growth_factor: Factor to increase scale by on successful steps
        backoff_factor: Factor to decrease scale by on overflow
        growth_interval: Number of successful steps before scaling up
        enabled: Whether to enable loss scaling
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled

        self._init_scale = init_scale
        self._growth_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision.

        Args:
            loss: Scalar loss tensor

        Returns:
            Scaled loss
        """
        if not self.enabled:
            return loss

        return loss * self.scale

    def unscale_grads(self, optimizer) -> None:
        """
        Unscale gradients.

        Args:
            optimizer: Optimizer with gradients to unscale
        """
        if not self.enabled:
            return

        # Divide gradients by scale
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)

    def step(self, optimizer, overflow: bool = False) -> None:
        """
        Update loss scale.

        Args:
            optimizer: Optimizer (for parameter access)
            overflow: Whether overflow was detected
        """
        if not self.enabled:
            return

        if overflow:
            # Reduce scale and skip update
            self.scale *= self.backoff_factor
            self._growth_steps = 0

            # Zero gradients to skip this update
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        param.grad.zero_()

        else:
            # Increment successful step counter
            self._growth_steps += 1

            # Increase scale periodically
            if self._growth_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_steps = 0

    def has_overflow(self, *args) -> bool:
        """
        Check if overflow occurred in gradients.

        Args:
            *args: Tensors to check for NaN/Inf

        Returns:
            True if any value is NaN or Inf
        """
        for tensor in args:
            if tensor is None:
                continue

            if isinstance(tensor, (list, tuple)):
                for t in tensor:
                    if self.has_overflow(t):
                        return True
            else:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    return True

        return False

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "scale": self.scale,
            "growth_steps": self._growth_steps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.scale = state_dict["scale"]
        self._growth_steps = state_dict.get("growth_steps", 0)
