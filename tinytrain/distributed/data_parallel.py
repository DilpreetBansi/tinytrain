"""
Data Parallelism

Each GPU holds a complete model copy. Batch is split across GPUs.
Gradients are synchronized after backward pass.
"""

import torch
import torch.nn as nn
from typing import Optional
from tinytrain.distributed.comm import (
    is_distributed,
    allreduce,
    get_world_size,
    get_rank,
)


class DataParallel(nn.Module):
    """
    Data parallel wrapper for models.

    Synchronizes gradients across all ranks using all-reduce
    after the backward pass. Supports gradient accumulation.

    Args:
        module: The model to wrap
        sync_gradients: Whether to synchronize gradients across ranks
        bucket_size_mb: Gradient bucketing size for communication efficiency
    """

    def __init__(
        self,
        module: nn.Module,
        sync_gradients: bool = True,
        bucket_size_mb: int = 25,
    ) -> None:
        super().__init__()

        self.module = module
        self.sync_gradients = sync_gradients
        self.bucket_size_mb = bucket_size_mb
        self.world_size = get_world_size() if is_distributed() else 1

        # For gradient accumulation
        self.accumulated_grads = 0
        self.accumulation_steps = 1

    def forward(self, *args, **kwargs):
        """Forward pass (delegates to wrapped module)."""
        return self.module(*args, **kwargs)

    def synchronize_gradients(self) -> None:
        """
        Synchronize gradients across all ranks using all-reduce.

        This should be called after backward() to ensure all GPUs
        have the same gradient values.
        """
        if not self.sync_gradients or self.world_size == 1:
            return

        # Bucket gradients for communication efficiency
        self._allreduce_gradients()

    def _allreduce_gradients(self) -> None:
        """All-reduce gradients across all ranks."""
        world_size = self.world_size

        # Collect all gradients
        grads = []
        for param in self.module.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.flatten())

        if not grads:
            return

        # Concatenate all gradients
        all_grads = torch.cat(grads)

        # All-reduce
        allreduce(all_grads, op="sum")

        # Average
        all_grads.div_(world_size)

        # Distribute back to parameters
        offset = 0
        for param in self.module.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.data = all_grads[offset : offset + numel].view_as(
                    param.grad.data
                )
                offset += numel

    def set_gradient_accumulation_steps(self, steps: int) -> None:
        """
        Set number of accumulation steps.

        Gradients will only be synchronized after this many backward passes.

        Args:
            steps: Number of accumulation steps
        """
        self.accumulation_steps = steps

    def should_sync_gradients(self) -> bool:
        """
        Check if gradients should be synchronized now.

        Used for gradient accumulation: only sync every N backward passes.

        Returns:
            True if should synchronize
        """
        self.accumulated_grads += 1
        should_sync = self.accumulated_grads % self.accumulation_steps == 0

        if should_sync:
            self.accumulated_grads = 0

        return should_sync

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.module.zero_grad()

    def parameters(self):
        """Get module parameters."""
        return self.module.parameters()

    def state_dict(self):
        """Get module state dict."""
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load module state dict."""
        return self.module.load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        """Set train mode."""
        self.module.train(mode)
        return self

    def eval(self):
        """Set eval mode."""
        self.module.eval()
        return self
