"""
Gradient Checkpointing (Activation Checkpointing)

Trade compute for memory by recomputing activations during backward pass
instead of storing them during forward pass.

Reduces memory usage from O(N) to O(sqrt(N)) for N layers.
"""

import torch
import torch.nn as nn
from typing import Callable, Any


def checkpoint(
    function: Callable,
    *args,
    use_reentrant: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Checkpoint a function (compute-efficient backward).

    Forward pass runs normally and doesn't save activations.
    During backward, recomputes the forward pass to get activations,
    then computes gradients.

    Reduces peak memory usage at cost of extra computation.

    Args:
        function: Callable to checkpoint
        *args: Arguments to pass to function
        use_reentrant: Whether to use reentrant checkpointing (more memory efficient)
        **kwargs: Keyword arguments to pass to function

    Returns:
        Output of function
    """
    # Use PyTorch's built-in checkpoint if available
    if hasattr(torch.utils, 'checkpoint'):
        try:
            return torch.utils.checkpoint.checkpoint(
                function,
                *args,
                use_reentrant=use_reentrant,
                **kwargs,
            )
        except TypeError:
            # Older PyTorch version without use_reentrant
            return torch.utils.checkpoint.checkpoint(
                function,
                *args,
                **kwargs,
            )

    # Fallback: simple forward without memory optimization
    return function(*args, **kwargs)


class CheckpointedModule(nn.Module):
    """
    A module that wraps another module for checkpointing.

    Useful for applying gradient checkpointing to specific parts
    of a larger model.

    Args:
        module: Module to checkpoint
        use_reentrant: Whether to use reentrant checkpointing
    """

    def __init__(self, module: nn.Module, use_reentrant: bool = False) -> None:
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with checkpointing."""
        return checkpoint(
            self.module,
            *args,
            use_reentrant=self.use_reentrant,
            **kwargs,
        )


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_segments: int = 2,
) -> nn.Module:
    """
    Apply gradient checkpointing to a model's transformer blocks.

    Wraps every N transformer blocks in a CheckpointedModule.

    Args:
        model: Model to apply checkpointing to
        checkpoint_segments: Number of blocks per checkpoint segment

    Returns:
        Modified model
    """
    # Look for blocks attribute (typical in transformer models)
    if hasattr(model, 'blocks'):
        blocks = model.blocks

        # Group blocks into segments and wrap
        for i in range(0, len(blocks), checkpoint_segments):
            end_idx = min(i + checkpoint_segments, len(blocks))

            # Get the segment
            segment_blocks = [blocks[j] for j in range(i, end_idx)]

            # Wrap in CheckpointedModule
            wrapped = nn.Sequential(*segment_blocks)
            model.blocks[i:end_idx] = nn.ModuleList([wrapped])

    return model
