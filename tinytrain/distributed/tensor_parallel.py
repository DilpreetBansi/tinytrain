"""
Tensor Parallelism

Shard weight matrices across GPUs for very large models.
Supports column-parallel and row-parallel linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tinytrain.distributed.comm import allreduce, is_distributed


class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer.

    Shards the weight matrix by output dimension (columns).
    Each GPU computes a subset of output features.

    If input is replicated across GPUs:
    - Forward: Each GPU computes subset of outputs
    - Backward: Gradients are synchronized via all-reduce

    Args:
        in_features: Input dimension
        out_features: Output dimension (will be sharded across world_size)
        bias: Whether to use bias
        world_size: Number of shards
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int = 1,
    ) -> None:
        super().__init__()

        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by world_size ({world_size})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.out_features_local = out_features // world_size

        # Shard weight matrix
        self.weight = nn.Parameter(
            torch.empty(self.out_features_local, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_local))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., in_features)

        Returns:
            Output tensor, shape (..., out_features_local)
        """
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer.

    Shards the weight matrix by input dimension (rows).
    Each GPU computes outputs for subset of inputs.

    If output needs to be replicated:
    - Forward: All-reduce input projections
    - Backward: Gradient scatter

    Args:
        in_features: Input dimension (will be sharded across world_size)
        out_features: Output dimension
        bias: Whether to use bias
        world_size: Number of shards
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int = 1,
    ) -> None:
        super().__init__()

        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by world_size ({world_size})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.in_features_local = in_features // world_size

        # Shard weight matrix
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_local)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., in_features_local)

        Returns:
            Output tensor, shape (..., out_features)
        """
        # Compute local output
        output = F.linear(x, self.weight, None)

        # All-reduce to combine outputs from all ranks
        if self.world_size > 1 and is_distributed():
            allreduce(output, op="sum")

        # Add bias on last GPU
        if self.bias is not None:
            output = output + self.bias

        return output
