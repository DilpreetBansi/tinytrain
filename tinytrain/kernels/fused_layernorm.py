"""
Fused Layer Normalization

Combines mean/var computation, normalization, and scaling in fewer kernel launches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLayerNorm(nn.Module):
    """
    Fused LayerNorm that reduces memory bandwidth.

    Combines mean/var computation, normalization, and scaling/bias
    in a single pass where possible.

    Args:
        normalized_shape: Shape of normalized dimensions
        eps: Small value for numerical stability
        elementwise_affine: Whether to learn scale and bias parameters
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Normalized output
        """
        # Use PyTorch's built-in LayerNorm (already quite optimized)
        # In production, would use Triton kernel for true fusion
        return F.layer_norm(
            x,
            normalized_shape=(self.normalized_shape,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
