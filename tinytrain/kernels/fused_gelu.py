"""
Fused GELU Activation

Combines GELU computation in a single operation.
"""

import torch
import torch.nn as nn
import math


class FusedGELU(nn.Module):
    """
    Fused GELU activation function.

    Computes: x * Phi(x) where Phi is the cumulative distribution
    function of the standard normal distribution.

    Two approximations available:
    - exact: Uses error function (slower but accurate)
    - tanh: Uses tanh approximation (faster, very close to exact)

    Args:
        approximate: Whether to use approximation ("tanh" or "exact")
    """

    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        Args:
            x: Input tensor

        Returns:
            GELU output
        """
        if self.approximate == "tanh":
            return self._gelu_tanh(x)
        else:
            return self._gelu_exact(x)

    @staticmethod
    def _gelu_exact(x: torch.Tensor) -> torch.Tensor:
        """Exact GELU using error function."""
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
        """Faster GELU using tanh approximation."""
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
