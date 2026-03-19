"""
Transformer Block

A single transformer block consisting of:
1. Pre-norm layer normalization
2. Multi-head self-attention
3. Residual connection
4. Pre-norm layer normalization
5. Feed-forward network (MLP)
6. Residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tinytrain.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) component.

    Two linear layers with GELU activation in between.
    First layer expands to d_ff, second layer projects back to d_model.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension
        dropout: Dropout probability
        activation: Activation function name
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu_pytorch_tanh":
            # PyTorch GELU with tanh approximation (faster, closer to original)
            self.activation = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=0.02)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block (pre-norm architecture).

    Uses pre-normalization (applies LayerNorm before each sublayer),
    which improves training stability and allows for better scaling.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        activation: Activation function name
        bias: Whether to use bias
        use_memory_efficient_attention: Use memory-efficient attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
        use_memory_efficient_attention: bool = False,
    ) -> None:
        super().__init__()

        # Pre-norm layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            use_memory_efficient_attention=use_memory_efficient_attention,
        )

        # Feed-forward network
        self.mlp = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with pre-norm and residual connections.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            causal_mask: Optional causal attention mask
            use_cache: Whether to cache KV for inference

        Returns:
            Tuple of:
            - output: Output tensor, shape (batch_size, seq_len, d_model)
            - kv_cache: Cached (K, V) if use_cache=True
        """
        # Self-attention block (pre-norm + residual)
        attn_out, kv_cache = self.attn(
            self.norm1(x), causal_mask=causal_mask, use_cache=use_cache
        )
        x = x + attn_out

        # MLP block (pre-norm + residual)
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x, kv_cache
