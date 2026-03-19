"""
Multi-Head Self-Attention Mechanism

Implements scaled dot-product attention with causal masking for language modeling.
Supports both standard and memory-efficient attention paths.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    Implements scaled dot-product attention:
    Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V

    For language modeling, applies causal mask so each position can only
    attend to past positions (including itself).

    Args:
        d_model: Total embedding dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        use_memory_efficient_attention: Whether to use memory-efficient attention
        use_flash_attention: Whether to use Flash Attention (requires compatible PyTorch)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_memory_efficient_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.use_flash_attention = use_flash_attention

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # KV cache for inference (optional)
        self.kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            causal_mask: Optional pre-computed causal mask
            use_cache: Whether to cache KV for next inference step

        Returns:
            Tuple of:
            - attn_output: Attention output, shape (batch_size, seq_len, d_model)
            - kv_cache: Cached (K, V) if use_cache=True, else None
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Cache KV if requested
        new_kv_cache = None
        if use_cache:
            new_kv_cache = (k, v)

        # Compute attention
        if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use Flash Attention if available (PyTorch 2.0+)
            attn_output = self._scaled_dot_product_attention_flash(
                q, k, v, causal_mask
            )
        elif self.use_memory_efficient_attention:
            # Use memory-efficient chunked attention
            attn_output = self._scaled_dot_product_attention_efficient(
                q, k, v, causal_mask
            )
        else:
            # Standard attention
            attn_output = self._scaled_dot_product_attention_standard(
                q, k, v, causal_mask
            )

        # Merge heads back
        # (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output, new_kv_cache

    def _scaled_dot_product_attention_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention.

        Args:
            q: Query, shape (batch_size, n_heads, seq_len, head_dim)
            k: Key, shape (batch_size, n_heads, seq_len, head_dim)
            v: Value, shape (batch_size, n_heads, seq_len, head_dim)
            causal_mask: Optional causal mask

        Returns:
            Attention output, shape (batch_size, n_heads, seq_len, head_dim)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if not provided
        if causal_mask is None:
            seq_len = scores.size(-1)
            causal_mask = self._get_causal_mask(seq_len, scores.device)

        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply to values
        attn_output = torch.matmul(attn_weights, v)

        return attn_output

    def _scaled_dot_product_attention_efficient(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient attention via chunked computation.

        Computes attention in chunks to avoid materializing the full
        attention matrix, reducing peak memory usage.

        Args:
            q: Query, shape (batch_size, n_heads, seq_len, head_dim)
            k: Key, shape (batch_size, n_heads, seq_len, head_dim)
            v: Value, shape (batch_size, n_heads, seq_len, head_dim)
            causal_mask: Optional causal mask

        Returns:
            Attention output, shape (batch_size, n_heads, seq_len, head_dim)
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        chunk_size = max(1, seq_len // 4)  # Process in 4 chunks

        output = torch.zeros_like(q)
        normalizer = torch.zeros(
            batch_size, n_heads, seq_len, 1, device=q.device, dtype=q.dtype
        )

        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end, :]

            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale

            # Apply causal mask
            if causal_mask is None:
                causal_mask = self._get_causal_mask(seq_len, scores.device)

            # Extract relevant portion of mask
            mask_chunk = causal_mask[i:end, :seq_len]
            scores = scores.masked_fill(mask_chunk == 0, float("-inf"))

            # Softmax with numerical stability
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Apply to values
            attn_output_chunk = torch.matmul(attn_weights, v)
            normalizer_chunk = attn_weights.sum(dim=-1, keepdim=True)

            output[:, :, i:end, :] = attn_output_chunk
            normalizer[:, :, i:end, :] = normalizer_chunk

        return output

    def _scaled_dot_product_attention_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flash Attention (PyTorch 2.0+ optimized kernel).

        Args:
            q: Query, shape (batch_size, n_heads, seq_len, head_dim)
            k: Key, shape (batch_size, n_heads, seq_len, head_dim)
            v: Value, shape (batch_size, n_heads, seq_len, head_dim)
            causal_mask: Optional causal mask (converted to attn_mask)

        Returns:
            Attention output, shape (batch_size, n_heads, seq_len, head_dim)
        """
        # Convert causal mask to PyTorch format if available
        attn_mask = None
        if causal_mask is None:
            seq_len = q.size(-2)
            attn_mask = self._get_causal_mask(seq_len, q.device)

        # Use scaled_dot_product_attention (PyTorch 2.0+)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.1 if self.training else 0.0
        )

        return attn_output

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask (lower triangular matrix).

        Mask is 1 where attention is allowed, 0 where not allowed.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask, shape (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
