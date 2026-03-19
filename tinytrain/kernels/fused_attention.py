"""
Fused Attention Kernels

Memory-efficient attention computation avoiding full matrix materialization.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def fused_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Memory-efficient scaled dot-product attention.

    For very large sequences, avoids materializing the full N×N
    attention matrix by processing in chunks.

    Args:
        q: Query, shape (batch_size, n_heads, seq_len, head_dim)
        k: Key, shape (batch_size, n_heads, seq_len, head_dim)
        v: Value, shape (batch_size, n_heads, seq_len, head_dim)
        causal_mask: Optional causal mask
        dropout_p: Dropout probability

    Returns:
        Attention output, shape (batch_size, n_heads, seq_len, head_dim)
    """
    # For short sequences, just use standard attention
    seq_len = q.size(2)
    if seq_len < 1024:  # Threshold for when chunking helps
        # Use PyTorch 2.0+ flash attention if available
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, dropout_p=dropout_p
            )
        else:
            # Fall back to standard attention
            return _standard_attention(q, k, v, causal_mask, dropout_p)

    # For long sequences, use chunked computation
    return _chunked_attention(q, k, v, causal_mask, dropout_p)


def _standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    output = torch.matmul(attn_weights, v)
    return output


def _chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Chunked attention to reduce memory usage.

    Process queries in chunks, computing attention with all keys/values
    for each chunk to maintain correctness while reducing peak memory.
    """
    batch_size, n_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype

    scale = head_dim ** -0.5
    output = torch.zeros_like(q)

    # Process queries in chunks
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        q_chunk = q[:, :, i:end, :]

        # Compute attention for this chunk
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

        # Apply causal mask if present
        if causal_mask is not None:
            # Only attend to positions up to current position for language modeling
            mask_chunk = torch.ones(
                (end - i, seq_len),
                device=device,
                dtype=torch.bool,
            )
            mask_chunk = torch.tril(mask_chunk, diagonal=i)
            scores = scores.masked_fill(~mask_chunk.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

        # Apply to values
        output[:, :, i:end, :] = torch.matmul(attn_weights, v)

    return output
