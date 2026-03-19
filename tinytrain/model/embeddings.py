"""
Token and Positional Embeddings

Implements token embeddings (vocabulary lookup) and learned positional embeddings.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class Embeddings(nn.Module):
    """
    Combined token and positional embeddings.

    Applies learnable positional embeddings to token embeddings. Uses the
    standard Transformer initialization.

    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length for positional embeddings
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learnable)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights."""
        # Token embeddings: normal distribution
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        # Positional embeddings: normal distribution
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token indices with positional information.

        Args:
            input_ids: Token indices, shape (batch_size, seq_len)

        Returns:
            Embeddings, shape (batch_size, seq_len, d_model)
        """
        seq_len = input_ids.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Create position indices
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        # Get token and position embeddings
        token_embs = self.token_emb(input_ids)
        pos_embs = self.pos_emb(positions)

        # Combine
        embeddings = token_embs + pos_embs

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings
