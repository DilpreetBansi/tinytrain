"""
GPT-2 Model Implementation from Scratch

Full transformer-based language model with token embeddings,
positional embeddings, transformer blocks, and language modeling head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tinytrain.model.config import GPTConfig
from tinytrain.model.embeddings import Embeddings
from tinytrain.model.transformer_block import TransformerBlock


class GPT(nn.Module):
    """
    GPT-2 Language Model.

    A transformer-based language model that predicts the next token given
    a sequence of tokens.

    Architecture:
    1. Token embeddings (learned)
    2. Positional embeddings (learned)
    3. N transformer blocks (self-attention + MLP)
    4. Layer normalization
    5. LM head (linear projection to vocabulary)

    The weight matrix of the LM head is tied with token embeddings
    for parameter efficiency.

    Args:
        config: GPTConfig object with hyperparameters
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config

        # Embeddings
        self.embeddings = Embeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ])

        # Output layer normalization
        self.norm = nn.LayerNorm(config.d_model)

        # Language modeling head (weight-tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_emb.weight

        # KV cache for efficient inference
        self.kv_cache: Optional[list] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (embeddings and head already initialized)."""
        # Note: embeddings are initialized in their own modules
        nn.init.normal_(self.norm.weight, mean=1.0, std=0.02)
        nn.init.zeros_(self.norm.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of GPT model.

        Args:
            input_ids: Token indices, shape (batch_size, seq_len)
            labels: Target token indices for loss computation (optional)
            use_cache: Whether to cache KV for inference
            return_loss: Whether to compute and return language modeling loss

        Returns:
            Tuple of:
            - logits: Unnormalized predictions, shape (batch_size, seq_len, vocab_size)
            - loss: Language modeling loss (if return_loss=True and labels provided)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.embeddings(input_ids)

        # Apply transformer blocks
        new_kv_cache = [] if use_cache else None

        for block in self.blocks:
            x, kv = block(x, use_cache=use_cache)
            if use_cache:
                new_kv_cache.append(kv)

        # Apply final layer norm
        x = self.norm(x)

        # Get logits
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if return_loss and labels is not None:
            # Flatten for cross-entropy: (batch_size * seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        # Update cache
        if use_cache:
            self.kv_cache = new_kv_cache

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token indices, shape (batch_size, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample only from top k tokens
            top_p: If set, sample only from tokens with cumulative prob >= top_p
            use_cache: Whether to use KV caching

        Returns:
            Generated token indices, shape (batch_size, prompt_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get model output
            logits, _ = self.forward(input_ids, use_cache=use_cache)

            # Only use last token's logits for next token prediction
            logits = logits[:, -1:, :]  # (batch_size, 1, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_logits[cumsum_probs > top_p] = float("-inf")
                logits = torch.scatter(
                    logits, -1, sorted_indices, sorted_logits
                )

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs[:, -1, :], num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, only_trainable: bool = False) -> int:
        """
        Count number of parameters.

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
