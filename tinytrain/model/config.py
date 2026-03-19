"""
GPT Model Configuration

Defines hyperparameters and provides preset configurations for different model sizes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """
    GPT model configuration.

    Attributes:
        vocab_size: Size of the vocabulary (default 50257 for GPT-2 tokenizer)
        max_seq_len: Maximum sequence length (context window)
        d_model: Hidden dimension / embedding dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward inner layer
        dropout: Dropout probability
        activation: Activation function name ("gelu", "relu", etc.)
        weight_init_std: Standard deviation for weight initialization
        bias: Whether to use bias in linear layers
        use_cache: Whether to cache KV during inference
    """

    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    weight_init_std: float = 0.02
    bias: bool = True
    use_cache: bool = False

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

        if self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("n_heads and n_layers must be positive")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """
        GPT-2 Small configuration (124M parameters).

        Returns:
            GPTConfig: Configuration matching GPT-2 small
        """
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            dropout=0.1,
        )

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """
        GPT-2 Medium configuration (355M parameters).

        Returns:
            GPTConfig: Configuration matching GPT-2 medium
        """
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            d_ff=4096,
            dropout=0.1,
        )

    @classmethod
    def gpt2_large(cls) -> "GPTConfig":
        """
        GPT-2 Large configuration (774M parameters).

        Returns:
            GPTConfig: Configuration matching GPT-2 large
        """
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=1280,
            n_layers=36,
            n_heads=20,
            d_ff=5120,
            dropout=0.1,
        )

    @classmethod
    def from_name(cls, name: str) -> "GPTConfig":
        """
        Create configuration from model name.

        Args:
            name: Model name ("gpt2-small", "gpt2-medium", "gpt2-large")

        Returns:
            GPTConfig: Configuration for the specified model

        Raises:
            ValueError: If model name is not recognized
        """
        configs = {
            "gpt2-small": cls.gpt2_small,
            "gpt2-medium": cls.gpt2_medium,
            "gpt2-large": cls.gpt2_large,
        }

        if name not in configs:
            raise ValueError(
                f"Unknown model name: {name}. "
                f"Available: {list(configs.keys())}"
            )

        return configs[name]()

    def num_parameters(self) -> int:
        """
        Estimate total number of parameters.

        Returns:
            int: Approximate number of trainable parameters
        """
        # Token embeddings: vocab_size * d_model
        token_emb = self.vocab_size * self.d_model

        # Position embeddings: max_seq_len * d_model
        pos_emb = self.max_seq_len * self.d_model

        # Per transformer block:
        # - Self-attention: 3*d_model^2 + 3*d_model (Q,K,V projections + output proj)
        # - MLP: d_model*d_ff + d_ff + d_ff*d_model + d_model
        # - LayerNorms: 4*d_model (2 per block)
        per_block = (
            3 * self.d_model * self.d_model + 3 * self.d_model +  # attention
            self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model +  # MLP
            4 * self.d_model  # LayerNorms
        )
        transformer_blocks = self.n_layers * per_block

        # Output LayerNorm: d_model
        output_norm = self.d_model

        # LM head (weight-tied with embeddings, typically no separate params)

        total = token_emb + pos_emb + transformer_blocks + output_norm

        return int(total)
