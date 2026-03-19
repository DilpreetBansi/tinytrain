"""
Training Configuration

Dataclass for all training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """
    Complete training configuration.

    Contains all hyperparameters for model, data, training, and distributed setup.
    """

    # Model
    model_name: str = "gpt2-small"
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1

    # Training
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: Optional[int] = None

    # Data
    data_path: Optional[str] = None
    num_workers: int = 0

    # Distributed
    distributed: bool = False
    num_nodes: int = 1
    gpus_per_node: int = 1
    strategy: str = "data_parallel"  # or "tensor_parallel", "pipeline_parallel"

    # Mixed Precision
    enable_mixed_precision: bool = False
    mixed_precision_dtype: str = "float16"  # or "bfloat16"

    # Gradient Checkpointing
    enable_gradient_checkpoint: bool = False

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1000
    load_checkpoint: Optional[str] = None

    # Logging
    log_interval: int = 100
    eval_interval: int = 500

    # Device
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
