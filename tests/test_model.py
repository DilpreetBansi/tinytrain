"""Tests for GPT model."""

import torch
import pytest
from tinytrain.model.config import GPTConfig
from tinytrain.model.gpt import GPT


class TestGPTConfig:
    """Test GPT configuration."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = GPTConfig()
        assert config.vocab_size == 50257
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12

    def test_config_gpt2_small(self):
        """Test GPT-2 small configuration."""
        config = GPTConfig.gpt2_small()
        assert config.d_model == 768
        assert config.n_layers == 12

    def test_config_gpt2_medium(self):
        """Test GPT-2 medium configuration."""
        config = GPTConfig.gpt2_medium()
        assert config.d_model == 1024
        assert config.n_layers == 24

    def test_config_gpt2_large(self):
        """Test GPT-2 large configuration."""
        config = GPTConfig.gpt2_large()
        assert config.d_model == 1280
        assert config.n_layers == 36

    def test_config_from_name(self):
        """Test loading config by name."""
        config = GPTConfig.from_name("gpt2-small")
        assert config.d_model == 768

        with pytest.raises(ValueError):
            GPTConfig.from_name("invalid-model")

    def test_config_head_dim(self):
        """Test head dimension calculation."""
        config = GPTConfig(d_model=768, n_heads=12)
        assert config.head_dim == 64

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            GPTConfig(d_model=768, n_heads=13)  # Not divisible

    def test_num_parameters(self):
        """Test parameter counting."""
        config = GPTConfig.gpt2_small()
        num_params = config.num_parameters()
        # GPT-2 small has ~124M parameters
        assert 100e6 < num_params < 150e6


class TestGPT:
    """Test GPT model."""

    def test_gpt_creation(self):
        """Test GPT model creation."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=512,
        )
        model = GPT(config)
        assert model.config == config

    def test_gpt_forward(self):
        """Test forward pass."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=512,
        )
        model = GPT(config)

        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(input_ids, return_loss=False)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is None

    def test_gpt_with_labels(self):
        """Test forward pass with labels."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=512,
        )
        model = GPT(config)

        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(input_ids, labels, return_loss=True)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is not None
        assert loss.item() > 0

    def test_gpt_generation(self):
        """Test generation."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=512,
        )
        model = GPT(config)

        batch_size = 2
        prompt_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, prompt_len))

        generated = model.generate(input_ids, max_new_tokens=20)

        assert generated.shape == (batch_size, prompt_len + 20)

    def test_gpt_num_params(self):
        """Test parameter counting."""
        config = GPTConfig.gpt2_small()
        model = GPT(config)

        num_params = model.get_num_params()
        assert num_params > 0

        num_trainable = model.get_num_params(only_trainable=True)
        assert num_trainable == num_params

    def test_gpt_device_movement(self):
        """Test moving model to different devices."""
        config = GPTConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
        )
        model = GPT(config)

        # Move to CPU (should always work)
        model = model.cpu()
        assert next(model.parameters()).device.type == "cpu"

        # Test forward pass on CPU
        input_ids = torch.randint(0, 100, (2, 32))
        logits, _ = model(input_ids)
        assert logits.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
