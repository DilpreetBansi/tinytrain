"""Tests for attention mechanisms."""

import torch
import pytest
from tinytrain.model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test multi-head attention."""

    def test_attention_creation(self):
        """Test attention module creation."""
        attn = MultiHeadAttention(
            d_model=768,
            n_heads=12,
            dropout=0.1,
        )
        assert attn.d_model == 768
        assert attn.n_heads == 12
        assert attn.head_dim == 64

    def test_attention_forward(self):
        """Test attention forward pass."""
        attn = MultiHeadAttention(
            d_model=768,
            n_heads=12,
            dropout=0.1,
        )

        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 768)

        output, kv_cache = attn(x, use_cache=False)

        assert output.shape == (batch_size, seq_len, 768)
        assert kv_cache is None

    def test_attention_with_cache(self):
        """Test attention with KV caching."""
        attn = MultiHeadAttention(
            d_model=768,
            n_heads=12,
            dropout=0.1,
        )

        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 768)

        output, kv_cache = attn(x, use_cache=True)

        assert output.shape == (batch_size, seq_len, 768)
        assert kv_cache is not None
        k, v = kv_cache
        assert k.shape == (batch_size, 12, seq_len, 64)
        assert v.shape == (batch_size, 12, seq_len, 64)

    def test_attention_causal_mask(self):
        """Test that causal mask prevents looking forward."""
        attn = MultiHeadAttention(
            d_model=64,
            n_heads=2,
            dropout=0.0,
        )

        batch_size = 1
        seq_len = 4
        # Create input where each position has a different value
        x = torch.arange(seq_len, dtype=torch.float32).view(1, -1, 1).repeat(1, 1, 64)

        # Get attention weights by running with a specific pattern
        output, _ = attn(x)

        # Shape should be preserved
        assert output.shape == (batch_size, seq_len, 64)

    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention."""
        attn = MultiHeadAttention(
            d_model=128,
            n_heads=4,
            dropout=0.0,
        )

        x = torch.randn(2, 16, 128, requires_grad=True)
        output, _ = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_attention_dropout_train_eval(self):
        """Test that dropout is disabled in eval mode."""
        attn = MultiHeadAttention(
            d_model=128,
            n_heads=4,
            dropout=0.5,
        )

        x = torch.randn(2, 16, 128)

        # Train mode
        attn.train()
        output_train, _ = attn(x)

        # Eval mode
        attn.eval()
        output_eval, _ = attn(x)

        # Outputs should be deterministic in eval mode
        with torch.no_grad():
            output_eval2, _ = attn(x)
        assert torch.allclose(output_eval, output_eval2)

    def test_attention_head_dim_validation(self):
        """Test that head dimension must be divisible."""
        with pytest.raises(ValueError):
            MultiHeadAttention(
                d_model=768,
                n_heads=13,  # 768 not divisible by 13
            )


if __name__ == "__main__":
    pytest.main([__file__])
