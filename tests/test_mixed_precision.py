"""Tests for mixed precision training."""

import torch
import pytest
from tinytrain.training.mixed_precision import GradScaler


class TestGradScaler:
    """Test gradient scaling for mixed precision."""

    def test_grad_scaler_creation(self):
        """Test GradScaler creation."""
        scaler = GradScaler(init_scale=65536.0)
        assert scaler.scale == 65536.0
        assert scaler.enabled is True

    def test_grad_scaler_disabled(self):
        """Test disabled GradScaler."""
        scaler = GradScaler(enabled=False)

        loss = torch.tensor(1.0)
        scaled_loss = scaler.scale_loss(loss)
        assert torch.allclose(scaled_loss, loss)

    def test_scale_loss(self):
        """Test loss scaling."""
        scaler = GradScaler(init_scale=1024.0)

        loss = torch.tensor(1.0)
        scaled_loss = scaler.scale_loss(loss)

        assert torch.allclose(scaled_loss, torch.tensor(1024.0))

    def test_unscale_grads(self):
        """Test gradient unscaling."""
        scaler = GradScaler(init_scale=1024.0)

        # Create simple optimizer
        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = torch.optim.SGD([param], lr=0.1)

        # Create gradient
        param.grad = torch.randn(2, 2)
        original_grad = param.grad.clone()

        # Unscale
        scaler.unscale_grads(optimizer)

        # Check that gradient was divided by scale
        expected = original_grad / 1024.0
        assert torch.allclose(param.grad, expected)

    def test_has_overflow(self):
        """Test overflow detection."""
        scaler = GradScaler()

        # Normal tensor
        normal = torch.randn(2, 2)
        assert not scaler.has_overflow(normal)

        # Tensor with NaN
        with_nan = torch.randn(2, 2)
        with_nan[0, 0] = float('nan')
        assert scaler.has_overflow(with_nan)

        # Tensor with Inf
        with_inf = torch.randn(2, 2)
        with_inf[0, 0] = float('inf')
        assert scaler.has_overflow(with_inf)

    def test_step_growth(self):
        """Test loss scale growth."""
        scaler = GradScaler(
            init_scale=1024.0,
            growth_factor=2.0,
            growth_interval=2,
        )

        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = torch.optim.SGD([param], lr=0.1)

        # First step (no overflow)
        scaler.step(optimizer, overflow=False)
        assert scaler.scale == 1024.0  # Not grown yet

        # Second step
        scaler.step(optimizer, overflow=False)
        assert scaler.scale == 2048.0  # Should grow

    def test_step_backoff(self):
        """Test loss scale backoff on overflow."""
        scaler = GradScaler(
            init_scale=1024.0,
            backoff_factor=0.5,
        )

        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = torch.optim.SGD([param], lr=0.1)

        # Overflow
        scaler.step(optimizer, overflow=True)
        assert scaler.scale == 512.0  # Should backoff

    def test_state_dict(self):
        """Test checkpoint state saving/loading."""
        scaler = GradScaler(init_scale=1024.0)
        scaler.scale = 2048.0
        scaler._growth_steps = 5

        state = scaler.state_dict()
        assert state['scale'] == 2048.0
        assert state['growth_steps'] == 5

        # Create new scaler and load state
        scaler2 = GradScaler()
        scaler2.load_state_dict(state)
        assert scaler2.scale == 2048.0
        assert scaler2._growth_steps == 5


if __name__ == "__main__":
    pytest.main([__file__])
