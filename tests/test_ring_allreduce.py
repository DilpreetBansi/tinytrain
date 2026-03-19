"""Tests for ring all-reduce communication."""

import torch
import pytest
from tinytrain.distributed.ring_allreduce import ring_allreduce


class TestRingAllreduce:
    """Test ring all-reduce implementation."""

    def test_ring_allreduce_single_rank(self):
        """Test ring all-reduce with single rank (no-op)."""
        tensors = [torch.ones(2, 2) for _ in range(3)]
        original = [t.clone() for t in tensors]

        # Single rank should not change values
        result = ring_allreduce(tensors, world_size=1, rank=0)

        for res, orig in zip(result, original):
            assert torch.allclose(res, orig)

    def test_ring_allreduce_shapes(self):
        """Test that shapes are preserved."""
        shapes = [(2, 3), (4, 5), (1, 10)]
        tensors = [torch.ones(shape) for shape in shapes]

        result = ring_allreduce(tensors, world_size=1, rank=0)

        for res, orig_shape in zip(result, shapes):
            assert res.shape == orig_shape

    def test_ring_allreduce_device(self):
        """Test that tensors stay on same device."""
        device = torch.device("cpu")
        tensors = [torch.ones(2, 2, device=device) for _ in range(2)]

        result = ring_allreduce(tensors, world_size=1, rank=0)

        for res in result:
            assert res.device == device

    def test_ring_allreduce_dtype_preservation(self):
        """Test that dtype is preserved."""
        dtypes = [torch.float32, torch.float64]
        tensors = [torch.ones(2, 2, dtype=dtype) for dtype in dtypes]

        result = ring_allreduce(tensors, world_size=1, rank=0)

        for res, dtype in zip(result, dtypes):
            assert res.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__])
