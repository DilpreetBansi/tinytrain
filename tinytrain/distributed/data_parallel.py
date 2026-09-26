"""
Data parallelism: every rank holds the whole model and a slice of the batch.

After backward, gradients are averaged across ranks in buckets of about `bucket_mb` megabytes
(fewer, larger messages), using either the hand-written ring all-reduce or the backend's
all-reduce. Parameters are broadcast from rank 0 at construction so all replicas start equal.
"""

from contextlib import contextmanager
from typing import Iterator, List

import torch
import torch.distributed as dist
import torch.nn as nn

from tinytrain.distributed.comm import get_world_size, is_distributed
from tinytrain.distributed.ring_allreduce import ring_allreduce


class DataParallel(nn.Module):
    def __init__(self, module: nn.Module, bucket_mb: float = 25.0, use_ring: bool = True) -> None:
        super().__init__()
        self.module = module
        self.bucket_bytes = int(bucket_mb * 2 ** 20)
        self.use_ring = use_ring
        self.world_size = get_world_size() if is_distributed() else 1
        self._sync = True
        if self.world_size > 1:
            with torch.no_grad():
                for p in self.module.state_dict().values():
                    dist.broadcast(p, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Skip gradient sync (for gradient accumulation); sync on the last micro-step."""
        self._sync = False
        try:
            yield
        finally:
            self._sync = True

    def _buckets(self) -> List[List[torch.Tensor]]:
        buckets, current, size = [], [], 0
        # Reverse order: the last layers' gradients are ready first in a real overlap schedule.
        for p in reversed([p for p in self.module.parameters() if p.grad is not None]):
            current.append(p.grad)
            size += p.grad.numel() * p.grad.element_size()
            if size >= self.bucket_bytes:
                buckets.append(current)
                current, size = [], 0
        if current:
            buckets.append(current)
        return buckets

    @torch.no_grad()
    def synchronize_gradients(self) -> None:
        if not self._sync or self.world_size == 1:
            return
        for bucket in self._buckets():
            flat = torch.cat([g.reshape(-1) for g in bucket])
            if self.use_ring:
                ring_allreduce(flat, op="avg")
            else:
                dist.all_reduce(flat)
                flat /= self.world_size
            offset = 0
            for g in bucket:
                g.copy_(flat[offset:offset + g.numel()].view_as(g))
                offset += g.numel()
