"""
Ring all-reduce built from point-to-point send/recv.

The flat buffer is split into N equal chunks (padded). Two phases of N-1 steps each:

  reduce-scatter: at step s, rank r sends chunk (r - s) to r+1 and adds the chunk it receives,
                  (r - s - 1), from r-1. Afterwards rank r holds the full sum of chunk (r + 1).
  all-gather:     at step s, rank r sends chunk (r + 1 - s) to r+1 and overwrites chunk (r - s)
                  with what it receives from r-1. Afterwards every rank holds every summed chunk.

Each rank sends 2 (N-1)/N of the data in total, independent of N, which is why the ring is
bandwidth-optimal.
"""

from typing import List

import torch
import torch.distributed as dist

from tinytrain.distributed.comm import get_rank, get_world_size, is_distributed


def _exchange(send: torch.Tensor, recv: torch.Tensor, rank: int, world: int) -> None:
    ops = [dist.P2POp(dist.isend, send, (rank + 1) % world), dist.P2POp(dist.irecv, recv, (rank - 1) % world)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()


@torch.no_grad()
def ring_allreduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """In-place all-reduce (sum or avg) of one tensor over the default process group."""
    if op not in ("sum", "avg"):
        raise ValueError(f"unsupported op {op!r}")
    if not is_distributed() or get_world_size() == 1:
        return tensor
    world, rank = get_world_size(), get_rank()

    flat = tensor.detach().reshape(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    buf = torch.zeros(chunk * world, dtype=flat.dtype, device=flat.device)
    buf[:n] = flat
    chunks = list(buf.view(world, chunk))
    recv = torch.empty(chunk, dtype=flat.dtype, device=flat.device)

    for s in range(world - 1):                       # reduce-scatter
        _exchange(chunks[(rank - s) % world].clone(), recv, rank, world)
        chunks[(rank - s - 1) % world] += recv
    for s in range(world - 1):                       # all-gather
        _exchange(chunks[(rank + 1 - s) % world].clone(), recv, rank, world)
        chunks[(rank - s) % world].copy_(recv)

    if op == "avg":
        buf /= world
    tensor.copy_(buf[:n].view_as(tensor))
    return tensor


def ring_allreduce_coalesced(tensors: List[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
    """All-reduce several tensors with one ring pass over a single flat buffer."""
    if not tensors or not is_distributed() or get_world_size() == 1:
        return tensors
    flat = torch.cat([t.detach().reshape(-1) for t in tensors])
    ring_allreduce(flat, op)
    offset = 0
    for t in tensors:
        t.copy_(flat[offset:offset + t.numel()].view_as(t))
        offset += t.numel()
    return tensors
