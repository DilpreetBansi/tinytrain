"""
Pipeline parallelism with a GPipe schedule across processes.

The GPT is cut into contiguous stages, one per rank. Each step splits the batch into M
micro-batches: all forwards run first (activations flow rank 0 -> N-1 via send/recv), then
all backwards run in reverse (gradients flow N-1 -> 0). The last stage computes the loss.

GPT ties the token embedding (first stage) to the LM head (last stage). Like Megatron, the last
stage keeps its own copy of that matrix and the two ranks all-reduce its gradient after the
backward pass, so both copies receive the same update.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tinytrain.distributed.comm import get_rank, get_world_size


def stage_bounds(n_layers: int, world: int) -> List[Tuple[int, int]]:
    """Split n_layers into `world` contiguous, nearly equal ranges."""
    base, extra = divmod(n_layers, world)
    bounds, start = [], 0
    for r in range(world):
        end = start + base + (1 if r < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


class GPTStage(nn.Module):
    """The part of a GPT that one pipeline rank owns."""

    def __init__(self, model: nn.Module, rank: int, world: int) -> None:
        super().__init__()
        self.rank, self.world = rank, world
        self.is_first, self.is_last = rank == 0, rank == world - 1
        lo, hi = stage_bounds(len(model.blocks), world)[rank]
        self.embeddings = model.embeddings if self.is_first else None
        self.blocks = nn.ModuleList(model.blocks[lo:hi])
        if self.is_last:
            self.norm = model.norm
            self.lm_head = nn.Linear(model.config.d_model, model.config.vocab_size, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(model.lm_head.weight)  # untied copy (see module docstring)
        self.d_model = model.config.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            x = self.embeddings(x)
        for block in self.blocks:
            x, _ = block(x)
        if self.is_last:
            x = self.lm_head(self.norm(x))
        return x

    def tied_weight(self) -> Optional[torch.Tensor]:
        if self.is_first:
            return self.embeddings.token_emb.weight
        if self.is_last:
            return self.lm_head.weight
        return None


def gpipe_step(stage: GPTStage, input_ids: torch.Tensor, labels: torch.Tensor, n_micro: int) -> Optional[float]:
    """One forward+backward over a batch. Returns the mean loss on the last rank, else None.

    Every rank passes the same input_ids/labels (only the first and last stage read them).
    Gradients accumulate in stage parameters; call optimizer.step() afterwards.
    """
    rank, world = stage.rank, stage.world
    ids_mb, labels_mb = input_ids.chunk(n_micro), labels.chunk(n_micro)
    saved = []
    total = 0.0

    for ids, lab in zip(ids_mb, labels_mb):                         # all forwards
        if stage.is_first:
            inp = ids
        else:
            inp = torch.empty(ids.shape[0], ids.shape[1], stage.d_model)
            dist.recv(inp, src=rank - 1)
            inp.requires_grad_(True)
        out = stage(inp)
        if stage.is_last:
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), lab.reshape(-1)) / n_micro
            total += loss.item()
            saved.append((inp, loss))
        else:
            dist.send(out.detach().contiguous(), dst=rank + 1)
            saved.append((inp, out))

    for inp, out in reversed(saved):                                  # all backwards
        if stage.is_last:
            out.backward()
        else:
            grad = torch.empty_like(out)
            dist.recv(grad, src=rank + 1)
            out.backward(grad)
        if not stage.is_first:
            dist.send(inp.grad.contiguous(), dst=rank - 1)

    if world > 1:                                                     # tied embedding / LM head
        w = stage.tied_weight()
        if stage.is_first or stage.is_last:
            grad = w.grad.clone()
            peer = world - 1 if stage.is_first else 0
            other = torch.empty_like(grad)
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, grad, peer), dist.P2POp(dist.irecv, other, peer)])
            for r in reqs:
                r.wait()
            w.grad += other
    return total if stage.is_last else None
