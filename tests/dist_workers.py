"""Worker bodies for the multi-process tests (module-level so spawn can pickle them)."""

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tinytrain.distributed.data_parallel import DataParallel
from tinytrain.distributed.pipeline_parallel import GPTStage, gpipe_step
from tinytrain.distributed.ring_allreduce import ring_allreduce, ring_allreduce_coalesced
from tinytrain.distributed.tensor_parallel import parallelize_gpt
from tinytrain.model.config import GPTConfig
from tinytrain.model.gpt import GPT


def tiny_gpt(seed: int = 0, n_layers: int = 4) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=97, max_seq_len=32, d_model=32, n_layers=n_layers, n_heads=4, dropout=0.0)
    return GPT(cfg).eval()


def batch(seed: int = 1, n: int = 8, seq: int = 16):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 97, (n, seq), generator=g)
    return ids, torch.roll(ids, -1, dims=1)


def loss_of(model, ids, labels):
    logits, _ = model(ids)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def ring_worker(rank, world, numel):
    g = torch.Generator().manual_seed(rank)
    x = torch.randn(numel, generator=g)
    expected = x.clone()
    dist.all_reduce(expected)
    ring_allreduce(x)
    parts = [torch.randn(3, 5, generator=g), torch.randn(7, generator=g)]
    ref = [p.clone() for p in parts]
    for r in ref:
        dist.all_reduce(r)
    ring_allreduce_coalesced(parts)
    return max((x - expected).abs().max().item(), *[(a - b).abs().max().item() for a, b in zip(parts, ref)])


def data_parallel_worker(rank, world, use_ring):
    model = tiny_gpt(seed=rank)                      # different init per rank on purpose
    dp = DataParallel(model, bucket_mb=0.01, use_ring=use_ring)   # broadcast makes them equal
    ids, labels = batch()
    shard = slice(rank * len(ids) // world, (rank + 1) * len(ids) // world)
    loss_of(dp, ids[shard], labels[shard]).backward()
    dp.synchronize_gradients()
    return {n: p.grad.clone() for n, p in model.named_parameters()}, {n: p.detach().clone() for n, p in model.named_parameters()}


def tensor_parallel_worker(rank, world):
    full = tiny_gpt(seed=0)
    ids, labels = batch()
    tp = parallelize_gpt(copy.deepcopy(full), rank, world)
    tp_loss = loss_of(tp, ids, labels)
    tp_loss.backward()
    # Gradients of replicated parameters (embeddings, norms, biases of row layers) should equal
    # the full model's; sharded weights should equal the matching shard of the full gradient.
    grads = {n: p.grad.clone() for n, p in tp.named_parameters()}
    return tp_loss.item(), grads


def pipeline_worker(rank, world, n_micro):
    full = tiny_gpt(seed=0, n_layers=4)
    stage = GPTStage(copy.deepcopy(full), rank, world)
    ids, labels = batch()
    loss = gpipe_step(stage, ids, labels, n_micro)
    return loss, {n: p.grad.clone() for n, p in stage.named_parameters() if p.grad is not None}, stage.is_first, stage.is_last
