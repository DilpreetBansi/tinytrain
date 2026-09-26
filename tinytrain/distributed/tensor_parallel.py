"""
Tensor (Megatron-style) parallelism for GPT blocks.

MLP:        Y = GeLU(X A) B.  A is split by columns and B by rows, so each rank computes
            GeLU(X A_i) B_i locally and one all-reduce sums the partial outputs.
Attention:  heads are split across ranks (q/k/v projections column-parallel, output
            projection row-parallel), again with a single all-reduce per block.

Two autograd functions carry the communication:
  copy_to_tp      forward: identity      backward: all-reduce the input gradient
  reduce_from_tp  forward: all-reduce    backward: identity
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tinytrain.distributed.comm import get_rank, get_world_size, is_distributed


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        if is_distributed() and get_world_size() > 1:
            grad = grad.clone()
            dist.all_reduce(grad)
        return grad


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if is_distributed() and get_world_size() > 1:
            x = x.clone()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


copy_to_tp = _CopyToTP.apply
reduce_from_tp = _ReduceFromTP.apply


def _shard(t: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    return t.chunk(world, dim=dim)[rank].clone()


class ColumnParallelLinear(nn.Module):
    """y_i = x A_i^T + b_i: this rank's slice of the output features."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 world_size: Optional[int] = None, rank: Optional[int] = None) -> None:
        super().__init__()
        self.world_size = world_size or get_world_size()
        self.rank = get_rank() if rank is None else rank
        if out_features % self.world_size:
            raise ValueError(f"out_features ({out_features}) must be divisible by world_size ({self.world_size})")
        self.weight = nn.Parameter(torch.empty(out_features // self.world_size, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features // self.world_size)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, world_size: int) -> "ColumnParallelLinear":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, world_size, rank)
        with torch.no_grad():
            layer.weight.copy_(_shard(linear.weight, 0, rank, world_size))
            if linear.bias is not None:
                layer.bias.copy_(_shard(linear.bias, 0, rank, world_size))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(copy_to_tp(x), self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """y = sum_i x_i B_i^T + b: this rank holds a slice of the input features."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 world_size: Optional[int] = None, rank: Optional[int] = None) -> None:
        super().__init__()
        self.world_size = world_size or get_world_size()
        self.rank = get_rank() if rank is None else rank
        if in_features % self.world_size:
            raise ValueError(f"in_features ({in_features}) must be divisible by world_size ({self.world_size})")
        self.weight = nn.Parameter(torch.empty(out_features, in_features // self.world_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, world_size: int) -> "RowParallelLinear":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, world_size, rank)
        with torch.no_grad():
            layer.weight.copy_(_shard(linear.weight, 1, rank, world_size))
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        y = reduce_from_tp(F.linear(x_local, self.weight))
        return y + self.bias if self.bias is not None else y


class ParallelMLP(nn.Module):
    def __init__(self, mlp: nn.Module, rank: int, world_size: int) -> None:
        super().__init__()
        self.fc_in = ColumnParallelLinear.from_linear(mlp.linear1, rank, world_size)
        self.fc_out = RowParallelLinear.from_linear(mlp.linear2, rank, world_size)
        self.activation = mlp.activation
        self.dropout = mlp.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.dropout(self.activation(self.fc_in(x)))))


class ParallelSelfAttention(nn.Module):
    """Causal self-attention with this rank's share of the heads."""

    def __init__(self, attn: nn.Module, rank: int, world_size: int) -> None:
        super().__init__()
        if attn.n_heads % world_size:
            raise ValueError(f"n_heads ({attn.n_heads}) must be divisible by world_size ({world_size})")
        self.local_heads = attn.n_heads // world_size
        self.head_dim = attn.head_dim
        self.q = ColumnParallelLinear.from_linear(attn.q_proj, rank, world_size)
        self.k = ColumnParallelLinear.from_linear(attn.k_proj, rank, world_size)
        self.v = ColumnParallelLinear.from_linear(attn.v_proj, rank, world_size)
        self.out = RowParallelLinear.from_linear(attn.out_proj, rank, world_size)
        self.dropout_p = attn.attn_dropout.p
        self.resid_dropout = attn.resid_dropout

    def forward(self, x: torch.Tensor, causal_mask=None, use_cache: bool = False) -> Tuple[torch.Tensor, None]:
        b, s, _ = x.shape
        q, k, v = (proj(x).view(b, s, self.local_heads, self.head_dim).transpose(1, 2) for proj in (self.q, self.k, self.v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        att = F.dropout(att.softmax(-1), self.dropout_p, self.training)
        y = (att @ v).transpose(1, 2).reshape(b, s, self.local_heads * self.head_dim)
        return self.resid_dropout(self.out(y)), None


def parallelize_block(block: nn.Module, rank: Optional[int] = None, world_size: Optional[int] = None) -> nn.Module:
    """Replace a TransformerBlock's attention and MLP with tensor-parallel versions (in place)."""
    rank = get_rank() if rank is None else rank
    world_size = world_size or get_world_size()
    block.attn = ParallelSelfAttention(block.attn, rank, world_size)
    block.mlp = ParallelMLP(block.mlp, rank, world_size)
    return block


def parallelize_gpt(model: nn.Module, rank: Optional[int] = None, world_size: Optional[int] = None) -> nn.Module:
    for block in model.blocks:
        parallelize_block(block, rank, world_size)
    return model
