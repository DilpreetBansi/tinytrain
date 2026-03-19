"""
Distributed Communication Primitives

Wrapper around torch.distributed for collective operations.
Handles initialization, rank management, and common collective ops.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List


def init_distributed() -> None:
    """
    Initialize distributed training.

    Sets up the distributed process group using environment variables
    set by torchrun launcher. Safe to call on single-GPU systems
    (will initialize with single rank).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    # Check if already initialized
    if dist.is_initialized():
        return

    # Initialize process group
    # torchrun sets: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    try:
        dist.init_process_group(backend="nccl")
    except RuntimeError:
        # Fallback to gloo if nccl unavailable (e.g., CPU-only or no GPUs)
        dist.init_process_group(backend="gloo")


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank (rank within machine)."""
    if not is_distributed():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def broadcast(tensor: torch.Tensor, src: int = 0) -> None:
    """
    Broadcast tensor from source rank to all ranks.

    Args:
        tensor: Tensor to broadcast (modified in-place)
        src: Source rank
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)


def allreduce(tensor: torch.Tensor, op: str = "sum") -> None:
    """
    All-reduce operation across all ranks.

    Args:
        tensor: Tensor to reduce (modified in-place)
        op: Reduction operation ("sum", "avg", "max", "min")
    """
    if not is_distributed():
        return

    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "avg":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(get_world_size())
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown reduction op: {op}")


def allgather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensor from all ranks.

    Args:
        tensor: Tensor to gather

    Returns:
        List of gathered tensors from each rank
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    return output


def send_recv(
    send_tensor: Optional[torch.Tensor] = None,
    src: int = 0,
    dst: int = 1,
) -> Optional[torch.Tensor]:
    """
    Send/receive tensor between two ranks.

    Args:
        send_tensor: Tensor to send (from src rank)
        src: Source rank
        dst: Destination rank

    Returns:
        Received tensor (for dst rank) or None
    """
    if not is_distributed():
        return send_tensor

    rank = get_rank()
    recv_tensor = None

    if rank == src:
        dist.send(send_tensor, dst)
    elif rank == dst:
        recv_tensor = torch.zeros_like(send_tensor)
        dist.recv(recv_tensor, src)

    return recv_tensor
