"""Run a test body in N CPU processes (gloo) and collect per-rank results."""

import os
import socket
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _to_plain(x):
    """Tensors -> numpy so results pickle by value (no shared-memory handles across processes)."""
    if isinstance(x, torch.Tensor):
        return ("__tensor__", x.detach().cpu().numpy())
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_plain(v) for v in x)
    return x


def _from_plain(x):
    if isinstance(x, tuple) and len(x) == 2 and x[0] == "__tensor__":
        return torch.from_numpy(x[1])
    if isinstance(x, dict):
        return {k: _from_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_from_plain(v) for v in x)
    return x


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, fn, args, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        queue.put((rank, "ok", _to_plain(fn(rank, world, *args))))
    except Exception:
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(fn, world: int, *args):
    """fn(rank, world, *args) must be a module-level function; returns results ordered by rank."""
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    procs = mp.start_processes(_worker, args=(world, _free_port(), fn, args, queue), nprocs=world,
                               join=False, start_method="spawn")
    # Drain the queue before joining: a child blocks in put() until its large result is read.
    results = sorted((queue.get() for _ in range(world)), key=lambda r: r[0])
    while not procs.join():
        pass
    errors = [r for r in results if r[1] == "error"]
    if errors:
        raise AssertionError(f"rank {errors[0][0]} failed:\n{errors[0][2]}")
    return [_from_plain(r[2]) for r in results]
