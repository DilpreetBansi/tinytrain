"""
Train a small character-level GPT on Tiny Shakespeare with one of four strategies.

    python scripts/train_parallel.py --strategy single
    torchrun --nproc_per_node=2 scripts/train_parallel.py --strategy dp
    torchrun --nproc_per_node=2 scripts/train_parallel.py --strategy tp
    torchrun --nproc_per_node=2 scripts/train_parallel.py --strategy pp --micro-batches 4

Every strategy sees the same global batches in the same order and starts from the same weights,
so their loss curves should match up to floating-point noise (see docs/loss_curves.png).
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinytrain.distributed.data_parallel import DataParallel  # noqa: E402
from tinytrain.distributed.pipeline_parallel import GPTStage, gpipe_step  # noqa: E402
from tinytrain.distributed.tensor_parallel import parallelize_gpt  # noqa: E402
from tinytrain.model.config import GPTConfig  # noqa: E402
from tinytrain.model.gpt import GPT  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["single", "dp", "tp", "pp"], default="single")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32, help="global batch size")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--micro-batches", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default=None, help="write the loss curve as JSON here")
    args = ap.parse_args()

    distributed = args.strategy != "single"
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1

    text = load_text(Path(__file__).resolve().parents[1] / "data" / "tiny_shakespeare.txt")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    torch.manual_seed(0)                        # identical initial weights on every rank/strategy
    cfg = GPTConfig(vocab_size=len(chars), max_seq_len=args.seq, d_model=args.d_model, n_layers=args.layers,
                    n_heads=args.heads, dropout=0.0)
    model = GPT(cfg)

    if args.strategy == "dp":
        model = DataParallel(model)
    elif args.strategy == "tp":
        model = parallelize_gpt(model, rank, world)
    elif args.strategy == "pp":
        model = GPTStage(model, rank, world)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    g = torch.Generator().manual_seed(1234)     # identical batch stream everywhere
    curve, start = [], time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(data) - args.seq - 1, (args.batch,), generator=g)
        x = torch.stack([data[i:i + args.seq] for i in idx])
        y = torch.stack([data[i + 1:i + args.seq + 1] for i in idx])

        if args.strategy == "pp":
            loss = gpipe_step(model, x, y, args.micro_batches)
            loss_t = torch.tensor([loss if loss is not None else 0.0])
            if distributed:
                dist.broadcast(loss_t, src=world - 1)
        else:
            if args.strategy == "dp":
                shard = slice(rank * args.batch // world, (rank + 1) * args.batch // world)
                x, y = x[shard], y[shard]
            logits, _ = model(x)
            loss_t = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss_t.backward()
            if args.strategy == "dp":
                model.synchronize_gradients()
                loss_t = loss_t.detach().clone()
                dist.all_reduce(loss_t)
                loss_t /= world
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            curve.append((step, float(loss_t)))
            if rank == 0:
                print(f"[{args.strategy} x{world}] step {step:4d}  loss {float(loss_t):.4f}  "
                      f"{(time.time() - start) / step * 1000:.0f} ms/step", flush=True)

    if rank == 0 and args.out:
        Path(args.out).write_text(json.dumps({"strategy": args.strategy, "world": world, "curve": curve}))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
