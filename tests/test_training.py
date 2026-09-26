"""Gradient checkpointing, loss scaling and the trainer's update rule."""

import copy

import torch
import torch.nn.functional as F

from tests.dist_workers import batch, loss_of, tiny_gpt
from tinytrain.training.gradient_checkpoint import CheckpointedModule
from tinytrain.training.mixed_precision import GradScaler


def test_gradient_checkpointing_gives_identical_gradients():
    model = tiny_gpt()
    ckpt = copy.deepcopy(model)
    ckpt.blocks = torch.nn.ModuleList(CheckpointedModule(b) for b in ckpt.blocks)
    ids, labels = batch()
    loss_of(model, ids, labels).backward()
    loss_of(ckpt, ids, labels).backward()
    for (n, p), (_, q) in zip(model.named_parameters(), ckpt.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


def test_scaled_then_unscaled_gradients_match_unscaled_training():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 3)
    ref = copy.deepcopy(lin)
    x = torch.randn(8, 4)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    scaler = GradScaler(init_scale=1024.0)
    scaler.scale_loss(lin(x).pow(2).mean()).backward()
    scaler.unscale_grads(opt)
    ref(x).pow(2).mean().backward()
    for p, q in zip(lin.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_overflow_skips_update_and_backs_off():
    lin = torch.nn.Linear(2, 1)
    before = [p.detach().clone() for p in lin.parameters()]
    opt = torch.optim.SGD(lin.parameters(), lr=1.0)
    scaler = GradScaler(init_scale=2.0 ** 16)
    for p in lin.parameters():
        p.grad = torch.full_like(p, float("inf"))
    assert scaler.has_overflow([p.grad for p in lin.parameters()])
    scaler.step(opt, overflow=True)
    assert scaler.scale == 2.0 ** 15
    opt.step()  # grads were zeroed, so parameters must not move
    for p, b in zip(lin.parameters(), before):
        torch.testing.assert_close(p.detach(), b)
