"""Multi-process correctness tests (CPU, gloo). Each parallel strategy must reproduce the
single-process result."""

import pytest
import torch

from tests.dist_utils import run_distributed
from tests.dist_workers import (batch, data_parallel_worker, loss_of, pipeline_worker, ring_worker,
                                tensor_parallel_worker, tiny_gpt)
from tinytrain.distributed.pipeline_parallel import stage_bounds


@pytest.mark.parametrize("world,numel", [(2, 10), (3, 1000), (4, 7), (4, 1)])
def test_ring_allreduce_matches_backend(world, numel):
    errors = run_distributed(ring_worker, world, numel)
    assert max(errors) < 1e-5


def full_grads():
    model = tiny_gpt(seed=0)
    ids, labels = batch()
    loss = loss_of(model, ids, labels)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("use_ring", [True, False])
def test_data_parallel_equals_full_batch(use_ring):
    results = run_distributed(data_parallel_worker, 2, use_ring)
    _, ref = full_grads()
    (g0, p0), (g1, p1) = results
    for name, grad in ref.items():
        torch.testing.assert_close(g0[name], grad, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(g1[name], grad, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(p0[name], p1[name])  # replicas identical after broadcast


def test_tensor_parallel_equals_full_model():
    ref_loss, ref = full_grads()
    world = 2
    results = run_distributed(tensor_parallel_worker, world)
    for rank, (loss, grads) in enumerate(results):
        assert loss == pytest.approx(ref_loss, rel=1e-5)
        for name, grad in grads.items():
            if ".attn." in name or ".mlp." in name:
                full_name = (name.replace("attn.q.", "attn.q_proj.").replace("attn.k.", "attn.k_proj.")
                             .replace("attn.v.", "attn.v_proj.").replace("attn.out.", "attn.out_proj.")
                             .replace("mlp.fc_in.", "mlp.linear1.").replace("mlp.fc_out.", "mlp.linear2."))
                full = ref[full_name]
                column = any(k in name for k in ("attn.q.", "attn.k.", "attn.v.", "mlp.fc_in."))
                if column:                                   # output features are sharded
                    expected = full.chunk(world, dim=0)[rank]
                elif name.endswith("weight"):                # row-parallel weight: input features sharded
                    expected = full.chunk(world, dim=1)[rank]
                else:                                        # row-parallel bias is replicated
                    expected = full
                torch.testing.assert_close(grad, expected, atol=1e-5, rtol=1e-4)
            else:
                torch.testing.assert_close(grad, ref[name], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("world,n_micro", [(2, 4), (4, 2)])
def test_gpipe_equals_full_model(world, n_micro):
    ref_loss, ref = full_grads()
    results = run_distributed(pipeline_worker, world, n_micro)
    bounds = stage_bounds(4, world)
    last_loss = results[-1][0]
    assert last_loss == pytest.approx(ref_loss, rel=1e-5)
    for rank, (_, grads, first, last) in enumerate(results):
        lo, _ = bounds[rank]
        for name, grad in grads.items():
            if name.startswith("blocks."):
                idx, rest = name.split(".", 2)[1:]
                full_name = f"blocks.{int(idx) + lo}.{rest}"
            elif name == "lm_head.weight":
                full_name = "embeddings.token_emb.weight"     # tied: full grad = embed + head usage
            else:
                full_name = name
            torch.testing.assert_close(grad, ref[full_name], atol=1e-5, rtol=1e-4)


def test_stage_bounds_cover_all_layers():
    assert stage_bounds(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
