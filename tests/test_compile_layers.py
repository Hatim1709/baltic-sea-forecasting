"""Tests of --compile_layers (neural_lam/compile_layers.py), run on CPU"""

# Standard library
import os
import shutil
import sys

# Third-party
import pytest
import torch

# First-party
from neural_lam import checkpointing, utils
from neural_lam.compile_layers import compile_layers

sys.path.insert(0, os.path.dirname(__file__))
# pylint: disable=wrong-import-position
from test_checkpoint_policy import (  # noqa: E402
    BATCH,
    DIM,
    NUM_NODES,
    _forward_backward,
    _make_cases,
    _rand,
    _same_level_edges,
)

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None,
    reason="torch.compile on CPU needs a C++ compiler",
)
# Cases: interaction, interaction without edge update, propagation, flexible x2
CASES = [0, 2, 3]


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("case_index", CASES)
def test_layer_has_no_graph_breaks(case_index):
    """The whole layer body is one graph (the num_rec fix, d1a7b05).
    Kept before the compile tests: dynamo's explain reports nothing once a
    selective-checkpoint run has gone through it in the same process."""
    _, layer, inputs = _make_cases()[case_index]
    inputs = [x.detach().clone() for x in inputs]
    report = torch._dynamo.explain(layer._forward_impl)(*inputs)
    assert report.graph_break_count == 0, report.break_reasons


@pytest.mark.parametrize("case_index", CASES)
@pytest.mark.parametrize("ckpt", [False, True, checkpointing.SELECTIVE])
def test_compiled_layer_matches_eager(case_index, ckpt):
    """Outputs and gradients are unchanged, also under each layer policy"""
    _, layer, inputs = _make_cases()[case_index]
    layer.ckpt_inner = ckpt
    reference = _forward_backward(layer, inputs)
    assert compile_layers(layer) == 1
    checkpointing.reset_counts()
    compiled = _forward_backward(layer, inputs)
    for new_group, ref_group in zip(compiled, reference):
        for new, ref in zip(new_group, ref_group):
            torch.testing.assert_close(new, ref, rtol=1e-7, atol=1e-9)
    # the audit counters still see every execution (1, or 2 with recompute)
    expected = 1 if ckpt is False else 2
    assert checkpointing.get_counts()["layer_calls"] == expected


def test_compile_layers_can_be_switched_off_again():
    _, layer, inputs = _make_cases()[0]
    eager = layer._forward_impl
    compile_layers(layer)
    assert layer._forward_impl is not eager
    compile_layers(layer, enabled=False)
    assert layer._forward_impl == eager
    assert not hasattr(layer, "_forward_impl_eager")


def test_stack_under_step_checkpoint_matches_eager_and_all_layers_compile():
    """A processor-like stack of layers, nested in a step-level checkpoint"""
    edge_index = _same_level_edges()
    seq = utils.make_gnn_seq(edge_index, 10, 1, DIM).double().train()
    nodes = _rand(BATCH, NUM_NODES, DIM, seed=1)
    edges = _rand(BATCH, edge_index.shape[1], DIM, seed=2)

    def run():
        x = nodes.clone().requires_grad_(True)
        seq.zero_grad()
        out, _ = torch.utils.checkpoint.checkpoint(
            seq, x, edges, use_reentrant=False
        )
        (out**2).sum().backward()
        return [x.grad.clone()] + [p.grad.clone() for p in seq.parameters()]

    reference = run()
    assert compile_layers(seq) == 10
    # 10 layers exceed dynamo's default limit of 8 graphs per function
    assert torch._dynamo.config.cache_size_limit >= 10
    for new, ref in zip(run(), reference):
        torch.testing.assert_close(new, ref, rtol=1e-7, atol=1e-9)
