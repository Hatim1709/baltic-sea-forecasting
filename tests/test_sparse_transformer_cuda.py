"""CUDA training-step checks against the CPU float64 loop reference."""

# Standard library
import copy
import importlib.util
from pathlib import Path

# Third-party
import pytest
import torch

# Reuse the existing fixture and reference without importing neural_lam.
REFERENCE_PATH = Path(__file__).with_name("test_sparse_transformer.py")
SPEC = importlib.util.spec_from_file_location(
    "sparse_transformer_cpu_reference", REFERENCE_PATH
)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)

graph_case = REFERENCE.graph_case
Processor = REFERENCE.SparseGraphTransformerProcessor


def check_training_step(graph_case, device, dtype, atol, rtol):
    """Compare outputs, loss, gradients, and an SGD update across devices."""
    edge_index, initial_nodes, initial_edges, initial_target = graph_case
    torch.manual_seed(79)
    actual = Processor(
        edge_index,
        hidden_dim=8,
        edge_dim=3,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ).to(dtype=dtype)
    # Preserve the actual parameter rounding before promoting the reference.
    reference = copy.deepcopy(actual).double()
    actual = actual.to(device)

    nodes = initial_nodes.to(device=device, dtype=dtype).clone()
    edges = initial_edges.to(device=device, dtype=dtype).clone()
    nodes.requires_grad_()
    edges.requires_grad_()
    ref_nodes = nodes.detach().cpu().double().clone().requires_grad_()
    ref_edges = edges.detach().cpu().double().clone().requires_grad_()
    target = initial_target.to(device=device, dtype=dtype)
    ref_target = target.cpu().double()

    def compare(observed, expected, label):
        assert observed is not None, label
        assert expected is not None, label
        assert torch.isfinite(observed).all(), label
        assert torch.isfinite(expected).all(), label
        torch.testing.assert_close(
            observed.detach().cpu().double(),
            expected.detach().cpu().double(),
            atol=atol,
            rtol=rtol,
            msg=label,
        )

    optimizer = torch.optim.SGD(actual.parameters(), lr=0.03, momentum=0.9)
    ref_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.03, momentum=0.9
    )
    output, returned_edges = actual(nodes, edges)
    ref_output, _ = REFERENCE.reference_processor(
        reference, ref_nodes, ref_edges, edge_index
    )
    assert output.device == device
    assert output.dtype == dtype
    assert returned_edges is edges
    assert all(block.edge_index.device == device for block in actual.blocks)
    compare(output, ref_output, "node outputs")

    loss = (output - target).square().mean()
    ref_loss = (ref_output - ref_target).square().mean()
    compare(loss, ref_loss, "scalar loss")
    loss.backward()
    ref_loss.backward()

    compare(nodes.grad, ref_nodes.grad, "node-input gradients")
    compare(edges.grad, ref_edges.grad, "edge-input gradients")
    assert edges.grad.abs().sum().item() > 1e-8
    parameters = dict(actual.named_parameters())
    ref_parameters = dict(reference.named_parameters())
    assert parameters.keys() == ref_parameters.keys()
    for name, parameter in parameters.items():
        compare(parameter.grad, ref_parameters[name].grad, name + " gradient")
    for block in actual.blocks:
        assert block.q_proj.weight.grad.norm().item() > 1e-8
        assert block.k_proj.weight.grad.norm().item() > 1e-8
        assert block.edge_bias.weight.grad.norm().item() > 1e-8

    before = actual.blocks[0].q_proj.weight.detach().clone()
    optimizer.step()
    ref_optimizer.step()
    for name, parameter in parameters.items():
        compare(parameter, ref_parameters[name], name + " after SGD")
    assert not torch.equal(before, actual.blocks[0].q_proj.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        pytest.param(torch.float64, 1e-10, 1e-8, id="float64"),
        pytest.param(torch.float32, 2e-6, 2e-4, id="float32"),
    ],
)
def test_cuda_training_step(graph_case, dtype, atol, rtol):
    """Run the production processor on CUDA against the CPU reference."""
    device = torch.device("cuda", torch.cuda.current_device())
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Test ordinary float32 before introducing reduced-precision matmul.
        torch.backends.cuda.matmul.allow_tf32 = False
        check_training_step(graph_case, device, dtype, atol, rtol)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
