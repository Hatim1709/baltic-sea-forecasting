"""Independent CPU float64 checks for sparse graph attention."""

# Standard library
import copy
import importlib.util
from pathlib import Path

# Third-party
import pytest
import torch
import torch.nn.functional as F

# Load only this module: neural_lam.__init__ imports forecasting dependencies.
MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "neural_lam" / "sparse_transformer.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sparse_transformer_under_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SparseGraphAttentionBlock = MODULE.SparseGraphAttentionBlock
SparseGraphTransformerProcessor = MODULE.SparseGraphTransformerProcessor

ATOL = 1e-10
RTOL = 1e-8


def assert_close(actual, expected):
    """Declared tolerances for CPU float64 equivalence."""
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.fixture
def graph_case():
    """Two samples, directed edges, degrees [0, 3, 2, 1, 0]."""
    generator = torch.Generator().manual_seed(1729)
    edge_index = torch.tensor(
        [[0, 2, 3, 1, 0, 4], [1, 1, 1, 2, 2, 3]],
        dtype=torch.long,
    )
    nodes = torch.randn(2, 5, 8, generator=generator, dtype=torch.float64)
    edges = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 5, 8, generator=generator, dtype=torch.float64)
    return edge_index, nodes, edges, target


def reference_block(block, nodes, edges, edge_index):
    """Evaluate the equations using explicit sample and receiver loops.

    This reference requires (B, N, D) nodes and dropout zero.
    It never calls the production forward, PyG softmax, or scatter/index_add.
    Only learned parameter tensors are shared with the specified architecture.
    """
    batch_size, num_nodes, width = nodes.shape
    heads = block.num_heads
    head_dim = width // heads
    normalized = F.layer_norm(
        nodes,
        (width,),
        block.norm1.weight,
        block.norm1.bias,
        eps=1e-5,
    )
    shape = (batch_size, num_nodes, heads, head_dim)
    queries = F.linear(normalized, block.q_proj.weight).reshape(shape)
    keys = F.linear(normalized, block.k_proj.weight).reshape(shape)
    values = F.linear(normalized, block.v_proj.weight).reshape(shape)
    edge_bias = F.linear(edges, block.edge_bias.weight)

    samples = []
    for sample in range(batch_size):
        receivers = []
        for receiver in range(num_nodes):
            incoming = (edge_index[1] == receiver).nonzero(as_tuple=True)[0]
            if incoming.numel() == 0:
                receivers.append(nodes.new_zeros(width))
                continue
            senders = edge_index[0, incoming]
            logits = torch.einsum(
                "hd,ehd->eh", queries[sample, receiver], keys[sample, senders]
            )
            logits = logits / head_dim**0.5 + edge_bias[sample, incoming]
            weights = torch.softmax(logits, dim=0)
            weighted = torch.einsum(
                "eh,ehd->hd", weights, values[sample, senders]
            )
            receivers.append(weighted.reshape(width))
        samples.append(torch.stack(receivers))
    aggregated = torch.stack(samples)

    attended = nodes + F.linear(aggregated, block.out_proj.weight)
    normalized = F.layer_norm(
        attended,
        (width,),
        block.norm2.weight,
        block.norm2.bias,
        eps=1e-5,
    )
    hidden = F.gelu(
        F.linear(normalized, block.ffn[0].weight, block.ffn[0].bias)
    )
    return attended + F.linear(hidden, block.ffn[2].weight, block.ffn[2].bias)


def reference_processor(processor, nodes, edges, edge_index):
    """Keep edge features fixed across independently evaluated blocks."""
    for block in processor.blocks:
        nodes = reference_block(block, nodes, edges, edge_index)
    return nodes, edges


@pytest.mark.parametrize("num_layers,num_heads", [(1, 2), (2, 4)])
def test_reference_forward_backward_and_update(
    graph_case, num_layers, num_heads
):
    """Compare the complete training calculation to the loop reference."""
    edge_index, initial_nodes, initial_edges, target = graph_case
    torch.manual_seed(41)
    actual = SparseGraphTransformerProcessor(
        edge_index,
        hidden_dim=8,
        edge_dim=3,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_expansion=3,
        dropout=0.0,
    ).double()
    reference = copy.deepcopy(actual)
    nodes = initial_nodes.clone().requires_grad_()
    edges = initial_edges.clone().requires_grad_()
    ref_nodes = initial_nodes.clone().requires_grad_()
    ref_edges = initial_edges.clone().requires_grad_()

    optimizer = torch.optim.SGD(actual.parameters(), lr=0.03, momentum=0.9)
    ref_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.03, momentum=0.9
    )
    output, returned_edges = actual(nodes, edges)
    ref_output, ref_returned_edges = reference_processor(
        reference, ref_nodes, ref_edges, edge_index
    )
    assert returned_edges is edges
    assert ref_returned_edges is ref_edges
    torch.testing.assert_close(edges, initial_edges, atol=0, rtol=0)
    assert_close(output, ref_output)
    assert_close(returned_edges, ref_returned_edges)

    # The denominator includes every sample, node, and feature.
    loss = (output - target).square().mean()
    ref_loss = (ref_output - target).square().mean()
    assert_close(loss, ref_loss)
    loss.backward()
    ref_loss.backward()

    assert nodes.grad is not None and ref_nodes.grad is not None
    assert edges.grad is not None and ref_edges.grad is not None
    assert_close(nodes.grad, ref_nodes.grad)
    assert_close(edges.grad, ref_edges.grad)
    assert edges.grad.abs().sum().item() > 1e-8

    actual_parameters = dict(actual.named_parameters())
    ref_parameters = dict(reference.named_parameters())
    assert actual_parameters.keys() == ref_parameters.keys()
    for name, parameter in actual_parameters.items():
        ref_parameter = ref_parameters[name]
        assert parameter.grad is not None, name
        assert ref_parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert_close(parameter.grad, ref_parameter.grad)

    # Prevent a degenerate fixture from hiding a broken score gradient.
    for block in actual.blocks:
        assert block.q_proj.weight.grad.norm().item() > 1e-8
        assert block.k_proj.weight.grad.norm().item() > 1e-8
        assert block.edge_bias.weight.grad.norm().item() > 1e-8

    before_update = actual.blocks[0].q_proj.weight.detach().clone()
    optimizer.step()
    ref_optimizer.step()
    for name, parameter in actual_parameters.items():
        assert_close(parameter, ref_parameters[name])
    assert not torch.equal(before_update, actual.blocks[0].q_proj.weight)


def test_edge_order_permutation_preserves_outputs_and_gradients(graph_case):
    """Reordering edges and their features must not change the operator."""
    edge_index, initial_nodes, initial_edges, target = graph_case
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    torch.manual_seed(53)
    original = SparseGraphAttentionBlock(edge_index, 8, 3).double()
    reordered = SparseGraphAttentionBlock(
        edge_index[:, permutation], 8, 3
    ).double()
    reordered.load_state_dict(original.state_dict())

    nodes = initial_nodes.clone().requires_grad_()
    edges = initial_edges.clone().requires_grad_()
    other_nodes = initial_nodes.clone().requires_grad_()
    other_edges = initial_edges[:, permutation].clone().requires_grad_()
    output = original(nodes, edges)
    other_output = reordered(other_nodes, other_edges)
    assert_close(output, other_output)

    loss = (output - target).square().mean()
    other_loss = (other_output - target).square().mean()
    assert_close(loss, other_loss)
    loss.backward()
    other_loss.backward()
    assert_close(nodes.grad, other_nodes.grad)
    assert_close(edges.grad[:, permutation], other_edges.grad)

    other_parameters = dict(reordered.named_parameters())
    for name, parameter in original.named_parameters():
        assert parameter.grad is not None, name
        assert_close(parameter.grad, other_parameters[name].grad)


def test_samples_are_independent(graph_case):
    """Batched and individual runs agree without cross-sample gradients."""
    edge_index, initial_nodes, initial_edges, target = graph_case
    torch.manual_seed(59)
    block = SparseGraphAttentionBlock(edge_index, 8, 3).double()
    nodes = initial_nodes.clone().requires_grad_()
    edges = initial_edges.clone().requires_grad_()

    batched = block(nodes, edges)
    individual = torch.stack(
        [
            block(nodes[sample], edges[sample])
            for sample in range(nodes.shape[0])
        ]
    )
    assert_close(batched, individual)

    # Only sample 0 participates in this loss.
    loss = (batched[0] - target[0]).square().mean()
    node_grad, edge_grad = torch.autograd.grad(loss, (nodes, edges))
    assert node_grad[0].abs().sum().item() > 1e-8
    assert edge_grad[0].abs().sum().item() > 1e-8
    assert torch.equal(node_grad[1], torch.zeros_like(node_grad[1]))
    assert torch.equal(edge_grad[1], torch.zeros_like(edge_grad[1]))


@pytest.mark.parametrize("shift", [-10000.0, 10000.0])
def test_extreme_scores_match_reference_without_overflow(graph_case, shift):
    """Huge common score offsets must cancel within each neighborhood.

    Unlike a saturated one-hot fixture, this keeps competing edges active,
    so score derivatives must still be finite and nonzero.
    """
    edge_index, initial_nodes, initial_edges, target = graph_case
    torch.manual_seed(61)
    actual = SparseGraphAttentionBlock(edge_index, 8, 3).double()
    with torch.no_grad():
        actual.edge_bias.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.2, -0.1],
                    [-1.0, 0.1, 0.3],
                    [0.5, -0.4, 0.2],
                    [-0.5, 0.3, -0.2],
                ],
                dtype=torch.float64,
            )
        )
        baseline = actual(initial_nodes, initial_edges)
    reference = copy.deepcopy(actual)

    shifted_edges = initial_edges.clone()
    shifted_edges[:, :, 0] += shift
    nodes = initial_nodes.clone().requires_grad_()
    edges = shifted_edges.clone().requires_grad_()
    ref_nodes = initial_nodes.clone().requires_grad_()
    ref_edges = shifted_edges.clone().requires_grad_()
    bias = actual.edge_bias(edges)
    assert bias.max().item() > 1000.0
    assert bias.min().item() < -1000.0

    output = actual(nodes, edges)
    ref_output = reference_block(reference, ref_nodes, ref_edges, edge_index)
    assert torch.isfinite(output).all()
    assert_close(output, baseline)
    assert_close(output, ref_output)

    loss = (output - target).square().mean()
    ref_loss = (ref_output - target).square().mean()
    assert_close(loss, ref_loss)
    gradients = torch.autograd.grad(loss, (nodes, edges, *actual.parameters()))
    ref_gradients = torch.autograd.grad(
        ref_loss, (ref_nodes, ref_edges, *reference.parameters())
    )
    for gradient, ref_gradient in zip(gradients, ref_gradients):
        assert torch.isfinite(gradient).all()
        assert torch.isfinite(ref_gradient).all()
        assert_close(gradient, ref_gradient)
    assert gradients[1].abs().sum().item() > 1e-8


def test_empty_graph_has_only_the_nodewise_ffn(graph_case):
    """No edges means zero attention contribution for every receiver."""
    _, initial_nodes, _, _ = graph_case
    torch.manual_seed(67)
    block = SparseGraphAttentionBlock(
        torch.empty(2, 0, dtype=torch.long), 8, 3
    ).double()
    nodes = initial_nodes.clone().requires_grad_()
    edges = torch.empty(2, 0, 3, dtype=torch.float64, requires_grad=True)

    output = block(nodes, edges)
    expected = nodes + block.ffn(block.norm2(nodes))
    assert_close(output, expected)
    output.square().mean().backward()
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
    assert edges.grad is not None and edges.grad.shape == edges.shape

    for module in (
        block.norm1,
        block.q_proj,
        block.k_proj,
        block.v_proj,
        block.edge_bias,
        block.out_proj,
    ):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))


def test_zero_depth_preserves_inputs_and_gradients(graph_case):
    """The identity processor preserves both objects and autograd paths."""
    edge_index, initial_nodes, initial_edges, _ = graph_case
    processor = SparseGraphTransformerProcessor(
        edge_index, hidden_dim=8, edge_dim=3, num_layers=0
    )
    nodes = initial_nodes.clone().requires_grad_()
    edges = initial_edges.clone().requires_grad_()

    output, returned_edges = processor(nodes, edges)
    assert output is nodes
    assert returned_edges is edges
    assert sum(p.numel() for p in processor.parameters()) == 0

    (output.sum() + returned_edges.sum()).backward()
    assert torch.equal(nodes.grad, torch.ones_like(nodes))
    assert torch.equal(edges.grad, torch.ones_like(edges))


def test_shared_geometry_keeps_gradients_through_multiple_blocks(graph_case):
    """An upstream edge embedder receives summed batch contributions."""
    edge_index, initial_nodes, initial_edges, target = graph_case
    torch.manual_seed(71)
    actual = SparseGraphTransformerProcessor(
        edge_index, hidden_dim=8, edge_dim=3, num_layers=2
    ).double()
    embedder = torch.nn.Linear(3, 3, bias=False).double()
    reference = copy.deepcopy(actual)
    ref_embedder = copy.deepcopy(embedder)

    nodes = initial_nodes.clone().requires_grad_()
    ref_nodes = initial_nodes.clone().requires_grad_()
    geometry = initial_edges[0].clone().requires_grad_()
    ref_geometry = initial_edges[0].clone().requires_grad_()
    batch_size = nodes.shape[0]

    # Like static graph features: embed once, share across batch samples.
    edges = embedder(geometry).unsqueeze(0).expand(batch_size, -1, -1)
    ref_edges = (
        ref_embedder(ref_geometry).unsqueeze(0).expand(batch_size, -1, -1)
    )
    assert edges.stride(0) == 0
    edges.retain_grad()
    ref_edges.retain_grad()
    before = edges.detach().clone()

    output, returned_edges = actual(nodes, edges)
    ref_output, _ = reference_processor(
        reference, ref_nodes, ref_edges, edge_index
    )
    assert returned_edges is edges
    torch.testing.assert_close(edges, before, atol=0, rtol=0)
    assert_close(output, ref_output)

    # No direct edge loss: these derivatives must come through attention.
    (output - target).square().mean().backward()
    (ref_output - target).square().mean().backward()
    assert_close(nodes.grad, ref_nodes.grad)
    assert_close(edges.grad, ref_edges.grad)
    assert_close(geometry.grad, ref_geometry.grad)
    assert_close(embedder.weight.grad, ref_embedder.weight.grad)
    assert geometry.grad.abs().sum().item() > 1e-8
    assert embedder.weight.grad.norm().item() > 1e-8
    for block in actual.blocks:
        assert block.edge_bias.weight.grad.norm().item() > 1e-8


def test_incompatible_head_width_is_rejected(graph_case):
    """A width of seven cannot be split evenly into four heads."""
    edge_index, _, _, _ = graph_case
    with pytest.raises(ValueError, match="divisible by num_heads"):
        SparseGraphAttentionBlock(
            edge_index, hidden_dim=7, edge_dim=3, num_heads=4
        )
