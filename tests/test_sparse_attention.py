"""Reference tests for StaticEdgeAttention (decoder option A).

The reference below is an explicit loop over samples, receivers, heads and
incoming edges. It reuses only the layer's parameter tensors and its node MLP
(``aggr_mlp``, the existing InteractionNet MLP), so it checks the attention
equations independently of PyG MessagePassing:

    q_i  = W_Q h_i + b_Q
    k_ji = W_K h_j + b_K + W_E  e_ji
    v_ji = W_V h_j + b_V + W_E' e_ji
    a_i  = concat_heads  sum_{j->i} softmax_j(q_i . k_ji / sqrt(d_k)) v_ji
    h_i' = h_i + MLP([h_i, a_i]),   e' = e (unchanged)

Shapes follow the GraphEFM decoder: nodes (B, N, d), edges (B, M, d).
"""

# Standard library
import copy
import math

# Third-party
import pytest
import torch
import torch.nn.functional as F

# First-party
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet
from neural_lam.sparse_attention import StaticEdgeAttention, build_intra_layer

ATOL = 1e-5
RTOL = 1e-4

# Unequal in-degrees (4, 3, 0, 1, 3, 2), a duplicated edge 4->1 and an
# isolated receiver (node 2). The highest node index (5) receives edges, as
# StaticEdgeAttention and InteractionNet derive num_rec = max(receiver) + 1.
SENDERS = [1, 2, 3, 5, 0, 4, 4, 0, 2, 3, 1, 4, 0]
RECEIVERS = [0, 0, 0, 0, 1, 1, 1, 3, 4, 4, 4, 5, 5]
B, N, D, H = 2, 6, 8, 2


def assert_close(actual, expected, atol=ATOL, rtol=RTOL):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.fixture
def edge_index():
    return torch.tensor([SENDERS, RECEIVERS], dtype=torch.long)


def make_inputs(seed, num_edges=len(SENDERS)):
    gen = torch.Generator().manual_seed(seed)
    nodes = torch.randn(B, N, D, generator=gen)
    edges = torch.randn(B, num_edges, D, generator=gen)
    target = torch.randn(B, N, D, generator=gen)
    return nodes, edges, target


def reference_forward(layer, nodes, edges, edge_index):
    """Loop implementation using only the layer's parameters and node MLP."""
    sender, receiver = edge_index.tolist()
    d_k = layer.d_k
    q = F.linear(nodes, layer.q_proj.weight, layer.q_proj.bias)
    k = F.linear(nodes, layer.k_proj.weight, layer.k_proj.bias)
    v = F.linear(nodes, layer.v_proj.weight, layer.v_proj.bias)
    k_e = F.linear(edges, layer.k_edge.weight)
    v_e = F.linear(edges, layer.v_edge.weight)

    samples = []
    for b in range(nodes.shape[0]):
        rows = []
        for i in range(nodes.shape[1]):
            incoming = [m for m, r in enumerate(receiver) if r == i]
            if not incoming:
                rows.append(nodes.new_zeros(nodes.shape[-1]))
                continue
            heads = []
            for h in range(layer.num_heads):
                sl = slice(h * d_k, (h + 1) * d_k)
                keys = torch.stack(
                    [k[b, sender[m], sl] + k_e[b, m, sl] for m in incoming]
                )
                values = torch.stack(
                    [v[b, sender[m], sl] + v_e[b, m, sl] for m in incoming]
                )
                scores = keys @ q[b, i, sl] / math.sqrt(d_k)
                weights = torch.softmax(scores, dim=0)
                heads.append(weights @ values)
            rows.append(torch.cat(heads))
        samples.append(torch.stack(rows))
    aggregated = torch.stack(samples)
    return nodes + layer.aggr_mlp(torch.cat((nodes, aggregated), dim=-1)), edges


def loss_fn(output, target):
    weights = torch.linspace(0.5, 1.5, output.shape[-1], device=output.device)
    return ((output - target) ** 2 * weights).mean()


def test_training_step_matches_reference(edge_index):
    """Outputs, loss, input + parameter gradients and one AdamW update."""
    torch.manual_seed(41)
    layer = StaticEdgeAttention(edge_index, D, H, hidden_layers=1)
    reference = copy.deepcopy(layer)
    nodes0, edges0, target = make_inputs(1)

    nodes = nodes0.clone().requires_grad_()
    edges = edges0.clone().requires_grad_()
    ref_nodes = nodes0.clone().requires_grad_()
    ref_edges = edges0.clone().requires_grad_()

    out, out_edges = layer(nodes, nodes, edges)
    ref_out, _ = reference_forward(reference, ref_nodes, ref_edges, edge_index)

    assert out.shape == (B, N, D)
    assert out_edges is edges  # static edges: same tensor, same autograd node
    assert_close(out, ref_out)

    loss = loss_fn(out, target)
    ref_loss = loss_fn(ref_out, target)
    assert_close(loss, ref_loss)

    loss.backward()
    ref_loss.backward()
    assert_close(nodes.grad, ref_nodes.grad)
    assert_close(edges.grad, ref_edges.grad)
    ref_params = dict(reference.named_parameters())
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert_close(param.grad, ref_params[name].grad)

    # eps=1e-6: k_proj.bias has a mathematically zero gradient (softmax is
    # shift-invariant per receiver), so its float noise must not dominate the
    # first Adam step, which divides by |g| + eps.
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, eps=1e-6)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3, eps=1e-6)
    opt.step()
    ref_opt.step()
    for name, param in layer.named_parameters():
        assert_close(param, ref_params[name])


def test_isolated_receiver_gets_mlp_of_zero_aggregate(edge_index):
    torch.manual_seed(43)
    layer = StaticEdgeAttention(edge_index, D, H)
    nodes, edges, _ = make_inputs(2)
    out, _ = layer(nodes, nodes, edges)
    h_isolated = nodes[:, 2]  # node 2 has no incoming edges
    expected = h_isolated + layer.aggr_mlp(
        torch.cat((h_isolated, torch.zeros_like(h_isolated)), dim=-1)
    )
    assert_close(out[:, 2], expected)


def test_edge_order_permutation(edge_index):
    """Reordering edges (with their features) must not change anything."""
    torch.manual_seed(47)
    layer = StaticEdgeAttention(edge_index, D, H)
    perm = torch.randperm(edge_index.shape[1])
    permuted = StaticEdgeAttention(edge_index[:, perm], D, H)
    permuted.load_state_dict(layer.state_dict())

    nodes0, edges0, target = make_inputs(3)
    nodes = nodes0.clone().requires_grad_()
    edges = edges0.clone().requires_grad_()
    p_nodes = nodes0.clone().requires_grad_()
    p_edges = edges0[:, perm].clone().requires_grad_()

    out, _ = layer(nodes, nodes, edges)
    p_out, _ = permuted(p_nodes, p_nodes, p_edges)
    assert_close(out, p_out)

    loss_fn(out, target).backward()
    loss_fn(p_out, target).backward()
    assert_close(nodes.grad, p_nodes.grad)
    assert_close(edges.grad[:, perm], p_edges.grad)
    p_params = dict(permuted.named_parameters())
    for name, param in layer.named_parameters():
        assert_close(param.grad, p_params[name].grad)


def test_samples_are_independent(edge_index):
    torch.manual_seed(53)
    layer = StaticEdgeAttention(edge_index, D, H)
    nodes, edges, _ = make_inputs(4)
    batched, _ = layer(nodes, nodes, edges)
    for b in range(B):
        single, _ = layer(nodes[b : b + 1], nodes[b : b + 1], edges[b : b + 1])
        assert_close(batched[b : b + 1], single)


@pytest.mark.parametrize("shift", [1e2, 1e4])
def test_large_scores_stay_finite_and_match_reference(edge_index, shift):
    """Scaled queries give very peaked softmax weights without overflow."""
    torch.manual_seed(59)
    layer = StaticEdgeAttention(edge_index, D, H)
    with torch.no_grad():
        layer.q_proj.weight.mul_(shift)
    nodes, edges, _ = make_inputs(5)
    out, _ = layer(nodes, nodes, edges)
    ref_out, _ = reference_forward(layer, nodes, edges, edge_index)
    assert torch.isfinite(out).all()
    assert_close(out, ref_out, atol=1e-4, rtol=1e-4)


def test_gnn_seq_with_attention_matches_stacked_reference(edge_index):
    """The decoder hook: make_gnn_seq(layer_type='attention'), 2 layers."""
    torch.manual_seed(61)
    seq = utils.make_gnn_seq(
        edge_index, 2, 1, D, layer_type="attention", attention_heads=H
    )
    layers = [m for m in seq.modules() if isinstance(m, StaticEdgeAttention)]
    assert len(layers) == 2

    nodes0, edges0, target = make_inputs(6)
    nodes = nodes0.clone().requires_grad_()
    edges = edges0.clone().requires_grad_()
    ref_nodes = nodes0.clone().requires_grad_()
    ref_edges = edges0.clone().requires_grad_()

    out, out_edges = seq(nodes, edges)
    ref_out = ref_nodes
    for layer in layers:
        ref_out, _ = reference_forward(layer, ref_out, ref_edges, edge_index)
    assert out_edges is edges
    assert_close(out, ref_out)

    loss_fn(out, target).backward()
    nodes_grad, edges_grad = nodes.grad.clone(), edges.grad.clone()
    param_grads = [p.grad.clone() for p in seq.parameters()]
    seq.zero_grad()
    loss_fn(ref_out, target).backward()
    assert_close(nodes_grad, ref_nodes.grad)
    assert_close(edges_grad, ref_edges.grad)
    for grad, param in zip(param_grads, seq.parameters()):
        assert_close(grad, param.grad)


def test_default_layer_is_unchanged_interaction_net(edge_index):
    """Regression: the default hook path builds the original InteractionNet."""
    torch.manual_seed(67)
    built = build_intra_layer("interaction", edge_index, D, 1)
    torch.manual_seed(67)
    original = InteractionNet(edge_index, D, hidden_layers=1)
    assert type(built) is InteractionNet
    built_state, original_state = built.state_dict(), original.state_dict()
    assert built_state.keys() == original_state.keys()
    for key in original_state:
        torch.testing.assert_close(
            built_state[key], original_state[key], atol=0, rtol=0
        )

    nodes, edges, _ = make_inputs(7)
    built_out = built(nodes, nodes, edges)
    original_out = original(nodes, nodes, edges)
    for actual, expected in zip(built_out, original_out):  # (nodes, edges)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_head_configuration_is_rejected(edge_index):
    with pytest.raises(ValueError):
        build_intra_layer("attention", edge_index, D, 1, attention_heads=None)
    with pytest.raises(AssertionError):
        StaticEdgeAttention(edge_index, D, 3)  # 8 is not divisible by 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_bf16_autocast_training_step(edge_index):
    """bf16 autocast forward/backward stays close to the float32 reference."""
    device = torch.device("cuda")
    torch.manual_seed(71)
    layer = StaticEdgeAttention(edge_index, D, H).to(device)
    reference = copy.deepcopy(layer)
    nodes0, edges0, target = (t.to(device) for t in make_inputs(8))

    nodes = nodes0.clone().requires_grad_()
    edges = edges0.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = layer(nodes, nodes, edges)
        loss = loss_fn(out.float(), target)
    loss.backward()

    ref_nodes = nodes0.clone().requires_grad_()
    ref_edges = edges0.clone().requires_grad_()
    ref_out, _ = reference_forward(
        reference, ref_nodes, ref_edges, edge_index.to(device)
    )
    ref_loss = loss_fn(ref_out, target)
    ref_loss.backward()

    assert torch.isfinite(out).all()
    assert_close(out.float(), ref_out, atol=5e-2, rtol=5e-2)
    assert_close(loss, ref_loss, atol=5e-2, rtol=5e-2)
    assert_close(nodes.grad, ref_nodes.grad, atol=5e-2, rtol=5e-2)
    assert_close(edges.grad, ref_edges.grad, atol=5e-2, rtol=5e-2)
