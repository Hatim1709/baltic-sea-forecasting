"""Reference tests for BipartiteEdgeAttention.

The reference is an explicit loop over samples, receivers, heads and incoming
edges. It reuses only the layer's parameter tensors and its node MLP, so it
checks the attention equations independently of the gather/scatter code:

    q_i  = W_Q h_i + b_Q
    k_ji = W_K h_j  (+ W_E  e_ji)        v_ji = W_V h_j  (+ W_E' e_ji)
    s_ji = q_i . k_ji / sqrt(d_k)  (+ w_h . e_ji)
    a_i  = concat_heads  sum_{j->i} softmax_j(s_ji) v_ji
    h_i' = h_i + MLP([h_i, a_i])   or   a_i + MLP([h_i, a_i])

Most tests run in float64 so that the comparison is tight and one default
AdamW step is not disturbed by float32 round-off.
"""

# Standard library
import copy
import math

# Third-party
import pytest
import torch
import torch.nn.functional as F

# First-party
from neural_lam.bipartite_attention import BipartiteEdgeAttention
from neural_lam.sparse_attention import StaticEdgeAttention

# Bipartite fixture: 5 senders, 6 receivers, different widths everywhere.
# In-degrees of the receivers: 4, 3, 0, 1, 3, 0. Receivers 2 and 5 have no
# incoming edge, and 5 is the last index, so a receiver count derived from
# the edge list (max + 1 = 5) would be wrong. Edge 3->1 appears twice.
SENDERS = [0, 1, 2, 4, 3, 3, 1, 0, 2, 4, 1]
RECEIVERS = [0, 0, 0, 0, 1, 1, 1, 3, 4, 4, 4]
SINGLE_EDGE = 7  # the only edge into receiver 3
ISOLATED = (2, 5)
B, N_SEND, N_REC = 2, 5, 6
SEND_DIM, REC_DIM, EDGE_DIM, HEADS = 12, 8, 6, 2

MODES = [
    (edge_mode, residual)
    for edge_mode in ("key_value", "bias")
    for residual in ("receiver", "aggregate")
]


def assert_close(actual, expected, atol=1e-10, rtol=1e-9):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.fixture
def edge_index():
    return torch.tensor([SENDERS, RECEIVERS], dtype=torch.long)


def make_layer(edge_index, edge_mode, residual, seed=0, **kwargs):
    torch.manual_seed(seed)
    return BipartiteEdgeAttention(
        edge_index,
        SEND_DIM,
        REC_DIM,
        EDGE_DIM,
        HEADS,
        num_rec=N_REC,
        residual=residual,
        edge_mode=edge_mode,
        **kwargs,
    ).double()


def make_inputs(seed, out_dim, num_edges=len(SENDERS)):
    gen = torch.Generator().manual_seed(seed)
    send = torch.randn(B, N_SEND, SEND_DIM, generator=gen).double()
    rec = torch.randn(B, N_REC, REC_DIM, generator=gen).double()
    edges = torch.randn(B, num_edges, EDGE_DIM, generator=gen).double()
    target = torch.randn(B, N_REC, out_dim, generator=gen).double()
    return send, rec, edges, target


def reference_aggregate(layer, send, rec, edges, edge_index):
    """a_i for every sample and receiver, by explicit loops."""
    sender, receiver = edge_index.tolist()
    d_k = layer.head_dim
    query = F.linear(rec, layer.q_proj.weight, layer.q_proj.bias)
    key = F.linear(send, layer.k_proj.weight, layer.k_proj.bias)
    value = F.linear(send, layer.v_proj.weight, layer.v_proj.bias)
    key_value = layer.edge_mode == "key_value"
    if key_value:
        key_edge = F.linear(edges, layer.k_edge.weight)
        value_edge = F.linear(edges, layer.v_edge.weight)
    else:
        score_bias = F.linear(edges, layer.edge_bias.weight)  # (B, M, H)

    samples = []
    for b in range(rec.shape[0]):
        rows = []
        for i in range(rec.shape[1]):
            incoming = [m for m, r in enumerate(receiver) if r == i]
            if not incoming:
                rows.append(rec.new_zeros(layer.out_dim))
                continue
            heads = []
            for h in range(layer.num_heads):
                sl = slice(h * d_k, (h + 1) * d_k)
                keys, values, biases = [], [], []
                for m in incoming:
                    k_m = key[b, sender[m], sl]
                    v_m = value[b, sender[m], sl]
                    if key_value:
                        k_m = k_m + key_edge[b, m, sl]
                        v_m = v_m + value_edge[b, m, sl]
                        biases.append(rec.new_zeros(()))
                    else:
                        biases.append(score_bias[b, m, h])
                    keys.append(k_m)
                    values.append(v_m)
                scores = torch.stack(keys) @ query[b, i, sl] / math.sqrt(d_k)
                weights = torch.softmax(scores + torch.stack(biases), dim=0)
                heads.append(weights @ torch.stack(values))
            rows.append(torch.cat(heads))
        samples.append(torch.stack(rows))
    return torch.stack(samples)


def reference_forward(layer, send, rec, edges, edge_index):
    aggr = reference_aggregate(layer, send, rec, edges, edge_index)
    update = layer.aggr_mlp(torch.cat((rec, aggr), dim=-1))
    return (rec if layer.residual == "receiver" else aggr) + update


def loss_fn(output, target):
    weights = torch.linspace(0.5, 1.5, output.shape[-1], dtype=output.dtype)
    return ((output - target) ** 2 * weights).mean()


@pytest.mark.parametrize("edge_mode,residual", MODES)
def test_training_step_matches_reference(edge_index, edge_mode, residual):
    """Outputs, loss, input + parameter gradients and one AdamW update."""
    layer = make_layer(edge_index, edge_mode, residual, seed=41)
    reference = copy.deepcopy(layer)
    out_dim = REC_DIM if residual == "receiver" else SEND_DIM
    assert layer.out_dim == out_dim
    inputs = make_inputs(1, out_dim)
    target = inputs[3]

    send, rec, edges = (t.clone().requires_grad_() for t in inputs[:3])
    r_send, r_rec, r_edges = (t.clone().requires_grad_() for t in inputs[:3])

    out = layer(send, rec, edges)
    ref_out = reference_forward(reference, r_send, r_rec, r_edges, edge_index)
    assert out.shape == (B, N_REC, out_dim)
    assert_close(out, ref_out)

    loss = loss_fn(out, target)
    ref_loss = loss_fn(ref_out, target)
    assert_close(loss, ref_loss)

    loss.backward()
    ref_loss.backward()
    assert_close(send.grad, r_send.grad)
    assert_close(rec.grad, r_rec.grad)
    assert_close(edges.grad, r_edges.grad)
    ref_params = dict(reference.named_parameters())
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().max() > 0, name
        assert_close(param.grad, ref_params[name].grad)

    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    opt.step()
    ref_opt.step()
    for name, param in layer.named_parameters():
        assert_close(param, ref_params[name])


@pytest.mark.parametrize("edge_mode,residual", MODES)
def test_receivers_without_edges(edge_index, edge_mode, residual):
    """a_i = 0, so the output is h_i + MLP([h_i, 0]) or MLP([h_i, 0])."""
    layer = make_layer(edge_index, edge_mode, residual, seed=43)
    send, rec, edges, _ = make_inputs(2, layer.out_dim)
    out = layer(send, rec, edges)
    for i in ISOLATED:
        zero_aggr = rec.new_zeros(B, layer.out_dim)
        update = layer.aggr_mlp(torch.cat((rec[:, i], zero_aggr), dim=-1))
        expected = rec[:, i] + update if residual == "receiver" else update
        assert_close(out[:, i], expected)


def test_receiver_count_comes_from_the_inputs(edge_index):
    """Without num_rec, trailing receivers without edges still work."""
    torch.manual_seed(47)
    layer = BipartiteEdgeAttention(
        edge_index, SEND_DIM, REC_DIM, EDGE_DIM, HEADS
    ).double()
    assert layer.min_rec == N_REC - 1  # the edge list alone is one short
    send, rec, edges, _ = make_inputs(3, REC_DIM)
    out = layer(send, rec, edges)
    assert out.shape == (B, N_REC, REC_DIM)
    assert_close(out, reference_forward(layer, send, rec, edges, edge_index))


@pytest.mark.parametrize("edge_mode,residual", MODES)
def test_edge_order_permutation(edge_index, edge_mode, residual):
    """Reordering edges (with their features) must not change anything."""
    layer = make_layer(edge_index, edge_mode, residual, seed=53)
    perm = torch.randperm(edge_index.shape[1])
    permuted = make_layer(edge_index[:, perm], edge_mode, residual)
    permuted.load_state_dict(layer.state_dict())

    inputs = make_inputs(4, layer.out_dim)
    target = inputs[3]
    send, rec, edges = (t.clone().requires_grad_() for t in inputs[:3])
    p_send, p_rec = (t.clone().requires_grad_() for t in inputs[:2])
    p_edges = inputs[2][:, perm].clone().requires_grad_()

    out = layer(send, rec, edges)
    p_out = permuted(p_send, p_rec, p_edges)
    assert_close(out, p_out)

    loss_fn(out, target).backward()
    loss_fn(p_out, target).backward()
    assert_close(send.grad, p_send.grad)
    assert_close(rec.grad, p_rec.grad)
    assert_close(edges.grad[:, perm], p_edges.grad)
    p_params = dict(permuted.named_parameters())
    for name, param in layer.named_parameters():
        assert_close(param.grad, p_params[name].grad)


@pytest.mark.parametrize("edge_mode,residual", MODES)
def test_samples_are_independent(edge_index, edge_mode, residual):
    layer = make_layer(edge_index, edge_mode, residual, seed=59)
    send, rec, edges, _ = make_inputs(5, layer.out_dim)
    batched = layer(send, rec, edges)
    for b in range(B):
        one = slice(b, b + 1)
        assert_close(batched[one], layer(send[one], rec[one], edges[one]))


def test_bias_mode_uses_edges_only_in_the_weights(edge_index):
    """A receiver with one incoming edge has weight 1 whatever the score.

    In bias mode that edge's features therefore cannot affect the output and
    get an exactly zero gradient. In key_value mode they enter the value.
    """
    for edge_mode, expect_zero in (("bias", True), ("key_value", False)):
        layer = make_layer(edge_index, edge_mode, "receiver", seed=61)
        send, rec, edges, target = make_inputs(6, REC_DIM)
        edges.requires_grad_()
        loss_fn(layer(send, rec, edges), target).backward()
        single = edges.grad[:, SINGLE_EDGE].abs().max().item()
        others = edges.grad[:, :SINGLE_EDGE].abs().max().item()
        assert others > 0
        assert (single == 0) == expect_zero, (edge_mode, single)


def test_reproduces_static_edge_attention():
    """Same node set, one width, receiver residual: the decoder layer."""
    senders = [1, 2, 3, 5, 0, 4, 4, 0, 2, 3, 1, 4, 0]
    receivers = [0, 0, 0, 0, 1, 1, 1, 3, 4, 4, 4, 5, 5]
    index = torch.tensor([senders, receivers], dtype=torch.long)
    nodes_n, dim, heads = 6, 8, 2

    torch.manual_seed(67)
    original = StaticEdgeAttention(index, dim, heads)
    general = BipartiteEdgeAttention(
        index,
        dim,
        dim,
        dim,
        heads,
        key_bias=True,
        value_bias=True,
        return_edges=True,
    )
    general.load_state_dict(original.state_dict())  # same parameter names

    gen = torch.Generator().manual_seed(7)
    nodes0 = torch.randn(B, nodes_n, dim, generator=gen)
    edges0 = torch.randn(B, len(senders), dim, generator=gen)
    target = torch.randn(B, nodes_n, dim, generator=gen)

    nodes = nodes0.clone().requires_grad_()
    edges = edges0.clone().requires_grad_()
    o_nodes = nodes0.clone().requires_grad_()
    o_edges = edges0.clone().requires_grad_()

    out, out_edges = general(nodes, nodes, edges)
    o_out, _ = original(o_nodes, o_nodes, o_edges)
    assert out_edges is edges
    assert_close(out, o_out, atol=1e-5, rtol=1e-4)

    loss_fn(out, target).backward()
    loss_fn(o_out, target).backward()
    assert_close(nodes.grad, o_nodes.grad, atol=1e-5, rtol=1e-4)
    assert_close(edges.grad, o_edges.grad, atol=1e-5, rtol=1e-4)
    o_params = dict(original.named_parameters())
    for name, param in general.named_parameters():
        assert_close(param.grad, o_params[name].grad, atol=1e-5, rtol=1e-4)


def test_default_parameters_and_count(edge_index):
    """No key or value bias by default; parameter count as documented."""
    for edge_mode in ("key_value", "bias"):
        layer = make_layer(edge_index, edge_mode, "aggregate")
        assert layer.k_proj.bias is None and layer.v_proj.bias is None
        out = SEND_DIM
        attention = REC_DIM * out + out + 2 * SEND_DIM * out
        if edge_mode == "key_value":
            attention += 2 * EDGE_DIM * out
        else:
            attention += EDGE_DIM * HEADS
        # Linear, SiLU, Linear, LayerNorm
        mlp = (REC_DIM + out) * out + out + out * out + out + 2 * out
        assert sum(p.numel() for p in layer.parameters()) == attention + mlp


def test_invalid_configuration_is_rejected(edge_index):
    def build(**kwargs):
        args = {"num_heads": HEADS}
        args.update(kwargs)
        return BipartiteEdgeAttention(
            edge_index, SEND_DIM, REC_DIM, EDGE_DIM, **args
        )

    with pytest.raises(ValueError):
        build(num_heads=3)  # 8 is not divisible by 3
    with pytest.raises(ValueError):
        build(residual="sender")
    with pytest.raises(ValueError):
        build(edge_mode="value")
    with pytest.raises(ValueError):
        build(num_rec=N_REC - 2)  # edge list needs at least 5 receivers

    layer = build(num_rec=N_REC).double()
    send, rec, edges, _ = make_inputs(8, REC_DIM)
    with pytest.raises(ValueError):
        layer(send, rec[:, :-1], edges)  # one receiver missing
    with pytest.raises(ValueError):
        layer(send[:, :-1], rec, edges)  # sender index out of range


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("edge_mode,residual", MODES)
def test_cuda_bf16_autocast_training_step(edge_index, edge_mode, residual):
    """bf16 autocast forward/backward stays close to the float32 loop."""
    device = torch.device("cuda")
    layer = make_layer(edge_index, edge_mode, residual, seed=71)
    layer = layer.float().to(device)
    inputs = [t.float().to(device) for t in make_inputs(9, layer.out_dim)]
    target = inputs[3]

    send, rec, edges = (t.clone().requires_grad_() for t in inputs[:3])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(send, rec, edges)
        loss = loss_fn(out.float(), target)
    loss.backward()

    r_send, r_rec, r_edges = (t.clone().requires_grad_() for t in inputs[:3])
    ref_out = reference_forward(
        layer, r_send, r_rec, r_edges, edge_index.to(device)
    )
    ref_loss = loss_fn(ref_out, target)
    ref_loss.backward()

    assert torch.isfinite(out).all()
    tol = {"atol": 5e-2, "rtol": 5e-2}
    assert_close(out.float(), ref_out, **tol)
    assert_close(loss, ref_loss, **tol)
    assert_close(send.grad, r_send.grad, **tol)
    assert_close(rec.grad, r_rec.grad, **tol)
    assert_close(edges.grad, r_edges.grad, **tol)
