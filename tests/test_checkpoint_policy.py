"""
Tests for the configurable gradient checkpointing (--ckpt_policy).

Two things are checked:
1. The policy logic: which policy switches which kind of checkpointing on.
2. Checkpointing never changes results. Outputs and gradients (of inputs and
   parameters) must be the same with and without per-layer checkpointing, and
   the recomputation must really take place (counted through a forward hook).

Everything runs on CPU in double precision on a tiny random graph.
"""

# Third-party
import pytest
import torch
import torch.utils.checkpoint
from torch import nn

# First-party
from neural_lam import checkpointing, utils
from neural_lam.interaction_net import (
    FlexibleNet,
    InteractionNet,
    PropagationNet,
)
from neural_lam.models.ar_model import ARModel

BATCH = 2
NUM_NODES = 6
DIM = 8


def _same_level_edges(num_nodes=NUM_NODES, num_extra=8, seed=0):
    """Edges of a small graph where every node has an incoming edge"""
    gen = torch.Generator().manual_seed(seed)
    ring = torch.stack(
        (torch.arange(num_nodes), (torch.arange(num_nodes) + 1) % num_nodes)
    )
    extra = torch.randint(0, num_nodes, (2, num_extra), generator=gen)
    return torch.cat((ring, extra), dim=1)


def _rand(*shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


def _make_cases():
    """
    One entry per layer type: (name, layer, inputs), where inputs are the
    tensors passed to layer(send, rec, edge).
    """
    edge_index = _same_level_edges()
    num_edges = edge_index.shape[1]
    nodes = _rand(BATCH, NUM_NODES, DIM, seed=1)
    edges = _rand(BATCH, num_edges, DIM, seed=2)

    # Bipartite graph for FlexibleNet, with different node and edge widths
    num_send, num_rec, d_send, d_rec, d_edge = 7, 5, 6, 4, 3
    bip_edges = torch.stack(
        (
            torch.randint(0, num_send, (12,)),
            torch.cat((torch.arange(num_rec), torch.randint(0, num_rec, (7,)))),
        )
    )
    bip_send = _rand(BATCH, num_send, d_send, seed=3)
    bip_rec = _rand(BATCH, num_rec, d_rec, seed=4)
    bip_edge_rep = _rand(BATCH, bip_edges.shape[1], d_edge, seed=5)

    cases = [
        (
            "interaction",
            InteractionNet(edge_index, DIM),
            (nodes, nodes.clone(), edges),
        ),
        (
            "interaction_no_edge_update",
            InteractionNet(edge_index, DIM, update_edges=False),
            (nodes, nodes.clone(), edges),
        ),
        (
            "propagation",
            PropagationNet(edge_index, DIM),
            (nodes, nodes.clone(), edges),
        ),
        (
            "flexible_propagation",
            FlexibleNet(
                bip_edges,
                d_send,
                d_rec,
                d_edge,
                num_rec=num_rec,
                propagation=True,
            ),
            (bip_send, bip_rec, bip_edge_rep),
        ),
        (
            "flexible_non_propagation",
            FlexibleNet(
                bip_edges,
                d_send,
                d_rec,
                d_edge,
                num_rec=num_rec,
                propagation=False,
                aggr="sum",
            ),
            (bip_send, bip_rec, bip_edge_rep),
        ),
    ]
    for _, layer, _ in cases:
        layer.double().train()
    return cases


def _forward_backward(layer, inputs, seed=0):
    """
    Run layer on copies of inputs, backpropagate a scalar loss, and return
    (outputs, gradients of the inputs, gradients of the parameters).
    """
    inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
    layer.zero_grad()
    torch.manual_seed(seed)  # for dropout inside MLPs, if any
    outputs = layer(*inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    loss = sum((out**2).sum() for out in outputs)
    loss.backward()
    input_grads = [x.grad.clone() for x in inputs]
    param_grads = [p.grad.clone() for p in layer.parameters()]
    return [o.detach() for o in outputs], input_grads, param_grads


def _count_edge_mlp_calls(layer):
    """Count forward calls of the edge MLP; returns a one-element list"""
    calls = [0]
    layer.edge_mlp.register_forward_hook(
        lambda *_: calls.__setitem__(0, calls[0] + 1)
    )
    return calls


@pytest.mark.parametrize(
    "policy, force, steps, expected",
    [
        # auto = behaviour before --ckpt_policy existed
        ("auto", False, 1, (False, False)),
        ("auto", False, 2, (True, False)),
        ("auto", True, 1, (True, False)),
        ("off", True, 7, (False, False)),
        ("unroll", False, 1, (True, False)),
        ("internal", False, 5, (False, True)),
        ("unroll+internal", False, 1, (True, True)),
    ],
)
def test_policy_flags(policy, force, steps, expected):
    assert checkpointing.policy_flags(policy, force, steps) == expected


def test_policy_flags_unknown_policy():
    with pytest.raises(ValueError):
        checkpointing.policy_flags("everything", False, 1)


def test_internal_checkpointing_is_off_by_default():
    for _, layer, _ in _make_cases():
        assert layer.ckpt_inner is False


@pytest.mark.parametrize("case_index", range(5))
def test_layer_results_unchanged_by_checkpointing(case_index):
    """Outputs and all gradients must agree with and without checkpointing"""
    _, layer, inputs = _make_cases()[case_index]
    ref_out, ref_in_grads, ref_param_grads = _forward_backward(layer, inputs)

    layer.ckpt_inner = True
    out, in_grads, param_grads = _forward_backward(layer, inputs)

    assert len(out) == len(ref_out)
    for new, ref in zip(
        out + in_grads + param_grads, ref_out + ref_in_grads + ref_param_grads
    ):
        torch.testing.assert_close(new, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case_index", range(5))
def test_layer_is_recomputed_exactly_once_in_backward(case_index):
    """The edge MLP runs once without and twice with checkpointing"""
    _, layer, inputs = _make_cases()[case_index]
    calls = _count_edge_mlp_calls(layer)

    _forward_backward(layer, inputs)
    assert calls[0] == 1

    calls[0] = 0
    layer.ckpt_inner = True
    _forward_backward(layer, inputs)
    assert calls[0] == 2


def test_no_checkpointing_in_eval_mode_or_without_grad():
    _, layer, inputs = _make_cases()[0]
    layer.ckpt_inner = True
    calls = _count_edge_mlp_calls(layer)

    layer.eval()
    layer(*inputs)
    assert calls[0] == 1

    layer.train()
    with torch.no_grad():
        layer(*inputs)
    assert calls[0] == 2


def test_random_dropout_inside_layer_is_replayed_identically():
    """
    The MLPs can contain MC dropout (random). The recomputation must draw the
    same random numbers, otherwise the gradients would be wrong.
    """
    _, layer, inputs = _make_cases()[0]
    utils.set_mc_dropout(layer, 0.3)
    ref = _forward_backward(layer, inputs, seed=7)
    # Dropout really is active: another seed gives other outputs
    other = _forward_backward(layer, inputs, seed=8)
    assert not torch.allclose(ref[0][0], other[0][0])

    layer.ckpt_inner = True
    new = _forward_backward(layer, inputs, seed=7)
    for new_t, ref_t in zip(new[0] + new[1] + new[2], ref[0] + ref[1] + ref[2]):
        torch.testing.assert_close(new_t, ref_t, rtol=1e-10, atol=1e-12)


def test_gnn_sequence_and_nested_checkpointing():
    """
    A stack of layers (as in the processor), also nested inside a step-level
    checkpoint, i.e. policy "unroll+internal". Results must be unchanged; the
    forward pass is then run three times in total (once, plus once per
    checkpoint level in backward).
    """
    edge_index = _same_level_edges()
    seq = utils.make_gnn_seq(edge_index, 2, 1, DIM).double().train()
    nodes = _rand(BATCH, NUM_NODES, DIM, seed=1)
    edges = _rand(BATCH, edge_index.shape[1], DIM, seed=2)

    def run(unroll):
        x = nodes.clone().requires_grad_(True)
        seq.zero_grad()
        if unroll:
            out, _ = torch.utils.checkpoint.checkpoint(
                seq, x, edges, use_reentrant=False
            )
        else:
            out, _ = seq(x, edges)
        (out**2).sum().backward()
        return [x.grad.clone()] + [p.grad.clone() for p in seq.parameters()]

    reference = run(unroll=False)

    first_layer = seq[0]
    calls = _count_edge_mlp_calls(first_layer)
    assert checkpointing.set_internal_checkpointing(seq, True) == 2
    nested = run(unroll=True)
    for new, ref in zip(nested, reference):
        torch.testing.assert_close(new, ref, rtol=1e-10, atol=1e-12)
    assert calls[0] == 3


def test_update_checkpointing_applies_policy_to_all_layers():
    """ARModel.update_checkpointing sets both the step and layer switches"""

    class Stub(nn.Module):
        """Minimal stand-in with the attributes update_checkpointing uses"""

        update_checkpointing = ARModel.update_checkpointing

        def __init__(self, policy, force):
            super().__init__()
            self.ckpt_policy = policy
            self.args = type("Args", (), {"grad_checkpointing": force})()
            self.grad_checkpointing = None
            edge_index = _same_level_edges()
            self.layers = nn.ModuleList(
                [
                    InteractionNet(edge_index, DIM),
                    PropagationNet(edge_index, DIM),
                    nn.Linear(DIM, DIM),  # not a GNN layer, must be skipped
                ]
            )

    stub = Stub("internal", False)
    stub.update_checkpointing(1)
    assert stub.grad_checkpointing is False
    assert [getattr(m, "ckpt_inner", None) for m in stub.layers] == [
        True,
        True,
        None,
    ]

    # Phase change under "auto": step-level on for 2 steps, layers untouched
    stub = Stub("auto", False)
    stub.update_checkpointing(1)
    assert stub.grad_checkpointing is False
    stub.update_checkpointing(2)
    assert stub.grad_checkpointing is True
    assert not any(getattr(m, "ckpt_inner", False) for m in stub.layers)

    # Going from "internal" phases back to none must switch layers off again
    stub = Stub("unroll+internal", False)
    stub.update_checkpointing(1)
    assert stub.grad_checkpointing is True
    assert all(m.ckpt_inner for m in stub.layers[:2])
    stub.ckpt_policy = "off"
    stub.update_checkpointing(1)
    assert stub.grad_checkpointing is False
    assert not any(getattr(m, "ckpt_inner", False) for m in stub.layers)


def _covering_edges(num_send, num_rec, num_extra, seed):
    """
    Random edges from num_send sender nodes to num_rec receiver nodes where
    every receiver gets at least one incoming edge (the layers infer the
    number of receivers from the edges).
    """
    gen = torch.Generator().manual_seed(seed)
    receivers = torch.cat(
        (
            torch.arange(num_rec),
            torch.randint(0, num_rec, (num_extra,), generator=gen),
        )
    )
    senders = torch.randint(0, num_send, (receivers.shape[0],), generator=gen)
    return torch.stack((senders, receivers))


def test_hierarchical_decoder_unchanged_by_checkpointing():
    """
    The real hierarchical decoder of Njord (grid-to-mesh, up/down sweep over
    three mesh levels, mesh-to-grid) on a small random graph: results and all
    gradients must be identical with per-layer checkpointing, and also when
    nested inside a step-level checkpoint (policy "unroll+internal").
    """
    # Imported here since it pulls in the model package
    # First-party
    from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder

    num_grid, level_sizes = 12, [8, 4, 2]
    d_h, d_grid, d_mesh, d_edge, d_latent, d_out = 8, 6, 5, 3, 3, 2
    intra_layers = 2

    g2m = _covering_edges(num_grid, level_sizes[0], 10, seed=1)
    m2g = _covering_edges(level_sizes[0], num_grid, 10, seed=2)
    m2m = [
        _covering_edges(n, n, n, seed=10 + i) for i, n in enumerate(level_sizes)
    ]
    up = [
        _covering_edges(level_sizes[i], level_sizes[i + 1], 2, seed=20 + i)
        for i in range(2)
    ]
    down = [
        _covering_edges(level_sizes[i + 1], level_sizes[i], 2, seed=30 + i)
        for i in range(2)
    ]

    decoder = (
        HiGraphLatentDecoder(
            g2m,
            m2m,
            m2g,
            up,
            down,
            hidden_dim=d_h,
            hidden_dim_grid=d_grid,
            hidden_dim_mesh_nodes=d_mesh,
            hidden_dim_edge=d_edge,
            latent_dim=d_latent,
            output_dim=d_out,
            intra_level_layers=intra_layers,
        )
        .double()
        .train()
    )

    seed = 100
    inputs = {
        "grid_rep": _rand(BATCH, num_grid, d_grid, seed=seed),
        "grid_rep_interior": _rand(BATCH, num_grid, d_grid, seed=seed + 1),
        "latent": _rand(BATCH, level_sizes[-1], d_latent, seed=seed + 2),
        "last_state": _rand(BATCH, num_grid, d_out, seed=seed + 3),
        "g2m": _rand(BATCH, g2m.shape[1], d_edge, seed=seed + 4),
        "m2g": _rand(BATCH, m2g.shape[1], d_edge, seed=seed + 5),
        "mesh0": _rand(BATCH, level_sizes[0], d_mesh, seed=seed + 6),
        "mesh1": _rand(BATCH, level_sizes[1], d_h, seed=seed + 7),
        "mesh2": _rand(BATCH, level_sizes[2], d_h, seed=seed + 8),
    }
    for i, edges in enumerate(m2m):
        inputs[f"m2m{i}"] = _rand(
            BATCH, edges.shape[1], d_h, seed=seed + 10 + i
        )
    for i, edges in enumerate(up):
        inputs[f"up{i}"] = _rand(BATCH, edges.shape[1], d_h, seed=seed + 20 + i)
    for i, edges in enumerate(down):
        inputs[f"down{i}"] = _rand(
            BATCH, edges.shape[1], d_h, seed=seed + 30 + i
        )

    def run(unroll_checkpoint):
        x = {k: v.clone().requires_grad_(True) for k, v in inputs.items()}

        def step(*tensors):
            t = dict(zip(x.keys(), tensors))
            graph_emb = {
                "g2m": t["g2m"],
                "m2g": t["m2g"],
                "m2m": [t[f"m2m{i}"] for i in range(3)],
                "mesh": [t["mesh0"], t["mesh1"], t["mesh2"]],
                "mesh_up": [t["up0"], t["up1"]],
                "mesh_down": [t["down0"], t["down1"]],
            }
            return decoder(
                t["grid_rep"],
                t["grid_rep_interior"],
                t["latent"],
                t["last_state"],
                graph_emb,
            )

        decoder.zero_grad()
        checkpointing.reset_counts()
        if unroll_checkpoint:
            mean, std = torch.utils.checkpoint.checkpoint(
                step, *x.values(), use_reentrant=False
            )
        else:
            mean, std = step(*x.values())
        ((mean**2).sum() + (std**2).sum()).backward()
        # The top-level mesh embedding is not used (the latent sample takes its
        # place), so it has no gradient: count that as zero
        input_grads = [
            torch.zeros_like(v) if v.grad is None else v.grad.clone()
            for v in x.values()
        ]
        param_grads = [p.grad.clone() for p in decoder.parameters()]
        run.counts = checkpointing.get_counts()
        return [mean.detach(), std.detach()] + input_grads + param_grads

    reference = run(unroll_checkpoint=False)
    # No checkpointing: every layer runs once (the "off" policy)
    assert run.counts["layer_calls"] == 1 + 1 + 2 + 2 + (3 + 2) * intra_layers
    assert run.counts["layer_blocks"] == 0

    # 1 grid-to-mesh + 1 mesh-to-grid + 2 up + 2 down
    # + 3 intra-level stacks going up and 2 going down, 2 layers each
    expected_layers = 1 + 1 + 2 + 2 + (3 + 2) * intra_layers
    assert (
        checkpointing.set_internal_checkpointing(decoder, True)
        == expected_layers
    )

    for unroll in (False, True):
        result = run(unroll_checkpoint=unroll)
        assert len(result) == len(reference)
        for new, ref in zip(result, reference):
            torch.testing.assert_close(new, ref, rtol=1e-9, atol=1e-11)
        n = expected_layers
        if unroll:
            # Whole step is recomputed once, and inside it every layer's own
            # block is recomputed as well: 3N layer runs, 2N layer blocks
            assert run.counts["layer_calls"] == 3 * n
            assert run.counts["layer_blocks"] == 2 * n
        else:
            # Layers only: N blocks, each run once more in backward -> 2N
            assert run.counts["layer_calls"] == 2 * n
            assert run.counts["layer_blocks"] == n


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("selective", (False, checkpointing.SELECTIVE)),
        ("unroll+selective", (True, checkpointing.SELECTIVE)),
    ],
)
def test_selective_policy_flags(policy, expected):
    assert checkpointing.policy_flags(policy, False, 1) == expected


@pytest.mark.parametrize("case_index", range(5))
def test_selective_results_unchanged(case_index):
    """Selective checkpointing must not change outputs or gradients"""
    _, layer, inputs = _make_cases()[case_index]
    ref_out, ref_in_grads, ref_param_grads = _forward_backward(layer, inputs)

    layer.ckpt_inner = checkpointing.SELECTIVE
    out, in_grads, param_grads = _forward_backward(layer, inputs)
    for new, ref in zip(
        out + in_grads + param_grads, ref_out + ref_in_grads + ref_param_grads
    ):
        torch.testing.assert_close(new, ref, rtol=1e-10, atol=1e-12)


def _count_ops(layer, inputs):
    """
    Number of times each low-level operation runs in one forward+backward,
    as {op name: count}, measured with a dispatch-level counter.
    """
    # Third-party
    from torch.utils._python_dispatch import TorchDispatchMode

    counts = {}

    class Counter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counts[str(func)] = counts.get(str(func), 0) + 1
            return func(*args, **(kwargs or {}))

    with Counter():
        _forward_backward(layer, inputs)
    return counts


def _matmul_count(counts):
    """Executions of the forward matrix multiplications (mm/addmm/bmm)"""
    names = {str(op) for op in checkpointing.SAVED_OPS}
    return sum(n for name, n in counts.items() if name in names)


@pytest.mark.parametrize("case_index", range(5))
def test_selective_saves_matmuls_but_recomputes_cheap_ops(case_index):
    """
    Plain: every operation runs once in the forward pass. Full per-layer
    checkpointing: the forward matmuls run twice. Selective: the forward
    matmuls run only once (their results are kept) while the cheap
    operations (the LayerNorm) are still recomputed.
    """
    _, layer, inputs = _make_cases()[case_index]
    plain = _count_ops(layer, inputs)

    layer.ckpt_inner = True
    full = _count_ops(layer, inputs)

    layer.ckpt_inner = checkpointing.SELECTIVE
    selective = _count_ops(layer, inputs)

    # backward matmuls are counted in all three runs; the difference between
    # runs is exactly the recomputed forward matmuls
    assert _matmul_count(selective) == _matmul_count(plain)
    assert _matmul_count(full) > _matmul_count(plain)

    norm = "aten.native_layer_norm.default"
    assert full[norm] == 2 * plain[norm]
    assert selective[norm] == 2 * plain[norm]
