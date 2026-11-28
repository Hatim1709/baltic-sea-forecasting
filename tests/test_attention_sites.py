"""Wiring tests for attention sites in the hierarchical GraphEFM sub-models.

The encoder (also used as prior) and the decoder are built directly on a tiny
synthetic two-level graph, so no dataset is needed:

    grid: 10 nodes    mesh level 0: 5 nodes    mesh level 1: 3 nodes

The attention equations themselves are checked in test_bipartite_attention.py.
Here we check that every site holds the intended layer, that the original
layers are kept by default, and that data and gradients flow through all
sites.
"""

# Third-party
import pytest
import torch

# First-party
from neural_lam.attention_sites import ALL_SITES, SITE_KINDS, resolve_sites
from neural_lam.bipartite_attention import BipartiteEdgeAttention
from neural_lam.interaction_net import (
    FlexibleNet,
    InteractionNet,
    PropagationNet,
)
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder
from neural_lam.sparse_attention import StaticEdgeAttention

B, N_GRID, N_MESH = 2, 10, (5, 3)
D, D_GRID, D_MESH0, D_EDGE, LATENT, N_STATE, HEADS = 16, 8, 8, 4, 4, 3, 2
DEC_LAYERS, ENC_LAYERS = 2, 1

ORIGINAL = {
    "decoder.g2m": FlexibleNet,
    "decoder.intra": InteractionNet,
    "decoder.up": InteractionNet,
    "decoder.down": PropagationNet,
    "decoder.m2g": FlexibleNet,
    "prior.g2m": FlexibleNet,
    "prior.intra": InteractionNet,
    "prior.up": PropagationNet,
    "encoder.g2m": FlexibleNet,
    "encoder.intra": InteractionNet,
    "encoder.up": PropagationNet,
}
RESIDUAL = {
    "decoder.g2m": "aggregate",
    "decoder.intra": "receiver",
    "decoder.up": "receiver",
    "decoder.down": "aggregate",
    "decoder.m2g": "receiver",
    "prior.g2m": "aggregate",
    "prior.intra": "receiver",
    "prior.up": "aggregate",
    "encoder.g2m": "aggregate",
    "encoder.intra": "receiver",
    "encoder.up": "aggregate",
}
GNN_LAYERS = (
    FlexibleNet,
    InteractionNet,
    BipartiteEdgeAttention,
    StaticEdgeAttention,
)


def edges(pairs):
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def ring(num_nodes):
    """Both directions of a ring, so that every node sends and receives."""
    forward = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    return edges(forward + [(j, i) for i, j in forward])


@pytest.fixture
def graph():
    # Every grid node is connected to mesh node i // 2, and some to a second
    # mesh node, so receivers have different numbers of incoming edges.
    g2m = [(i, i // 2) for i in range(N_GRID)]
    g2m += [(i, (i // 2 + 1) % 5) for i in range(0, N_GRID, 2)]
    m2g = [(i // 2, i) for i in range(N_GRID)]
    m2g += [((i // 2 + 1) % 5, i) for i in range(1, N_GRID, 2)]
    up = [(i, i * 3 // 5) for i in range(5)]
    up += [(i, (i * 3 // 5 + 1) % 3) for i in (0, 2, 4)]
    return {
        "g2m": edges(g2m),
        "m2g": edges(m2g),
        "m2m": [ring(5), ring(3)],
        "up": [edges(up)],
        "down": [edges([(j, i) for i, j in up])],
    }


def make_decoder(graph, seed=0, **kwargs):
    torch.manual_seed(seed)
    kwargs.setdefault("attention_heads", HEADS)
    return HiGraphLatentDecoder(
        graph["g2m"],
        graph["m2m"],
        graph["m2g"],
        graph["up"],
        graph["down"],
        D,
        D_GRID,
        D_MESH0,
        D_EDGE,
        LATENT,
        N_STATE,
        DEC_LAYERS,
        hidden_layers=1,
        output_std=True,
        num_grid_con_mesh_nodes=N_MESH[0],
        num_interior_nodes=N_GRID,
        **kwargs,
    )


def make_encoder(graph, submodel, seed=0, **kwargs):
    torch.manual_seed(seed)
    kwargs.setdefault("attention_heads", HEADS)
    return HiGraphLatentEncoder(
        LATENT,
        graph["g2m"],
        graph["m2m"],
        graph["up"],
        D,
        D_GRID,
        D_MESH0,
        D_EDGE,
        ENC_LAYERS,
        hidden_layers=1,
        output_dist="diagonal",
        num_grid_con_mesh_nodes=N_MESH[0],
        submodel=submodel,
        **kwargs,
    )


def make_inputs(graph, seed=1):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(B, *shape, generator=gen).requires_grad_()

    graph_emb = {
        "mesh": [randn(N_MESH[0], D_MESH0), randn(N_MESH[1], D)],
        "g2m": randn(graph["g2m"].shape[1], D_EDGE),
        "m2g": randn(graph["m2g"].shape[1], D_EDGE),
        "m2m": [randn(index.shape[1], D) for index in graph["m2m"]],
        "mesh_up": [randn(graph["up"][0].shape[1], D)],
        "mesh_down": [randn(graph["down"][0].shape[1], D)],
    }
    return {
        "grid": randn(N_GRID, D_GRID),
        "latent": randn(N_MESH[1], LATENT),
        "last_state": torch.randn(B, N_GRID, N_STATE, generator=gen),
        "graph_emb": graph_emb,
    }


def run_decoder(decoder, inputs):
    return decoder(
        inputs["grid"],
        inputs["grid"],
        inputs["latent"],
        inputs["last_state"],
        inputs["graph_emb"],
    )


def site_layers(model, submodel):
    """The GNN layers at each kind of site of a sub-model."""

    def gnn_layers(seqs):
        return [
            module
            for seq in seqs
            for module in seq.modules()
            if isinstance(module, GNN_LAYERS)
        ]

    layers = {"g2m": [model.g2m_gnn], "up": list(model.mesh_up_gnns)}
    if submodel == "decoder":
        layers["m2g"] = [model.m2g_gnn]
        layers["down"] = list(model.mesh_down_gnns)
        layers["intra"] = gnn_layers(
            list(model.intra_up_gnns) + list(model.intra_down_gnns)
        )
        assert len(layers["intra"]) == 3 * DEC_LAYERS  # levels 0, 1 up; 0 down
    else:
        layers["intra"] = gnn_layers(model.intra_level_gnns)
        assert len(layers["intra"]) == 2 * ENC_LAYERS
    return layers


def build_all(graph, **kwargs):
    return {
        "decoder": make_decoder(graph, **kwargs),
        "prior": make_encoder(graph, "prior", **kwargs),
        "encoder": make_encoder(graph, "encoder", **kwargs),
    }


def test_resolve_sites():
    assert resolve_sites(()) == frozenset()
    assert resolve_sites(None) == frozenset()
    assert resolve_sites(["all"]) == frozenset(ALL_SITES)
    assert resolve_sites(["prior.up", "decoder.g2m", "prior.up"]) == {
        "prior.up",
        "decoder.g2m",
    }
    assert set(ALL_SITES) == set(ORIGINAL)
    with pytest.raises(ValueError):
        resolve_sites(["decoder.g2g"])


@pytest.mark.parametrize("selection", [()] + [(s,) for s in ALL_SITES])
def test_each_site_holds_the_intended_layer(graph, selection):
    """Only the selected site changes; every other site keeps its layer."""
    models = build_all(graph, attention_sites=selection)
    for submodel, kinds in SITE_KINDS.items():
        layers = site_layers(models[submodel], submodel)
        assert set(layers) == set(kinds)
        for kind in kinds:
            site = f"{submodel}.{kind}"
            for layer in layers[kind]:
                if site in selection:
                    assert type(layer) is BipartiteEdgeAttention, site
                    assert layer.residual == RESIDUAL[site], site
                    assert layer.return_edges == (kind == "intra"), site
                else:
                    assert type(layer) is ORIGINAL[site], site

    # Original settings that differ between sites of the same class
    if not selection:
        decoder = models["decoder"]
        assert decoder.g2m_gnn.propagation and not decoder.m2g_gnn.propagation
        assert not decoder.mesh_up_gnns[0].update_edges
        assert site_layers(decoder, "decoder")["intra"][0].update_edges


@pytest.mark.parametrize("edge_mode", ["key_value", "bias"])
def test_all_sites_use_attention_with_the_given_edge_mode(graph, edge_mode):
    models = build_all(
        graph, attention_sites=("all",), attention_edge_mode=edge_mode
    )
    for submodel, model in models.items():
        for kind, layers in site_layers(model, submodel).items():
            for layer in layers:
                assert type(layer) is BipartiteEdgeAttention, kind
                assert layer.edge_mode == edge_mode
                assert layer.num_heads == HEADS
    # Widths at the two grid sites follow the FlexibleNet rule
    decoder = models["decoder"]
    assert decoder.g2m_gnn.out_dim == D_GRID  # sender width (aggregate)
    assert decoder.m2g_gnn.out_dim == D_GRID  # receiver width
    assert decoder.g2m_gnn.num_rec == N_MESH[0]
    assert decoder.m2g_gnn.num_rec == N_GRID


CONFIGS = [
    {},
    {"attention_sites": ("all",)},
    {"attention_sites": ("all",), "attention_edge_mode": "bias"},
]


def assert_all_parameters_trained(model):
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().max() > 0, name


@pytest.mark.parametrize("config", CONFIGS)
def test_decoder_forward_and_backward(graph, config):
    decoder = make_decoder(graph, **config)
    inputs = make_inputs(graph)
    mean, std = run_decoder(decoder, inputs)
    assert mean.shape == (B, N_GRID, N_STATE)
    assert std.shape == (B, N_GRID, N_STATE)
    assert torch.isfinite(mean).all() and (std > 0).all()

    (mean.square().mean() + std.mean()).backward()
    assert_all_parameters_trained(decoder)

    # Every input the decoder is meant to read influences the output. The
    # top-level mesh embedding is replaced by the latent in the decoder.
    emb = inputs["graph_emb"]
    used = [inputs["grid"], inputs["latent"], emb["mesh"][0]]
    used += [emb["g2m"], emb["m2g"], emb["mesh_up"][0], emb["mesh_down"][0]]
    used += emb["m2m"]
    for tensor in used:
        assert tensor.grad is not None and tensor.grad.abs().max() > 0
    assert emb["mesh"][1].grad is None


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("submodel", ["prior", "encoder"])
def test_encoder_forward_and_backward(graph, config, submodel):
    encoder = make_encoder(graph, submodel, **config)
    inputs = make_inputs(graph)
    dist = encoder(inputs["grid"], graph_emb=inputs["graph_emb"])
    assert dist.mean.shape == (B, N_MESH[1], LATENT)
    assert torch.isfinite(dist.mean).all() and (dist.stddev > 0).all()

    (dist.mean.square().mean() + dist.stddev.mean()).backward()
    assert_all_parameters_trained(encoder)

    emb = inputs["graph_emb"]
    used = [inputs["grid"], emb["g2m"], emb["mesh_up"][0]]
    used += emb["mesh"] + emb["m2m"]
    for tensor in used:
        assert tensor.grad is not None and tensor.grad.abs().max() > 0
    assert emb["m2g"].grad is None and emb["mesh_down"][0].grad is None


@pytest.mark.parametrize("config", CONFIGS)
def test_samples_are_independent(graph, config):
    decoder = make_decoder(graph, **config)
    encoder = make_encoder(graph, "encoder", **config)
    inputs = make_inputs(graph)
    with torch.no_grad():
        mean, std = run_decoder(decoder, inputs)
        latent_mean = encoder(
            inputs["grid"], graph_emb=inputs["graph_emb"]
        ).mean
        for b in range(B):
            one = {
                "grid": inputs["grid"][b : b + 1],
                "latent": inputs["latent"][b : b + 1],
                "last_state": inputs["last_state"][b : b + 1],
                "graph_emb": {
                    key: (
                        [t[b : b + 1] for t in value]
                        if isinstance(value, list)
                        else value[b : b + 1]
                    )
                    for key, value in inputs["graph_emb"].items()
                },
            }
            one_mean, one_std = run_decoder(decoder, one)
            one_latent = encoder(one["grid"], graph_emb=one["graph_emb"]).mean
            tol = {"atol": 1e-5, "rtol": 1e-4}
            torch.testing.assert_close(mean[b : b + 1], one_mean, **tol)
            torch.testing.assert_close(std[b : b + 1], one_std, **tol)
            torch.testing.assert_close(
                latent_mean[b : b + 1], one_latent, **tol
            )


def test_decoder_intra_layer_option_still_works(graph):
    """--decoder_intra_layer attention keeps building StaticEdgeAttention."""
    decoder = make_decoder(graph, intra_layer_type="attention")
    for layer in site_layers(decoder, "decoder")["intra"]:
        assert type(layer) is StaticEdgeAttention

    # It can be combined with attention at other sites, ...
    mixed = make_decoder(
        graph,
        intra_layer_type="attention",
        attention_sites=("decoder.g2m", "decoder.down"),
    )
    assert type(mixed.g2m_gnn) is BipartiteEdgeAttention
    assert type(mixed.mesh_down_gnns[0]) is BipartiteEdgeAttention
    assert type(mixed.m2g_gnn) is FlexibleNet
    mean, _ = run_decoder(mixed, make_inputs(graph))
    assert torch.isfinite(mean).all()

    # ... but not with an attention site at the same place
    with pytest.raises(ValueError):
        make_decoder(
            graph,
            intra_layer_type="attention",
            attention_sites=("decoder.intra",),
        )


def test_attention_sites_need_heads(graph):
    with pytest.raises(ValueError):
        make_decoder(
            graph, attention_sites=("decoder.g2m",), attention_heads=None
        )
    # Sites of other sub-models do not concern this one
    decoder = make_decoder(
        graph, attention_sites=("prior.up",), attention_heads=None
    )
    assert type(decoder.g2m_gnn) is FlexibleNet
