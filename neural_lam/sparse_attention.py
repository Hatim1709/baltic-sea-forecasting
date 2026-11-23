# neural_lam/sparse_attention.py
"""Alternative same-level layers for the GraphEFM decoder."""

import torch
import torch_geometric as pyg
from torch import nn

from . import utils
from .interaction_net import InteractionNet

INTRA_LAYER_TYPES = ("interaction", "interaction_static_edges", "attention", "attention_uniform")


class StaticEdgeInteractionNet(InteractionNet):
    """Control: InteractionNet without edge-state update.

    Same parameters and messages as InteractionNet, but the edge
    representation is passed through unchanged instead of e + m.
    """

    def __init__(self, edge_index, input_dim, hidden_layers=1):
        super().__init__(
            edge_index, input_dim, update_edges=False, hidden_layers=hidden_layers
        )

    def forward(self, send_rep, rec_rep, edge_rep, emb=None):
        rec_rep = super().forward(send_rep, rec_rep, edge_rep, emb)
        return rec_rep, edge_rep  # unchanged edge state


def build_intra_layer(layer_type, edge_index, hidden_dim, hidden_layers,
                      attention_heads=None):
    """Build one same-level layer of the requested type."""
    if layer_type == "interaction":
        # Must stay exactly the original call (same RNG use -> same init as R0)
        return InteractionNet(edge_index, hidden_dim, hidden_layers=hidden_layers)
    if layer_type == "interaction_static_edges":
        return StaticEdgeInteractionNet(edge_index, hidden_dim, hidden_layers=hidden_layers)
    if layer_type == "attention":
        if(attention_heads == None):
            raise ValueError(f"Missing attention heads")
        return StaticEdgeAttention(edge_index, hidden_dim, attention_heads, hidden_layers=hidden_layers)
    if layer_type == "attention_uniform":
        if attention_heads is None:
            raise ValueError("--decoder_intra_layer attention_uniform needs --attention_heads")
        return StaticEdgeAttention(edge_index, hidden_dim, attention_heads,
                                hidden_layers=hidden_layers, uniform_weights=True)

    raise ValueError(f"Unknown intra-level layer type: {layer_type}")



class StaticEdgeAttention(pyg.nn.MessagePassing):
    """
    Multi-head Q/K/V attention over incoming edges, with static edge features:
    q_i  = W_Q h_i                       per node    (B, N, d)
    k_ji = W_K h_j + W_E  e⁰_ji          per edge    (B, M, d)
    v_ji = W_V h_j + W_E' e⁰_ji          per edge    (B, M, d)
            → split each into heads: (B, M, H, d_k)
    s_ji = q_i · k_ji / √d_k                         (B, M, H)
    α_ji = softmax of s over the incoming edges of i  (B, M, H)
    a_i  = concat over heads of Σ_j α_ji v_ji        (B, N, d)
    h_i' = h_i + MLP_n([h_i, a_i])                   same node MLP as InteractionNet
    e'   = e⁰                                        returned unchanged

    """

    def __init__(self, edge_index, input_dim, num_heads, hidden_layers=1, uniform_weights = False):
        assert(input_dim % num_heads == 0)
        super().__init__(aggr = "sum")

        self.num_heads = num_heads
        self.d_k = input_dim//num_heads
        self.scale = self.d_k ** -0.5

        # Store number of receiver nodes according to edge_index
        self.num_rec = int(edge_index[1].max()) + 1
    
        # any edge_index used here must start sender and rec. nodes at index 0
        edge_index = torch.stack(
            (edge_index[0] + self.num_rec, edge_index[1]), dim=0
        )  # Make sender indices after rec
    
        self.register_buffer("edge_index", edge_index, persistent=False)

        d = input_dim
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.k_edge = nn.Linear(d, d, bias = False)
        self.v_edge = nn.Linear(d, d, bias = False)

        self.aggr_mlp = utils.make_mlp([2 * d] + [d] * (hidden_layers + 1))
        self.uniform_weights = uniform_weights
        if uniform_weights:
        # Not used when the weights are uniform. Freeze them so DDP and the
        # optimizer ignore them (DDP requires a gradient for every trainable parameter).
            for module in (self.q_proj, self.k_proj, self.k_edge):
                module.requires_grad_(False)


    def forward(self, send_rep, rec_rep, edge_rep, emb=None):
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        aggr = self.propagate(
            self.edge_index,
            q=self.q_proj(node_reps), k=self.k_proj(node_reps), v=self.v_proj(node_reps),
            k_e=self.k_edge(edge_rep), v_e=self.v_edge(edge_rep),
        )
        rec_rep = rec_rep + self.aggr_mlp(torch.cat((rec_rep, aggr), dim=-1))
        return rec_rep, edge_rep          # edge_rep unchanged, as in R1

    def message(self, q_i, k_j, v_j, k_e, v_e, index):
        heads = lambda t: t.view(*t.shape[:-1], self.num_heads, self.d_k)   # (B, M, H, d_k)
        q, k, v = heads(q_i), heads(k_j + k_e), heads(v_j + v_e)
        if self.uniform_weights:
            scores = torch.zeros(q.shape[:-1], device=q.device)   # (B, M, H): softmax -> 1/in-degree
        else:
            scores = (q * k).sum(-1) * self.scale                 # (B, M, H)
        alpha = pyg.utils.softmax(scores.float(), index, num_nodes=self.num_rec, dim=1)
        return (alpha.to(v.dtype).unsqueeze(-1) * v).flatten(-2)             # (B, M, d)

    def aggregate(self, inputs, index, ptr, dim_size):
        return super().aggregate(inputs, index, ptr, self.num_rec)
