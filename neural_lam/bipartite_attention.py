"""Sparse attention over a directed edge set between two node sets."""

# Third-party
import torch
import torch_geometric as pyg
from torch import nn

# Local
from . import utils

EDGE_MODES = ("key_value", "bias")
RESIDUAL_MODES = ("receiver", "aggregate")


class BipartiteEdgeAttention(nn.Module):
    """
    Multi-head attention of receiver nodes over their incoming edges.

    Senders and receivers may be different node sets with different widths
    (cross-attention, e.g. grid to mesh), or the same set passed twice
    (self-attention over graph neighbours). Edge features are read but never
    updated.

    Shapes, with any leading sample dimensions (...):
        send_rep (..., N_send, send_dim)
        rec_rep  (..., N_rec, rec_dim)
        edge_rep (..., M, edge_dim)
        output   (..., N_rec, out_dim)

    out_dim is rec_dim for residual="receiver" and send_dim for
    residual="aggregate", the same rule as FlexibleNet. It is split into
    num_heads heads of width d_k = out_dim / num_heads.

        q_i  = W_Q h_i                                    receiver i
        k_ji = W_K h_j  (+ W_E  e_ji if edge_mode="key_value")
        v_ji = W_V h_j  (+ W_E' e_ji if edge_mode="key_value")
        s_ji = q_i . k_ji / sqrt(d_k)  (+ w_h . e_ji if edge_mode="bias")
        alpha_ji = softmax of s_ji over the edges into i, per head
        a_i  = concatenated heads of sum_j alpha_ji v_ji

        residual="receiver":   h_i' = h_i + MLP([h_i, a_i])
        residual="aggregate":  h_i' = a_i + MLP([h_i, a_i])

    A receiver without incoming edges gets a_i = 0. Duplicate edges stay
    distinct. The node MLP is the one used by InteractionNet and FlexibleNet
    (LayerNorm on its output).
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        edge_index,
        send_dim,
        rec_dim,
        edge_dim,
        num_heads,
        hidden_layers=1,
        num_rec=None,
        residual="receiver",
        edge_mode="key_value",
        key_bias=False,
        value_bias=False,
        return_edges=False,
    ):
        """
        edge_index: (2, M), row 0 indexes senders and row 1 receivers, both
            starting at 0
        send_dim, rec_dim, edge_dim: widths of the three inputs
        num_heads: number of attention heads, must divide out_dim
        hidden_layers: number of hidden layers in the node MLP
        num_rec: number of receiver nodes. If given, forward checks it. If
            None, the number of rows of rec_rep is used, so receivers without
            incoming edges are handled either way.
        residual: "receiver" or "aggregate", see class docstring
        edge_mode: "key_value" adds projected edge features to keys and
            values. "bias" adds one scalar per head to the score only.
        key_bias: bias on W_K. It shifts all scores of a receiver equally,
            so the softmax cancels it; only for parameter compatibility.
        value_bias: bias on W_V. Adds the same vector to every a_i.
        return_edges: return (rec_rep, edge_rep) with edge_rep unchanged, as
            expected by utils.make_gnn_seq
        """
        super().__init__()
        if residual not in RESIDUAL_MODES:
            raise ValueError(f"Unknown residual mode: {residual}")
        if edge_mode not in EDGE_MODES:
            raise ValueError(f"Unknown edge mode: {edge_mode}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, M)")

        out_dim = rec_dim if residual == "receiver" else send_dim
        if out_dim % num_heads != 0:
            raise ValueError(
                f"Output width {out_dim} is not divisible by "
                f"{num_heads} attention heads"
            )

        self.residual = residual
        self.edge_mode = edge_mode
        self.return_edges = return_edges
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Smallest node counts that the edge list is valid for
        has_edges = edge_index.numel() > 0
        self.min_send = int(edge_index[0].max()) + 1 if has_edges else 0
        self.min_rec = int(edge_index[1].max()) + 1 if has_edges else 0
        self.num_rec = None if num_rec is None else int(num_rec)
        if self.num_rec is not None and self.num_rec < self.min_rec:
            raise ValueError("edge_index has a receiver index >= num_rec")

        # Follows module.to(device), but is not a learned state
        self.register_buffer(
            "edge_index", edge_index.detach().clone(), persistent=False
        )

        self.q_proj = nn.Linear(rec_dim, out_dim)
        self.k_proj = nn.Linear(send_dim, out_dim, bias=key_bias)
        self.v_proj = nn.Linear(send_dim, out_dim, bias=value_bias)
        if edge_mode == "key_value":
            self.k_edge = nn.Linear(edge_dim, out_dim, bias=False)
            self.v_edge = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)

        self.aggr_mlp = utils.make_mlp(
            [rec_dim + out_dim] + [out_dim] * (hidden_layers + 1)
        )

    def forward(self, send_rep, rec_rep, edge_rep, emb=None):
        """
        send_rep: (..., N_send, send_dim), sender node representations
        rec_rep: (..., N_rec, rec_dim), receiver node representations
        edge_rep: (..., M, edge_dim), edge representations
        emb: unused, present for call-compatibility with the other GNN layers

        Returns:
        rec_rep: (..., N_rec, out_dim), updated receiver representations
        (if return_edges) edge_rep: the unchanged input edge representations
        """
        num_rec = rec_rep.shape[-2]
        if self.num_rec is not None and num_rec != self.num_rec:
            raise ValueError(
                f"Expected {self.num_rec} receiver nodes, got {num_rec}"
            )
        if num_rec < self.min_rec or send_rep.shape[-2] < self.min_send:
            raise ValueError("edge_index has an index outside the given nodes")

        sender, receiver = self.edge_index
        head_shape = (self.num_heads, self.head_dim)

        # Project per node, then gather the two ends of every edge: (..., M, d)
        query = self.q_proj(rec_rep).index_select(-2, receiver)
        key = self.k_proj(send_rep).index_select(-2, sender)
        value = self.v_proj(send_rep).index_select(-2, sender)
        if self.edge_mode == "key_value":
            key = key + self.k_edge(edge_rep)
            value = value + self.v_edge(edge_rep)

        # Split into heads: (..., M, H, d_k)
        query = query.unflatten(-1, head_shape)
        key = key.unflatten(-1, head_shape)
        value = value.unflatten(-1, head_shape)

        scores = (query * key).sum(dim=-1) * self.scale  # (..., M, H)
        if self.edge_mode == "bias":
            scores = scores + self.edge_bias(edge_rep)

        # Normalise in at least float32: bf16 is too coarse for the softmax
        if scores.dtype not in (torch.float32, torch.float64):
            scores = scores.float()
        alpha = pyg.utils.softmax(
            scores, receiver, num_nodes=num_rec, dim=-2
        )  # (..., M, H), sums to 1 over the edges into each receiver

        messages = (alpha.to(value.dtype).unsqueeze(-1) * value).flatten(-2)
        aggr = messages.new_zeros(
            messages.shape[:-2] + (num_rec, self.out_dim)
        ).index_add(
            -2, receiver, messages
        )  # (..., N_rec, out_dim)

        update = self.aggr_mlp(torch.cat((rec_rep, aggr), dim=-1), emb)
        if self.residual == "receiver":
            new_rec_rep = rec_rep + update
        else:
            new_rec_rep = aggr + update

        if self.return_edges:
            return new_rec_rep, edge_rep
        return new_rec_rep
