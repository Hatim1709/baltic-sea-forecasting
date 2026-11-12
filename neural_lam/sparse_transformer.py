"""Sparse, edge-biased attention for a single mesh level."""

# Standard library
import math

# Third-party
import torch
from torch import nn
from torch_geometric.utils import softmax


class SparseGraphAttentionBlock(nn.Module):
    """Pre-LayerNorm attention and FFN with unchanged input edge features.

    Nodes have shape (..., N, hidden_dim), and edges have shape
    (..., M, edge_dim), with identical leading sample dimensions.
    edge_index has shape (2, M): row 0 sends, row 1 receives.
    Both rows index the same N nodes, starting at zero.

    Only supplied edges participate. Duplicate edges remain distinct.
    Isolated receivers get zero attention contribution, then the FFN.
    The return value is the updated node tensor. edge_rep is neither
    modified nor detached.

    All attention projections are bias-free. The FFN has biases.
    Dropout applies to the two residual branches, not attention weights.
    """

    def __init__(
        self,
        edge_index,
        hidden_dim,
        edge_dim,
        num_heads=4,
        ffn_expansion=4,
        dropout=0.0,
    ):
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("edge_dim", edge_dim),
            ("num_heads", num_heads),
            ("ffn_expansion", ffn_expansion),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("edge_index must be a tensor")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, M)")
        if edge_index.numel() and edge_index.min().item() < 0:
            raise ValueError("edge_index must contain nonnegative indices")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ffn_expansion = ffn_expansion
        self._max_node_index = (
            int(edge_index.max().item()) if edge_index.numel() else -1
        )
        # Connectivity follows module.to(device), but is not a learned state.
        # Clone so later edits to the caller's tensor cannot change this graph.
        self.register_buffer(
            "edge_index", edge_index.detach().clone(), persistent=False
        )

        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_expansion * hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_expansion * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_rep, edge_rep):
        """Return updated nodes, preserving sample axes and node order."""
        if node_rep.ndim < 2 or node_rep.shape[-1] != self.hidden_dim:
            raise ValueError("node_rep must have shape (..., N, hidden_dim)")
        expected_edges = node_rep.shape[:-2] + (
            self.edge_index.shape[1],
            self.edge_dim,
        )
        if edge_rep.shape != expected_edges:
            raise ValueError(
                f"edge_rep must have shape {tuple(expected_edges)}, "
                f"got {tuple(edge_rep.shape)}"
            )
        if not node_rep.is_floating_point() or not edge_rep.is_floating_point():
            raise TypeError("node_rep and edge_rep must be floating point")
        if node_rep.dtype != edge_rep.dtype:
            raise TypeError("node_rep and edge_rep must have the same dtype")
        if (
            node_rep.device != edge_rep.device
            or node_rep.device != self.edge_index.device
        ):
            raise ValueError(
                "nodes, edges, and connectivity must share a device"
            )
        num_nodes = node_rep.shape[-2]
        if self._max_node_index >= num_nodes:
            raise ValueError("edge_index contains an index outside node_rep")

        sender, receiver = self.edge_index
        normalized = self.norm1(node_rep)
        head_shape = node_rep.shape[:-1] + (self.num_heads, self.head_dim)
        queries = self.q_proj(normalized).reshape(head_shape)
        keys = self.k_proj(normalized).reshape(head_shape)
        values = self.v_proj(normalized).reshape(head_shape)

        # (..., M, H, d): gather only the endpoints of supplied edges.
        receiver_queries = queries.index_select(-3, receiver)
        sender_keys = keys.index_select(-3, sender)
        sender_values = values.index_select(-3, sender)

        # (..., M, H): independent softmax per receiver, sample, and head.
        scores = (receiver_queries * sender_keys).sum(dim=-1)
        scores = scores / math.sqrt(self.head_dim) + self.edge_bias(edge_rep)
        weights = softmax(scores, receiver, num_nodes=num_nodes, dim=-2)

        # Flatten only heads and head features, retaining the edge axis.
        messages = (weights.unsqueeze(-1) * sender_values).flatten(-2)
        aggregated = messages.new_zeros(node_rep.shape).index_add(
            -2, receiver, messages
        )

        # Bias-free out_proj maps an empty neighborhood's zero sum to zero.
        attended = node_rep + self.dropout(self.out_proj(aggregated))
        return attended + self.dropout(self.ffn(self.norm2(attended)))


class SparseGraphTransformerProcessor(nn.Module):
    """Stack attention blocks with the existing two-tensor interface.

    forward(mesh_rep, edge_rep) returns (updated_mesh_rep, edge_rep).
    Every block receives the original edge tensor, including its autograd
    history. Each block has independent trainable parameters.

    With num_layers=0, return both inputs unchanged, matching the identity
    behavior of make_gnn_seq. Other configuration and tensor validation is
    performed by the blocks only when num_layers is positive.
    """

    def __init__(
        self,
        edge_index,
        hidden_dim,
        edge_dim,
        num_layers=1,
        num_heads=4,
        ffn_expansion=4,
        dropout=0.0,
    ):
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be a nonnegative integer")
        if num_layers < 0:
            raise ValueError("num_layers must be nonnegative")

        self.blocks = nn.ModuleList(
            [
                SparseGraphAttentionBlock(
                    edge_index=edge_index,
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, mesh_rep, edge_rep):
        """Update nodes and return the exact original edge tensor."""
        for block in self.blocks:
            mesh_rep = block(mesh_rep, edge_rep)
        return mesh_rep, edge_rep
