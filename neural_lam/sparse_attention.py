# neural_lam/sparse_attention.py
"""Alternative same-level layers for the GraphEFM decoder."""

from .interaction_net import InteractionNet

INTRA_LAYER_TYPES = ("interaction", "interaction_static_edges")


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
    raise ValueError(f"Unknown intra-level layer type: {layer_type}")
