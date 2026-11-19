"""Build each GNN site of the hierarchical GraphEFM sub-models.

A site is one place where a GNN layer acts, e.g. "decoder.g2m". For every
site the builder returns either the original layer, constructed with exactly
the original arguments, or a BipartiteEdgeAttention layer configured to stand
in for it:

    site                     original layer            attention residual
    g2m                      FlexibleNet (mean)        aggregate
    m2g                      FlexibleNet (sum)         receiver
    intra                    InteractionNet            receiver
    up (decoder)             InteractionNet            receiver
    up (prior, encoder)      PropagationNet            aggregate
    down                     PropagationNet            aggregate
"""

# Local
from . import utils
from .bipartite_attention import EDGE_MODES, BipartiteEdgeAttention
from .interaction_net import FlexibleNet, InteractionNet, PropagationNet

SITE_KINDS = {
    "decoder": ("g2m", "intra", "up", "down", "m2g"),
    "prior": ("g2m", "intra", "up"),
    "encoder": ("g2m", "intra", "up"),
}
ALL_SITES = tuple(
    f"{submodel}.{kind}"
    for submodel, kinds in SITE_KINDS.items()
    for kind in kinds
)
SITE_CHOICES = ALL_SITES + ("all",)


def resolve_sites(names):
    """
    Turn a list of site names into a set of full names, expanding "all".
    """
    sites = set()
    for name in names or ():
        if name == "all":
            sites.update(ALL_SITES)
        elif name in ALL_SITES:
            sites.add(name)
        else:
            raise ValueError(
                f"Unknown attention site: {name}. "
                f"Choose from {', '.join(SITE_CHOICES)}"
            )
    return frozenset(sites)


class SiteBuilder:
    """
    Builds the GNN layers of one sub-model ("decoder", "prior" or "encoder").
    """

    def __init__(
        self,
        submodel,
        hidden_layers=1,
        attention_sites=(),
        attention_heads=None,
        attention_edge_mode="key_value",
    ):
        """
        submodel: "decoder", "prior" or "encoder"
        hidden_layers: number of hidden layers in all MLPs
        attention_sites: site names that use attention, e.g. "decoder.g2m"
            or "all". Sites of other sub-models are ignored.
        attention_heads: number of heads, required if any site of this
            sub-model uses attention
        attention_edge_mode: "key_value" or "bias", see BipartiteEdgeAttention
        """
        if submodel not in SITE_KINDS:
            raise ValueError(f"Unknown sub-model: {submodel}")
        if attention_edge_mode not in EDGE_MODES:
            raise ValueError(f"Unknown edge mode: {attention_edge_mode}")

        self.submodel = submodel
        self.hidden_layers = hidden_layers
        self.heads = attention_heads
        self.edge_mode = attention_edge_mode
        self.kinds = frozenset(
            site.split(".", 1)[1]
            for site in resolve_sites(attention_sites)
            if site.startswith(f"{submodel}.")
        )
        if self.kinds and attention_heads is None:
            raise ValueError(
                "Attention sites need a number of heads (--attention_heads)"
            )

    def uses_attention(self, kind):
        """If the site of the given kind uses attention in this sub-model"""
        return kind in self.kinds

    def _attention(self, edge_index, send_dim, rec_dim, edge_dim, **kwargs):
        return BipartiteEdgeAttention(
            edge_index,
            send_dim,
            rec_dim,
            edge_dim,
            self.heads,
            hidden_layers=self.hidden_layers,
            edge_mode=self.edge_mode,
            **kwargs,
        )

    def g2m(self, edge_index, send_dim, rec_dim, edge_dim, num_rec):
        """Grid to bottom mesh level"""
        if self.uses_attention("g2m"):
            return self._attention(
                edge_index,
                send_dim,
                rec_dim,
                edge_dim,
                num_rec=num_rec,
                residual="aggregate",
            )
        return FlexibleNet(
            edge_index=edge_index,
            send_node_dim=send_dim,
            rec_node_dim=rec_dim,
            edge_dim=edge_dim,
            hidden_layers=self.hidden_layers,
            num_rec=num_rec,
            propagation=True,
            aggr="mean",
        )

    def m2g(self, edge_index, send_dim, rec_dim, edge_dim, num_rec):
        """Bottom mesh level to grid"""
        if self.uses_attention("m2g"):
            return self._attention(
                edge_index,
                send_dim,
                rec_dim,
                edge_dim,
                num_rec=num_rec,
                residual="receiver",
            )
        return FlexibleNet(
            edge_index=edge_index,
            send_node_dim=send_dim,
            rec_node_dim=rec_dim,
            edge_dim=edge_dim,
            hidden_layers=self.hidden_layers,
            num_rec=num_rec,
            propagation=False,
            aggr="sum",
        )

    def up(self, edge_index, hidden_dim):
        """Mesh level l to level l+1"""
        # The decoder uses InteractionNets here, prior and encoder
        # PropagationNets
        decoder = self.submodel == "decoder"
        if self.uses_attention("up"):
            return self._attention(
                edge_index,
                hidden_dim,
                hidden_dim,
                hidden_dim,
                residual="receiver" if decoder else "aggregate",
            )
        layer_class = InteractionNet if decoder else PropagationNet
        return layer_class(
            edge_index,
            hidden_dim,
            hidden_layers=self.hidden_layers,
            update_edges=False,
        )

    def down(self, edge_index, hidden_dim):
        """Mesh level l+1 to level l"""
        if self.uses_attention("down"):
            return self._attention(
                edge_index,
                hidden_dim,
                hidden_dim,
                hidden_dim,
                residual="aggregate",
            )
        return PropagationNet(
            edge_index,
            hidden_dim,
            hidden_layers=self.hidden_layers,
            update_edges=False,
        )

    def intra(
        self,
        edge_index,
        num_layers,
        hidden_dim,
        layer_type="interaction",
        layer_heads=None,
    ):
        """
        Sequence of same-level layers, mapping (mesh_rep, edge_rep) to
        (mesh_rep, edge_rep).

        layer_type, layer_heads: the --decoder_intra_layer option, which
            selects StaticEdgeAttention or the static-edge control. It can
            not be combined with an attention site here.
        """
        if not self.uses_attention("intra"):
            return utils.make_gnn_seq(
                edge_index,
                num_layers,
                self.hidden_layers,
                hidden_dim,
                layer_type=layer_type,
                attention_heads=layer_heads,
            )
        if layer_type != "interaction":
            raise ValueError(
                f"{self.submodel}.intra is an attention site, so the "
                f"same-level layer type must stay 'interaction', "
                f"got '{layer_type}'"
            )
        return utils.make_gnn_seq(
            edge_index,
            num_layers,
            self.hidden_layers,
            hidden_dim,
            layer_factory=lambda: self._attention(
                edge_index,
                hidden_dim,
                hidden_dim,
                hidden_dim,
                residual="receiver",
                return_edges=True,
            ),
        )
