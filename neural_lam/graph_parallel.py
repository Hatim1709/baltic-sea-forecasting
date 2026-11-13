"""Two-rank receiver-owned execution of the existing Njord graph layers.

This integration bridge keeps the model's global tensor interface: layer
boundaries and feature embedders remain replicated. Edge/receiver MLP work is
partitioned, with differentiable halo exchange, before outputs are assembled
back into global order. It is not persistent sharding of all GraphEFM state.

Each rank must execute the same model and scalar loss divided by group size.
After backward, call ``sum_parameter_gradients`` exactly once. Do not also
wrap this model in DistributedDataParallel, which would reduce twice.
"""

# Standard library
from dataclasses import dataclass
from itertools import count
from typing import Dict

# Third-party
import torch
import torch.distributed as dist
from torch import nn

# Local
from .interaction_net import FlexibleNet, InteractionNet

_TAGS = count()


def _group_info(group):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed before graph sharding")
    if dist.get_world_size(group) != 2:
        raise ValueError(
            "The initial graph-parallel backend requires two ranks"
        )
    rank = dist.get_rank(group)
    peer = 1 - rank
    global_peer = peer if group is None else dist.get_global_rank(group, peer)
    return rank, global_peer


def _integer_ids(value, label):
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"{label} must be an integer tensor")
    return value.detach().to(device="cpu", dtype=torch.long).contiguous()


@dataclass
class ReceiverPartition:
    """CPU index metadata; sender and receiver IDs have separate namespaces."""

    owned_sender_ids: torch.Tensor
    owned_node_ids: torch.Tensor
    halo_node_ids: torch.Tensor
    sender_global_ids: torch.Tensor
    owned_edge_ids: torch.Tensor
    src_local: torch.Tensor
    dst_local: torch.Tensor
    send_local_indices: torch.Tensor
    send_ids: torch.Tensor
    recv_ids: torch.Tensor
    receiver_degree: torch.Tensor
    sender_count: int
    receiver_count: int
    edge_count: int


def build_receiver_partition(edge_index, sender_owner, receiver_owner, rank):
    """Keep all incoming edges of owned receivers, in original edge order."""
    edges = _integer_ids(edge_index, "edge_index")
    send_owner = _integer_ids(sender_owner, "sender_owner")
    recv_owner = _integer_ids(receiver_owner, "receiver_owner")
    if rank not in (0, 1):
        raise ValueError("Rank must be zero or one")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    for owners in (send_owner, recv_owner):
        if owners.ndim != 1 or not ((owners == 0) | (owners == 1)).all():
            raise ValueError("Ownership must be a vector of zero/one labels")
    for endpoints, owners in zip(edges, (send_owner, recv_owner)):
        if endpoints.numel() and (
            endpoints.min() < 0 or endpoints.max() >= owners.numel()
        ):
            raise ValueError("Edge endpoint outside its node namespace")

    src, dst = edges
    owned_src = torch.where(send_owner == rank)[0]
    owned_dst = torch.where(recv_owner == rank)[0]
    edge_ids = torch.where(recv_owner[dst] == rank)[0]
    local_src, local_dst = src[edge_ids], dst[edge_ids]
    halo = torch.unique(local_src[send_owner[local_src] != rank], sorted=True)
    senders = torch.cat((owned_src, halo))
    src_map = torch.full_like(send_owner, -1)
    src_map[senders] = torch.arange(senders.numel())
    dst_map = torch.full_like(recv_owner, -1)
    dst_map[owned_dst] = torch.arange(owned_dst.numel())
    send_ids = torch.unique(
        src[(send_owner[src] == rank) & (recv_owner[dst] != rank)],
        sorted=True,
    )
    return ReceiverPartition(
        owned_sender_ids=owned_src,
        owned_node_ids=owned_dst,
        halo_node_ids=halo,
        sender_global_ids=senders,
        owned_edge_ids=edge_ids,
        src_local=src_map[local_src],
        dst_local=dst_map[local_dst],
        send_local_indices=src_map[send_ids],
        send_ids=send_ids,
        recv_ids=halo,
        receiver_degree=torch.bincount(dst, minlength=recv_owner.numel())[
            owned_dst
        ],
        sender_count=send_owner.numel(),
        receiver_count=recv_owner.numel(),
        edge_count=edges.shape[1],
    )


def _exchange(send, recv, peer, group, tag):
    ops = []
    if recv.numel():
        ops.append(dist.P2POp(dist.irecv, recv, peer, group, tag))
    if send.numel():
        ops.append(dist.P2POp(dist.isend, send, peer, group, tag))
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()


class _HaloExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, send_indices, halo_count, peer, group, tag):
        ctx.save_for_backward(send_indices)
        ctx.owned_count = owned.shape[-2]
        ctx.peer, ctx.group, ctx.tag = peer, group, tag
        send = owned.index_select(-2, send_indices).contiguous()
        shape = list(owned.shape)
        shape[-2] = halo_count
        recv = owned.new_empty(shape)
        _exchange(send, recv, peer, group, tag)
        return torch.cat((owned, recv), dim=-2)

    @staticmethod
    def backward(ctx, gradient):
        (send_indices,) = ctx.saved_tensors
        owned_gradient = gradient[..., : ctx.owned_count, :].contiguous()
        owned_gradient = owned_gradient.clone()
        send = gradient[..., ctx.owned_count :, :].contiguous()
        shape = list(gradient.shape)
        shape[-2] = send_indices.numel()
        recv = gradient.new_empty(shape)
        _exchange(send, recv, ctx.peer, ctx.group, ctx.tag + 1)
        if send_indices.numel():
            owned_gradient.index_add_(-2, send_indices, recv)
        return owned_gradient, None, None, None, None, None


class _AssembleGlobal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, global_ids, total_count, group):
        ctx.save_for_backward(global_ids)
        ctx.group = group
        shape = list(local.shape)
        shape[-2] = total_count
        result = local.new_zeros(shape)
        result.index_copy_(-2, global_ids, local)
        if result.numel():
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        (global_ids,) = ctx.saved_tensors
        summed = gradient.contiguous().clone()
        if summed.numel():
            dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=ctx.group)
        return summed.index_select(-2, global_ids), None, None, None


class ReceiverOwnedLayer(nn.Module):
    """Execute an existing layer locally and preserve its global tensor API.

    ``edge_index`` uses separate zero-based sender and receiver namespaces;
    it must not contain InteractionNet's internal concatenation offset.
    The passed layer becomes ``local_layer`` and retains its MLP parameters.
    Install this adapter after loading ordinary model weights and before
    creating the optimizer. Wrapped checkpoint keys gain ``local_layer``.
    """

    def __init__(
        self, layer, edge_index, sender_owner, receiver_owner, group=None
    ):
        super().__init__()
        rank, peer = _group_info(group)
        if not isinstance(layer, (InteractionNet, FlexibleNet)):
            raise TypeError("Expected an InteractionNet or FlexibleNet layer")
        if hasattr(layer.edge_mlp, "chunk_sizes") or hasattr(
            layer.aggr_mlp, "chunk_sizes"
        ):
            raise NotImplementedError("SplitMLPs require a separate adapter")
        if int(layer.num_rec) != len(receiver_owner):
            raise ValueError("Receiver ownership disagrees with layer.num_rec")
        self.plan = build_receiver_partition(
            edge_index, sender_owner, receiver_owner, rank
        )
        self.group, self.peer = group, peer
        self.tag = 2 * next(_TAGS)
        self.local_layer = layer
        for field in (
            "owned_sender_ids",
            "owned_node_ids",
            "owned_edge_ids",
            "send_local_indices",
        ):
            self.register_buffer(
                "_" + field, getattr(self.plan, field), persistent=False
            )
        local_edges = torch.stack((self.plan.src_local, self.plan.dst_local))
        count_rec = self.plan.owned_node_ids.numel()
        if isinstance(layer, InteractionNet):
            local_edges[0] += count_rec
        layer.edge_index = local_edges.to(layer.edge_index.device)
        layer.num_rec = count_rec

    def forward(self, send_rep, rec_rep, edge_rep, emb=None):
        if any(
            getattr(module, "mc_dropout_p", 0.0) > 0.0
            or (
                isinstance(module, nn.Dropout)
                and module.training
                and module.p > 0.0
            )
            for module in self.local_layer.modules()
        ):
            raise NotImplementedError(
                "Graph-parallel layers require deterministic MLPs; "
                "globally indexed dropout is not implemented"
            )
        if send_rep.shape[-2] != self.plan.sender_count:
            raise ValueError("Wrong global sender count")
        if rec_rep.shape[-2] != self.plan.receiver_count:
            raise ValueError("Wrong global receiver count")
        if edge_rep.shape[-2] != self.plan.edge_count:
            raise ValueError("Wrong global edge count")
        source_ids = self._owned_sender_ids.to(send_rep.device)
        receiver_ids = self._owned_node_ids.to(rec_rep.device)
        edge_ids = self._owned_edge_ids.to(edge_rep.device)
        send_indices = self._send_local_indices.to(send_rep.device)
        owned_senders = send_rep.index_select(-2, source_ids)
        senders = _HaloExchange.apply(
            owned_senders,
            send_indices,
            self.plan.halo_node_ids.numel(),
            self.peer,
            self.group,
            self.tag,
        )
        local = self.local_layer(
            senders,
            rec_rep.index_select(-2, receiver_ids),
            edge_rep.index_select(-2, edge_ids),
            emb,
        )
        # Empty-edge ranks must participate in the halo backward too.
        dependency = senders[..., :0, :].sum()
        if isinstance(local, tuple):
            nodes, edges = local
            nodes = _AssembleGlobal.apply(
                nodes + dependency,
                receiver_ids,
                self.plan.receiver_count,
                self.group,
            )
            edges = _AssembleGlobal.apply(
                edges + dependency,
                edge_ids,
                self.plan.edge_count,
                self.group,
            )
            return nodes, edges
        return _AssembleGlobal.apply(
            local + dependency,
            receiver_ids,
            self.plan.receiver_count,
            self.group,
        )


def sum_parameter_gradients(module, group=None):
    """Sum partial derivatives once; no averaging and no additional DDP."""
    _group_info(group)
    for parameter in module.parameters():
        flag = torch.tensor(
            int(parameter.grad is not None),
            device=parameter.device,
            dtype=torch.int64,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.SUM, group=group)
        if not flag.item():
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)


def _levels(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [value[index] for index in range(len(value))]


def install_graph_parallel(model, owners: Dict[str, torch.Tensor], group=None):
    """Adapt all existing Interaction/Propagation/Flexible layers in a model.

    Owner keys are ``grid``, ``mesh_0``, ``mesh_1``, ... . Each relation is
    matched against the model's loaded global connectivity, preventing an
    accidental cross-level namespace match. The initial bridge supports a
    common input/output grid; separate auxiliary grid namespaces are pending.
    Returns operator counts by graph relation for run provenance.
    """
    _group_info(group)
    if any(isinstance(m, ReceiverOwnedLayer) for m in model.modules()):
        raise ValueError("Graph-parallel adapters are already installed")
    relations = []
    mesh_edges = _levels(model.m2m_edge_index)
    for level, edge_index in enumerate(mesh_edges):
        key = f"mesh_{level}"
        relations.append((f"m2m_{level}", edge_index, key, key))
    relations.extend(
        [
            ("g2m", model.g2m_edge_index, "grid", "mesh_0"),
            ("m2g", model.m2g_edge_index, "mesh_0", "grid"),
        ]
    )
    for direction in ("up", "down"):
        edge_list = _levels(getattr(model, f"mesh_{direction}_edge_index", []))
        for level, edge_index in enumerate(edge_list):
            source, receiver = level, level + 1
            if direction == "down":
                source, receiver = receiver, source
            relations.append(
                (
                    f"mesh_{direction}_{level}",
                    edge_index,
                    f"mesh_{source}",
                    f"mesh_{receiver}",
                )
            )
    required = {
        key for _, _, source, target in relations for key in (source, target)
    }
    if not required.issubset(owners):
        raise ValueError(f"Missing owner namespaces: {required - set(owners)}")
    # Validate every relation before mutating the model.
    normalized = []
    for name, edges, source, target in relations:
        build_receiver_partition(edges, owners[source], owners[target], 0)
        normalized.append((name, edges.detach().cpu(), source, target))
    counts = {name: 0 for name, _, _, _ in normalized}
    replacements = {}

    def adapt(parent):
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            if isinstance(child, (InteractionNet, FlexibleNet)):
                if id(child) in replacements:
                    parent._modules[name] = replacements[id(child)]
                    continue
                raw = child.edge_index.detach().cpu().clone()
                count_rec = int(child.num_rec)
                if isinstance(child, InteractionNet):
                    raw[0] -= count_rec
                matches = [
                    row
                    for row in normalized
                    if len(owners[row[3]]) == count_rec
                    and torch.equal(row[1], raw)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Cannot uniquely match layer {name} to graph relation"
                    )
                relation, edges, source, target = matches[0]
                wrapped = ReceiverOwnedLayer(
                    child, edges, owners[source], owners[target], group
                )
                replacements[id(child)] = wrapped
                parent._modules[name] = wrapped
                counts[relation] += 1
            else:
                adapt(child)

    adapt(model)
    if not replacements:
        raise ValueError("No supported graph layers found in the model")
    return counts
