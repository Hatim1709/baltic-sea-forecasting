"""
Optional torch.compile of the GNN layer computations (--compile_layers).

torch.compile turns the operations of a layer into fused GPU kernels. For
Njord's layers most of the time is spent moving data (gathering node values
onto edges, concatenating, SiLU, LayerNorm, summing over edges), so fusing
them can make the forward pass, and therefore also every recomputation of a
checkpointing policy, cheaper.

Only the numerical body of each layer (``_forward_impl``) is compiled. The
checkpoint blocks and the event counters in ``forward``/``_run_layer`` stay in
ordinary Python, so all checkpoint policies and their audits keep working.
The mathematical result is unchanged.
"""

# Third-party
import torch


def compile_layers(root, enabled=True, **compile_kwargs):
    """
    Compile (or restore) ``_forward_impl`` of every GNN layer below `root`.

    compile_kwargs: passed to torch.compile (default: dynamic=False, since
    shapes are fixed during training).
    Returns the number of layers switched.
    """
    kwargs = {"dynamic": False, **compile_kwargs}
    count = 0
    # Every layer instance can need its own compiled graph; with dynamo's
    # default limit of 8 per function the later layers would silently stay
    # eager.
    if enabled:
        total = sum(1 for m in root.modules() if hasattr(m, "ckpt_inner"))
        dynamo_config = torch._dynamo.config
        for name in ("cache_size_limit", "accumulated_cache_size_limit"):
            if hasattr(dynamo_config, name):
                setattr(
                    dynamo_config,
                    name,
                    max(getattr(dynamo_config, name), 4 * total + 16),
                )
    for module in root.modules():
        if not hasattr(module, "ckpt_inner"):  # only the GNN layer classes
            continue
        if enabled:
            eager = getattr(module, "_forward_impl_eager", None)
            if eager is None:
                eager = module._forward_impl
                module._forward_impl_eager = eager
            module._forward_impl = torch.compile(eager, **kwargs)
        elif hasattr(module, "_forward_impl_eager"):
            module._forward_impl = module._forward_impl_eager
            del module._forward_impl_eager
        count += 1
    return count
