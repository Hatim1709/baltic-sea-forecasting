"""
Configurable activation (gradient) checkpointing.

Background
----------
Training needs the intermediate values ("activations") of the forward pass to
compute gradients. Normally PyTorch keeps all of them in GPU memory until the
backward pass. *Checkpointing* keeps only the inputs of a chosen block, throws
the block's internal values away, and recomputes them during backward. This
saves memory and costs extra compute. It does not change the model's maths.

Where checkpoints can be placed
-------------------------------
* ``unroll``   -- around one whole forecast step (this existed before, see
  ``ARModel.unroll_ckpt_func``). The memory kept between steps is small, but
  while one step is recomputed all of its internal values exist at once.
* ``internal`` -- around every single GNN layer (``InteractionNet``,
  ``PropagationNet``, ``FlexibleNet``). Only the inputs/outputs of each layer
  are kept, so the peak is the size of ONE layer's internals, at the price of
  one extra forward pass.

Policies (the value of ``--ckpt_policy``)
-----------------------------------------
``auto``             Previous behaviour: step-level checkpointing when more
                     than one step is unrolled, or when ``--grad_checkpointing``
                     is given. No internal checkpoints. This is the default, so
                     nothing changes unless a policy is asked for.
``off``              No checkpointing at all (even for multi-step rollouts).
``unroll``           Step-level checkpointing, also for 1-step phases.
``internal``         Per-GNN-layer checkpointing only.
``unroll+internal``  Both, nested. Smallest memory, but the forward pass is
                     recomputed twice, so the slowest.
``selective``        Like ``internal``, but with *selective* activation
                     checkpointing (SAC, PyTorch >= 2.5): inside each GNN layer
                     the outputs of the expensive matrix multiplications (the
                     Linear layers of the edge/node MLPs) are KEPT, and only
                     the cheap operations (gathering node values onto edges,
                     concatenation, SiLU, LayerNorm, summing over edges) are
                     recomputed. Less recompute time than ``internal``, more
                     memory.
``unroll+selective`` Step-level checkpointing with the selective layers nested.

Checkpointing is only ever active while training with gradients enabled, never
in evaluation or under ``torch.no_grad()``.
"""

# Standard library
from functools import partial

# Third-party
import torch
import torch.utils.checkpoint
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

CKPT_POLICIES = (
    "auto",
    "off",
    "unroll",
    "internal",
    "unroll+internal",
    "selective",
    "unroll+selective",
)
# Value of a layer's ``ckpt_inner`` switch for selective checkpointing
# (plain True means ordinary full per-layer checkpointing)
SELECTIVE = "selective"

# policy -> (unroll checkpointing, internal checkpointing); "auto" is special
_FIXED_POLICIES = {
    "off": (False, False),
    "unroll": (True, False),
    "internal": (False, True),
    "unroll+internal": (True, True),
    "selective": (False, SELECTIVE),
    "unroll+selective": (True, SELECTIVE),
}


def policy_flags(policy, force_unroll, ar_steps_train):
    """
    Translate a policy name into two switches.

    policy: one of CKPT_POLICIES
    force_unroll: value of the older --grad_checkpointing flag (only used by
        the "auto" policy)
    ar_steps_train: number of unrolled steps in the current training phase

    Returns:
    (unroll, internal): step-level switch (bool) and per-layer switch (False,
    True for full layer checkpointing, or SELECTIVE)
    """
    if policy == "auto":
        return bool(force_unroll) or ar_steps_train > 1, False
    if policy not in _FIXED_POLICIES:
        raise ValueError(
            f"Unknown checkpoint policy '{policy}', choose from {CKPT_POLICIES}"
        )
    return _FIXED_POLICIES[policy]


def step_is_checkpointed(step, unroll_steps):
    """
    Whether forecast step ``step`` is a checkpointed block when step-level
    checkpointing is on. ``unroll_steps`` K: only steps 0..K-1 are
    checkpointed; -1 (or no step index) means every step.
    """
    return step is None or unroll_steps < 0 or step < unroll_steps


# Event counters, used by the benchmark to check that a policy really did
# what it claims (e.g. that a recomputation happened). They count Python-level
# events, cost nothing measurable and never affect results.
#   layer_calls   executions of a GNN layer's computation, including those
#                 repeated during backward (recomputation)
#   layer_blocks  per-layer checkpoint blocks started in a forward pass
#   unroll_blocks per-step checkpoint blocks started in a forward pass
_COUNTS = {"layer_calls": 0, "layer_blocks": 0, "unroll_blocks": 0}


def reset_counts():
    """Set all event counters to zero"""
    for key in _COUNTS:
        _COUNTS[key] = 0


def get_counts():
    """Return a copy of the event counters"""
    return dict(_COUNTS)


def count_layer_call():
    """Called each time a GNN layer's computation runs"""
    _COUNTS["layer_calls"] += 1


def count_unroll_block():
    """Called each time a forecast step is started as a checkpointed block"""
    _COUNTS["unroll_blocks"] += 1


def internal_active(module):
    """
    Whether a GNN layer should currently run as a checkpointed block.
    Needs the layer's own switch, training mode and recording of gradients
    (checkpointing would only add overhead in evaluation / no_grad).
    """
    return module.ckpt_inner and module.training and torch.is_grad_enabled()


# Operations whose results are kept (not recomputed) by selective
# checkpointing: the matrix multiplications behind every torch.nn.Linear. They
# dominate the compute of a GNN layer, while everything else (index gathers,
# concatenation, SiLU, LayerNorm, scatter-sum) is cheap to redo.
_aten = torch.ops.aten
SAVED_OPS = (_aten.mm.default, _aten.addmm.default, _aten.bmm.default)


def selective_policy(ctx, op, *args, **kwargs):
    """
    Decide per operation whether SAC keeps its result or recomputes it
    (this is the ``policy_fn`` of PyTorch's SAC API).
    """
    if op in SAVED_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def run_checkpointed(func, *args, selective=False):
    """
    Call func(*args) as a checkpointed block: its internal values are not kept
    for backward but recomputed from args when needed. The non-reentrant
    variant is used, as recommended by PyTorch and as in ARModel.

    selective: if True, use selective checkpointing: results of the
        operations in SAVED_OPS are kept, all others recomputed.
    """
    _COUNTS["layer_blocks"] += 1
    extra = {}
    if selective:
        extra["context_fn"] = partial(
            create_selective_checkpoint_contexts, selective_policy
        )
    return torch.utils.checkpoint.checkpoint(
        func, *args, use_reentrant=False, **extra
    )


def set_internal_checkpointing(root, enabled):
    """
    Switch per-layer checkpointing (off, on, or SELECTIVE) for every GNN layer
    below `root`.

    Only the layer classes define the attribute ``ckpt_inner``, so this finds
    exactly those. Returns the number of layers that were switched.
    """
    count = 0
    for module in root.modules():
        if hasattr(module, "ckpt_inner"):
            # False / True / SELECTIVE (see run_checkpointed)
            module.ckpt_inner = (
                SELECTIVE if enabled == SELECTIVE else bool(enabled)
            )
            count += 1
    return count
