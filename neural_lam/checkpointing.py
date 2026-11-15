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

Checkpointing is only ever active while training with gradients enabled, never
in evaluation or under ``torch.no_grad()``.
"""

# Third-party
import torch
import torch.utils.checkpoint

CKPT_POLICIES = ("auto", "off", "unroll", "internal", "unroll+internal")

# policy -> (unroll checkpointing, internal checkpointing); "auto" is special
_FIXED_POLICIES = {
    "off": (False, False),
    "unroll": (True, False),
    "internal": (False, True),
    "unroll+internal": (True, True),
}


def policy_flags(policy, force_unroll, ar_steps_train):
    """
    Translate a policy name into two switches.

    policy: one of CKPT_POLICIES
    force_unroll: value of the older --grad_checkpointing flag (only used by
        the "auto" policy)
    ar_steps_train: number of unrolled steps in the current training phase

    Returns:
    (unroll, internal): booleans, step-level and per-layer checkpointing
    """
    if policy == "auto":
        return bool(force_unroll) or ar_steps_train > 1, False
    if policy not in _FIXED_POLICIES:
        raise ValueError(
            f"Unknown checkpoint policy '{policy}', choose from {CKPT_POLICIES}"
        )
    return _FIXED_POLICIES[policy]


def internal_active(module):
    """
    Whether a GNN layer should currently run as a checkpointed block.
    Needs the layer's own switch, training mode and recording of gradients
    (checkpointing would only add overhead in evaluation / no_grad).
    """
    return module.ckpt_inner and module.training and torch.is_grad_enabled()


def run_checkpointed(func, *args):
    """
    Call func(*args) as a checkpointed block: its internal values are not kept
    for backward but recomputed from args when needed. The non-reentrant
    variant is used, as recommended by PyTorch and as in ARModel.
    """
    return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)


def set_internal_checkpointing(root, enabled):
    """
    Switch per-layer checkpointing on or off for every GNN layer below `root`.

    Only the layer classes define the attribute ``ckpt_inner``, so this finds
    exactly those. Returns the number of layers that were switched.
    """
    count = 0
    for module in root.modules():
        if hasattr(module, "ckpt_inner"):
            module.ckpt_inner = bool(enabled)
            count += 1
    return count
