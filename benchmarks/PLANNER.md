# Planner: proposed interface (not implemented)

Goal: choose a checkpoint policy per GNN layer so that a training step fits
the memory of each GPU with as little recomputation as possible. Today the
policy is one global setting (`--ckpt_policy`), plus `--ckpt_unroll_steps`.

## Inputs
- memory per GPU (MiB) and a safety margin
- per GPU: local node and edge counts of every graph level (grid, mesh
  levels, grid-mesh edges: the paper calls these the dominant memory cost),
  and halo (boundary) sizes
- widths, number of processing layers, forecast steps T, precision

## Output
- `unroll_steps` K: how many forecast steps to checkpoint
- per layer: `keep` | `selective` | `recompute` (full checkpoint)

## Cost model
Per layer measure (or estimate from node/edge counts) saved memory m_i
and recompute time t_i for each option. Peak = fixed + sum of kept m_i
+ largest single recomputed block. Choose options minimizing sum t_i
subject to peak <= budget (greedy by saved MiB per ms, like the Jane Street
approach; small enough to solve exactly if needed).
First version: `benchmarks/plan_unroll_steps.py` (step level only).

## Hook points
- `ARModel.update_checkpointing(ar_steps_train)`: apply the plan per phase
- `checkpointing.set_internal_checkpointing(model, mode)`: currently one mode
  for all layers; needs a per-layer mapping
- `ARModel.unroll_ckpt_func(..., step=i)`: already takes the step index

## Open questions
- Partitioned graph: store halo features (memory) or recompute them
  (repeats communication)? Plan per GPU, since partitions are unequal.
- Does the 1.8x recompute factor hold at 0.25° and on MI250X?
- Whole-model torch.compile + budget: untested; needs one clean graph.
