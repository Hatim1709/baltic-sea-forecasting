# Checkpointing results (reference workload, 1 x A100 80 GB)

Workload: GraphEFM, hierarchical graph, d=1024, bf16, CRPS 0, latent 16
(Hatim's shared reference, 1° grid). Job 87918 on piora2, branch `nils`,
commit `e78376e`, torch 2.6.0+cu124. Memory in MiB, time relative to `off`.

## What the policies mean

| Policy | What is thrown away and recomputed in backward |
|---|---|
| `off` | nothing (all activations kept) |
| `unroll` | everything inside one forecast step (only the state between steps is kept) |
| `internal` | everything inside each GNN layer |
| `selective` | per layer, but matrix-multiplication results are kept (PyTorch SAC) |
| `unroll+selective` | both (nested) |

## Peak memory

| Policy | T=2 | T=4 | T=6 | growth per extra step |
|---|---|---|---|---|
| off | 29692 | 51619 | 73540 | ~10.7 GiB |
| selective | 21859 | 34567 | 47270 | ~6.2 GiB |
| internal | 17218 | 24626 | 32028 | ~3.6 GiB |
| unroll | 18734 | 18780 | 18814 | ~20 MiB |
| unroll+selective | 19074 | 24418 | 29763 | ~2.7 GiB |

Time: `internal` 1.79x, `selective` 1.76x, `unroll` 1.81x,
`unroll+selective` about 2.57x of `off` (off: 1424 / 2827 / 4242 ms).

## Findings

1. Every useful policy costs about 1.8x time. Nothing is cheaper than
   `unroll` for memory, and `selective` saves little over `internal`.
2. Memory is linear in T: peak = S + T*a (S about 7.8 GiB fixed, a about
   10.7 GiB per step). `unroll` pays S + one step, because only the small
   state between steps is stored (hence ~20 MiB per extra step).
3. Nesting (`unroll+selective`) is dominated: more time, no memory gain.
4. Extrapolated fit on 80 GB: `off` up to T=6, `selective` about T=11,
   `internal` about T=19, `unroll` unbounded (floor: one step must fit).
5. Cheap gain available: checkpoint only as many steps as the budget needs
   (`--ckpt_unroll_steps K`, `plan_unroll_steps.py`). Predicted, not yet run
   on a GPU: T=6, K=3 gives about 40,654 MiB at about 1.4x time.

## torch.compile

It ran only with `CC=/usr/bin/gcc CXX=/usr/bin/g++` (now set in the sbatch).
The 84 graph breaks came from one line: `num_rec = edge_index[1].max() + 1`
is a tensor, but PyG's scatter needs a plain size. Fixed with `int(...)`
(`d1a7b05`); zero graph breaks afterwards, same gradients. Whole-model
compile with a memory budget is still untested.

## Caveats

- Benchmark is 1° with d=1024; the paper's production run is 0.25°, so the
  1.8x may not transfer. The paper trains mostly with T=1; T=7 only in the
  final fine-tuning (about 1,970 GPUh of about 8,930).
- Recompute costs about 0.78 of a forward+backward step; why is unexplained
  (a per-operation profile would tell).
- Single GPU only. On a partitioned graph recomputation can repeat
  communication unless halo data is stored (see PLANNER.md).
