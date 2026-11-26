# Checkpointing work: status and roadmap

Status as of 2026-11-22. "Verified" means run on CPU in tests; Hatim's GPU
results (steps 1-3) are in RESULTS.md.

## Done (branch `nils`, on top of the pinned commit `288164d`)

- [x] `a09c656` (Hatim) step timing and GPU memory written to a CSV
- [x] `2614837` `--ckpt_policy`: `auto` (old behaviour), `off`, `unroll`
      (whole forecast step), `internal` (each GNN layer), `unroll+internal`;
      tests check identical outputs and gradients (CPU)
- [x] `a393c1b` merge of the timing callback into `nils`
- [x] `e8fe696` event counters + benchmark runner (fresh process per launch,
      audit that each policy really recomputed, bootstrap intervals)
- [x] `eae063d` selective checkpointing (SAC): `selective`,
      `unroll+selective`; `compile_budget_probe.py` for `torch.compile` with a
      memory budget (single layer)
- [x] `ee391c9` shared reference-workload check, `preflight.py`, one-job script
      `piora_ckpt_steps_1_3.sbatch` (steps 1-3)
- [x] `877b876` tidy-up
- [x] Steps 1-3 run on Piora (job 87918), see RESULTS.md
- [x] `d1a7b05` `num_rec` as Python int: removes torch.compile graph breaks
- [x] `--ckpt_unroll_steps K` + `plan_unroll_steps.py` (partial unroll)
- [x] Planner interface written down: PLANNER.md (not implemented)

## Next (in order)

Decision after the supervisor meeting (2026-11-26): first find out whether more
checkpointing work beats plain `unroll` at all.

- [x] Profiling tools written (CPU-tested): `neural_lam/profiling.py`
      (`--profile_dir`), `benchmarks/profile_peak.py`, `analyze_profile.py`,
      `analyze_snapshot.py`, `dinfk_profile_peak.sbatch`
- [ ] Run `profile_peak.py` on one GPU (student cluster): peak per phase of
      `unroll`, saved memory and forward time per region
- [ ] Go/no-go from `analyze_profile.py`: keep some regions stored if about
      1 GiB/step buys at least about 15% less time than `unroll`;
      otherwise report that `unroll` is the answer
- [ ] If go: split the step at the latent, keep cheap-to-store regions, or
      `torch.compile` a layer, depending on where the peak is

- [ ] Optional GPU check (Hatim): T=6, `unroll` with K=3, predicted about
      40.7 GiB and 1.4x time
- [ ] Profile memory and time per layer / mesh level
- [ ] Per-layer planner: keep, selective, or recompute per layer under a
      memory budget (today the policy is one global setting)
- [ ] Automatic planner after the Jane Street blog post (greedy drop of saved
      values under a budget, via `torch.compile` custom partitioner). Start
      on a separate branch; only useful if `torch.compile` works on Piora.
      Compare against `selective` / `internal` / PyTorch's budget setting.

## Possible extensions (proposals, not started)

- [ ] Avoid storing the large edge-message input (gather + concat of edge,
      sender, receiver features): fused kernel or chunked edge processing
- [ ] Offload saved activations to CPU memory and overlap the copy with compute
- [ ] CUDA graphs or streams to cut launch overhead of recomputation
- [ ] GPU-partitioned mesh: for boundary (halo) nodes, compare recomputing
      against communicating; overlap recompute with communication
- [ ] Per-GPU planner: memory budget per partition (unequal partitions)
- [ ] Larger width / longer rollout: find where each policy stops fitting
