# Checkpointing policy benchmark (single GPU)

Compares the four gradient checkpointing policies of
`neural_lam/checkpointing.py` on time and GPU memory per training update:

| policy | what is checkpointed | expected cost |
|---|---|---|
| `off` | nothing | fastest, most memory |
| `unroll` | each whole forecast step | about +1 forward pass |
| `internal` | each single GNN layer | about +1 forward pass |
| `unroll+internal` | both, nested | about +2 forward passes, least memory |

`run_ckpt_policy_benchmark.py` starts a fresh training process per launch
(random order inside each repeat), collects the per-update CSV written by
`StepMeasurements`, and **audits** it: event counters prove that each policy
really recomputed what it should (layer executions 1x / 2x / 2x / 3x of a plain
forward). A launch failing the audit is marked `audit_failed` and must not be
used. Out-of-memory launches are kept as results (`oom`).

## Run it (Piora, from the `nils` branch)

Easiest: the one-job script below. By hand, from the folder that contains
`sample_data/` and inside a one-GPU allocation:

```bash
cd ~/njord-dphpc/external/njord && git fetch origin && git switch nils && git pull --ff-only
source /home/hatim1709/.venvs/njord-learning/bin/activate
cd /scratch/hatim1709/njord
python ~/njord-dphpc/external/njord/benchmarks/run_ckpt_policy_benchmark.py \
  --workdir /scratch/hatim1709/njord --output_dir ckpt_policy_runs \
  --width 1024 --steps 2 --launches 6 --measured 20   # add --dry_run to only see the plan
```

Before starting, the script compares its command with the project's
`experiments/reference_workload.py` (found in `~/njord-dphpc`, or `--reference
PATH`) and stops if the workload differs. Width, steps and checkpoint policy
are study settings and may differ. Repeat with `--steps 4` and `--steps 6`;
larger `--steps` show where `off` stops fitting (OOM launches are recorded).

## What to send back

The whole session folder `ckpt_policy_runs/ckpt_policy_<timestamp>/`:
`summary.md` (table), `summary.csv/json`, `launches.csv` (one row per launch,
check the `status` and `audit_problems` columns), `manifest.json` (code
version, GPU, plan) and the per-launch logs/CSVs.
`--summarize_only <session_dir>` recomputes the tables from `launches.csv`.

## Notes

* Time ratios are ratios of mean update time over matched repeats, with a 95%
  bootstrap interval over launches (10,000 draws); memory is the peak
  `allocated` value of the measured updates.
* `max_rel_loss_diff_vs_off` compares each launch's loss with `off` from the
  same repeat (same seed). Small differences from GPU arithmetic are normal.
* Default policy `auto` (no `--ckpt_policy`) behaves exactly as before.
* Nothing here has been run on a GPU yet; the tests
  (`tests/test_checkpoint_policy.py`, `tests/test_ckpt_benchmark.py`) run on
  CPU with a fake training program.

## Selective checkpointing (SAC) and torch.compile

Two more policies exist: `selective` and `unroll+selective`. Per GNN layer they
keep the results of the matrix multiplications (the Linear layers) and
recompute only the cheap operations (gathers, concatenation, SiLU, LayerNorm,
summing over edges). Expect time between `off` and `internal`, memory between
`internal` and `off`. Add them to the benchmark with
`--policies off internal selective unroll unroll+selective`.

`compile_budget_probe.py` is a quick one-GPU check (single layer) of eager vs.
the two layer checkpointing modes vs. `torch.compile` with
`activation_memory_budget` 1.0 / 0.5 / 0.0. It prints time, peak memory and
gradient agreement, and reports compile failures instead of crashing. Run it
first to decide whether a whole-model compile comparison is worthwhile:
`PYTHONPATH=. python benchmarks/compile_budget_probe.py --nodes 200000 --edges 1200000 --dim 512`

## Steps 1-3 in one job (Hatim: only this is needed)

One script does preflight, the compile probe, and the benchmark at T=2, 4, 6 on
the shared reference workload (d=1024, bf16, CRPS 0, latent 16). It refuses to
start if the workload differs from `experiments/reference_workload.py`.

```bash
# project repo: get the reference workload (once)
cd ~/njord-dphpc && git switch main && git pull

# fork checkout: get the code
cd ~/njord-dphpc/external/njord
git fetch origin && git switch nils && git pull --ff-only

# no GPU needed: shows the plan and runs the reference check
DRY_RUN=1 bash benchmarks/piora_ckpt_steps_1_3.sbatch

# real run (about 3-6 h on one A100; resubmitting skips finished stages)
module load slurm/slurm/23.02.7
sbatch benchmarks/piora_ckpt_steps_1_3.sbatch
```

Send back the printed `ckpt_steps_1_3_<timestamp>.tar.gz` (in the work folder)
plus the branch and commit from `run_info.txt`. Options (environment
variables): `STAGES="preflight probe t2 t4 t6"`, `POLICIES`, `WIDTH`,
`LAUNCHES_T2`, `LAUNCHES_LONG`, `FORCE=1`. `preflight.py` can also be run alone
(`PYTHONPATH=. python benchmarks/preflight.py --workdir ...`); exit code 2 means
only `torch.compile` is unavailable, and the probe is then skipped.
Nothing in this job has been run on a GPU yet.

## Where is the peak of `unroll`? (profiling on any one GPU)

`profile_peak.py` runs `off` (1 forecast step) and `unroll` (2 steps) with
`--profile_dir`, then prints `report.txt`: peak memory of the forward, backward
and optimizer phases, memory saved and forward time per region of the step,
a prediction of keeping some regions stored, and what was alive at the peak.
On a Slurm cluster use `benchmarks/dinfk_profile_peak.sbatch` (set REPO, DATA,
VENV at the top; DATA is the folder with `sample_data/`). Not yet run on a GPU;
the CPU tests are in `tests/test_profiling.py`.
