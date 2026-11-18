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

```bash
# 1. get the code (fork of the model, branch nils)
cd ~/njord-dphpc/external/njord
git fetch origin && git checkout nils && git pull --ff-only

# 2. environment as in experiments/README.md
source /home/hatim1709/.venvs/njord-learning/bin/activate
export WANDB_MODE=offline

# 3. look at the plan first (starts nothing)
python benchmarks/run_ckpt_policy_benchmark.py --dry_run

# 4. real run on ONE GPU, from the folder that contains sample_data/
cd /scratch/hatim1709/njord
srun <your usual single-GPU options> \
  python ~/njord-dphpc/external/njord/benchmarks/run_ckpt_policy_benchmark.py \
    --workdir /scratch/hatim1709/njord --output_dir ckpt_policy_runs \
    --width 1024 --steps 2 --launches 6 --measured 20
```

Repeat with `--steps 4` and `--steps 6` to compare with the earlier ON/OFF
scaling study (use the same `--width`). To find where `off` stops fitting, add
larger `--steps`; the OOM launches are recorded.


## Continue after interrupted CSV collection

CSV collection supports separate work and output filesystems, such as
`/scratch` and `/home`. For an existing session whose training completed before
CSV collection failed, first recover and audit it on the login node:

```bash
python benchmarks/resume_ckpt_policy_benchmark.py /path/to/session --dry_run
```

Then run the same command without `--dry_run` inside a one-GPU allocation.
It restores arguments from `manifest.json`, preserves the original schedule,
and skips launches already in its recovery journal. A recovered launch requires
a complete CSV that passes the policy and finite-loss checks, an identical
training command, and an explicit training-completion log. An incomplete
launch requires inspection.

Recovery leaves the original exit code and whole-process duration unknown.
Each recovered launch has a `recovery.json`, and each continuation records
placement and software provenance. Review this allocation information when
interpreting comparisons whose repeats span the interruption. The original
manifest remains the record of the initial collection.

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
* Nothing here has been run on a GPU yet; the tests (`tests/test_checkpoint_policy.py`,
  `tests/test_ckpt_benchmark.py`) run on CPU with a fake training program.

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
