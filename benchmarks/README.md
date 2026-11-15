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
