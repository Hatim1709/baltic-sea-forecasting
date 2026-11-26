#!/bin/bash
#SBATCH --job-name=profile-peak
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=profile-peak-%j.out
#
# ETH D-INFK student cluster (isg.inf.ethz.ch, "Student Cluster"): where does
# the memory peak of `unroll` occur? See benchmarks/profile_peak.py.
# The cluster's GPUs are small (RTX 5060 Ti 16 GiB, 2080 Ti / 1080 Ti 11 GiB),
# so the default width is 512; the GB10 (128 GiB) can run WIDTH=1024.
#
# Set up once (see benchmarks/README.md), then:
#   export REPO=/scratch/users/$USER/baltic-sea-forecasting
#   export DATA=/scratch/users/$USER/njord-data      # contains sample_data/
#   export VENV=/scratch/users/$USER/venv
#   sbatch -A <your course/project tag> benchmarks/dinfk_profile_peak.sbatch
# Other card:  sbatch -A <tag> --gpus=gb10:1 ...   (WIDTH=1024 for gb10)
# Dry check on the login node:  DRY_RUN=1 bash benchmarks/dinfk_profile_peak.sbatch
set -eu
REPO=${REPO:-$HOME/baltic-sea-forecasting}
DATA=${DATA:-$HOME/njord-data}
VENV=${VENV:-$HOME/venvs/njord}
WIDTH=${WIDTH:-512}
OUT=${OUT:-$DATA/profile_runs_$(date -u +%Y%m%dT%H%M%SZ)}

. /etc/profile.d/modules.sh 2>/dev/null || true
module add cuda/12.9 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++}
[ -d "$DATA/sample_data" ] || { echo "no sample_data/ in $DATA" >&2; exit 1; }
echo "commit $(git -C "$REPO" rev-parse --short HEAD), branch $(git -C "$REPO" branch --show-current)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
extra=""
[ "${DRY_RUN:-0}" = 1 ] && extra="--dry_run"
python "$REPO/benchmarks/profile_peak.py" --workdir "$DATA" \
    --output_dir "$OUT" --width "$WIDTH" $extra
