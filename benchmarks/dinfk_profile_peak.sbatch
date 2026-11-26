#!/bin/bash
#SBATCH --job-name=profile-peak
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=profile-peak-%j.out
#
# D-INFK student cluster (or any Slurm GPU node): where does the memory peak
# of `unroll` occur? See benchmarks/profile_peak.py.
#
# Set up once (adapt the first two lines to your cluster account):
#   export REPO=$HOME/baltic-sea-forecasting     # checkout, branch nils
#   export DATA=$HOME/njord-data                 # folder containing sample_data/
#   export VENV=$HOME/venvs/njord                # python env with the project deps
# Then:
#   sbatch [--partition=<gpu partition>] [--account=<account>] \
#       benchmarks/dinfk_profile_peak.sbatch
# Smaller cards: WIDTH=512 sbatch ...   (dry check: DRY_RUN=1 bash ...)
set -eu
REPO=${REPO:-$HOME/baltic-sea-forecasting}
DATA=${DATA:-$HOME/njord-data}
VENV=${VENV:-$HOME/venvs/njord}
WIDTH=${WIDTH:-1024}
OUT=${OUT:-$DATA/profile_runs_$(date -u +%Y%m%dT%H%M%SZ)}

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
