#!/bin/bash
#SBATCH --job-name=ckpt-steps-1-3
#SBATCH --partition=defq
#SBATCH --nodelist=piora2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/hatim1709/njord/slurm-ckpt-steps-1-3-%j.out
#
# Checkpointing follow-up, steps 1-3, on the shared reference workload
# (GraphEFM, hierarchical graph, d=1024, bf16, CRPS 0, latent 16):
#   preflight  - checks code, GPU, data, reference workload, torch.compile
#   probe      - compile_budget_probe.py (torch.compile + memory budget, 1 layer)
#   t2         - policies off internal selective unroll unroll+selective, T=2
#   t4, t6     - the same policies at T=4 and T=6 (rollout scaling)
#
# Start it from the checkout root (the folder that contains benchmarks/):
#   module load slurm/slurm/23.02.7
#   sbatch benchmarks/piora_ckpt_steps_1_3.sbatch
# or, inside a GPU allocation you already have:
#   bash benchmarks/piora_ckpt_steps_1_3.sbatch
#
# Everything is optional and set through environment variables, e.g.
#   STAGES="t2" sbatch --export=ALL,STAGES=t2 benchmarks/piora_ckpt_steps_1_3.sbatch
#   sbatch --time=03:00:00 benchmarks/piora_ckpt_steps_1_3.sbatch
#   DRY_RUN=1 bash benchmarks/piora_ckpt_steps_1_3.sbatch   (no GPU needed)
# Stages that finished are skipped when the job is submitted again (FORCE=1
# repeats them), so a job that hit the time limit can simply be resubmitted.

set -u

STAGES=${STAGES:-"preflight probe t2 t4 t6"}
POLICIES=${POLICIES:-"off internal selective unroll unroll+selective"}
WIDTH=${WIDTH:-1024}
MEASURED=${MEASURED:-20}        # after 5 warmup updates (25 per launch)
LAUNCHES_T2=${LAUNCHES_T2:-6}   # fresh processes per policy at T=2
LAUNCHES_LONG=${LAUNCHES_LONG:-3}  # ... at T=4 and T=6 (longer updates)
WORKDIR=${WORKDIR:-/scratch/hatim1709/njord}   # contains sample_data/
VENV=${VENV:-/home/hatim1709/.venvs/njord-learning}
OUT=${OUT:-$WORKDIR/ckpt_steps_1_3}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

die() { echo "ERROR: $*" >&2; exit 1; }

# --- where is the checkout? (under sbatch this file is a copy in the spool)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ]; then :
elif [ -f "$here/benchmarks/run_ckpt_policy_benchmark.py" ]; then REPO=$here
elif [ -f "${SLURM_SUBMIT_DIR:-.}/benchmarks/run_ckpt_policy_benchmark.py" ]; then
    REPO=${SLURM_SUBMIT_DIR:-.}
else die "run from the checkout root (folder with benchmarks/), or set REPO=..."; fi
REPO="$(cd "$REPO" && pwd)"

# --- environment (a dry run only needs plain python)
if [ "$DRY_RUN" = 1 ]; then
    STAGES="t2 t4 t6"
    [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
else
    [ -f "$VENV/bin/activate" ] || die "virtual environment not found: $VENV"
    source "$VENV/bin/activate"
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
# torch.compile needs a C++ compiler. On Piora the gcc/13.3 module points to a
# path that does not exist, so use the system compiler (set before sourcing
# anything else to override: CC=... CXX=... sbatch ...).
export CC=${CC:-/usr/bin/gcc}
export CXX=${CXX:-/usr/bin/g++}
[ -d "$WORKDIR" ] || die "WORKDIR not found: $WORKDIR"
mkdir -p "$OUT" || die "cannot create $OUT"
cd "$WORKDIR" || die "cannot enter $WORKDIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
FAILED=""
COMPILE_OK=1

{
    echo "started   : $(date -u +%FT%TZ) on $(hostname)"
    echo "checkout  : $REPO"
    echo "branch    : $(git -C "$REPO" branch --show-current)"
    echo "commit    : $(git -C "$REPO" rev-parse HEAD)"
    echo "dirty     : $(git -C "$REPO" status --porcelain | wc -l) changed files"
    echo "stages    : $STAGES"
    echo "policies  : $POLICIES   width=$WIDTH measured=$MEASURED"
    echo "launches  : T=2: $LAUNCHES_T2, T=4/6: $LAUNCHES_LONG"
    echo "output    : $OUT"
} | tee "$OUT/run_info.txt"
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
    echo "WARNING: the checkout has uncommitted changes; results are not tied to the commit above."
fi

finish() {
    echo
    echo "=== summary $(date -u +%FT%TZ)"
    for f in "$OUT"/t*/ckpt_policy_*/summary.md; do
        [ -f "$f" ] || continue
        echo; echo "--- $f"; cat "$f"
    done
    [ -f "$OUT/probe.txt" ] && { echo; echo "--- probe"; cat "$OUT/probe.txt"; }
    if [ -n "$FAILED" ]; then echo; echo "STAGES THAT FAILED:$FAILED"; fi
    if [ "$DRY_RUN" != 1 ]; then
        tarball="$WORKDIR/ckpt_steps_1_3_$STAMP.tar.gz"
        tar czf "$tarball" -C "$(dirname "$OUT")" "$(basename "$OUT")" \
            && echo && echo "Results to send back: $tarball"
    fi
}
trap finish EXIT
trap 'exit 143' TERM INT

# run_stage NAME COMMAND... : skip finished stages, remember failures
run_stage() {
    local name=$1; shift
    if [ "$DRY_RUN" != 1 ] && [ -f "$OUT/$name.done" ] && [ "$FORCE" != 1 ]; then
        echo "[skip] stage $name already finished (FORCE=1 repeats it)"; return 0
    fi
    [ -d "$OUT/$name" ] && mv "$OUT/$name" "$OUT/$name.incomplete.$(date +%s)"
    echo; echo "=== stage $name  $(date -u +%FT%TZ)"
    if "$@"; then
        [ "$DRY_RUN" = 1 ] || touch "$OUT/$name.done"
    else
        local rc=$?
        echo "[FAILED] stage $name (exit code $rc)"
        FAILED="$FAILED $name"
    fi
}

stage_preflight() {
    python "$REPO/benchmarks/preflight.py" --workdir "$WORKDIR" 2>&1 \
        | tee "$OUT/preflight.txt"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = 2 ]; then
        COMPILE_OK=0
        echo "torch.compile unavailable: the probe stage will be skipped."
        return 0
    fi
    [ "$rc" = 0 ] || { echo "Blocking problem, stopping."; ABORT=1; }
    return "$rc"
}

stage_probe() {
    if [ "$COMPILE_OK" != 1 ]; then echo "skipped (torch.compile unavailable)"; return 0; fi
    (cd "$REPO" && python benchmarks/compile_budget_probe.py \
        --nodes 200000 --edges 1200000 --dim 512) 2>&1 | tee "$OUT/probe.txt"
    return "${PIPESTATUS[0]}"
}

# stage_bench STEPS LAUNCHES
stage_bench() {
    local steps=$1 launches=$2 dry=""
    [ "$DRY_RUN" = 1 ] && dry="--dry_run"
    # shellcheck disable=SC2086  (POLICIES is a word list)
    python "$REPO/benchmarks/run_ckpt_policy_benchmark.py" \
        --workdir "$WORKDIR" --output_dir "$OUT/t$steps" \
        --width "$WIDTH" --steps "$steps" --launches "$launches" \
        --measured "$MEASURED" --policies $POLICIES $dry
}

ABORT=0
for stage in $STAGES; do
    [ "$ABORT" = 1 ] && break
    case "$stage" in
        preflight) run_stage preflight stage_preflight ;;
        probe)     run_stage probe stage_probe ;;
        t2)        run_stage t2 stage_bench 2 "$LAUNCHES_T2" ;;
        t4)        run_stage t4 stage_bench 4 "$LAUNCHES_LONG" ;;
        t6)        run_stage t6 stage_bench 6 "$LAUNCHES_LONG" ;;
        *)         die "unknown stage '$stage' (preflight probe t2 t4 t6)" ;;
    esac
done

[ -z "$FAILED" ]
