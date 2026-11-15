"""
Single-GPU benchmark of the four gradient checkpointing policies.

For every policy in {off, unroll, internal, unroll+internal} (see
neural_lam/checkpointing.py) this script trains GraphEFM for a few updates in a
FRESH Python process, and compares time and GPU memory per training update.

What it does
------------
1. Builds a plan: `--launches` repeats, each containing every policy once, in
   random order inside each repeat (so slow drifts of the GPU do not favour a
   policy). The repeat index pairs the launches of different policies.
2. Starts `python -m neural_lam.train_model ...` once per launch. Training
   writes `step-measurements-*.csv` (see StepMeasurements in train_model.py);
   the script collects that file and the log into one folder per launch.
3. Audits each launch: the CSV must have the right number of rows, record the
   requested policy, and show the expected amount of recomputation (the event
   counters, see `infer_layer_executions`). A policy that silently did not take
   effect would otherwise produce plausible but meaningless numbers.
4. Summarizes per policy: mean update time with a 95% bootstrap interval over
   launch means, peak allocated/reserved memory, and paired time/memory ratios
   against `off`, following docs/benchmarking.md of the project repository.

Out-of-memory launches are recorded (status "oom") and the sweep continues:
"this policy does not fit" is a result. Any other failure is recorded too.

It never edits source files and needs exactly one visible GPU.
Run `--dry_run` first to see the plan and commands without starting anything.
"""

# Standard library
import argparse
import csv
import json
import os
import random
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

POLICIES = ("off", "unroll", "internal", "unroll+internal")
# Selective checkpointing (SAC) policies; not run by default, add them with
# --policies. They recompute the same layers as the full variants (the
# counters are identical); what differs is which operations are recomputed.
SELECTIVE_POLICIES = ("selective", "unroll+selective")
ALL_POLICIES = POLICIES + SELECTIVE_POLICIES
WARMUP = 5  # fixed in StepMeasurements (train_model.py)
BOOTSTRAP_DRAWS = 10_000
SUMMARY_FIELDS = (
    "policy",
    "launches_completed",
    "launches_planned",
    "mean_ms",
    "mean_ms_ci_low",
    "mean_ms_ci_high",
    "peak_allocated_mib_max",
    "peak_allocated_mib_mean",
    "peak_reserved_mib_max",
    "time_ratio_vs_off",
    "time_ratio_ci_low",
    "time_ratio_ci_high",
    "memory_ratio_vs_off",
)
LAUNCH_FIELDS = (
    "run",
    "repeat",
    "policy",
    "width",
    "forecast_steps",
    "status",
    "returncode",
    "measured_steps",
    "mean_ms",
    "median_ms",
    "stdev_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "layer_calls",
    "layer_blocks",
    "unroll_blocks",
    "layer_executions_plain_forward",
    "max_rel_loss_diff_vs_off",
    "audit_problems",
    "process_seconds",
)

# Same training setup as the existing width study of the project repository
# (experiments/run_width_study.py), so results are comparable with it.
# Options that this script sweeps or sets itself are added in training_args.
COMMON = shlex.split(
    """
--config_path sample_data/global_ocean_1.yaml
--model graph_efm --graph global_hierarchical_graph
--latent_dim 16 --processor_layers 3 --encoder_processor_layers 1
--prior_processor_layers 1 --hidden_layers 1 --input_steps 2
--batch_size 1 --num_workers 1 --precision bf16-mixed --num_nodes 1
--seed 42 --optimizer adamw --lr 0.001 --min_lr 0.001
--kl_beta 0.1 --crps_weight 0
--ar_steps_eval 2 --val_steps_to_log 1 2
--limit_val_batches 0 --num_sanity_val_steps 0
"""
)


# ---------------------------------------------------------------- planning


def build_plan(policies, launches, seed):
    """
    List of launches. Each repeat holds all policies once, in an order that is
    shuffled per repeat (blocked randomization). Reproducible through `seed`.
    """
    rng = random.Random(seed)
    plan = []
    for repeat in range(1, launches + 1):
        order = list(policies)
        rng.shuffle(order)
        for policy in order:
            plan.append(
                {
                    "run": f"{len(plan) + 1:02d}_{policy.replace('+', '-')}"
                    f"_r{repeat}",
                    "repeat": repeat,
                    "policy": policy,
                }
            )
    return plan


def training_args(width, steps, policy, measured, extra_args=""):
    """
    Command-line arguments of neural_lam.train_model for one launch.
    Widths follow the width study: D / D/2 / D/2 / D/4 for the main, grid,
    mesh-node and edge features. One epoch of `WARMUP + measured` updates.
    """
    args = list(COMMON)
    args += ["--hidden_dim", str(width)]
    args += ["--hidden_dim_grid", str(width // 2)]
    args += ["--hidden_dim_mesh_nodes", str(width // 2)]
    args += ["--hidden_dim_edge", str(width // 4)]
    args += ["--ar_steps_train", str(steps)]
    args += ["--epochs", "1"]
    args += ["--limit_train_batches", str(WARMUP + measured)]
    args += ["--ckpt_policy", policy]
    return args + shlex.split(extra_args)


# ------------------------------------------------------------------ audit


def infer_layer_executions(policy, counts, forecast_steps):
    """
    Check the event counters of one update against what the policy implies
    and return (N, problems). N is the number of GNN layer executions of a
    plain forward pass, recovered from the counters; it must be the same for
    all policies, because the model and its work are the same.

    Counters (checkpointing.py), with N = layer executions in a plain forward:
      off              layer_calls = N      blocks 0           unroll 0
      unroll           layer_calls = 2N     blocks 0           unroll = steps
      internal         layer_calls = 2N     blocks N           unroll 0
      unroll+internal  layer_calls = 3N     blocks 2N          unroll = steps
    (A checkpointed region is executed once in the forward pass and once more
    when backward needs its internals; nested regions repeat the inner one.)
    """
    calls = counts["layer_calls"]
    blocks = counts["layer_blocks"]
    unrolls = counts["unroll_blocks"]
    divisor, block_factor, unroll_expected = {
        "off": (1, 0, 0),
        "unroll": (2, 0, forecast_steps),
        "internal": (2, 1, 0),
        "unroll+internal": (3, 2, forecast_steps),
        "selective": (2, 1, 0),
        "unroll+selective": (3, 2, forecast_steps),
    }[policy]
    problems = []
    if calls % divisor or calls == 0:
        return None, [f"layer_calls={calls} not a multiple of {divisor}"]
    n = calls // divisor
    if blocks != block_factor * n:
        problems.append(f"layer_blocks={blocks}, expected {block_factor * n}")
    if unrolls != unroll_expected:
        problems.append(f"unroll_blocks={unrolls}, expected {unroll_expected}")
    return n, problems


def audit_launch(rows, item, width, steps, measured):
    """
    Validate the rows of one step-measurements CSV (list of dicts).
    Returns (measured_rows, counts, problems).
    """
    problems = []
    policy = item["policy"]
    if len(rows) != WARMUP + measured:
        problems.append(f"{len(rows)} rows, expected {WARMUP + measured}")
    for index, row in enumerate(rows, 1):
        expected_phase = "warmup" if index <= WARMUP else "measured"
        if int(row["step"]) != index or row["phase"] != expected_phase:
            problems.append(f"unexpected step/phase at row {index}")
            break
    if any(row["ckpt_policy"] != policy for row in rows):
        problems.append("recorded ckpt_policy differs from the requested one")
    want_unroll = policy.startswith("unroll")
    # Recorded per-layer mode: "True", "False" or "selective"
    want_internal = {
        "internal": "True",
        "unroll+internal": "True",
        "selective": "selective",
        "unroll+selective": "selective",
    }.get(policy, "False")
    if any((row["checkpointing"] == "True") != want_unroll for row in rows):
        problems.append("step-level checkpointing flag does not match policy")
    if any(row["internal_ckpt"] != want_internal for row in rows):
        problems.append("per-layer checkpointing flag does not match policy")

    measured_rows = rows[WARMUP:]
    count_keys = ("layer_calls", "layer_blocks", "unroll_blocks")
    per_update = {
        tuple(int(row[key]) for key in count_keys) for row in measured_rows
    }
    if len(per_update) != 1:
        problems.append("event counters differ between measured updates")
    counts = dict(zip(count_keys, sorted(per_update)[0])) if per_update else {}
    if counts:
        _, count_problems = infer_layer_executions(policy, counts, steps)
        problems += count_problems
    return measured_rows, counts, problems


# --------------------------------------------------------------- statistics


def bootstrap_ci(values, statistic, seed, draws=BOOTSTRAP_DRAWS):
    """
    95% percentile bootstrap interval of statistic(sample), resampling whole
    launches with replacement. `values` is a list; statistic maps a list to a
    number. With fewer than two values no interval can be given.
    """
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    stats = sorted(
        statistic([rng.choice(values) for _ in values]) for _ in range(draws)
    )
    return stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]


def summarize(launches, policies, planned, seed):
    """
    Per-policy summary rows from the launch records (dicts, numbers as float).
    Time is summarized as the mean of launch means with a bootstrap interval;
    the ratio to `off` is the ratio of mean times over matched repeats
    (aggregate the costs first, then take the ratio).
    """
    done = [r for r in launches if r["status"] == "completed"]
    by_policy = {
        p: {r["repeat"]: r for r in done if r["policy"] == p} for p in policies
    }
    rows = []
    for policy in policies:
        runs = by_policy[policy]
        row = dict.fromkeys(SUMMARY_FIELDS)
        row.update(
            policy=policy,
            launches_completed=len(runs),
            launches_planned=planned,
        )
        if runs:
            times = [r["mean_ms"] for r in runs.values()]
            low, high = bootstrap_ci(times, statistics.mean, seed)
            row.update(
                mean_ms=statistics.mean(times),
                mean_ms_ci_low=low,
                mean_ms_ci_high=high,
                peak_allocated_mib_max=max(
                    r["peak_allocated_mib"] for r in runs.values()
                ),
                peak_allocated_mib_mean=statistics.mean(
                    r["peak_allocated_mib"] for r in runs.values()
                ),
                peak_reserved_mib_max=max(
                    r["peak_reserved_mib"] for r in runs.values()
                ),
            )
        reference = by_policy.get("off", {})
        paired = sorted(set(runs) & set(reference))
        if policy != "off" and paired:
            pairs = [
                (runs[k]["mean_ms"], reference[k]["mean_ms"]) for k in paired
            ]
            ratio = lambda sample: (  # noqa: E731
                sum(a for a, _ in sample) / sum(b for _, b in sample)
            )
            low, high = bootstrap_ci(pairs, ratio, seed + 1)
            row.update(
                time_ratio_vs_off=ratio(pairs),
                time_ratio_ci_low=low,
                time_ratio_ci_high=high,
                memory_ratio_vs_off=(
                    sum(runs[k]["peak_allocated_mib"] for k in paired)
                    / sum(reference[k]["peak_allocated_mib"] for k in paired)
                ),
            )
        rows.append(row)
    return rows


def markdown_table(rows):
    """Readable summary table (plain markdown)"""

    def fmt(value, digits=1):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        "| policy | done | mean ms [95% CI] | peak alloc MiB (max) | "
        "time vs off [95% CI] | memory vs off |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        interval = (
            f" [{fmt(r['mean_ms_ci_low'])}, {fmt(r['mean_ms_ci_high'])}]"
            if r["mean_ms_ci_low"] is not None
            else ""
        )
        ratio_interval = (
            f" [{fmt(r['time_ratio_ci_low'], 3)}, "
            f"{fmt(r['time_ratio_ci_high'], 3)}]"
            if r["time_ratio_ci_low"] is not None
            else ""
        )
        lines.append(
            f"| {r['policy']} | {r['launches_completed']}/"
            f"{r['launches_planned']} | {fmt(r['mean_ms'])}{interval} | "
            f"{fmt(r['peak_allocated_mib_max'], 0)} | "
            f"{fmt(r['time_ratio_vs_off'], 3)}{ratio_interval} | "
            f"{fmt(r['memory_ratio_vs_off'], 3)} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- launching


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def capture(command, **kwargs):
    """Run a short command and return its output; never raises"""
    try:
        done = subprocess.run(
            command, text=True, capture_output=True, timeout=30, **kwargs
        )
        return {"returncode": done.returncode, "stdout": done.stdout}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def provenance(source):
    """Read-only record of the code and hardware that produced the results"""
    git = ["git", "-C", str(source)]
    return {
        "time": utcnow(),
        "python": sys.version,
        "source": str(source),
        "git_sha": capture(git + ["rev-parse", "HEAD"]),
        "git_branch": capture(git + ["branch", "--show-current"]),
        "git_status": capture(git + ["status", "--porcelain"]),
        "gpu": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ]
        ),
        "hostname": os.uname().nodename,
    }


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ran_out_of_memory(log_path):
    """Whether the tail of a log shows a CUDA out-of-memory error"""
    with Path(log_path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 400_000))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return "out of memory" in tail or "outofmemoryerror" in tail


def stop(process):
    """Stop a child and everything it started"""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_launch(item, args, session):
    """
    Run one fresh training process, collect its CSV and log, audit and
    summarize it. Returns the launch record (a dict of LAUNCH_FIELDS).
    """
    run_dir = session / item["run"]
    run_dir.mkdir()
    command = [sys.executable, "-m", args.train_module] + training_args(
        args.width, args.steps, item["policy"], args.measured, args.extra_args
    )
    (run_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.source)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("WANDB_MODE", "offline")

    workdir = Path(args.workdir)
    before = set(workdir.glob("step-measurements-*.csv"))
    record = dict.fromkeys(LAUNCH_FIELDS)
    record.update(
        run=item["run"],
        repeat=item["repeat"],
        policy=item["policy"],
        width=args.width,
        forecast_steps=args.steps,
    )
    started = time.time()
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=args.timeout)
            record["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            stop(process)
            record["status"] = "timeout"
        except BaseException:
            stop(process)
            raise
    record["process_seconds"] = round(time.time() - started, 1)

    new_csvs = sorted(set(workdir.glob("step-measurements-*.csv")) - before)
    for path in new_csvs:  # keep each launch's CSV with its log
        path.replace(run_dir / path.name)
    if record["status"] == "timeout":
        return record
    if record["returncode"] != 0:
        record["status"] = "oom" if ran_out_of_memory(log_path) else "failed"
        return record
    if len(new_csvs) != 1:
        record["status"] = "failed"
        record["audit_problems"] = f"{len(new_csvs)} measurement CSVs found"
        return record

    rows = read_csv(run_dir / new_csvs[0].name)
    measured_rows, counts, problems = audit_launch(
        rows, item, args.width, args.steps, args.measured
    )
    times = [float(r["step_ms"]) for r in measured_rows]
    record.update(
        status="completed" if not problems else "audit_failed",
        measured_steps=len(times),
        audit_problems="; ".join(problems) or None,
        **counts,
    )
    if times:
        record.update(
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            peak_allocated_mib=max(
                float(r["allocated_mib"]) for r in measured_rows
            ),
            peak_reserved_mib=max(
                float(r["reserved_mib"]) for r in measured_rows
            ),
        )
    if counts:
        n, _ = infer_layer_executions(item["policy"], counts, args.steps)
        record["layer_executions_plain_forward"] = n
    record["_losses"] = [r["loss"] for r in rows]
    return record


def add_loss_comparison(records):
    """
    Informational check that checkpointing does not change training: the loss
    of every update is compared with the `off` launch of the same repeat (same
    seed and data order). Small differences are expected on a GPU; large ones
    deserve a look. Stores the largest relative difference per launch.
    """
    reference = {
        r["repeat"]: r["_losses"]
        for r in records
        if r["policy"] == "off" and r.get("_losses")
    }
    for record in records:
        losses, ref = record.get("_losses"), reference.get(record["repeat"])
        if not losses or not ref or record["policy"] == "off":
            continue
        try:
            diffs = [
                abs(float(a) - float(b)) / max(abs(float(b)), 1e-12)
                for a, b in zip(losses, ref)
            ]
        except ValueError:  # an empty loss column
            continue
        record["max_rel_loss_diff_vs_off"] = max(diffs)


def consistency_problems(records):
    """Session-level audit: every policy must see the same plain forward"""
    sizes = {
        r["layer_executions_plain_forward"]
        for r in records
        if r["status"] == "completed"
        and r["layer_executions_plain_forward"] is not None
    }
    if len(sizes) > 1:
        return [f"inconsistent layer executions per forward pass: {sizes}"]
    return []


def write_results(session, records, args, policies, planned_per_policy):
    """Write launches.csv, summary.csv/json/md for the records so far"""
    clean = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in records
    ]
    with (session / "launches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=LAUNCH_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(clean)
    numeric = []
    for r in clean:  # CSV gives strings back when re-analysing; use floats
        r = dict(r)
        for key in ("mean_ms", "peak_allocated_mib", "peak_reserved_mib"):
            r[key] = float(r[key]) if r[key] not in (None, "") else None
        r["repeat"] = int(r["repeat"])
        numeric.append(r)
    rows = summarize(numeric, policies, planned_per_policy, args.seed)
    with (session / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    problems = consistency_problems(clean)
    (session / "summary.json").write_text(
        json.dumps({"summary": rows, "session_problems": problems}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (session / "summary.md").write_text(markdown_table(rows), encoding="utf-8")
    return rows, problems


# -------------------------------------------------------------------- main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--workdir",
        help="directory to run training in; "
        "it must contain the data paths used by the config "
        "(for example sample_data/)",
    )
    parser.add_argument("--output_dir", default="ckpt_policy_runs")
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parents[1]),
        help="checkout that provides neural_lam (default: "
        "the checkout containing this script)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="hidden_dim D; grid/mesh D/2, edge D/4",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="training rollout length (--ar_steps_train)",
    )
    parser.add_argument(
        "--launches", type=int, default=6, help="fresh processes per policy"
    )
    parser.add_argument(
        "--measured",
        type=int,
        default=20,
        help=f"measured updates per launch (after {WARMUP} warmup)",
    )
    parser.add_argument(
        "--policies", nargs="+", choices=ALL_POLICIES, default=list(POLICIES)
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261115,
        help="seed for run order and the bootstrap",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="seconds per launch, including start-up",
    )
    parser.add_argument("--train_module", default="neural_lam.train_model")
    parser.add_argument(
        "--extra_args",
        default="",
        help="further train_model arguments, as one string",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the plan and commands, run nothing",
    )
    parser.add_argument(
        "--summarize_only",
        metavar="SESSION_DIR",
        help="recompute the summary of a finished session",
    )
    return parser.parse_args(argv)


def summarize_only(args):
    session = Path(args.summarize_only)
    records = read_csv(session / "launches.csv")
    policies = [
        p for p in ALL_POLICIES if any(r["policy"] == p for r in records)
    ]
    planned = len({r["repeat"] for r in records})
    rows, problems = write_results(session, records, args, policies, planned)
    print(markdown_table(rows))
    for problem in problems:
        print("SESSION PROBLEM:", problem)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize_only:
        return summarize_only(args)
    if not args.workdir and not args.dry_run:
        raise SystemExit("--workdir is required")
    plan = build_plan(args.policies, args.launches, args.seed)
    if args.dry_run:
        for item in plan:
            command = [sys.executable, "-m", args.train_module] + training_args(
                args.width,
                args.steps,
                item["policy"],
                args.measured,
                args.extra_args,
            )
            print(item["run"], shlex.join(command), sep="\n  ")
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = (
        Path(args.output_dir).expanduser().resolve() / f"ckpt_policy_{stamp}"
    )
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "plan": plan,
                "provenance": provenance(Path(args.source)),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Session folder:", session, flush=True)

    records = []
    for index, item in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {item['run']} ...", flush=True)
        record = run_launch(item, args, session)
        records.append(record)
        add_loss_comparison(records)
        write_results(session, records, args, args.policies, args.launches)
        print(
            f"    {record['status']}  mean_ms={record['mean_ms']}  "
            f"peak_alloc_mib={record['peak_allocated_mib']}",
            flush=True,
        )
        if record["audit_problems"]:
            print("    AUDIT:", record["audit_problems"], flush=True)

    rows, problems = write_results(
        session, records, args, args.policies, args.launches
    )
    print()
    print(markdown_table(rows))
    for problem in problems:
        print("SESSION PROBLEM:", problem)
    print("Results folder to share:", session)
    return None


if __name__ == "__main__":
    main()
