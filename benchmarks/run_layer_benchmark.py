"""
Single-GPU benchmark of GNN layer variants in GraphEFM.

Compares time and GPU memory per training update between the original GNN
layers and sparse attention (--attention_sites), each with gradient
checkpointing on and off, on the team's reference workload (GraphEFM,
hierarchical graph, two forecast steps, bf16, latent width 16).

What it does
------------
1. Builds a plan: `--repeats` repeats, each containing every cell (variant x
   checkpointing) once, in random order inside each repeat, so slow drifts of
   the GPU do not favour a cell. The repeat index pairs launches of
   different cells.
2. Starts one FRESH training process per launch. Training writes
   `step-measurements-*.csv` (StepMeasurements in train_model.py): time per
   update with GPU synchronisation, and peak allocated and reserved memory.
3. Runs the training entry point unchanged, but adds one callback that
   - sets the checkpointing flag explicitly. The code switches checkpointing
     on by itself whenever more than one step is unrolled, so "off" needs an
     override, as in the baseline study;
   - fixes the order of the training batches, so that it does not depend on
     how many random numbers a variant draws to initialise its weights;
   - records what really ran: parameter count, layer classes, the runtime
     checkpointing flag, the loss of every update, and how often one forecast
     step was computed (twice per step with checkpointing, once without).
4. Audits each launch against the request. A launch that fails the audit is
   marked `audit_failed` and is not used.
5. Summarises per cell: mean update time with a 95% bootstrap interval over
   launch means, peak memory, and paired ratios against the original layers
   at the same checkpointing setting.

Out-of-memory launches are recorded (status "oom") and the sweep continues.
Each launch runs in its own folder. Its model checkpoints are deleted
afterwards (they are large and not part of the result).

Run `--dry_run` first to see the plan and the commands without starting
anything. The parent process needs only the standard library.
"""

# Standard library
import argparse
import csv
import json
import math
import os
import random
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WARMUP = 5  # fixed in StepMeasurements
BATCHES_PER_EPOCH = 5  # the reference short run is 5 epochs x 5 batches
BOOTSTRAP_DRAWS = 10_000
REFERENCE_VARIANT = "default"
GNN_CLASSES = (
    "InteractionNet",
    "PropagationNet",
    "FlexibleNet",
    "BipartiteEdgeAttention",
    "StaticEdgeAttention",
    "StaticEdgeInteractionNet",
)

# The team's reference workload (experiments/run_width_study.py COMMON in the
# project repository). Width, forecast steps, run length and checkpointing
# are added in training_args. Check a printed command with
# experiments/reference_workload.py --check.
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

LAUNCH_FIELDS = [
    "run",
    "repeat",
    "variant",
    "checkpointing",
    "status",
    "problems",
    "returncode",
    "process_seconds",
    "measured_steps",
    "mean_ms",
    "median_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "parameters",
    "first_loss",
    "last_loss",
]
SUMMARY_FIELDS = [
    "variant",
    "checkpointing",
    "launches_ok",
    "launches_planned",
    "statuses",
    "parameters",
    "mean_ms",
    "mean_ms_ci_low",
    "mean_ms_ci_high",
    "mean_peak_allocated_mib",
    "max_peak_allocated_mib",
    "mean_peak_reserved_mib",
    "time_ratio",
    "time_ratio_ci_low",
    "time_ratio_ci_high",
    "memory_ratio",
    "memory_ratio_ci_low",
    "memory_ratio_ci_high",
    "matched_repeats",
]


# ---------------------------------------------------------------- planning


def variant_flags(heads):
    """Extra training arguments of each layer variant."""
    attention = ["--attention_sites", "all", "--attention_heads", str(heads)]
    return {
        "default": [],
        "attention-key-value": attention,
        "attention-bias": attention + ["--attention_edge_mode", "bias"],
    }


def build_plan(variants, checkpointing, repeats, seed):
    """
    One launch per (repeat, variant, checkpointing), in an order shuffled
    inside each repeat (blocked randomisation). Reproducible through `seed`.
    """
    rng = random.Random(seed)
    plan = []
    for repeat in range(1, repeats + 1):
        cells = [(v, c) for v in variants for c in checkpointing]
        rng.shuffle(cells)
        for variant, on in cells:
            name = f"{len(plan) + 1:02d}_{variant}_ckpt-{'on' if on else 'off'}"
            plan.append(
                {
                    "run": f"{name}_r{repeat}",
                    "repeat": repeat,
                    "variant": variant,
                    "checkpointing": on,
                }
            )
    return plan


def training_args(width, steps, checkpointing, variant, measured, heads):
    """
    Arguments of neural_lam.train_model for one launch. Widths follow the
    baseline: d, d/2, d/2, d/4 for main, grid, mesh-node and edge features.
    """
    updates = WARMUP + measured
    if updates % BATCHES_PER_EPOCH:
        raise ValueError(
            f"warmup + measured must be a multiple of {BATCHES_PER_EPOCH}"
        )
    if width < 4 or width % 4:
        raise ValueError("width must be a positive multiple of 4")
    args = list(COMMON)
    args += ["--ar_steps_train", str(steps)]
    args += ["--epochs", str(updates // BATCHES_PER_EPOCH)]
    args += ["--limit_train_batches", str(BATCHES_PER_EPOCH)]
    args += ["--hidden_dim", str(width)]
    args += ["--hidden_dim_grid", str(width // 2)]
    args += ["--hidden_dim_mesh_nodes", str(width // 2)]
    args += ["--hidden_dim_edge", str(width // 4)]
    if checkpointing:
        args.append("--grad_checkpointing")
    return args + variant_flags(heads)[variant]


# ------------------------------------------------------------------- child


def child(audit_path, checkpointing, argv):
    """
    Run the unchanged training entry point with one extra callback, see the
    module docstring. Imports the training stack, so only used in the
    launched process.
    """
    # Standard library
    import collections
    import runpy

    # Third-party
    import pytorch_lightning as pl
    import torch

    # First-party
    from neural_lam.weather_dataset import WeatherDataModule

    report = {
        "completed": False,
        "completed_batches": 0,
        "optimizer_updates": 0,
        "step_loss_calls": 0,
        "losses": [],
        "data_shuffle_seed": 42,
        "worker_generator_seed": 4242,
    }

    def save():
        Path(audit_path).write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )

    original_loader = WeatherDataModule.train_dataloader

    def fixed_order_loader(self):
        loader = original_loader(self)
        if not hasattr(loader.sampler, "generator"):
            raise RuntimeError("Expected the single-GPU RandomSampler")
        # Batch order must not depend on the variant's initialisation draws
        loader.sampler.generator = torch.Generator().manual_seed(42)
        loader.generator = torch.Generator().manual_seed(4242)
        return loader

    WeatherDataModule.train_dataloader = fixed_order_loader

    class Audit(pl.Callback):
        """Sets the checkpointing flag and records what really ran."""

        def on_train_start(self, trainer, model):
            # Explicit runtime policy: the model sets this flag to
            # `--grad_checkpointing or ar_steps_train > 1` on its own.
            model.grad_checkpointing = checkpointing

            original_step = model.compute_step_loss

            def counted_step(*args, **kwargs):
                report["step_loss_calls"] += 1
                return original_step(*args, **kwargs)

            model.compute_step_loss = counted_step

            classes = collections.Counter(
                type(module).__name__ for module in model.modules()
            )
            report.update(
                parameters=sum(p.numel() for p in model.parameters()),
                gnn_layers={n: classes[n] for n in GNN_CLASSES if classes[n]},
                attention_sites=list(
                    getattr(model.args, "attention_sites", [])
                ),
                attention_edge_mode=getattr(
                    model.args, "attention_edge_mode", None
                ),
                attention_heads=getattr(model.args, "attention_heads", None),
                ar_steps_train=model.args.ar_steps_train,
                hidden_dim=model.args.hidden_dim,
                runtime_checkpointing=bool(model.grad_checkpointing),
                precision=str(trainer.precision),
                torch=torch.__version__,
                gpu=(
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            )
            save()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is None or not bool(torch.isfinite(loss.detach()).all()):
                raise RuntimeError("Missing or non-finite training loss")
            report["losses"].append(float(loss.detach()))
            report["completed_batches"] += 1
            report["optimizer_updates"] = int(trainer.global_step)

        def on_train_end(self, trainer, model):
            report["completed"] = True
            report["runtime_checkpointing_end"] = bool(model.grad_checkpointing)
            save()

        def on_exception(self, trainer, model, exception):
            report["exception"] = repr(exception)
            save()

    original_init = pl.Trainer.__init__

    def audited_init(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        names = [type(callback).__name__ for callback in callbacks]
        if names.count("StepMeasurements") != 1:
            raise RuntimeError("Expected exactly one StepMeasurements callback")
        if callbacks[names.index("StepMeasurements")].warmup != WARMUP:
            raise RuntimeError(f"Expected {WARMUP} warmup updates")
        # The timing callback finishes before this one reads the loss
        callbacks.append(Audit())
        kwargs["callbacks"] = callbacks
        original_init(self, *args, **kwargs)

    pl.Trainer.__init__ = audited_init
    sys.argv = ["neural_lam.train_model"] + argv
    runpy.run_module("neural_lam.train_model", run_name="__main__")


# ------------------------------------------------------------------- audit


def audit_launch(rows, audit, item, steps, measured):
    """
    Compare what a launch recorded with what was asked for. Returns a list of
    problems; an empty list means the launch can be used.
    """
    problems = []
    updates = WARMUP + measured
    requested = bool(item["checkpointing"])

    if len(rows) != updates:
        problems.append(f"{len(rows)} measurement rows, expected {updates}")
    for index, row in enumerate(rows, start=1):
        phase = "warmup" if index <= WARMUP else "measured"
        if int(row["step"]) != index or row["phase"] != phase:
            problems.append(f"row {index}: unexpected step or phase")
            break
        if row["checkpointing"] != str(requested):
            problems.append(
                f"row {index}: checkpointing {row['checkpointing']}"
            )
            break
        values = [float(row[k]) for k in ("step_ms", "allocated_mib")]
        if not all(math.isfinite(v) and v > 0 for v in values):
            problems.append(f"row {index}: invalid measurement")
            break

    if not audit.get("completed"):
        problems.append("training did not complete")
    for key in ("completed_batches", "optimizer_updates"):
        if audit.get(key) != updates:
            problems.append(f"{key} {audit.get(key)}, expected {updates}")
    if len(audit.get("losses", [])) != updates:
        problems.append("not one finite loss per update")
    for key in ("runtime_checkpointing", "runtime_checkpointing_end"):
        if audit.get(key) is not requested:
            problems.append(f"{key} {audit.get(key)}, requested {requested}")
    # With checkpointing every forecast step is computed twice: once in the
    # forward pass and once again during backward.
    expected_calls = updates * steps * (2 if requested else 1)
    if audit.get("step_loss_calls") != expected_calls:
        problems.append(
            f"{audit.get('step_loss_calls')} forecast-step computations, "
            f"expected {expected_calls}"
        )

    layers = audit.get("gnn_layers") or {}
    original = sum(
        layers.get(n, 0)
        for n in ("InteractionNet", "PropagationNet", "FlexibleNet")
    )
    attention = layers.get("BipartiteEdgeAttention", 0)
    if item["variant"] == "default":
        if attention or not original or audit.get("attention_sites"):
            problems.append(f"default variant has layers {layers}")
    else:
        mode = "bias" if item["variant"].endswith("bias") else "key_value"
        if original or not attention:
            problems.append(f"attention variant has layers {layers}")
        if audit.get("attention_edge_mode") != mode:
            problems.append(f"edge mode {audit.get('attention_edge_mode')}")
    return problems


# -------------------------------------------------------------- statistics


def bootstrap_ci(values, statistic, seed, draws=BOOTSTRAP_DRAWS):
    """
    95% percentile bootstrap interval of statistic(sample), resampling whole
    launches (or matched pairs). (None, None) with fewer than two values.
    """
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    estimates = sorted(
        statistic([rng.choice(values) for _ in values]) for _ in range(draws)
    )
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]


def ratio_of_means(pairs, index):
    """Ratio of mean(variant) to mean(reference) over matched pairs."""
    return statistics.mean(p[0][index] for p in pairs) / statistics.mean(
        p[1][index] for p in pairs
    )


def summarize(records, cells, planned, seed):
    """
    One row per cell. Time is the mean of launch means with a bootstrap
    interval. Ratios compare with the reference variant at the same
    checkpointing setting over repeats where both launches are usable.
    """
    usable = {
        (r["variant"], r["checkpointing"], r["repeat"]): (
            float(r["mean_ms"]),
            float(r["peak_allocated_mib"]),
        )
        for r in records
        if r["status"] == "ok"
    }
    rows = []
    for variant, on in cells:
        mine = [
            r
            for r in records
            if (r["variant"], r["checkpointing"]) == (variant, on)
        ]
        good = [r for r in mine if r["status"] == "ok"]
        statuses = sorted({r["status"] for r in mine})
        row = dict.fromkeys(SUMMARY_FIELDS)
        row.update(
            variant=variant,
            checkpointing="on" if on else "off",
            launches_ok=len(good),
            launches_planned=planned,
            statuses=" ".join(statuses),
        )
        if good:
            times = [float(r["mean_ms"]) for r in good]
            peaks = [float(r["peak_allocated_mib"]) for r in good]
            reserved = [float(r["peak_reserved_mib"]) for r in good]
            low, high = bootstrap_ci(times, statistics.mean, seed)
            row.update(
                parameters=good[0]["parameters"],
                mean_ms=statistics.mean(times),
                mean_ms_ci_low=low,
                mean_ms_ci_high=high,
                mean_peak_allocated_mib=statistics.mean(peaks),
                max_peak_allocated_mib=max(peaks),
                mean_peak_reserved_mib=statistics.mean(reserved),
            )
        if variant != REFERENCE_VARIANT:
            pairs = [
                (value, usable[(REFERENCE_VARIANT, on, repeat)])
                for (v, c, repeat), value in sorted(usable.items())
                if (v, c) == (variant, on)
                and (REFERENCE_VARIANT, on, repeat) in usable
            ]
            row["matched_repeats"] = len(pairs)
            if pairs:
                for name, index in (("time", 0), ("memory", 1)):

                    def statistic(sample, index=index):
                        return ratio_of_means(sample, index)

                    low, high = bootstrap_ci(pairs, statistic, seed + 1 + index)
                    row[f"{name}_ratio"] = statistic(pairs)
                    row[f"{name}_ratio_ci_low"] = low
                    row[f"{name}_ratio_ci_high"] = high
        rows.append(row)
    return rows


def consistency_notes(records):
    """Things that should agree across the launches of one cell."""
    notes = []
    cells = {}
    for record in records:
        if record["status"] == "ok":
            key = (record["variant"], record["checkpointing"])
            cells.setdefault(key, []).append(record)
    for (variant, on), group in sorted(cells.items()):
        name = f"{variant}, checkpointing {'on' if on else 'off'}"
        if len({r["parameters"] for r in group}) > 1:
            notes.append(f"{name}: parameter count differs between launches")
        first = [float(r["first_loss"]) for r in group]
        spread = (max(first) - min(first)) / max(
            abs(statistics.mean(first)), 1e-30
        )
        notes.append(
            f"{name}: first-update loss {statistics.mean(first):.6g}, "
            f"relative spread over {len(group)} launches {spread:.2e}"
        )
    by_variant = {}
    for record in records:
        if record["status"] == "ok":
            by_variant.setdefault(record["variant"], set()).add(
                record["parameters"]
            )
    counts = [next(iter(v)) for v in by_variant.values() if len(v) == 1]
    if len(counts) > 1 and len(set(counts)) != len(counts):
        notes.append("two variants report the same parameter count")
    return notes


def fmt(value, digits=1):
    return "" if value is None else f"{value:.{digits}f}"


def markdown_table(rows):
    lines = [
        "| Variant | Checkpointing | Launches | Parameters | Mean ms "
        "[95% CI] | Mean peak allocated MiB | Largest allocated MiB | "
        "Time vs default [95% CI] | Memory vs default [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        time_ci = (
            f"{fmt(r['mean_ms'])} [{fmt(r['mean_ms_ci_low'])}, "
            f"{fmt(r['mean_ms_ci_high'])}]"
            if r["mean_ms"] is not None
            else r["statuses"]
        )
        ratios = []
        for name in ("time", "memory"):
            if r[f"{name}_ratio"] is None:
                ratios.append("")
            else:
                ratios.append(
                    f"{fmt(r[name + '_ratio'], 3)} "
                    f"[{fmt(r[name + '_ratio_ci_low'], 3)}, "
                    f"{fmt(r[name + '_ratio_ci_high'], 3)}]"
                )
        lines.append(
            f"| {r['variant']} | {r['checkpointing']} | "
            f"{r['launches_ok']}/{r['launches_planned']} | "
            f"{r['parameters'] or ''} | {time_ci} | "
            f"{fmt(r['mean_peak_allocated_mib'])} | "
            f"{fmt(r['max_peak_allocated_mib'])} | {ratios[0]} | {ratios[1]} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------- running


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def capture(command, **kwargs):
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=30, **kwargs
        )
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def provenance(source):
    """Read-only record of code and machine; never changes a checkout."""
    return {
        "time": utcnow(),
        "host": capture(["hostname"]),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "source": str(source),
        "sha": capture(["git", "rev-parse", "HEAD"], cwd=source),
        "branch": capture(["git", "branch", "--show-current"], cwd=source),
        "status": capture(["git", "status", "--short"], cwd=source),
        "python": sys.version,
        "gpus": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader",
            ]
        ),
    }


def ran_out_of_memory(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "out of memory" in text or "OutOfMemoryError" in text


def stop(process):
    """Stop a launch and everything it started."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=20)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def run_launch(item, args, session):
    """
    Run one fresh training process in its own folder, then audit and
    summarise it. Returns the launch record (a dict of LAUNCH_FIELDS).
    """
    run_dir = session / item["run"]
    run_dir.mkdir()
    # The reference command names the config relative to the working folder
    data_link = run_dir / "sample_data"
    data_link.symlink_to(Path(args.workdir) / "sample_data", True)
    audit_path = run_dir / "audit.json"
    train_args = training_args(
        args.width,
        args.steps,
        item["checkpointing"],
        item["variant"],
        args.measured,
        args.heads,
    )
    command = [sys.executable, "-u", args.child_program, "--child"]
    command += [str(audit_path), "on" if item["checkpointing"] else "off"]
    command += ["--"] + train_args
    (run_dir / "command.txt").write_text(
        "python -m neural_lam.train_model " + shlex.join(train_args) + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.source)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("WANDB_MODE", "offline")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    record = dict.fromkeys(LAUNCH_FIELDS)
    record.update(
        run=item["run"],
        repeat=item["repeat"],
        variant=item["variant"],
        checkpointing=item["checkpointing"],
    )
    started = time.time()
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
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
        finally:
            data_link.unlink()
            if not args.keep_checkpoints:
                shutil.rmtree(run_dir / "saved_models", ignore_errors=True)
    record["process_seconds"] = round(time.time() - started, 1)
    if record["status"] == "timeout":
        return record

    audit = {}
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    csvs = sorted(run_dir.glob("step-measurements-*.csv"))
    # The training entry point can catch an exception and still exit 0
    if ran_out_of_memory(log_path):
        record["status"] = "oom"
        return record
    if record["returncode"] != 0 or len(csvs) != 1 or not audit:
        record["status"] = "failed"
        record["problems"] = f"{len(csvs)} measurement files"
        return record

    with csvs[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    problems = audit_launch(rows, audit, item, args.steps, args.measured)
    if problems:
        record["status"] = "audit_failed"
        record["problems"] = "; ".join(problems)
        return record

    measured = rows[WARMUP:]
    times = [float(r["step_ms"]) for r in measured]
    record.update(
        status="ok",
        measured_steps=len(measured),
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        peak_allocated_mib=max(float(r["allocated_mib"]) for r in measured),
        peak_reserved_mib=max(float(r["reserved_mib"]) for r in measured),
        parameters=audit["parameters"],
        first_loss=audit["losses"][0],
        last_loss=audit["losses"][-1],
    )
    return record


def write_results(session, records, args, cells):
    with (session / "launches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=LAUNCH_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    rows = summarize(records, cells, args.repeats, args.seed)
    with (session / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    notes = consistency_notes(records)
    text = [
        f"# Layer variant benchmark: width {args.width}, "
        f"{args.steps} forecast steps",
        "",
        f"{len(records)} of {len(cells) * args.repeats} planned launches done. "
        f"Each launch: {WARMUP} warmup and {args.measured} measured updates. "
        "Intervals: 95% percentile bootstrap over launches "
        f"({BOOTSTRAP_DRAWS} draws). Ratios use matched repeats.",
        "",
        markdown_table(rows),
        "",
        "## Consistency",
        "",
    ] + [f"- {note}" for note in notes]
    (session / "summary.md").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    return rows


def parse_args(argv=None):
    here = Path(__file__).resolve()
    names = list(variant_flags(1))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--workdir",
        default=".",
        help="folder that contains sample_data/ (default: current folder)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="where the session folder is created "
        "(default: WORKDIR/layer_benchmark_runs)",
    )
    parser.add_argument(
        "--source",
        default=str(here.parents[1]),
        help="implementation checkout to measure (default: this checkout)",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=names, default=names)
    parser.add_argument(
        "--checkpointing",
        nargs="+",
        choices=("on", "off"),
        default=["on", "off"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261119,
        help="seed for the launch order and the bootstrap",
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds per launch"
    )
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--child_program", default=str(here), help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if REFERENCE_VARIANT not in args.variants:
        print(f"note: no '{REFERENCE_VARIANT}' variant, so no ratios")
    switches = [name == "on" for name in args.checkpointing]
    cells = [(v, c) for v in args.variants for c in switches]
    plan = build_plan(args.variants, switches, args.repeats, args.seed)

    if args.dry_run:
        print(f"{len(plan)} launches, {WARMUP}+{args.measured} updates each:")
        for item in plan:
            print(" ", item["run"])
        print("\nTraining command of each cell (run from a folder that")
        print("contains sample_data/):")
        for variant, on in cells:
            flags = training_args(
                args.width, args.steps, on, variant, args.measured, args.heads
            )
            print(f"\n# {variant}, checkpointing {'on' if on else 'off'}")
            print("python -m neural_lam.train_model " + shlex.join(flags))
        return 0

    workdir = Path(args.workdir).resolve()
    args.workdir = str(workdir)
    source = Path(args.source).resolve()
    args.source = str(source)
    if not (workdir / "sample_data" / "global_ocean_1.yaml").is_file():
        print(f"error: no sample_data/global_ocean_1.yaml in {workdir}")
        return 2
    train_model = source / "neural_lam" / "train_model.py"
    if "class StepMeasurements" not in train_model.read_text(encoding="utf-8"):
        print("error: this source lacks the StepMeasurements callback")
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or workdir / "layer_benchmark_runs")
    session = output.resolve() / f"layer_bench_{stamp}"
    session.mkdir(parents=True)
    design = {key: value for key, value in vars(args).items()}
    design.update(plan=plan, warmup=WARMUP, common=COMMON)
    (session / "plan.json").write_text(
        json.dumps(design, indent=2) + "\n", encoding="utf-8"
    )
    (session / "provenance.json").write_text(
        json.dumps(provenance(source), indent=2) + "\n", encoding="utf-8"
    )
    print(f"session: {session}", flush=True)

    records = []
    for number, item in enumerate(plan, start=1):
        record = run_launch(item, args, session)
        records.append(record)
        write_results(session, records, args, cells)
        print(
            f"[{number}/{len(plan)}] {item['run']}: {record['status']}"
            + (
                f", {record['mean_ms']:.1f} ms, "
                f"{record['peak_allocated_mib']:.0f} MiB allocated"
                if record["status"] == "ok"
                else f" {record['problems'] or ''}"
            )
            + f", {record['process_seconds']} s",
            flush=True,
        )

    print("\n" + (session / "summary.md").read_text(encoding="utf-8"))
    return 0 if all(r["status"] in ("ok", "oom") for r in records) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        separator = sys.argv.index("--")
        child(sys.argv[2], sys.argv[3] == "on", sys.argv[separator + 1 :])
    else:
        sys.exit(main())
