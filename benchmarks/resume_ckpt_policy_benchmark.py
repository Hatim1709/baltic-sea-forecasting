"""Continue a checkpoint-policy session after CSV collection was interrupted.

Use --dry_run to recover and audit saved results without starting training.
The original plan and completed launches are retained. A recovered launch has
an unknown return code and process duration, recorded explicitly in recovery.json.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

import run_ckpt_policy_benchmark as benchmark


def recover_launch(session, item, args):
    folder = session / item["run"]
    log = (folder / "train.log").read_text()
    expected_command = [sys.executable, "-m", args.train_module] + benchmark.training_args(
        args.width, args.steps, item["policy"], args.measured, args.extra_args
    )
    if shlex.split((folder / "command.txt").read_text()) != expected_command:
        raise RuntimeError("Saved training command differs: " + item["run"])
    names = re.findall(r"^Measurements CSV: (step-measurements-[0-9]+\.csv)\s*$", log, re.MULTILINE)
    if len(names) != 1 or "`Trainer.fit` stopped:" not in log:
        raise RuntimeError("Launch lacks an unambiguous completion log: " + item["run"])
    saved = folder / names[0]
    source = Path(args.workdir) / names[0]
    if not saved.exists():
        shutil.copy2(source, saved)
    if source.exists() and source.read_bytes() != saved.read_bytes():
        raise RuntimeError("Work-directory and collected CSV differ: " + item["run"])
    rows = benchmark.read_csv(saved)
    measured, counts, problems = benchmark.audit_launch(
        rows, item, args.width, args.steps, args.measured
    )
    if problems or not measured:
        raise RuntimeError("Recovered launch failed audit: " + repr(problems))
    if any(not math.isfinite(float(row["loss"])) for row in rows):
        raise RuntimeError("Non-finite recovered loss: " + item["run"])
    times = [float(row["step_ms"]) for row in measured]
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise RuntimeError("Invalid recovered update time: " + item["run"])
    record = dict.fromkeys(benchmark.LAUNCH_FIELDS)
    record.update(
        run=item["run"], repeat=item["repeat"], policy=item["policy"],
        width=args.width, forecast_steps=args.steps, status="completed",
        measured_steps=len(measured), mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        peak_allocated_mib=max(float(row["allocated_mib"]) for row in measured),
        peak_reserved_mib=max(float(row["reserved_mib"]) for row in measured),
        _losses=[row["loss"] for row in rows], **counts,
    )
    record["layer_executions_plain_forward"] = benchmark.infer_layer_executions(
        item["policy"], counts, args.steps
    )[0]
    recovery_file = folder / "recovery.json"
    if not recovery_file.exists():
        recovery_file.write_text(json.dumps({
            "recovered_at": benchmark.utcnow(),
            "basis": "Full CSV passed audit and finite-loss checks; training completion appears in log",
            "original_returncode": None,
            "original_process_seconds": None,
            "csv_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        }, indent=2) + "\n")
    return record


def check_model_source(args, manifest):
    original = manifest["provenance"]["git_sha"]["stdout"].strip()
    paths = ["neural_lam", "pyproject.toml", "setup.py", "setup.cfg"]
    command = ["git", "-C", args.source, "diff", "--exit-code", original, "--"] + paths
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Model or dependency source differs from the measured revision")
    untracked = subprocess.run(
        ["git", "-C", args.source, "ls-files", "--others", "--exclude-standard", "neural_lam"],
        text=True, capture_output=True, check=True,
    )
    if any(name.endswith(".py") for name in untracked.stdout.splitlines()):
        raise RuntimeError("Untracked Python model source needs inspection")


def continue_session(session, dry_run=False):
    manifest = json.loads((session / "manifest.json").read_text())
    args = argparse.Namespace(**manifest["args"])
    plan = manifest["plan"]
    if plan != benchmark.build_plan(args.policies, args.launches, args.seed):
        raise RuntimeError("Saved plan differs from the original schedule")
    check_model_source(args, manifest)
    journal = session / "resume_records.json"
    if journal.exists():
        records = json.loads(journal.read_text())
    elif (session / "launches.csv").exists():
        raise RuntimeError("Session already has a results journal; inspect it before using this recovery helper")
    else:
        records = []
    expected_prefix = [item["run"] for item in plan[:len(records)]]
    if [record["run"] for record in records] != expected_prefix:
        raise RuntimeError("Recovered journal differs from the original plan")
    for item in plan[len(records):]:
        if not (session / item["run"]).exists():
            break
        records.append(recover_launch(session, item, args))
    recorded = {record["run"] for record in records}
    unexpected = [item["run"] for item in plan if (session / item["run"]).exists() and item["run"] not in recorded]
    if unexpected:
        raise RuntimeError("Non-sequential launch directories need inspection: " + repr(unexpected))

    def save():
        benchmark.add_loss_comparison(records)
        temporary = journal.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(journal)
        benchmark.write_results(session, records, args, args.policies, args.launches)

    save()
    remaining = plan[len(records):]
    print(f"Saved launches: {len(records)}/{len(plan)}; remaining: {len(remaining)}", flush=True)
    if dry_run or not remaining:
        return
    if args.train_module == "neural_lam.train_model":
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one allocated GPU'"], check=True)
    continuation = session / f"continuation-{time.time_ns()}.json"
    continuation.write_text(json.dumps({
        "saved_launches_before": len(records),
        "remaining_plan": remaining,
        "provenance": benchmark.provenance(Path(args.source)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": "Continuation may use a different allocation. Recovered process exit and duration remain unknown.",
    }, indent=2) + "\n")
    for item in remaining:
        print(f"[{len(records) + 1}/{len(plan)}] {item['run']} ...", flush=True)
        record = benchmark.run_launch(item, args, session)
        records.append(record)
        save()
        print(f"    {record['status']}  mean_ms={record['mean_ms']}", flush=True)
    print("Results folder:", session, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    cli = parser.parse_args()
    with (cli.session / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        continue_session(cli.session, cli.dry_run)


if __name__ == "__main__":
    main()
