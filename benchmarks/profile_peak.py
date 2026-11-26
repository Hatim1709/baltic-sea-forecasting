"""
Where does the memory peak of `unroll` occur, and is more checkpointing
effort worth it? Two short launches on ONE GPU, then the analysis:

  1. `off`    at 1 forecast step: memory saved per region + forward time
  2. `unroll` at 2 forecast steps: peak per phase + allocator snapshot

Run from the folder that contains sample_data/ (needs the GPU, the project
environment and PYTHONPATH set to the checkout root):

  python /path/to/checkout/benchmarks/profile_peak.py --workdir . \
      --output_dir profile_runs --width 1024

Any GPU works for the breakdown (shares and ratios); absolute memory is for
that GPU's width. Use --width 512 on cards with less than about 24 GB. Times
are those of the GPU used, so compare them only with runs on the same GPU.
Use --dry_run to see the commands without running them.
"""

# Standard library
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Local
import run_ckpt_policy_benchmark as bench  # noqa: E402


def launch_command(width, steps, policy, profile_dir, snapshot, extra):
    """Training command of one profiling launch (10 updates, profile #3)"""
    args = bench.training_args(width, steps, policy, 5, extra)
    args += ["--profile_dir", str(profile_dir), "--profile_batch", "3"]
    if snapshot:
        args.append("--profile_snapshot")
    return [sys.executable, "-m", "neural_lam.train_model"] + args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--output_dir", default="profile_runs")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--off_steps", type=int, default=1)
    parser.add_argument("--unroll_steps", type=int, default=2)
    parser.add_argument("--extra_args", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output_dir).expanduser().resolve()
    jobs = [
        ("off", args.off_steps, False),
        ("unroll", args.unroll_steps, True),
    ]
    for policy, steps, snapshot in jobs:
        profile_dir = out / f"{policy}_t{steps}"
        command = launch_command(
            args.width, steps, policy, profile_dir, snapshot, args.extra_args
        )
        print(f"\n== {policy}, {steps} forecast step(s)")
        print(shlex.join(command))
        if args.dry_run:
            continue
        profile_dir.mkdir(parents=True, exist_ok=True)
        log = profile_dir / "train.log"
        with open(log, "w", encoding="utf-8") as file:
            code = subprocess.call(
                command, cwd=args.workdir, stdout=file,
                stderr=subprocess.STDOUT,
            )
        if code != 0 or not (profile_dir / "profile_step.json").exists():
            raise SystemExit(f"{policy} failed (exit code {code}), see {log}")
    if args.dry_run:
        return
    off = out / f"off_t{args.off_steps}" / "profile_step.json"
    unroll = out / f"unroll_t{args.unroll_steps}"
    report = out / "report.txt"
    commands = [
        [sys.executable, str(HERE / "analyze_profile.py"), str(off),
         "--unroll", str(unroll / "profile_step.json")],
    ]
    if (unroll / "memory_snapshot.pickle").exists():
        commands.append(
            [sys.executable, str(HERE / "analyze_snapshot.py"),
             str(unroll / "memory_snapshot.pickle")]
        )
    with open(report, "w", encoding="utf-8") as file:
        for command in commands:
            file.write(subprocess.run(
                command, capture_output=True, text=True, check=False
            ).stdout + "\n")
    print(report.read_text())
    print(f"Send back the folder {out} (report.txt + the json files).")


if __name__ == "__main__":
    main()
