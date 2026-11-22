"""Manual real-data integration validation with a test-only CPU transport.

This harness imports the full project and invokes the production training
runner. It only substitutes transport for socket-restricted test environments.
It cannot establish Gloo/NCCL correctness or performance. The optional float64
cast is a diagnostic comparison, not a supported production precision mode.
"""

# Standard library
import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

# Third-party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _worker(rank, args, store_path, diagnostic_float64):
    # Local imports keep this harness separate from the lightweight fixtures.
    # Third-party
    from graph_parallel_file_backend import register_file_test_backend

    # First-party
    from neural_lam import train_graph_parallel as runner

    register_file_test_backend()
    torch.set_num_threads(args.gp_cpu_threads)
    dist.init_process_group(
        "filetest",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=args.gp_timeout_seconds),
    )
    if diagnostic_float64:
        original_build = runner._build_data_and_model
        original_move = runner._move_batch

        def build_double(arguments):
            model, loader, graph_dir = original_build(arguments)
            return model.double(), loader, graph_dir

        def move_double(batch, device):
            # The original recursive mover calls this replacement on children.
            result = original_move(batch, device)
            if isinstance(result, torch.Tensor) and result.is_floating_point():
                return result.double()
            return result

        runner._build_data_and_model = build_double
        runner._move_batch = move_double
    try:
        runner._run(args, torch.device("cpu"))
        if rank == 0:
            target = args.gp_output_dir / "graph_parallel_run.json"
            report = json.loads(target.read_text())
            report["validation_harness"] = {
                "transport": "test-only CPU FileStore simulation",
                "floating_dtype": (
                    "float64 diagnostic override"
                    if diagnostic_float64
                    else "float32"
                ),
                "scope": "actual config, datastores, DataLoader and GraphEFM",
                "gloo_nccl_validated": False,
                "performance_evidence": False,
            }
            target.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main():
    # First-party
    from neural_lam.train_graph_parallel import parse_args

    diagnostic_parser = ArgumentParser(add_help=False, allow_abbrev=False)
    diagnostic_parser.add_argument("--diagnostic-float64", action="store_true")
    diagnostic, remaining = diagnostic_parser.parse_known_args()
    args = parse_args(remaining)
    if os.environ.get("NJORD_TEST_BACKEND") != "filetest":
        raise RuntimeError(
            "Explicitly set NJORD_TEST_BACKEND=filetest for this test harness"
        )
    if args.gp_device != "cpu" or args.gp_skip_first_step_check:
        raise RuntimeError(
            "Real-data validation requires --gp-device cpu and the "
            "default first-step serial comparison"
        )
    args.gp_output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="test-transport-", dir=args.gp_output_dir
    ) as directory:
        mp.spawn(
            _worker,
            args=(
                args,
                str(Path(directory) / "store"),
                diagnostic.diagnostic_float64,
            ),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    main()
