"""
Quick check (about a minute, one GPU) before the checkpointing benchmarks.

It answers, in plain words, the questions that would otherwise cost hours:
  * Is this the right code (branch with the selective policies)?
  * Is exactly one GPU visible, and is it the reference GPU (A100 80GB)?
  * Is the sample data where the training expects it?
  * Is the project's reference_workload.py found (needed for comparable runs)?
  * Does torch.compile work here? (needed only for compile_budget_probe.py)

Exit code: 0 = all fine, 1 = a blocking problem (do not start the benchmark),
2 = only torch.compile is unavailable (the benchmark itself can still run;
skip compile_budget_probe.py).

Usage (inside the GPU allocation, with the environment activated):
    PYTHONPATH=. python benchmarks/preflight.py --workdir /scratch/hatim1709/njord
"""

# Standard library
import argparse
import importlib
import sys
from pathlib import Path

# Make the benchmark script importable (it finds the reference workload)
sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, WARN, FAIL, COMPILE_FAIL = "OK  ", "WARN", "FAIL", "COMPILE"
REFERENCE_GPU_MIB = 80 * 1024 * 0.97  # an "80GB" A100 shows ~81 GiB


def check_python():
    version = sys.version.split()[0]
    return [(OK, f"python {version} ({sys.executable})")]


def check_code():
    """The selective policies exist only on the nils branch"""
    try:
        checkpointing = importlib.import_module("neural_lam.checkpointing")
    except Exception as error:  # wrong PYTHONPATH, missing package, ...
        return [
            (
                FAIL,
                "cannot import neural_lam.checkpointing "
                f"({type(error).__name__}: {error}). Run from the checkout "
                "with PYTHONPATH set to its root, environment activated.",
            )
        ]
    if not hasattr(checkpointing, "SELECTIVE"):
        return [
            (
                FAIL,
                "neural_lam.checkpointing has no selective policy: wrong "
                "checkout/branch (need branch nils).",
            )
        ]
    return [(OK, f"checkpoint policies found ({checkpointing.__file__})")]


def check_gpu(torch):
    results = [(OK, f"torch {torch.__version__}, CUDA {torch.version.cuda}")]
    if not torch.cuda.is_available():
        return results + [(FAIL, "no CUDA GPU visible (not on a GPU node?)")]
    count = torch.cuda.device_count()
    if count != 1:
        results.append(
            (FAIL, f"{count} GPUs visible, the benchmark needs exactly 1")
        )
    properties = torch.cuda.get_device_properties(0)
    total_mib = properties.total_memory / 2**20
    message = f"GPU: {properties.name}, {total_mib / 1024:.0f} GiB"
    if "A100" in properties.name and total_mib >= REFERENCE_GPU_MIB:
        results.append((OK, message + " (reference GPU)"))
    else:
        results.append(
            (
                WARN,
                message + " is NOT the reference A100 80GB: numbers are fine "
                "for development but not comparable with the baseline.",
            )
        )
    return results


def check_data(workdir):
    config = Path(workdir) / "sample_data" / "global_ocean_1.yaml"
    if config.is_file():
        return [(OK, f"sample data found ({config})")]
    return [
        (
            FAIL,
            f"{config} missing: --workdir must contain sample_data/ "
            "(e.g. /scratch/hatim1709/njord).",
        )
    ]


def check_reference(explicit, source):
    import run_ckpt_policy_benchmark as bench

    path = bench.find_reference(explicit, source)
    if path is None:
        return [
            (
                FAIL,
                "reference_workload.py not found: run `git switch main && "
                "git pull` in ~/njord-dphpc, or set NJORD_REFERENCE_WORKLOAD.",
            )
        ]
    return [(OK, f"reference workload found ({path})")]


def check_compile(torch):
    """A tiny forward+backward through torch.compile with a memory budget"""
    problem = None
    try:
        import triton  # noqa: F401  (inductor needs it on GPU)

        import torch._functorch.config as functorch_config

        if not hasattr(functorch_config, "activation_memory_budget"):
            problem = (
                f"torch {torch.__version__} has no activation_memory_budget "
                "(needs torch >= 2.4)"
            )
        else:
            functorch_config.activation_memory_budget = 0.5

            def function(x, w):
                return torch.nn.functional.silu(x @ w).sum()

            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            w = torch.randn(64, 64, device="cuda", requires_grad=True)
            torch._dynamo.reset()
            torch.compile(function, backend="inductor", dynamic=False)(
                x, w
            ).backward()
            torch.cuda.synchronize()
            if x.grad is None or w.grad is None:
                problem = "compiled backward produced no gradients"
    except Exception as error:  # a compile problem is a result, not a crash
        problem = f"{type(error).__name__}: {str(error)[:200]}"
    if problem:
        return [
            (
                COMPILE_FAIL,
                "torch.compile does not work here "
                f"({problem}). Skip compile_budget_probe.py; the "
                "checkpointing benchmark itself is unaffected.",
            )
        ]
    return [(OK, "torch.compile with activation_memory_budget works")]


def exit_code(results):
    """1 if anything blocks the benchmark, 2 if only compile fails, else 0"""
    statuses = {status for status, _ in results}
    if FAIL in statuses:
        return 1
    return 2 if COMPILE_FAIL in statuses else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--workdir", required=True, help="folder containing sample_data/"
    )
    parser.add_argument(
        "--reference", help="path of the project's reference_workload.py"
    )
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]

    results = check_python() + check_code()
    results += check_data(args.workdir)
    results += check_reference(args.reference, source)
    try:
        import torch
    except ImportError as error:
        results.append((FAIL, f"cannot import torch ({error})"))
    else:
        gpu = check_gpu(torch)
        results += gpu
        if not any(status == FAIL for status, _ in gpu):
            results += check_compile(torch)

    for status, message in results:
        print(f"[{status}] {message}")
    code = exit_code(results)
    print(
        {
            0: "PREFLIGHT: all fine",
            1: "PREFLIGHT: blocking problem, fix the FAIL lines first",
            2: "PREFLIGHT: benchmark can run, torch.compile unavailable",
        }[code]
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
