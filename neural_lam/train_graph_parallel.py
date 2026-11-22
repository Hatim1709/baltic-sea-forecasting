"""Two-rank GraphEFM training through receiver-owned interaction layers.

This experimental bridge keeps the original GraphEFM training computation and
global tensor interfaces. Local interaction computation and halo exchange are
distributed; full node/edge representations are assembled at layer boundaries.
It is not a fully persistently sharded model. See docs/graph_parallel.md.
"""

# Standard library
import copy
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import timedelta
from itertools import chain
from pathlib import Path
from types import MethodType

# Third-party
import numpy as np
import torch
import torch.distributed as dist

# Local
from .config import load_config_and_datastores
from .graph_parallel import install_graph_parallel, sum_parameter_gradients
from .models.graph_efm import GraphEFM
from .partition_owners import graph_fingerprint, load_owners
from .train_model import build_parser
from .weather_dataset import WeatherDataModule


def parse_args(argv=None):
    """Reuse model/data arguments while keeping this smoke runner bounded."""
    parser = build_parser()
    parser.description = __doc__.splitlines()[0]
    parser.set_defaults(
        model="graph_efm",
        hidden_dim=16,
        hidden_dim_grid=8,
        hidden_dim_mesh_nodes=8,
        hidden_dim_edge=4,
        latent_dim=16,
        epochs=1,
        batch_size=1,
        num_workers=0,
        ar_steps_train=1,
        ar_steps_eval=1,
        n_example_pred=0,
        crps_weight=0,
        grad_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    parser.add_argument(
        "--partition-dir",
        type=Path,
        required=True,
        help="Directory containing owners.pt, or its path",
    )
    parser.add_argument(
        "--gp-output-dir",
        type=Path,
        required=True,
        help="External run directory for per-rank results",
    )
    parser.add_argument(
        "--gp-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--gp-max-steps",
        type=int,
        default=1,
        help="Maximum optimizer steps across all epochs",
    )
    parser.add_argument("--gp-cpu-threads", type=int, default=1)
    parser.add_argument("--gp-timeout-seconds", type=int, default=180)
    parser.add_argument("--gp-rtol", type=float, default=1e-4)
    parser.add_argument("--gp-atol", type=float, default=1e-5)
    parser.add_argument(
        "--gp-skip-first-step-check",
        action="store_true",
        help="Skip serial equivalence only after a small run "
        "has passed; useful when the serial model OOMs",
    )
    args = parser.parse_args(argv)
    for field in ("var_leads_metrics_watch", "var_leads_val_plot"):
        value = getattr(args, field)
        setattr(args, field, {int(k): v for k, v in json.loads(value).items()})
    if not args.config_path or args.model != "graph_efm":
        parser.error("--config_path and --model graph_efm are required")
    if args.epochs < 1 or args.gp_max_steps < 1 or args.batch_size < 1:
        parser.error("epochs, batch_size and gp-max-steps must be positive")
    if args.gp_atol < 0 or args.gp_rtol < 0:
        parser.error("comparison tolerances must be nonnegative")
    if args.num_workers < 0 or args.gp_cpu_threads < 1:
        parser.error("num_workers must be nonnegative; cpu threads positive")
    if args.gp_timeout_seconds < 1:
        parser.error("gp-timeout-seconds must be positive")
    if args.limit_train_batches is not None and args.limit_train_batches <= 0:
        parser.error("limit_train_batches must be positive when supplied")
    if args.grad_checkpointing or args.ar_steps_train != 1:
        parser.error(
            "this initial runner supports one forecast step with "
            "gradient checkpointing disabled"
        )
    if args.crps_weight != 0 or args.train_distr_crps:
        parser.error("CRPS and data-parallel/paired samplers are not supported")
    if args.eval or args.load or args.restore_opt:
        parser.error("evaluation and checkpoint restoration are not supported")
    if args.optimizer != "adamw" or str(args.precision) not in (
        "32",
        "32-true",
    ):
        parser.error("the initial correctness path uses AdamW and float32")
    if args.n_example_pred != 0 or args.use_atmosphere_g2m:
        parser.error(
            "plotting and separate atmosphere-grid nodes are unsupported"
        )
    if args.devices != ["auto"]:
        parser.error(
            "select visible GPUs with CUDA_VISIBLE_DEVICES; LOCAL_RANK "
            "selects each graph rank's GPU"
        )
    return args


def _gather(value):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def _same_on_all_ranks(value, label):
    values = _gather(value)
    if any(item != values[0] for item in values[1:]):
        raise RuntimeError(f"Ranks disagree on {label}")


def _digest(value):
    """Hash actual batch values, not just shapes or sampler seeds."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                update(entry)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _rng_state(device):
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _sync_rng(device):
    states = [_rng_state(device) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(states, src=0)
    _set_rng_state(states[0], device)
    return states[0]


def _move_batch(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(_move_batch(x, device) for x in batch)
    if isinstance(batch, list):
        return [_move_batch(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: _move_batch(v, device) for k, v in batch.items()}
    return batch


def _capture_log_dict(model, dictionary, **_kwargs):
    """Keep training_step's math; replace only its Lightning logger hook."""
    model._graph_parallel_metrics = {
        key: (
            float(value.detach().cpu())
            if isinstance(value, torch.Tensor)
            else float(value)
        )
        for key, value in dictionary.items()
    }


def _step_loss(model, batch, capture_predictions=False):
    if not capture_predictions:
        return model.training_step(batch), None
    original = model.compute_step_loss
    predictions = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        predictions.append(
            tuple(
                x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x
                for x in result[2:]
            )
        )
        return result

    model.compute_step_loss = capture
    try:
        return model.training_step(batch), predictions
    finally:
        model.compute_step_loss = original


def _reference_step(model, batch):
    """Evaluate a disposable serial copy before its distributed counterpart."""
    # Datastores hold read-only xarray/dask/filesystem handles. Do not copy
    # those clients or reopen their data; only clone model state and weights.
    memo = {
        id(value): value
        for name, value in model.__dict__.items()
        if "datastore" in name and value is not None
    }
    reference = copy.deepcopy(model, memo)
    reference.log_dict = MethodType(_capture_log_dict, reference)
    optimizer = reference.configure_optimizers()["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    loss, predictions = _step_loss(reference, batch, capture_predictions=True)
    loss.backward()
    gradients = {
        name: None if p.grad is None else p.grad.detach().cpu().clone()
        for name, p in reference.named_parameters()
    }
    optimizer.step()
    parameters = {
        name: p.detach().cpu().clone()
        for name, p in reference.named_parameters()
    }
    result = {
        "loss": loss.detach().cpu().clone(),
        "predictions": predictions,
        "gradients": gradients,
        "parameters": parameters,
    }
    del loss, optimizer, reference
    gc.collect()
    return result


def _compare_gradient(actual, expected, rtol, atol, label="gradient"):
    """Check tensor-scaled bounds, allowing float32 sum cancellation.

    Partitioning changes the order of large shared-parameter reductions.
    Require both an L2 bound and a maximum-error bound for each tensor;
    retain ordinary pointwise mismatches as diagnostics. Accumulate these
    statistics in float64 so the comparison itself does not lose precision.
    """
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: gradient presence differs")
        return None
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: gradient dtypes differ")
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: gradient shapes differ")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: nonfinite gradient")
    delta = (actual - expected).abs()
    count = expected.numel()
    norm_error = float(delta.norm())
    reference_norm = float(expected.norm())
    maximum_error = float(delta.max()) if count else 0.0
    reference_maximum = float(expected.abs().max()) if count else 0.0
    norm_bound = atol * count**0.5 + rtol * reference_norm
    maximum_bound = atol + rtol * reference_maximum
    if norm_error > norm_bound or maximum_error > maximum_bound:
        raise AssertionError(
            f"{label}: L2 error {norm_error:.9g} "
            f"(bound {norm_bound:.9g}); maximum error {maximum_error:.9g} "
            f"(bound {maximum_bound:.9g})"
        )
    return {
        "max_absolute_error": maximum_error,
        "relative_l2_error": (
            norm_error / reference_norm if reference_norm else None
        ),
        "scaled_l2_error": norm_error / norm_bound if norm_bound else 0.0,
        "scaled_max_error": (
            maximum_error / maximum_bound if maximum_bound else 0.0
        ),
        "pointwise_mismatches": int(
            (delta > atol + rtol * expected.abs()).sum()
        ),
        "elements": count,
    }


def _check_reference(reference, loss, predictions, named_parameters, args):
    """Run the same check on every rank, then fail collectively."""
    errors = {}
    gradient_errors = {}
    failure = None

    def close(label, actual, expected):
        if actual is None or expected is None:
            if actual is not expected:
                raise AssertionError(f"{label}: gradient presence differs")
            return
        actual = actual.detach().cpu()
        errors[label] = (
            float((actual - expected).abs().max()) if actual.numel() else 0.0
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=args.gp_rtol,
            atol=args.gp_atol,
            msg=lambda message: f"{label}\n{message}",
        )

    try:
        close("loss", loss, reference["loss"])
        if len(predictions) != len(reference["predictions"]):
            raise AssertionError("different forecast step count")
        for step, (actual, expected) in enumerate(
            zip(predictions, reference["predictions"])
        ):
            for index, (x, y) in enumerate(zip(actual, expected)):
                close(f"prediction.{step}.{index}", x, y)
        for name, parameter in named_parameters:
            expected = reference["gradients"][name]
            actual = parameter.grad
            statistics = _compare_gradient(
                actual, expected, args.gp_rtol, args.gp_atol, name
            )
            if statistics is not None:
                gradient_errors[name] = statistics
            close(f"optimizer.{name}", parameter, reference["parameters"][name])
    except (AssertionError, RuntimeError) as exc:
        failure = str(exc)
    failures = _gather(failure)
    if any(failures):
        raise RuntimeError("First-step equivalence failed: " + repr(failures))
    return {
        "passed": True,
        "rtol": args.gp_rtol,
        "atol": args.gp_atol,
        "max_absolute_error": max(errors.values(), default=0.0),
        "gradient_comparison": {
            "criterion": "per_tensor_l2_and_max_bounds",
            "max_absolute_error": max(
                (row["max_absolute_error"] for row in gradient_errors.values()),
                default=0.0,
            ),
            "worst_relative_l2_error": max(
                (
                    row["relative_l2_error"]
                    for row in gradient_errors.values()
                    if row["relative_l2_error"] is not None
                ),
                default=0.0,
            ),
            "worst_scaled_l2_error": max(
                (row["scaled_l2_error"] for row in gradient_errors.values()),
                default=0.0,
            ),
            "worst_scaled_max_error": max(
                (row["scaled_max_error"] for row in gradient_errors.values()),
                default=0.0,
            ),
            "pointwise_mismatched_elements": sum(
                row["pointwise_mismatches"] for row in gradient_errors.values()
            ),
            "checked_elements": sum(
                row["elements"] for row in gradient_errors.values()
            ),
            "per_tensor": gradient_errors,
        },
        "checked": "loss, predictions, parameter gradients, AdamW update",
    }


def _build_data_and_model(args):
    (
        config,
        datastore,
        boundary,
        atmosphere,
        stats,
        stats_boundary,
        stats_atmosphere,
    ) = load_config_and_datastores(config_path=args.config_path)
    if boundary is not None:
        raise ValueError(
            "Separate boundary input-grid datastores are not yet supported"
        )
    module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=boundary,
        datastore_atmosphere=atmosphere,
        statistics_datastore=stats,
        statistics_datastore_boundary=stats_boundary,
        statistics_datastore_atmosphere=stats_atmosphere,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        input_steps=args.input_steps,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        current_forcing_step=bool(args.current_forcing_step),
        current_boundary_step=bool(args.current_boundary_step),
        current_atmosphere_step=bool(args.current_atmosphere_step),
        num_past_boundary_steps=args.num_past_boundary_steps,
        num_future_boundary_steps=args.num_future_boundary_steps,
        num_past_atmosphere_steps=args.num_past_atmosphere_steps,
        num_future_atmosphere_steps=args.num_future_atmosphere_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_atmosphere_g2m=False,
        density_channel=getattr(config.training, "density_channel", None),
        train_distr_crps=False,
        seed=args.seed,
    )
    module.setup("fit")
    # No Lightning Trainer and no DistributedSampler: both graph ranks must
    # process the exact same example. A dedicated generator isolates sampling
    # from the model's stochastic latent sampling.
    loader = torch.utils.data.DataLoader(
        module.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
    )
    model = GraphEFM(
        args,
        config=config,
        datastore=datastore,
        datastore_boundary=boundary,
        datastore_atmosphere=atmosphere,
        statistics_datastore=stats,
        statistics_datastore_boundary=stats_boundary,
        statistics_datastore_atmosphere=stats_atmosphere,
    )
    model.log_dict = MethodType(_capture_log_dict, model)
    return model, loader, Path(datastore.root_path) / "graphs" / args.graph


def main(argv=None):
    args = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError(
            "Launch exactly two graph ranks with torchrun --nproc-per-node=2"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    use_cuda = args.gp_device == "cuda" or (
        args.gp_device == "auto" and torch.cuda.is_available()
    )
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    )
    if use_cuda:
        torch.cuda.set_device(device)
    torch.set_num_threads(args.gp_cpu_threads)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        timeout=timedelta(seconds=args.gp_timeout_seconds),
    )
    try:
        _run(args, device)
    finally:
        dist.destroy_process_group()


def _run(args, device):
    rank = dist.get_rank()
    _same_on_all_ranks(
        json.dumps(vars(args), sort_keys=True, default=str), "arguments"
    )
    _same_on_all_ranks(
        hashlib.sha256(Path(args.config_path).read_bytes()).hexdigest(),
        "configuration bytes",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    model, loader, graph_dir = _build_data_and_model(args)
    if len(loader) == 0:
        raise RuntimeError("The configured training dataset is empty")
    owners = load_owners(args.partition_dir, graph_dir)
    if model.num_total_grid_nodes != model.num_grid_nodes:
        raise ValueError(
            "The initial owner schema requires identical "
            "input/output grid namespaces"
        )
    if len(owners["grid"]) != model.num_grid_nodes:
        raise ValueError("Grid owners do not match model data rows")
    _same_on_all_ranks(_digest(owners), "owner arrays")
    model = model.to(device).train()
    if any(
        getattr(module, "mc_dropout_p", 0.0) > 0.0
        or (isinstance(module, torch.nn.Dropout) and module.p > 0.0)
        for module in model.modules()
    ):
        raise ValueError(
            "Stochastic MLP dropout is not supported by the "
            "shared-RNG boundary bridge"
        )
    named_parameters = list(model.named_parameters())
    _same_on_all_ranks(
        _digest({name: p for name, p in named_parameters}), "initial parameters"
    )
    _sync_rng(device)
    first_iterator = iter(loader)
    first_cpu_batch = next(first_iterator)
    _same_on_all_ranks(_digest(first_cpu_batch), "first batch")
    first_batch = _move_batch(first_cpu_batch, device)
    reference = None
    if not args.gp_skip_first_step_check:
        initial_rng = _sync_rng(device)
        reference = _reference_step(model, first_batch)
        _set_rng_state(initial_rng, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    installed = install_graph_parallel(model, owners)
    if {id(p) for _, p in named_parameters} != {
        id(p) for p in model.parameters()
    }:
        raise RuntimeError(
            "Installation replaced/dropped original model parameters"
        )
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]
    args.gp_output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    verification = (
        {"passed": False, "skipped": True} if reference is None else None
    )
    step = 0
    if rank == 0:
        print(
            "Experimental graph-parallel boundary bridge: "
            "full boundary representations remain replicated.",
            flush=True,
        )
        print(f"Installed relation/operator counts: {installed}", flush=True)
        if reference is None:
            print(
                "First-step serial equivalence check explicitly skipped.",
                flush=True,
            )
    for epoch in range(args.epochs):
        batches = (
            chain((first_cpu_batch,), first_iterator)
            if epoch == 0
            else iter(loader)
        )
        for batch_index, cpu_batch in enumerate(batches):
            if args.limit_train_batches is not None:
                limit = args.limit_train_batches
                limit = (
                    int(limit)
                    if limit > 1
                    else max(1, int(len(loader) * limit))
                )
                if batch_index >= limit:
                    break
            _same_on_all_ranks(
                _digest(cpu_batch),
                f"batch at epoch {epoch}, index {batch_index}",
            )
            batch = first_batch if step == 0 else _move_batch(cpu_batch, device)
            if step == 0:
                # Continue the original iterator after the checked first
                # example; do not reshuffle or discard a second first batch.
                _same_on_all_ranks(_digest(batch), "checked first device batch")
            _sync_rng(device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            dist.barrier()
            start = time.perf_counter()
            checking = reference is not None
            loss, predictions = _step_loss(
                model, batch, capture_predictions=checking
            )
            (loss / dist.get_world_size()).backward()
            sum_parameter_gradients(model)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = 1000 * (time.perf_counter() - start)
            if reference is not None:
                verification = _check_reference(
                    reference, loss, predictions, named_parameters, args
                )
                reference = None
            local = {
                "rank": rank,
                "step_ms": elapsed_ms,
                "loss": float(loss.detach().cpu()),
                "peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else None
                ),
                "peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(device) / 2**20
                    if device.type == "cuda"
                    else None
                ),
            }
            per_rank = _gather(local)
            records.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "phase": "correctness_check" if checking else "smoke",
                    "per_rank": per_rank,
                    "slowest_rank_ms": max(x["step_ms"] for x in per_rank),
                }
            )
            if rank == 0:
                print(json.dumps(records[-1]), flush=True)
            step += 1
            if step >= args.gp_max_steps:
                break
        scheduler.step()
        if step >= args.gp_max_steps:
            break
    if step == 0:
        raise RuntimeError("No optimizer step was performed")
    try:
        source_root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=source_root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    report = {
        "schema_version": 1,
        "scope": "global_boundary_bridge",
        "model": "graph_efm",
        "world_size": dist.get_world_size(),
        "backend": dist.get_backend(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "implementation_revision": revision,
        "implementation_dirty": dirty,
        "implementation_sources_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "train_graph_parallel.py",
                "train_model.py",
                "graph_parallel.py",
                "partition_owners.py",
            )
        },
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "graph_fingerprint": graph_fingerprint(graph_dir),
        "node_counts": {key: len(value) for key, value in owners.items()},
        "arguments": vars(args),
        "installed": installed,
        "verification": verification,
        "steps": records,
        "limitations": (
            "Boundary tensors/embedders remain replicated; smoke records "
            "are not a warmed-up scaling benchmark."
        ),
    }
    if rank == 0:
        target = args.gp_output_dir / "graph_parallel_run.json"
        target.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
