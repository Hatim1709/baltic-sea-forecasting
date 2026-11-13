"""Numerical equivalence tests for the actual receiver-owned Njord operators.

Run this focused suite without the unrelated datastore-download conftest:

    python tests/test_graph_parallel.py
    pytest --noconftest tests/test_graph_parallel.py

Only PyTorch and PyG are required. Gloo is the default real transport. In
socket-restricted test environments, NJORD_TEST_BACKEND=filetest explicitly
selects a test-only FileStore transport; that is not evidence for Gloo/NCCL.
The lightweight loader executes unchanged
definitions from this checkout; it does not reimplement any layer or loss
equations. The GraphEFM fixture bypasses datastore/Lightning initialization,
but executes the real training_step, variational loss, hierarchical encoder,
learned prior, decoder and autoregressive feedback, with checkpointing OFF.
It is not a CLI/data or checkpoint-recomputation test.
"""

from __future__ import annotations

# Standard library
import ast
import copy
import importlib.util
import os
import sys
import tempfile
import time
import types
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Union

try:
    # Third-party
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError:
    torch = dist = mp = None


ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 2
RTOL = 2e-8
ATOL = 2e-9


def _source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent, _, attribute = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], attribute, module)
    return module


def _selected_definitions(path, names, namespace):
    """Execute exact source definitions, omitting unrelated import-time I/O."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node for node in tree.body if getattr(node, "name", None) in names
    ]
    found = {node.name for node in selected}
    if found != set(names):
        raise AssertionError(
            f"Source definitions changed: {set(names) - found}"
        )
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def _load_real_modules():
    """Load the actual numerical modules without plotting/Lightning imports."""
    if "neural_lam.graph_parallel" in sys.modules:
        return sys.modules["neural_lam.graph_parallel"]
    # Third-party
    import torch_geometric as pyg

    for name, directory in (
        ("neural_lam", ROOT / "neural_lam"),
        ("neural_lam.models", ROOT / "neural_lam/models"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package

    if "neural_lam.utils" not in sys.modules:
        utils = types.ModuleType("neural_lam.utils")
        utils.__file__ = str(ROOT / "neural_lam/utils.py")
        utils.__dict__.update(torch=torch, nn=torch.nn, pyg=pyg)
        names = {
            "BufferList",
            "ConditionalLayerNorm",
            "_mlp_layers",
            "_forward_mc_dropout",
            "MLP",
            "CondMLP",
            "make_mlp",
            "IdentityModule",
            "make_gnn_seq",
        }
        _selected_definitions(Path(utils.__file__), names, utils.__dict__)
        sys.modules[utils.__name__] = utils
        sys.modules["neural_lam"].utils = utils
    utils = sys.modules["neural_lam.utils"]
    if "neural_lam.interaction_net" not in sys.modules:
        interaction = _source_module(
            "neural_lam.interaction_net", ROOT / "neural_lam/interaction_net.py"
        )
        utils.interaction_net = interaction
    return _source_module(
        "neural_lam.graph_parallel", ROOT / "neural_lam/graph_parallel.py"
    )


def _assert_close(actual, expected, label):
    torch.testing.assert_close(
        actual, expected, rtol=RTOL, atol=ATOL, msg=label
    )


def _tuple_output(value):
    return value if isinstance(value, tuple) else (value,)


def _probe_loss(value):
    """A nonuniform scalar probe tests every output row and feature."""
    result = None
    for index, output in enumerate(_tuple_output(value)):
        weights = torch.linspace(
            -0.7, 1.1, output.numel(), dtype=output.dtype, device=output.device
        ).reshape(output.shape)
        contribution = (output * weights).sum() + 0.13 * output.square().sum()
        contribution = contribution * (index + 1)
        result = contribution if result is None else result + contribution
    return result


def _leaf(shape, device, noncontiguous=False):
    if noncontiguous:
        value = torch.randn(
            *shape[:-2],
            shape[-1],
            shape[-2],
            dtype=torch.float64,
            device=device,
        ).transpose(-1, -2)
    else:
        value = torch.randn(*shape, dtype=torch.float64, device=device)
    return value.detach().requires_grad_(True)


def _gradient(value):
    return torch.zeros_like(value) if value.grad is None else value.grad.clone()


def _compare_input_gradients(actual, expected, label):
    gradient = _gradient(actual)
    if gradient.numel():
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    _assert_close(gradient, _gradient(expected), label)


def _compare_parameters(actual, expected, label, gradients=False):
    actual_parameters = list(actual.parameters())
    expected_parameters = list(expected.parameters())
    if len(actual_parameters) != len(expected_parameters):
        raise AssertionError(f"{label}: parameter count changed")
    for index, (left, right) in enumerate(
        zip(actual_parameters, expected_parameters)
    ):
        if gradients:
            _assert_close(
                _gradient(left), _gradient(right), f"{label} grad {index}"
            )
        else:
            _assert_close(left, right, f"{label} parameter {index}")


def _edge_tensor(pairs, device):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()


def _check_one_layer(gp, case, device):
    # First-party
    from neural_lam.interaction_net import (
        FlexibleNet,
        InteractionNet,
        PropagationNet,
    )

    torch.manual_seed(case.get("seed", 31))
    edge_index = _edge_tensor(case["pairs"], device)
    sender_owner = torch.tensor(case["send_owner"], dtype=torch.long)
    receiver_owner = torch.tensor(case["recv_owner"], dtype=torch.long)
    send_dim, rec_dim, edge_dim = case.get("dims", (4, 4, 4))
    if case["kind"] == "flexible":
        original = FlexibleNet(
            edge_index,
            send_dim,
            rec_dim,
            edge_dim,
            hidden_layers=1,
            num_rec=len(receiver_owner),
            propagation=case["propagation"],
            aggr=case["aggregation"],
        )
    else:
        layer_class = (
            InteractionNet if case["kind"] == "interaction" else PropagationNet
        )
        original = layer_class(
            edge_index,
            send_dim,
            hidden_layers=1,
            num_rec=len(receiver_owner),
            aggr=case["aggregation"],
            update_edges=case["update_edges"],
        )
    original = original.to(device=device, dtype=torch.float64)
    reference = copy.deepcopy(original)
    parallel = gp.ReceiverOwnedLayer(
        original, edge_index, sender_owner, receiver_owner
    ).to(device)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.017)
    parallel_optimizer = torch.optim.SGD(parallel.parameters(), lr=0.017)

    # Repeated updates catch stale halo data and stale parameter references.
    for step in range(3):
        torch.manual_seed(401 + step)
        parallel_inputs = [
            _leaf((2, len(sender_owner), send_dim), device, noncontiguous=True),
            _leaf((2, len(receiver_owner), rec_dim), device),
            _leaf((2, edge_index.shape[1], edge_dim), device),
        ]
        reference_inputs = [
            value.detach().clone().requires_grad_() for value in parallel_inputs
        ]
        reference_optimizer.zero_grad(set_to_none=True)
        parallel_optimizer.zero_grad(set_to_none=True)
        expected = reference(*reference_inputs)
        actual = parallel(*parallel_inputs)
        for index, (left, right) in enumerate(
            zip(_tuple_output(actual), _tuple_output(expected))
        ):
            _assert_close(
                left, right, f"{case['name']} output {index}, step {step}"
            )
        expected_loss, actual_loss = _probe_loss(expected), _probe_loss(actual)
        _assert_close(actual_loss, expected_loss, f"{case['name']} loss")
        expected_loss.backward()
        (actual_loss / WORLD_SIZE).backward()
        gp.sum_parameter_gradients(parallel)
        for index, (left, right) in enumerate(
            zip(parallel_inputs, reference_inputs)
        ):
            _compare_input_gradients(
                left, right, f"{case['name']} input {index}"
            )
        _compare_parameters(parallel, reference, case["name"], gradients=True)
        reference_optimizer.step()
        parallel_optimizer.step()
        _compare_parameters(parallel, reference, case["name"])


def _layer_cases():
    # Receiver 5 is isolated. Sender 0 has multiple remote consumers, and
    # repeated edges remain separate messages in sum and mean aggregation.
    pairs = [
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 0),
        (2, 1),
        (3, 4),
        (4, 0),
        (5, 2),
        (5, 2),
    ]
    ownership = [0, 0, 1, 1, 1, 1]
    cases = []
    for aggregation in ("sum", "mean"):
        for update in (False, True):
            cases.append(
                dict(
                    name=f"interaction-{aggregation}-edges{update}",
                    kind="interaction",
                    aggregation=aggregation,
                    update_edges=update,
                    pairs=pairs,
                    send_owner=ownership,
                    recv_owner=ownership,
                )
            )
    for update in (False, True):
        cases.append(
            dict(
                name=f"propagation-mean-edges{update}",
                kind="propagation",
                aggregation="mean",
                update_edges=update,
                pairs=pairs,
                send_owner=ownership,
                recv_owner=ownership,
            )
        )
    return cases


def _boundary_cases():
    cases = []
    for name, pairs, source, receiver in (
        (
            "no-remote-neighbors",
            [(0, 1), (1, 0), (2, 3), (3, 2)],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ),
        (
            "one-sided-halo-empty-local-edges",
            [(0, 2), (1, 2), (2, 3), (3, 2)],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ),
        (
            "rank-without-owned-receivers",
            [(0, 0), (1, 1), (2, 2), (3, 3)],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ),
        ("entirely-empty-edge-set", [], [0, 0, 1, 1], [0, 0, 1, 1]),
    ):
        cases.append(
            dict(
                name=name,
                kind="interaction",
                aggregation="mean",
                update_edges=True,
                pairs=pairs,
                send_owner=source,
                recv_owner=receiver,
            )
        )
    return cases


def _bipartite_cases():
    cases = []
    for aggregation in ("sum", "mean"):
        for propagation in (False, True):
            cases.append(
                dict(
                    name=f"flexible-{aggregation}-propagation{propagation}",
                    kind="flexible",
                    aggregation=aggregation,
                    propagation=propagation,
                    dims=(3, 5, 2),
                    pairs=[
                        (0, 2),
                        (0, 3),
                        (1, 0),
                        (2, 1),
                        (3, 2),
                        (4, 0),
                        (5, 3),
                        (6, 4),
                        (6, 4),
                    ],
                    send_owner=[0, 0, 1, 1, 1, 0, 1],
                    recv_owner=[0, 1, 1, 0, 1],
                )
            )
    return cases


def _load_graph_efm_class():
    """Use original GraphEFM/ARModel math with a no-I/O initialization shell."""
    # First-party
    from neural_lam import utils

    metrics = _source_module(
        "neural_lam.metrics", ROOT / "neural_lam/metrics.py"
    )
    base_path = ROOT / "neural_lam/models/ar_model.py"
    tree = ast.parse(base_path.read_text(), filename=str(base_path))
    ar_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ARModel"
    )
    names = {
        "expand_to_batch",
        "get_clamped_new_state",
        "apply_density_threshold",
    }
    methods = [
        node for node in ar_class.body if getattr(node, "name", None) in names
    ]
    shell_class = ast.ClassDef(
        name="FixtureARBase",
        bases=[
            ast.Attribute(
                value=ast.Name(id="nn", ctx=ast.Load()),
                attr="Module",
                ctx=ast.Load(),
            )
        ],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    shell_namespace = {"torch": torch, "nn": torch.nn, "utils": utils}
    shell_tree = ast.fix_missing_locations(
        ast.Module(body=[shell_class], type_ignores=[])
    )
    exec(compile(shell_tree, str(base_path), "exec"), shell_namespace)
    base = shell_namespace["FixtureARBase"]
    base.log_dict = lambda self, *args, **kwargs: None

    modules = {}
    for name in (
        "base_latent_encoder",
        "constant_latent_encoder",
        "base_graph_latent_decoder",
        "graph_latent_encoder",
        "graph_latent_decoder",
        "hi_graph_latent_encoder",
        "hi_graph_latent_decoder",
    ):
        full_name = f"neural_lam.models.{name}"
        modules[name] = _source_module(
            full_name, ROOT / f"neural_lam/models/{name}.py"
        )
    namespace = {
        "__name__": "neural_lam.models.graph_efm_fixture",
        "torch": torch,
        "nn": torch.nn,
        "utils": utils,
        "metrics": metrics,
        "ARProbModel": base,
        "Union": Union,
        "NeuralLAMConfig": object,
        "BaseDatastore": object,
        "ConstantLatentEncoder": modules[
            "constant_latent_encoder"
        ].ConstantLatentEncoder,
        "GraphLatentEncoder": modules[
            "graph_latent_encoder"
        ].GraphLatentEncoder,
        "GraphLatentDecoder": modules[
            "graph_latent_decoder"
        ].GraphLatentDecoder,
        "HiGraphLatentEncoder": modules[
            "hi_graph_latent_encoder"
        ].HiGraphLatentEncoder,
        "HiGraphLatentDecoder": modules[
            "hi_graph_latent_decoder"
        ].HiGraphLatentDecoder,
    }
    _selected_definitions(
        ROOT / "neural_lam/models/graph_efm.py", {"GraphEFM"}, namespace
    )
    return namespace["GraphEFM"], modules, metrics


def _graph_efm_fixture(device):
    # First-party
    from neural_lam import utils

    graph_class, modules, metrics = _load_graph_efm_class()
    model = graph_class.__new__(graph_class)
    torch.nn.Module.__init__(model)
    model.hierarchical_graph = True
    model.boundary_forced = model.atmosphere_forced = False
    model.concat_atmosphere = model.use_atmosphere_g2m = False
    model.input_steps = 2
    model.output_std = model.embedd_m2m = True
    model.sample_obs_noise = model.use_density = False
    model.grad_checkpointing = False
    model.kl_beta, model.crps_weight = 0.2, 0.0
    model.loss = metrics.nll
    model.unroll_ckpt_func = lambda function, *args: function(*args)
    model.register_buffer("loss_mask", torch.ones(6, 2, dtype=torch.float64))
    model.register_buffer(
        "grid_static_features", torch.randn(6, 2, dtype=torch.float64)
    )
    for name in ("clamp_lower_upper_idx", "clamp_lower_idx", "clamp_upper_idx"):
        model.register_buffer(name, torch.empty(0, dtype=torch.long))

    g2m = _edge_tensor(
        [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (1, 2), (4, 4)], "cpu"
    )
    m2g = _edge_tensor(
        [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
        ],
        "cpu",
    )
    mesh0 = _edge_tensor(
        [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (0, 4),
            (4, 0),
        ],
        "cpu",
    )
    mesh1 = _edge_tensor([(0, 1), (1, 0)], "cpu")
    up = _edge_tensor([(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], "cpu")
    down = up.flip(0)
    model.register_buffer("g2m_edge_index", g2m)
    model.register_buffer("m2g_edge_index", m2g)
    model.m2m_edge_index = utils.BufferList([mesh0, mesh1])
    model.mesh_up_edge_index = utils.BufferList([up])
    model.mesh_down_edge_index = utils.BufferList([down])
    model.mesh_static_features = utils.BufferList(
        [torch.randn(5, 3), torch.randn(2, 3)]
    )
    model.m2m_features = utils.BufferList(
        [torch.randn(mesh0.shape[1], 2), torch.randn(mesh1.shape[1], 2)]
    )
    model.mesh_up_features = utils.BufferList([torch.randn(up.shape[1], 2)])
    model.mesh_down_features = utils.BufferList([torch.randn(down.shape[1], 2)])
    model.register_buffer("g2m_features", torch.randn(g2m.shape[1], 2))
    model.register_buffer("m2g_features", torch.randn(m2g.shape[1], 2))
    model.interior_embedder = utils.make_mlp([7, 4, 4])
    model.interior_current_embedder = utils.make_mlp([9, 4, 4])
    model.g2m_embedder = utils.make_mlp([2, 2, 2])
    model.m2g_embedder = utils.make_mlp([2, 2, 2])
    model.mesh_embedders = torch.nn.ModuleList(
        [utils.make_mlp([3, 3, 3]), utils.make_mlp([3, 5, 5])]
    )
    model.m2m_embedders = torch.nn.ModuleList(
        [utils.make_mlp([2, 5, 5]), utils.make_mlp([2, 5, 5])]
    )
    model.mesh_up_embedders = torch.nn.ModuleList([utils.make_mlp([2, 5, 5])])
    model.mesh_down_embedders = torch.nn.ModuleList([utils.make_mlp([2, 5, 5])])
    encoder_type = modules["hi_graph_latent_encoder"].HiGraphLatentEncoder
    encoder_arguments = (3, g2m, [mesh0, mesh1], [up], 5, 4, 3, 2, 2)
    model.encoder = encoder_type(
        *encoder_arguments, output_dist="diagonal", num_grid_con_mesh_nodes=5
    )
    model.prior_model = encoder_type(
        *encoder_arguments, output_dist="diagonal", num_grid_con_mesh_nodes=5
    )
    model.decoder = modules["hi_graph_latent_decoder"].HiGraphLatentDecoder(
        g2m,
        [mesh0, mesh1],
        m2g,
        [up],
        [down],
        5,
        4,
        3,
        2,
        3,
        2,
        2,
        num_grid_con_mesh_nodes=5,
        num_interior_nodes=6,
    )
    owners = {
        "grid": torch.tensor([0, 0, 1, 1, 0, 1]),
        "mesh_0": torch.tensor([0, 0, 1, 1, 1]),
        "mesh_1": torch.tensor([0, 1]),
    }
    return model.to(device=device, dtype=torch.float64), owners


def _capture_model_outputs(model):
    captured = []

    def capture(module, args, output):
        if isinstance(output, torch.distributions.Normal):
            values = (output.loc, output.scale)
        else:
            values = _tuple_output(output)
        captured.append(tuple(value.detach().clone() for value in values))

    for module in (model.encoder, model.prior_model, model.decoder):
        module.register_forward_hook(capture)
    return captured


def _check_graph_efm(gp, device):
    torch.manual_seed(812)
    parallel, owners = _graph_efm_fixture(device)
    reference = copy.deepcopy(parallel)
    original_parameters = list(parallel.parameters())
    counts = gp.install_graph_parallel(parallel, owners)
    if not counts or any(count == 0 for count in counts.values()):
        raise AssertionError(f"A GraphEFM relation was not adapted: {counts}")
    if {id(value) for value in original_parameters} != {
        id(value) for value in parallel.parameters()
    }:
        raise AssertionError(
            "Installation replaced or dropped model parameters"
        )
    expected_outputs = _capture_model_outputs(reference)
    actual_outputs = _capture_model_outputs(parallel)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.0007)
    actual_optimizer = torch.optim.SGD(parallel.parameters(), lr=0.0007)

    for step in range(2):
        torch.manual_seed(901 + step)
        actual_inputs = [
            _leaf((2, 2, 6, 2), device),
            _leaf((2, 2, 6, 2), device),
            _leaf((2, 2, 6, 1), device),
        ]
        expected_inputs = [
            value.detach().clone().requires_grad_() for value in actual_inputs
        ]
        empty = torch.empty((2, 2, 0, 1), dtype=torch.float64, device=device)
        actual_batch = (*actual_inputs, empty, empty, None)
        expected_batch = (*expected_inputs, empty, empty, None)
        expected_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)
        expected_outputs.clear()
        actual_outputs.clear()
        torch.manual_seed(1101 + step)
        expected_loss = reference.training_step(expected_batch)
        torch.manual_seed(1101 + step)
        actual_loss = parallel.training_step(actual_batch)
        _assert_close(
            actual_loss, expected_loss, "GraphEFM autoregressive ELBO"
        )
        if (
            len(expected_outputs) != len(actual_outputs)
            or len(actual_outputs) != 6
        ):
            raise AssertionError(
                "Expected encoder, prior and decoder outputs "
                "at both forecast steps"
            )
        for index, (actual, expected) in enumerate(
            zip(actual_outputs, expected_outputs)
        ):
            for left, right in zip(actual, expected):
                _assert_close(left, right, f"GraphEFM module output {index}")
        expected_loss.backward()
        (actual_loss / WORLD_SIZE).backward()
        gp.sum_parameter_gradients(parallel)
        for index, (actual, expected) in enumerate(
            zip(actual_inputs, expected_inputs)
        ):
            _compare_input_gradients(
                actual, expected, f"GraphEFM input {index}"
            )
        _compare_parameters(parallel, reference, "GraphEFM", gradients=True)
        expected_optimizer.step()
        actual_optimizer.step()
        _compare_parameters(parallel, reference, "GraphEFM")


def _worker(rank, rendezvous, suite, backend):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    elif backend == "filetest":
        helper = _source_module(
            "njord_graph_parallel_file_backend",
            ROOT / "tests/graph_parallel_file_backend.py",
        )
        helper.register_file_test_backend()
    dist.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=45),
    )
    try:
        gp = _load_real_modules()
        if suite == "graph_efm":
            _check_graph_efm(gp, device)
        else:
            cases = {
                "layers": _layer_cases,
                "boundaries": _boundary_cases,
                "bipartite": _bipartite_cases,
            }[suite]()
            for case in cases:
                _check_one_layer(gp, case, device)
    finally:
        dist.destroy_process_group()


def _run_two_ranks(suite, backend=None):
    backend = backend or os.environ.get("NJORD_TEST_BACKEND", "gloo")
    if backend not in ("gloo", "nccl", "filetest"):
        raise ValueError(f"Unsupported test backend: {backend}")
    with tempfile.TemporaryDirectory(
        prefix="njord-graph-parallel-"
    ) as directory:
        rendezvous = str(Path(directory) / "process-group")
        processes = mp.spawn(
            _worker,
            args=(rendezvous, suite, backend),
            nprocs=WORLD_SIZE,
            join=False,
        )
        deadline = time.monotonic() + 150
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                for process in processes.processes:
                    process.terminate()
                for process in processes.processes:
                    process.join(timeout=5)
                raise AssertionError(f"Distributed {suite} test timed out")


@unittest.skipUnless(
    torch is not None
    and dist.is_available()
    and (
        dist.is_gloo_available()
        or os.environ.get("NJORD_TEST_BACKEND") == "filetest"
    ),
    "PyTorch with Gloo or the explicit filetest transport is required",
)
class GraphParallelEquivalenceTests(unittest.TestCase):
    def test_actual_interaction_and_propagation_layers(self):
        _run_two_ranks("layers")

    def test_empty_and_one_sided_boundary_cases(self):
        _run_two_ranks("boundaries")

    def test_actual_flexible_bipartite_layers(self):
        _run_two_ranks("bipartite")

    def test_actual_graph_efm_two_step_elbo_and_optimizer(self):
        _run_two_ranks("graph_efm")

    @unittest.skipUnless(
        torch is not None
        and torch.cuda.device_count() >= 2
        and dist.is_nccl_available(),
        "CUDA/NCCL equivalence requires two CUDA devices",
    )
    def test_cuda_nccl_actual_layers(self):
        _run_two_ranks("layers", backend="nccl")

    @unittest.skipUnless(
        torch is not None
        and torch.cuda.device_count() >= 2
        and dist.is_nccl_available(),
        "CUDA/NCCL GraphEFM equivalence requires two CUDA devices",
    )
    def test_cuda_nccl_actual_graph_efm(self):
        _run_two_ranks("graph_efm", backend="nccl")


if __name__ == "__main__":
    print(
        "Distributed test transport:",
        os.environ.get("NJORD_TEST_BACKEND", "gloo"),
    )
    unittest.main(verbosity=2)
