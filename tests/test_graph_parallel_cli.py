"""Focused launch safeguards for the real-data graph-parallel entry point."""

# Standard library
import random

# Third-party
import numpy as np
import pytest
import torch

# First-party
from neural_lam.train_graph_parallel import (
    _compare_gradient,
    _digest,
    _rng_state,
    _set_rng_state,
    main,
    parse_args,
)
from neural_lam.train_model import build_parser

BASE_ARGS = [
    "--config_path",
    "config.yaml",
    "--partition-dir",
    "owners",
    "--gp-output-dir",
    "run",
]


def test_bridge_defaults_preserve_legacy_parser_defaults():
    bridge = parse_args(BASE_ARGS)
    assert bridge.model == "graph_efm"
    assert bridge.epochs == bridge.batch_size == bridge.ar_steps_train == 1
    assert bridge.gp_max_steps == 1 and bridge.num_workers == 0
    assert bridge.hidden_dim == 16 and bridge.hidden_dim_grid == 8
    assert bridge.lr == bridge.min_lr == 0.001
    assert bridge.precision == 32
    assert not bridge.gp_skip_first_step_check
    assert not bridge.grad_checkpointing and bridge.crps_weight == 0
    legacy = build_parser().parse_args([])
    assert legacy.model == "graphcast"
    assert legacy.epochs == 200 and legacy.batch_size == 4


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ar_steps_train", "2"],
        ["--grad_checkpointing"],
        ["--precision", "bf16-mixed"],
        ["--crps_weight", "1"],
        ["--devices", "0", "1"],
        ["--gp-max-steps", "0"],
        ["--limit_train_batches", "0"],
        ["--optimizer", "soap"],
        ["--model", "graph_fm"],
        ["--use_atmosphere_g2m"],
    ],
)
def test_reject_unsupported_or_empty_runs(arguments):
    with pytest.raises(SystemExit) as error:
        parse_args(BASE_ARGS + arguments)
    assert error.value.code == 2


def test_batch_digest_includes_values_and_shapes():
    batch = (torch.tensor([[1.0, 2.0]]), [torch.tensor([4])])
    assert _digest(batch) == _digest((batch[0].clone(), [batch[1][0].clone()]))
    assert _digest(batch) != _digest((batch[0] + 1, batch[1]))
    assert _digest(batch) != _digest((batch[0].reshape(2, 1), batch[1]))


def test_reference_rng_restore_replays_all_cpu_random_sources():
    device = torch.device("cpu")
    original = _rng_state(device)
    try:
        random.seed(314)
        np.random.seed(314)
        torch.manual_seed(314)
        before = _rng_state(device)
        expected = (random.random(), np.random.rand(3), torch.randn(3))
        _set_rng_state(before, device)
        actual = (random.random(), np.random.rand(3), torch.randn(3))
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    finally:
        _set_rng_state(original, device)


def test_single_rank_rejected_before_loading_data(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="Launch exactly two graph ranks"):
        main(BASE_ARGS)


def test_gradient_comparison_accepts_small_reduction_cancellation():
    expected = torch.tensor([1e6, -1e6, 3.5])
    actual = expected + torch.tensor([0.0625, -0.0625, 0.001])
    statistics = _compare_gradient(actual, expected, rtol=1e-4, atol=1e-5)
    assert statistics["pointwise_mismatches"] == 1
    assert statistics["relative_l2_error"] < 1e-6
    assert statistics["scaled_l2_error"] < 1
    assert statistics["scaled_max_error"] < 1


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([1.01e6, -1e6, 3.5], [1e6, -1e6, 3.5]),
        ([0.0, 0.0, 0.001], [0.0, 0.0, 0.0]),
        ([float("nan")], [1.0]),
    ],
)
def test_gradient_comparison_rejects_material_errors(actual, expected):
    with pytest.raises(AssertionError):
        _compare_gradient(
            torch.tensor(actual), torch.tensor(expected), rtol=1e-4, atol=1e-5
        )


def test_gradient_comparison_preserves_unused_gradient_semantics():
    assert _compare_gradient(None, None, rtol=1e-4, atol=1e-5) is None
    with pytest.raises(AssertionError, match="presence differs"):
        _compare_gradient(torch.zeros(1), None, rtol=1e-4, atol=1e-5)


def test_gradient_comparison_rejects_shape_and_dtype_changes():
    with pytest.raises(AssertionError, match="shapes differ"):
        _compare_gradient(
            torch.zeros(1, 2), torch.zeros(2), rtol=1e-4, atol=1e-5
        )
    with pytest.raises(AssertionError, match="dtypes differ"):
        _compare_gradient(
            torch.zeros(2), torch.zeros(2).double(), rtol=1e-4, atol=1e-5
        )
