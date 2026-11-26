"""CPU tests of the step profiler (neural_lam/profiling.py)"""

# Standard library
import json

# Third-party
import pytest
import torch
from torch import nn

# First-party
from neural_lam.profiling import MIB, RegionProfiler, StepProfiler


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.second = nn.Linear(64, 4)
        self.register_buffer("table", torch.zeros(1000))

    def forward(self, x):
        return self.second(self.first(x))


def _run(model, x, profiler=None):
    loss = model(x).sum()
    if profiler is not None:
        profiler.stop_saving()
    loss.backward()


def test_saved_memory_is_attributed_once_to_the_first_region():
    model = Tiny()
    x = torch.randn(256, 16)
    profiler = RegionProfiler(model, max_depth=2)
    profiler.start(batch=x)
    _run(model, x, profiler)
    profiler.stop()
    rows = {r["name"]: r for r in profiler.summary()}
    # first.0 saves the input (not counted: it is the batch) -> nothing;
    # first.1 (SiLU) saves its input (256x64 floats), second saves the
    # SiLU output (256x64 floats); weights/buffers are never counted.
    assert rows["first.0"]["saved_mib"] == 0
    expected = 256 * 64 * 4 / MIB
    assert rows["first.1"]["saved_mib"] == pytest.approx(expected)
    assert rows["second"]["saved_mib"] == pytest.approx(expected)
    assert all(r["calls"] == 1 for r in rows.values() if r["incl_ms"] >= 0)


def test_profiler_does_not_change_results_and_removes_hooks():
    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(8, 16)
    model(x).sum().backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    profiler = RegionProfiler(model)
    profiler.start(batch=x)
    _run(model, x, profiler)
    profiler.stop()
    for grad, ref in zip((p.grad for p in model.parameters()), reference):
        torch.testing.assert_close(grad, ref)
    assert not any(m._forward_hooks or m._forward_pre_hooks
                   for m in model.modules())


def test_own_time_excludes_regions_inside_and_calls_are_counted():
    model = Tiny()
    x = torch.randn(4, 16)
    profiler = RegionProfiler(model, max_depth=2)
    profiler.start(batch=x)
    model(x)
    model(x)
    profiler.stop()
    rows = {r["name"]: r for r in profiler.summary()}
    assert rows["first"]["calls"] == 2
    assert rows["first"]["self_ms"] <= rows["first"]["incl_ms"] + 1e-9
    inner = rows["first.0"]["incl_ms"] + rows["first.1"]["incl_ms"]
    assert rows["first"]["self_ms"] == pytest.approx(
        rows["first"]["incl_ms"] - inner, abs=1e-6
    )


def test_step_profiler_callback_writes_json(tmp_path):
    model = Tiny()
    model.ckpt_policy = "off"
    callback = StepProfiler(tmp_path, batch_idx=1)
    x = torch.randn(8, 16)
    for index in range(3):
        callback.on_train_batch_start(None, model, x, index)
        loss = model(x).sum()
        callback.on_before_backward(None, model, loss)
        loss.backward()
        callback.on_after_backward(None, model)
        callback.on_train_batch_end(None, model, None, x, index)
    result = json.loads((tmp_path / "profile_step.json").read_text())
    assert result["meta"]["batch_idx"] == 1
    assert result["meta"]["regions_saved_valid"] is True
    assert set(result["phases"]) == {"forward", "backward", "optimizer"}
    assert {r["name"] for r in result["regions"]} >= {"first", "second"}


def _load_script(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keep_table_prefers_cheap_to_store_slow_to_recompute():
    analyze = _load_script("analyze_profile")
    # two steps; "big" stores 4000 MiB for 20 ms, "small" 200 MiB for 40 ms
    result = {
        "regions": [
            {"name": "big.a", "saved_mib": 8000.0, "self_ms": 40.0},
            {"name": "small.a", "saved_mib": 400.0, "self_ms": 80.0},
        ]
    }
    groups = analyze.group_regions(result, 1)
    phases = {
        "forward": {"ms": 200.0},
        "backward": {"ms": 300.0},
        "optimizer": {"ms": 10.0},
    }
    rows = {
        r["budget_mib"]: r
        for r in analyze.keep_table(groups, phases, 2, budgets=(0, 250, 5000))
    }
    assert rows[0]["time_vs_unroll"] == 1.0 and rows[0]["kept"] == []
    assert rows[0]["time_vs_off"] == pytest.approx(710 / 510)
    assert rows[250]["kept"] == ["small"]
    # keeping "small" saves its 80 ms (both steps) of the 710 ms
    assert rows[250]["time_vs_unroll"] == pytest.approx(630 / 710)
    assert rows[5000]["kept"] == ["small", "big"]


def test_snapshot_replay_finds_peak_and_groups_by_site():
    analyze = _load_script("analyze_snapshot")
    frame = {"filename": "/x/neural_lam/interaction_net.py", "line": 7,
             "name": "forward"}
    events = [
        {"action": "alloc", "addr": 1, "size": 100, "frames": [frame]},
        {"action": "alloc", "addr": 2, "size": 50, "frames": []},
        {"action": "free_requested", "addr": 1},
        {"action": "free_completed", "addr": 1},
        {"action": "alloc", "addr": 3, "size": 10, "frames": [frame]},
    ]
    peak, index, live, _ = analyze.replay(events)
    assert (peak, index) == (150, 1)
    groups = dict(analyze.group_live(live))
    assert groups["interaction_net.py:forward:7"] == 100
    assert groups[analyze.NO_FRAME] == 50
