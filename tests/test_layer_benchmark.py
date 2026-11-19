"""
Tests for benchmarks/run_layer_benchmark.py that need no GPU and no data: the
plan, the training arguments, the audit, the statistics, and whole sessions
driven by a fake training program that writes what a real launch writes.
"""

# Standard library
import csv
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

# Third-party
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_layer_benchmark.py"
)
spec = importlib.util.spec_from_file_location("layer_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules["layer_bench"] = bench
spec.loader.exec_module(bench)

VARIANTS = ["default", "attention-key-value", "attention-bias"]
STEPS, MEASURED = 2, 20
UPDATES = bench.WARMUP + MEASURED


def make_rows(checkpointing, updates=UPDATES):
    return [
        {
            "step": str(i),
            "phase": "warmup" if i <= bench.WARMUP else "measured",
            "epoch": "1",
            "step_ms": "1000.0",
            "allocated_mib": "100.0",
            "reserved_mib": "120.0",
            "checkpointing": str(checkpointing),
        }
        for i in range(1, updates + 1)
    ]


def make_audit(variant, checkpointing, **overrides):
    attention = variant != "default"
    audit = {
        "completed": True,
        "completed_batches": UPDATES,
        "optimizer_updates": UPDATES,
        "losses": [1.0] * UPDATES,
        "runtime_checkpointing": checkpointing,
        "runtime_checkpointing_end": checkpointing,
        "step_loss_calls": UPDATES * STEPS * (2 if checkpointing else 1),
        "parameters": 123,
        "gnn_layers": (
            {"BipartiteEdgeAttention": 23}
            if attention
            else {"InteractionNet": 14, "PropagationNet": 3, "FlexibleNet": 4}
        ),
        "attention_sites": ["all"] if attention else [],
        "attention_edge_mode": (
            "bias" if variant.endswith("bias") else "key_value"
        ),
    }
    audit.update(overrides)
    return audit


def item(variant, checkpointing):
    return {"variant": variant, "checkpointing": checkpointing}


def test_plan_is_balanced_and_reproducible():
    plan = bench.build_plan(VARIANTS, [True, False], 3, seed=1)
    assert plan == bench.build_plan(VARIANTS, [True, False], 3, seed=1)
    assert plan != bench.build_plan(VARIANTS, [True, False], 3, seed=2)
    assert len(plan) == 18 and len({p["run"] for p in plan}) == 18
    for repeat in (1, 2, 3):
        cells = [
            (p["variant"], p["checkpointing"])
            for p in plan
            if p["repeat"] == repeat
        ]
        assert sorted(cells) == sorted(
            (v, c) for v in VARIANTS for c in (True, False)
        )


def test_training_args_follow_the_reference_short_run():
    args = bench.training_args(1024, 2, True, "default", 20, 4)
    flags = dict(zip(args, args[1:]))
    assert flags["--epochs"] == "5" and flags["--limit_train_batches"] == "5"
    assert flags["--ar_steps_train"] == "2"
    assert [flags[f"--hidden_dim{s}"] for s in ("", "_grid", "_edge")] == [
        "1024",
        "512",
        "256",
    ]
    assert "--grad_checkpointing" in args and "--attention_sites" not in args

    off = bench.training_args(1024, 2, False, "attention-bias", 20, 8)
    assert "--grad_checkpointing" not in off
    tail = off[off.index("--attention_sites") :]
    assert tail == [
        "--attention_sites",
        "all",
        "--attention_heads",
        "8",
        "--attention_edge_mode",
        "bias",
    ]
    key_value = bench.training_args(64, 2, False, "attention-key-value", 20, 4)
    assert "--attention_edge_mode" not in key_value
    with pytest.raises(ValueError):
        bench.training_args(1024, 2, True, "default", 18, 4)
    with pytest.raises(ValueError):
        bench.training_args(1022, 2, True, "default", 20, 4)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("checkpointing", [True, False])
def test_audit_accepts_a_correct_launch(variant, checkpointing):
    problems = bench.audit_launch(
        make_rows(checkpointing),
        make_audit(variant, checkpointing),
        item(variant, checkpointing),
        STEPS,
        MEASURED,
    )
    assert problems == []


def test_audit_detects_checkpointing_that_stayed_on():
    """Asked for off, but the model checkpointed anyway."""
    problems = bench.audit_launch(
        make_rows(True),
        make_audit("default", True),
        item("default", False),
        STEPS,
        MEASURED,
    )
    text = " ".join(problems)
    assert "checkpointing True" in text
    assert "runtime_checkpointing True" in text
    assert "forecast-step computations" in text


def test_audit_detects_wrong_layers_and_short_runs():
    wrong_layers = bench.audit_launch(
        make_rows(False),
        make_audit("default", False),
        item("attention-bias", False),
        STEPS,
        MEASURED,
    )
    assert any("attention variant has layers" in p for p in wrong_layers)

    wrong_mode = bench.audit_launch(
        make_rows(False),
        make_audit("attention-key-value", False),
        item("attention-bias", False),
        STEPS,
        MEASURED,
    )
    assert wrong_mode == ["edge mode key_value"]

    short = bench.audit_launch(
        make_rows(False, updates=10),
        make_audit("default", False, completed_batches=10),
        item("default", False),
        STEPS,
        MEASURED,
    )
    assert any("measurement rows" in p for p in short)
    assert any("completed_batches 10" in p for p in short)

    not_finished = bench.audit_launch(
        make_rows(False),
        make_audit("default", False, completed=False),
        item("default", False),
        STEPS,
        MEASURED,
    )
    assert not_finished == ["training did not complete"]


def test_bootstrap_interval_contains_mean_and_needs_two_values():
    assert bench.bootstrap_ci([1.0], max, 0) == (None, None)
    values = [1.0, 2.0, 3.0, 4.0]
    low, high = bench.bootstrap_ci(values, lambda v: sum(v) / len(v), 0)
    assert low < 2.5 < high
    assert bench.bootstrap_ci(values, max, 3) == bench.bootstrap_ci(
        values, max, 3
    )


def test_summary_ratios_are_paired_with_default():
    def record(variant, repeat, ms, mib, status="ok"):
        return {
            "variant": variant,
            "checkpointing": True,
            "repeat": repeat,
            "status": status,
            "mean_ms": ms,
            "peak_allocated_mib": mib,
            "peak_reserved_mib": mib,
            "parameters": 5,
        }

    records = [
        record("default", 1, 100.0, 1000.0),
        record("default", 2, 200.0, 1000.0),
        record("default", 3, 400.0, 1000.0),
        record("attention-bias", 1, 50.0, 500.0),
        record("attention-bias", 2, 100.0, 500.0),
        # Repeat 3 ran out of memory: it must not be paired
        record("attention-bias", 3, None, None, status="oom"),
    ]
    rows = {
        r["variant"]: r
        for r in bench.summarize(
            records, [("default", True), ("attention-bias", True)], 3, seed=0
        )
    }
    default, bias = rows["default"], rows["attention-bias"]
    assert default["launches_ok"] == 3 and default["time_ratio"] is None
    assert default["mean_ms"] == pytest.approx(700 / 3)
    assert bias["launches_ok"] == 2 and bias["statuses"] == "ok oom"
    assert bias["matched_repeats"] == 2
    # (50 + 100) / (100 + 200): repeat 3 of the default is left out
    assert bias["time_ratio"] == pytest.approx(0.5)
    assert bias["memory_ratio"] == pytest.approx(0.5)
    assert bias["time_ratio_ci_low"] <= 0.5 <= bias["time_ratio_ci_high"]


FAKE_PROGRAM = textwrap.dedent(
    '''
    """Writes what a real launch writes, without training anything."""
    import csv, json, os, sys

    audit_path, switch = sys.argv[2], sys.argv[3]
    args = sys.argv[sys.argv.index("--") + 1 :]
    flags = dict(zip(args, args[1:]))
    requested = switch == "on"
    mode = os.environ.get("FAKE_MODE", "ok")
    attention = "--attention_sites" in args
    bias = flags.get("--attention_edge_mode") == "bias"
    updates = int(flags["--epochs"]) * int(flags["--limit_train_batches"])
    steps = int(flags["--ar_steps_train"])
    assert os.path.isfile("sample_data/global_ocean_1.yaml")
    assert ("--grad_checkpointing" in args) == requested

    if mode == "oom" and bias:
        print("torch.OutOfMemoryError: CUDA out of memory.")
        sys.exit(1)
    actual = True if mode == "stuck_on" else requested

    factor = (0.5 if bias else 1.5) if attention else 1.0
    ms = (2000.0 if actual else 1000.0) * factor
    mib = (50.0 if actual else 100.0) * factor
    os.makedirs("saved_models/run")
    open("saved_models/run/last.ckpt", "w").write("large")
    with open("step-measurements-1.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "phase", "epoch", "step_ms",
                         "allocated_mib", "reserved_mib", "checkpointing"])
        for i in range(1, updates + 1):
            writer.writerow([i, "warmup" if i <= 5 else "measured", 1,
                             ms * (10 if i <= 5 else 1), mib, mib + 5, actual])
    json.dump({
        "completed": True, "completed_batches": updates,
        "optimizer_updates": updates, "losses": [3.0] * updates,
        "runtime_checkpointing": actual, "runtime_checkpointing_end": actual,
        "step_loss_calls": updates * steps * (2 if actual else 1),
        "parameters": 700 if bias else 900 if attention else 800,
        "gnn_layers": ({"BipartiteEdgeAttention": 3} if attention
                       else {"InteractionNet": 2, "FlexibleNet": 1}),
        "attention_sites": ["all"] if attention else [],
        "attention_edge_mode": "bias" if bias else "key_value",
    }, open(audit_path, "w"))
    print("Measurements CSV: step-measurements-1.csv")
    '''
)


def run_session(tmp_path, monkeypatch, mode, repeats=2):
    work = tmp_path / "work"
    (work / "sample_data").mkdir(parents=True)
    (work / "sample_data" / "global_ocean_1.yaml").write_text("x")
    fake = tmp_path / "fake_train.py"
    fake.write_text(FAKE_PROGRAM)
    monkeypatch.setenv("FAKE_MODE", mode)
    code = bench.main(
        [
            "--workdir",
            str(work),
            "--repeats",
            str(repeats),
            "--child_program",
            str(fake),
            "--timeout",
            "60",
        ]
    )
    (session,) = (work / "layer_benchmark_runs").iterdir()
    return code, session


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_full_session_with_fake_training_program(tmp_path, monkeypatch):
    code, session = run_session(tmp_path, monkeypatch, "ok")
    assert code == 0
    launches = read(session / "launches.csv")
    assert len(launches) == 12 and {r["status"] for r in launches} == {"ok"}
    # Warmup updates (ten times slower in the fake) are left out of the mean
    assert {float(r["mean_ms"]) for r in launches} == {
        500.0,
        1000.0,
        1500.0,
        2000.0,
        3000.0,
    }
    summary = {
        (r["variant"], r["checkpointing"]): r
        for r in read(session / "summary.csv")
    }
    assert len(summary) == 6
    default_on = summary[("default", "on")]
    assert float(default_on["mean_ms"]) == 2000.0
    assert float(default_on["mean_peak_allocated_mib"]) == 50.0
    assert default_on["time_ratio"] == ""
    bias_off = summary[("attention-bias", "off")]
    assert float(bias_off["time_ratio"]) == pytest.approx(0.5)
    assert float(bias_off["memory_ratio"]) == pytest.approx(0.5)
    assert (
        bias_off["parameters"] == "700" and bias_off["matched_repeats"] == "2"
    )
    key_value_on = summary[("attention-key-value", "on")]
    assert float(key_value_on["time_ratio"]) == pytest.approx(1.5)

    text = (session / "summary.md").read_text()
    assert "attention-key-value | on | 2/2" in text
    assert "first-update loss 3" in text
    plan = json.loads((session / "plan.json").read_text())
    assert len(plan["plan"]) == 12 and plan["width"] == 1024
    assert (session / "provenance.json").is_file()

    # Each launch keeps its evidence, but not the checkpoints or the data link
    for launch in launches:
        folder = session / launch["run"]
        assert (folder / "train.log").is_file()
        assert (folder / "audit.json").is_file()
        assert "neural_lam.train_model" in (folder / "command.txt").read_text()
        assert not (folder / "saved_models").exists()
        assert not (folder / "sample_data").exists()
    assert (tmp_path / "work" / "sample_data" / "global_ocean_1.yaml").is_file()


def test_out_of_memory_is_recorded_and_sweep_continues(tmp_path, monkeypatch):
    code, session = run_session(tmp_path, monkeypatch, "oom")
    assert code == 0  # "does not fit" is a result
    launches = read(session / "launches.csv")
    assert len(launches) == 12
    by_variant = {}
    for r in launches:
        by_variant.setdefault(r["variant"], set()).add(r["status"])
    assert by_variant == {
        "default": {"ok"},
        "attention-key-value": {"ok"},
        "attention-bias": {"oom"},
    }
    summary = {
        (r["variant"], r["checkpointing"]): r
        for r in read(session / "summary.csv")
    }
    assert summary[("attention-bias", "on")]["launches_ok"] == "0"
    assert summary[("attention-bias", "on")]["mean_ms"] == ""
    assert "oom" in (session / "summary.md").read_text()


def test_launch_is_rejected_when_checkpointing_stays_on(tmp_path, monkeypatch):
    code, session = run_session(tmp_path, monkeypatch, "stuck_on", repeats=1)
    assert code == 1
    launches = read(session / "launches.csv")
    off = [r for r in launches if r["checkpointing"] == "False"]
    on = [r for r in launches if r["checkpointing"] == "True"]
    assert {r["status"] for r in off} == {"audit_failed"}
    assert all("runtime_checkpointing True" in r["problems"] for r in off)
    assert {r["status"] for r in on} == {"ok"}
    summary = {
        (r["variant"], r["checkpointing"]): r
        for r in read(session / "summary.csv")
    }
    assert summary[("default", "off")]["launches_ok"] == "0"
    assert summary[("default", "on")]["launches_ok"] == "1"


def test_dry_run_starts_nothing(tmp_path, capsys):
    code = bench.main(["--dry_run", "--workdir", str(tmp_path / "missing")])
    assert code == 0
    out = capsys.readouterr().out
    assert "36 launches, 5+20 updates each" in out
    assert out.count("python -m neural_lam.train_model") == 6
    assert not (tmp_path / "missing").exists()
