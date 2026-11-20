"""
Tests for benchmarks/run_ckpt_policy_benchmark.py that need no GPU and no data:
the planning, the audit of the measurement files, the statistics, and a full
session driven by a fake training program that mimics what StepMeasurements
writes.
"""

# Standard library
import csv
import importlib.util
import sys
import textwrap
from pathlib import Path

# Third-party
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "run_ckpt_policy_benchmark.py"
)
spec = importlib.util.spec_from_file_location("ckpt_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules["ckpt_bench"] = bench
spec.loader.exec_module(bench)

N = 16  # layer executions of a plain forward pass in the fake program


def counters(policy, steps=2):
    """Counters a correct run of `policy` must report (see the script)"""
    calls = {
        "off": N,
        "unroll": 2 * N,
        "internal": 2 * N,
        "unroll+internal": 3 * N,
    }[policy]
    blocks = {"off": 0, "unroll": 0, "internal": N, "unroll+internal": 2 * N}[
        policy
    ]
    unroll = steps if policy.startswith("unroll") else 0
    return {
        "layer_calls": calls,
        "layer_blocks": blocks,
        "unroll_blocks": unroll,
    }


def make_rows(policy, measured=3, steps=2, **overrides):
    """Rows as written by StepMeasurements, as dicts of strings"""
    rows = []
    for i in range(1, bench.WARMUP + measured + 1):
        row = {
            "step": str(i),
            "phase": "warmup" if i <= bench.WARMUP else "measured",
            "epoch": "0",
            "step_ms": "100.0",
            "allocated_mib": "1000.0",
            "reserved_mib": "1200.0",
            "checkpointing": str(policy.startswith("unroll")),
            "ckpt_policy": policy,
            "internal_ckpt": str(policy.endswith("internal")),
            "loss": "1.0",
            **{k: str(v) for k, v in counters(policy, steps).items()},
        }
        row.update(overrides)
        rows.append(row)
    return rows


@pytest.mark.parametrize("policy", bench.POLICIES)
def test_audit_accepts_correct_run(policy):
    rows = make_rows(policy)
    measured, counts, problems = bench.audit_launch(
        rows, {"policy": policy}, 1024, 2, 3
    )
    assert problems == []
    assert len(measured) == 3
    assert counts == counters(policy)
    assert bench.infer_layer_executions(policy, counts, 2)[0] == N


def test_audit_detects_policy_that_did_not_take_effect():
    # "internal" requested, but the counters show a plain forward pass
    rows = make_rows("internal", layer_calls=str(N), layer_blocks="0")
    _, _, problems = bench.audit_launch(rows, {"policy": "internal"}, 1, 2, 3)
    assert problems  # layer_blocks / layer_calls do not match
    # Wrong flag recorded
    rows = make_rows("off", checkpointing="True")
    _, _, problems = bench.audit_launch(rows, {"policy": "off"}, 1, 2, 3)
    assert any("step-level" in p for p in problems)
    # Missing rows
    _, _, problems = bench.audit_launch(
        make_rows("off")[:-1], {"policy": "off"}, 1, 2, 3
    )
    assert any("rows" in p for p in problems)


def test_plan_is_balanced_and_reproducible():
    plan = bench.build_plan(bench.POLICIES, 3, seed=1)
    assert plan == bench.build_plan(bench.POLICIES, 3, seed=1)
    assert len(plan) == 12
    for repeat in (1, 2, 3):
        assert sorted(p["policy"] for p in plan if p["repeat"] == repeat) == (
            sorted(bench.POLICIES)
        )
    assert len({p["run"] for p in plan}) == 12


def test_bootstrap_interval_contains_mean_and_needs_two_values():
    assert bench.bootstrap_ci([1.0], max, 0) == (None, None)
    values = [10.0, 11.0, 9.5, 10.5, 10.2, 9.8]
    low, high = bench.bootstrap_ci(values, lambda v: sum(v) / len(v), 0)
    assert low < sum(values) / len(values) < high


def test_summary_ratio_is_paired_with_off():
    def rec(policy, repeat, ms, mem):
        return {
            "policy": policy,
            "repeat": repeat,
            "status": "completed",
            "mean_ms": ms,
            "peak_allocated_mib": mem,
            "peak_reserved_mib": mem,
        }

    launches = [rec("off", r, 100.0, 4000.0) for r in (1, 2, 3)]
    launches += [rec("unroll", r, 175.0, 1000.0) for r in (1, 2, 3)]
    rows = {
        r["policy"]: r
        for r in bench.summarize(launches, ["off", "unroll"], 3, seed=0)
    }
    assert rows["unroll"]["time_ratio_vs_off"] == pytest.approx(1.75)
    assert rows["unroll"]["memory_ratio_vs_off"] == pytest.approx(0.25)
    assert rows["off"]["time_ratio_vs_off"] is None


FAKE_TRAIN = textwrap.dedent(
    '''
    """Stand-in for neural_lam.train_model: writes the measurement CSV only"""
    import csv, sys, time

    def arg(name):
        return sys.argv[sys.argv.index(name) + 1]

    policy = arg("--ckpt_policy")
    total = int(arg("--limit_train_batches"))
    if policy == "internal" and "FAIL_OOM" in open("mode.txt").read():
        sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory.\\n")
        sys.exit(1)
    n = 16
    calls = {"off": n, "unroll": 2*n, "internal": 2*n, "unroll+internal": 3*n}
    blocks = {"off": 0, "unroll": 0, "internal": n, "unroll+internal": 2*n}
    steps = int(arg("--ar_steps_train"))
    with open(f"step-measurements-{time.time_ns()}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","phase","epoch","step_ms","allocated_mib",
                    "reserved_mib","checkpointing","ckpt_policy",
                    "internal_ckpt","layer_calls","layer_blocks",
                    "unroll_blocks","loss"])
        for i in range(1, total + 1):
            w.writerow([i, "warmup" if i <= 5 else "measured", 0,
                        100 * calls[policy] / n, 1000 / (1 + blocks[policy]),
                        1500, policy.startswith("unroll"), policy,
                        policy.endswith("internal"), calls[policy],
                        blocks[policy],
                        steps if policy.startswith("unroll") else 0, 1.0])
    '''
)


def run_fake_session(tmp_path, mode):
    source = tmp_path / "src"
    source.mkdir()
    (source / "fake_train.py").write_text(FAKE_TRAIN)
    work = tmp_path / "work"
    work.mkdir()
    (work / "mode.txt").write_text(mode)
    out = tmp_path / "out"
    bench.main(
        [
            "--workdir",
            str(work),
            "--output_dir",
            str(out),
            "--source",
            str(source),
            "--train_module",
            "fake_train",
            "--launches",
            "2",
            "--measured",
            "4",
            "--timeout",
            "60",
            "--skip_reference_check",
        ]
    )
    (session,) = list(out.iterdir())
    return session, work


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_full_session_with_fake_training_program(tmp_path):
    session, work = run_fake_session(tmp_path, "ok")
    launches = read(session / "launches.csv")
    assert len(launches) == 8
    assert all(r["status"] == "completed" for r in launches)
    assert all(r["layer_executions_plain_forward"] == "16" for r in launches)
    # Each launch's CSV was moved next to its log; none left behind
    assert not list(work.glob("step-measurements-*.csv"))
    assert len(list(session.glob("*/step-measurements-*.csv"))) == 8
    summary = {r["policy"]: r for r in read(session / "summary.csv")}
    # The fake program makes `unroll` twice as slow as `off`
    assert float(summary["unroll"]["time_ratio_vs_off"]) == pytest.approx(2.0)
    assert float(summary["unroll+internal"]["time_ratio_vs_off"]) == (
        pytest.approx(3.0)
    )
    assert (session / "summary.md").read_text().startswith("| policy")
    assert (session / "manifest.json").exists()
    # Re-analysis from the saved launches gives the same table
    bench.main(["--summarize_only", str(session)])


def test_out_of_memory_is_recorded_and_sweep_continues(tmp_path):
    session, _ = run_fake_session(tmp_path, "FAIL_OOM")
    launches = read(session / "launches.csv")
    status = {(r["policy"], r["status"]) for r in launches}
    assert ("internal", "oom") in status
    assert ("off", "completed") in status
    summary = {r["policy"]: r for r in read(session / "summary.csv")}
    assert summary["internal"]["launches_completed"] == "0"


def test_step_measurements_writes_policy_columns(tmp_path, monkeypatch):
    """The callback records policy, flags, counters and loss (CPU, fake GPU)"""
    # Third-party
    import torch
    from torch import nn

    # First-party
    from neural_lam import checkpointing, train_model

    monkeypatch.chdir(tmp_path)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**20 * 7)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2**20 * 9)

    model = nn.Linear(1, 1)
    model.grad_checkpointing = True
    model.ckpt_policy = "unroll"
    trainer = type("T", (), {"current_epoch": 0})()
    callback = train_model.StepMeasurements(warmup=1)
    callback.on_train_start(trainer, model)
    for _ in range(2):
        callback.on_train_batch_start(trainer, model, None, 0)
        checkpointing.count_layer_call()  # as a layer would
        callback.on_train_batch_end(trainer, model, torch.tensor(0.5), None, 0)
    rows = read(callback.csv_path)
    assert [r["phase"] for r in rows] == ["warmup", "measured"]
    assert rows[1]["ckpt_policy"] == "unroll"
    assert rows[1]["checkpointing"] == "True"
    assert rows[1]["internal_ckpt"] == "False"
    # counters restart every update, so they do not accumulate
    assert rows[1]["layer_calls"] == "1"
    assert float(rows[1]["allocated_mib"]) == 7.0
    assert float(rows[1]["loss"]) == 0.5


# ---------------------------------------------------- reference workload

REFERENCE = bench.find_reference(None, SCRIPT.parents[1]) or Path("missing")


def test_run_length_matches_reference_short_run():
    assert bench.run_length(20) == (5, 5)  # 25 updates, as --short
    assert bench.run_length(4) == (1, bench.WARMUP + 4)


def test_without_flag_removes_flag_and_value():
    tokens = ["--a", "1", "--ckpt_policy", "off", "--b", "2"]
    kept = bench.without_flag(tokens, "--ckpt_policy")
    assert kept == ["--a", "1", "--b", "2"]
    assert bench.without_flag(["--x"], "--x") == []


def test_find_reference_prefers_explicit_path(tmp_path):
    ref = tmp_path / "reference_workload.py"
    ref.write_text("")
    assert bench.find_reference(str(ref), tmp_path) == ref.resolve()


def test_preflight_exit_codes():
    sys.path.insert(0, str(SCRIPT.parent))
    import preflight

    assert preflight.exit_code([(preflight.OK, "x")]) == 0
    assert preflight.exit_code([(preflight.COMPILE_FAIL, "x")]) == 2
    assert preflight.exit_code(
        [(preflight.COMPILE_FAIL, "x"), (preflight.FAIL, "y")]
    ) == 1


@pytest.mark.skipif(not REFERENCE.is_file(), reason="project repo not here")
def test_reference_check_accepts_defaults_and_rejects_changes():
    module = bench.load_reference(REFERENCE)

    def args(**kw):
        ns = bench.parse_args(["--dry_run"])
        for key, value in kw.items():
            setattr(ns, key, value)
        return ns

    ok, _ = bench.check_reference(module, args(), "off")
    assert ok
    ok, _ = bench.check_reference(
        module, args(steps=4, width=2048), "selective"
    )
    assert ok  # width and steps are study settings
    ok, text = bench.check_reference(
        module, args(extra_args="--precision 32"), "off"
    )
    assert not ok and "--precision" in text
