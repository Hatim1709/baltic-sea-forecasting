"""
Read the profile_step.json files written by `train_model.py --profile_dir`
and answer: where is the memory peak, which parts of a forecast step store
a lot for little recompute time, and would keeping some of them stored beat
plain `unroll`?

  python benchmarks/analyze_profile.py off/profile_step.json \
      --unroll unroll/profile_step.json

The first file must come from `--ckpt_policy off` (only then are the saved
tensors visible per region). The optional second file comes from `unroll`
and gives the phase peaks of the policy we want to improve.

Model used for the "keep" table (a prediction, to be confirmed on a GPU):
  time(unroll)  = update time of `off` + forward time (one full recompute)
  keeping a set K of regions stored: the recompute of K is saved, and each
  forecast step stores size(K) more, so memory grows by size(K) per step.
"""

# Standard library
import argparse
import json
from collections import defaultdict

BUDGETS_MIB = (0, 250, 500, 1000, 2000, 4000)


def load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def phase_lines(result):
    lines = [
        f"  device {result['meta']['device']}, policy "
        f"{result['meta']['policy']}, steps {result['meta']['ar_steps_train']}"
    ]
    if result.get("static_mib") is not None:
        lines.append(
            f"  before the update (weights, optimizer, ...): "
            f"{result['static_mib']:.0f} MiB"
        )
    for name, phase in result["phases"].items():
        text = f"  {name:<10} {phase['ms']:9.1f} ms"
        if "peak_mib" in phase:
            text += (
                f"   peak {phase['peak_mib']:9.0f} MiB"
                f"   still allocated at end {phase['allocated_end_mib']:9.0f}"
            )
        lines.append(text)
    return lines


def group_regions(result, depth):
    """Merge regions whose names share the first `depth` path components"""
    groups = defaultdict(lambda: {"saved_mib": 0.0, "self_ms": 0.0})
    for row in result["regions"]:
        key = ".".join(row["name"].split(".")[:depth])
        groups[key]["saved_mib"] += row["saved_mib"]
        groups[key]["self_ms"] += row["self_ms"]
    return dict(groups)


def region_lines(groups, steps):
    total_mib = sum(g["saved_mib"] for g in groups.values()) or 1.0
    lines = [
        f"  {'region':<40}{'MiB/step':>10}{'share':>8}"
        f"{'fwd ms/step':>13}{'ms per GiB':>12}"
    ]
    for name, group in sorted(
        groups.items(), key=lambda item: -item[1]["saved_mib"]
    ):
        per_gib = (
            group["self_ms"] / (group["saved_mib"] / 1024)
            if group["saved_mib"] > 1
            else float("inf")
        )
        lines.append(
            f"  {name:<40}{group['saved_mib'] / steps:10.0f}"
            f"{100 * group['saved_mib'] / total_mib:7.1f}%"
            f"{group['self_ms'] / steps:13.1f}{per_gib:12.0f}"
        )
    return lines


def keep_table(groups, phases, steps, budgets=BUDGETS_MIB):
    """
    Greedy choice of regions to keep stored, best recompute time saved per
    MiB first, for several growth budgets (MiB per forecast step).
    Returns a list of dicts.
    """
    off_ms = sum(phases[p]["ms"] for p in ("forward", "backward", "optimizer"))
    unroll_ms = off_ms + phases["forward"]["ms"]
    items = sorted(
        (
            (name, g["saved_mib"] / steps, g["self_ms"] / steps)
            for name, g in groups.items()
        ),
        key=lambda item: -(item[2] / max(item[1], 1e-6)),
    )
    rows = []
    for budget in budgets:
        used, saved_ms, kept = 0.0, 0.0, []
        for name, mib, ms in items:
            if used + mib <= budget and ms > 0:
                used += mib
                saved_ms += ms
                kept.append(name)
        time_ms = unroll_ms - saved_ms * steps
        rows.append(
            {
                "budget_mib": budget,
                "growth_mib_per_step": used,
                "kept": kept,
                "time_vs_off": time_ms / off_ms,
                "time_vs_unroll": time_ms / unroll_ms,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("off_profile", help="profile_step.json of `off`")
    parser.add_argument("--unroll", help="profile_step.json of `unroll`")
    parser.add_argument("--depth", type=int, default=3,
                        help="path components that define a region")
    args = parser.parse_args()

    off = load(args.off_profile)
    steps = off["meta"]["ar_steps_train"] or 1
    print("Phases, policy off (one update):")
    print("\n".join(phase_lines(off)))
    if args.unroll:
        unroll = load(args.unroll)
        print("\nPhases, unroll (one update):")
        print("\n".join(phase_lines(unroll)))
    if not off["meta"]["regions_saved_valid"]:
        print("\nWARNING: this profile was not made with --ckpt_policy off;"
              " saved memory per region is not visible. Stopping.")
        return
    phases = off["phases"]
    fwd_share = phases["forward"]["ms"] / sum(
        phases[p]["ms"] for p in ("forward", "backward", "optimizer")
    )
    print(
        f"\nForward is {100 * fwd_share:.0f}% of an update, so a full "
        f"recompute (unroll) should cost about {1 + fwd_share:.2f}x."
    )
    groups = group_regions(off, args.depth)
    print(f"\nWhat each part of a forecast step stores (steps={steps}):")
    print("\n".join(region_lines(groups, steps)))
    print("\nKeep some regions stored instead of recomputing them (predicted):")
    print(f"  {'growth MiB/step':>16}{'time vs unroll':>16}"
          f"{'time vs off':>13}  kept")
    for row in keep_table(groups, phases, steps):
        print(
            f"  {row['growth_mib_per_step']:16.0f}"
            f"{row['time_vs_unroll']:16.2f}{row['time_vs_off']:13.2f}  "
            + (", ".join(row["kept"]) or "-")
        )


if __name__ == "__main__":
    main()
