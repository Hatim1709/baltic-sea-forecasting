"""
Find what is allocated at the moment of peak GPU memory.

Reads `memory_snapshot.pickle` (from `train_model.py --profile_dir D
--profile_snapshot`), replays every allocation and free recorded during the
profiled update, finds the moment of highest memory and lists what was alive
then, grouped by the place in neural_lam that allocated it.

  python benchmarks/analyze_snapshot.py D/memory_snapshot.pickle [--top 15]

Notes
* Recording starts at the beginning of the update, so weights and optimizer
  state (allocated earlier) are not in the list; add `static_mib` from
  profile_step.json to get the absolute peak.
* Allocations made by the backward pass have no Python stack; they are shown
  as "<no python frame (backward / C++)>". A large share of those at the
  peak means the peak is in backward.
* Frames are read innermost first (the order PyTorch stores them). If the
  names look upside down, pass --outermost.
"""

# Standard library
import argparse
import pickle
from collections import defaultdict

MIB = 2**20
NO_FRAME = "<no python frame (backward / C++)>"


def replay(events):
    """
    Walk through the allocator events of one device. Returns
    (peak_bytes, peak_index, live_at_peak, curve) where live_at_peak is a
    list of (size, frames) and curve a list of (index, bytes).
    """
    live = {}
    current = peak = 0
    peak_index = -1
    peak_live = []
    curve = []
    for index, event in enumerate(events):
        action = event["action"]
        if action == "alloc":
            live[event["addr"]] = (event["size"], event.get("frames", []))
            current += event["size"]
            if current > peak:
                peak, peak_index = current, index
                peak_live = list(live.values())
        elif action in ("free_requested", "free_completed"):
            # count a free once, at the request
            if action == "free_requested" and event["addr"] in live:
                current -= live.pop(event["addr"])[0]
        if index % max(1, len(events) // 40) == 0:
            curve.append((index, current))
    return peak, peak_index, peak_live, curve


def site(frames, outermost=False, package="neural_lam"):
    """Name of the allocation site: first frame inside `package`"""
    if not frames:
        return NO_FRAME
    ordered = list(reversed(frames)) if outermost else frames
    for frame in ordered:
        name = frame["filename"]
        if package in name and "profiling" not in name:
            short = name.split(package)[-1].lstrip("/")
            return f"{short}:{frame['name']}:{frame['line']}"
    frame = ordered[0]
    return f"{frame['filename'].split('/')[-1]}:{frame['name']}"


def group_live(live, outermost=False):
    totals = defaultdict(int)
    for size, frames in live:
        totals[site(frames, outermost)] += size
    return sorted(totals.items(), key=lambda item: -item[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--outermost", action="store_true")
    args = parser.parse_args()
    with open(args.snapshot, "rb") as file:
        snapshot = pickle.load(file)
    events = snapshot["device_traces"][args.device]
    peak, index, live, curve = replay(events)
    print(
        f"{len(events)} allocator events; peak {peak / MIB:.0f} MiB of "
        f"allocations made during the update, at event {index} "
        f"({100 * index / max(len(events), 1):.0f}% through the update)"
    )
    print("\nMemory over the update (event number, MiB):")
    print("  " + "  ".join(f"{i}:{b / MIB:.0f}" for i, b in curve))
    print(f"\nAlive at the peak, by allocation site (top {args.top}):")
    for name, size in group_live(live, args.outermost)[: args.top]:
        print(f"  {size / MIB:9.0f} MiB  {100 * size / max(peak, 1):5.1f}%  "
              f"{name}")


if __name__ == "__main__":
    main()
