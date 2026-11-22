"""
Pick how many forecast steps to checkpoint for a given GPU memory budget.

Background (measured on the reference workload, A100, d=1024, see RESULTS.md):
memory and time grow linearly with the number of forecast steps T.
  no checkpointing : peak = S + T * a          time = T * t
  every step ckpt. : peak = S + a              time = T * t * (1 + r)
with S the fixed part (weights, data, graph), a the activations of one
forecast step, and r the extra cost of recomputing one step (about 0.78).
Checkpointing only the FIRST k steps (--ckpt_unroll_steps k) gives
  peak = S + max(T - k, 1) * a        time = T * t * (1 + r * k / T)
because the backward pass frees the late, uncheckpointed steps first.
This is a prediction from the linear model, not yet measured on a GPU.

Example:  python benchmarks/plan_unroll_steps.py --steps 6 --budget_mib 40000
"""

# Standard library
import argparse

# Defaults fitted to the reference workload (MiB, ratio)
STATIC_MIB = 7768.0
STEP_MIB = 10962.0
RECOMPUTE = 0.78


def predict(steps, k, static=STATIC_MIB, step=STEP_MIB, recompute=RECOMPUTE):
    """Predicted (peak memory in MiB, time relative to no checkpointing)"""
    k = max(0, min(k, steps))
    peak = static + max(steps - k, 1) * step
    return peak, 1.0 + recompute * k / steps


def plan_unroll_steps(steps, budget_mib, **model):
    """
    Smallest number of checkpointed steps k whose predicted peak fits the
    budget. Returns (k, peak, relative_time), or None if even checkpointing
    every step does not fit (one forecast step must fit on the GPU).
    """
    for k in range(steps + 1):
        peak, time = predict(steps, k, **model)
        if peak <= budget_mib:
            return k, peak, time
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--budget_mib", type=float, required=True)
    parser.add_argument("--static_mib", type=float, default=STATIC_MIB)
    parser.add_argument("--step_mib", type=float, default=STEP_MIB)
    parser.add_argument("--recompute", type=float, default=RECOMPUTE)
    args = parser.parse_args()
    result = plan_unroll_steps(
        args.steps,
        args.budget_mib,
        static=args.static_mib,
        step=args.step_mib,
        recompute=args.recompute,
    )
    if result is None:
        raise SystemExit(
            "Does not fit: even one forecast step exceeds the budget."
        )
    k, peak, time = result
    print(f"--ckpt_policy unroll --ckpt_unroll_steps {k}")
    print(f"predicted peak {peak:.0f} MiB, time {time:.2f}x of no ckpt")


if __name__ == "__main__":
    main()
