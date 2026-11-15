"""
Feasibility probe: torch.compile with an activation memory budget on one
Njord GNN layer (InteractionNet), compared with eager and with our per-layer
checkpointing policies. Meant to be run once on the GPU (Piora) before deciding
whether a whole-model compile comparison is worth the effort.

Budget (torch._functorch.config.activation_memory_budget): 1.0 = compiled
default (keeps what is fast), 0.0 = recompute as much as possible (like full
checkpointing), values in between trade the two. See
https://pytorch.org/blog/activation-checkpointing-techniques/

For each variant it prints: gradient agreement with eager, peak GPU memory and
time per forward+backward. Variants that fail to compile are reported, not
fatal. Nothing here changes the model or the training code.

Usage (from the repository root):
    PYTHONPATH=. python benchmarks/compile_budget_probe.py --nodes 200000 \
        --edges 1200000 --dim 512
"""

# Standard library
import argparse
import time

# Third-party
import torch
import torch._functorch.config as functorch_config

# First-party
from neural_lam import checkpointing
from neural_lam.interaction_net import InteractionNet


def make_layer(nodes, edges, dim, device):
    """A random graph and one InteractionNet on it, in training mode"""
    receivers = torch.cat(
        (
            torch.arange(nodes),
            torch.randint(0, nodes, (edges - nodes,)),
        )
    )
    senders = torch.randint(0, nodes, (edges,))
    layer = InteractionNet(torch.stack((senders, receivers)), dim)
    return layer.to(device).train()


def measure(fn, inputs, device, repeats):
    """Mean time (ms) and peak memory (MiB) of forward+backward, plus grads"""
    for tensor in inputs:
        tensor.grad = None

    def step():
        out = fn(*inputs)
        sum((o.float() ** 2).sum() for o in out).backward()

    step()  # warm-up, includes compilation
    grads = [t.grad.clone() for t in inputs]
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    return elapsed, peak, grads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--edges", type=int, default=120000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.0]
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    layer = make_layer(args.nodes, args.edges, args.dim, device)
    nodes = torch.randn(1, args.nodes, args.dim, device=device)
    edges = torch.randn(1, args.edges, args.dim, device=device)
    inputs = [nodes.requires_grad_(), edges.requires_grad_()]

    def call(*tensors):
        # nodes serve as senders and receivers, as in a same-level layer
        return layer(tensors[0], tensors[0], tensors[1])

    results = {}
    reference = None
    print(f"device={device} backend={args.backend}")
    print(f"{'variant':<34}{'ms':>9}{'peak MiB':>11}  max grad diff")

    def report(name, fn):
        nonlocal reference
        try:
            ms, peak, grads = measure(fn, inputs, device, args.repeats)
        except Exception as error:  # compile problems are a result
            print(
                f"{name:<34}FAILED: {type(error).__name__}: {str(error)[:90]}"
            )
            return
        if reference is None:
            reference = grads
        diff = max((g - r).abs().max().item() for g, r in zip(grads, reference))
        results[name] = (ms, peak)
        print(f"{name:<34}{ms:>9.1f}{peak:>11.0f}  {diff:.2e}")

    for name, mode in (
        ("eager", False),
        ("eager + full layer ckpt", True),
        ("eager + selective layer ckpt", checkpointing.SELECTIVE),
    ):
        layer.ckpt_inner = mode
        report(name, call)
    layer.ckpt_inner = False

    for budget in args.budgets:
        torch._dynamo.reset()
        functorch_config.activation_memory_budget = budget

        def compiled(*tensors):
            # compiled lazily so that a failure to compile is reported
            return torch.compile(call, backend=args.backend, dynamic=False)(
                *tensors
            )

        report(f"compile, memory budget {budget}", compiled)


if __name__ == "__main__":
    main()
