"""
Profile ONE training update: where does the memory peak occur, and how much
does each part of a forecast step store for the backward pass?

Two tools, both only used when asked for (--profile_dir in train_model.py):

RegionProfiler  hooks into named sub-modules ("regions") and records, per
                region, the time of its forward pass and the memory that
                autograd keeps for backward ("saved tensors"). Run it with
                the policy `off`: with checkpointing the saved tensors are
                hidden inside the checkpoint blocks.
StepProfiler    Lightning callback that applies the RegionProfiler to one
                chosen update and also records, on a GPU, the peak memory of
                the forward pass, the backward pass and the optimizer step,
                and optionally an allocator snapshot (every allocation with
                its Python stack) for benchmarks/analyze_snapshot.py.

The profiler never changes results: the hooks only look at tensors.
Attribution rule: saved memory is counted once per storage and goes to the
innermost region that was running when it was first saved. Parameters,
buffers and the input batch are not counted (they exist anyway).
"""

# Standard library
import json
import time
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch

# Local
from .interaction_net import FlexibleNet, InteractionNet, PropagationNet

MIB = 2**20
# Sub-modules that are always a region, whatever their depth
LAYER_TYPES = (FlexibleNet, InteractionNet, PropagationNet)


def _storage(tensor):
    """(pointer, size in bytes) of a tensor's storage, or None"""
    try:
        storage = tensor.untyped_storage()
        return storage.data_ptr(), storage.nbytes()
    except Exception:  # pylint: disable=broad-except
        return None


def _tensors_in(obj):
    """All tensors inside nested lists/tuples/dicts (e.g. a batch)"""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _tensors_in(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _tensors_in(value)


class RegionProfiler:
    """
    Time and saved-tensor memory per region of one forward pass.

    Use: start() before the forward pass, stop_saving() just before backward,
    stop() when done, then summary(). Regions are all sub-modules at most
    `max_depth` levels below the model (the model's own children are depth 1)
    plus every GNN layer, at any depth.
    """

    def __init__(self, model, max_depth=2):
        self.model = model
        self.max_depth = max_depth
        self.cuda = any(p.is_cuda for p in model.parameters())
        self.calls = []  # one dict per call of a region
        self.saved = {}  # region name -> bytes saved for backward
        self.saved_dtype = {}  # dtype name -> bytes saved for backward
        self._stack = []  # indices into self.calls of the running regions
        self._seen = set()  # storage pointers that are not counted again
        self._handles = []
        self._context = None

    def _is_region(self, name, module):
        if not name:
            return False
        return name.count(".") < self.max_depth or isinstance(
            module, LAYER_TYPES
        )

    def _now(self):
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _pre(self, name):
        def hook(_module, _args):
            parent = self._stack[-1] if self._stack else -1
            self.calls.append(
                {"name": name, "parent": parent, "start": self._now()}
            )
            self._stack.append(len(self.calls) - 1)

        return hook

    def _post(self):
        def hook(_module, _args, _output):
            index = self._stack.pop()
            self.calls[index]["end"] = self._now()

        return hook

    def _pack(self, tensor):
        info = _storage(tensor)
        if info is not None and info[0] and info[0] not in self._seen:
            self._seen.add(info[0])
            region = (
                self.calls[self._stack[-1]]["name"]
                if self._stack
                else "<outside regions>"
            )
            self.saved[region] = self.saved.get(region, 0) + info[1]
            dtype = str(tensor.dtype).replace("torch.", "")
            self.saved_dtype[dtype] = (
                self.saved_dtype.get(dtype, 0) + info[1]
            )
        return tensor

    @staticmethod
    def _unpack(tensor):
        return tensor

    def start(self, batch=None):
        """Install the hooks. `batch`: input tensors, not counted as saved"""
        for tensor in list(self.model.parameters()) + list(
            self.model.buffers()
        ):
            info = _storage(tensor)
            if info is not None:
                self._seen.add(info[0])
        for tensor in _tensors_in(batch):
            info = _storage(tensor)
            if info is not None:
                self._seen.add(info[0])
        for name, module in self.model.named_modules():
            if self._is_region(name, module):
                self._handles.append(
                    module.register_forward_pre_hook(self._pre(name))
                )
                self._handles.append(
                    module.register_forward_hook(self._post())
                )
        self._context = torch.autograd.graph.saved_tensors_hooks(
            self._pack, self._unpack
        )
        self._context.__enter__()

    def stop_saving(self):
        """Stop recording saved tensors (call before backward)"""
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None

    def stop(self):
        """Remove all hooks and turn the recorded times into milliseconds"""
        self.stop_saving()
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self.cuda:
            torch.cuda.synchronize()
        for call in self.calls:
            if "end" not in call:  # a region that raised
                call["end"] = call["start"]
            if self.cuda:
                call["ms"] = call["start"].elapsed_time(call["end"])
            else:
                call["ms"] = (call["end"] - call["start"]) * 1000
        for call in self.calls:
            del call["start"], call["end"]

    def dtype_summary(self):
        """MiB saved for backward per data type (e.g. float32, bfloat16)"""
        return {k: v / MIB for k, v in sorted(self.saved_dtype.items())}

    def summary(self):
        """
        One row per region: calls, saved MiB (own, not counting regions
        inside it), inclusive time, own time (inclusive minus the regions
        inside it) in ms.
        """
        children_ms = [0.0] * len(self.calls)
        for call in self.calls:
            if call["parent"] >= 0:
                children_ms[call["parent"]] += call["ms"]
        rows = {}
        for index, call in enumerate(self.calls):
            row = rows.setdefault(
                call["name"],
                {
                    "name": call["name"],
                    "calls": 0,
                    "saved_mib": 0.0,
                    "incl_ms": 0.0,
                    "self_ms": 0.0,
                },
            )
            row["calls"] += 1
            row["incl_ms"] += call["ms"]
            row["self_ms"] += call["ms"] - children_ms[index]
        for name, size in self.saved.items():
            row = rows.setdefault(
                name,
                {
                    "name": name,
                    "calls": 0,
                    "saved_mib": 0.0,
                    "incl_ms": 0.0,
                    "self_ms": 0.0,
                },
            )
            row["saved_mib"] = size / MIB
        return sorted(rows.values(), key=lambda row: -row["saved_mib"])


class StepProfiler(pl.Callback):
    """
    Profile the training update number `batch_idx` (0-based; choose >= 2 so
    that the optimizer state already exists) and write
    `<out_dir>/profile_step.json`. On a GPU it also records the peak memory
    of each phase of the update and, with `snapshot`, an allocator snapshot
    (`memory_snapshot.pickle`, analysed by benchmarks/analyze_snapshot.py).
    """

    def __init__(
        self, out_dir, batch_idx=3, snapshot=False, max_depth=2, kernels=False
    ):
        self.kernels = kernels
        self.out_dir = Path(out_dir)
        self.batch_idx = batch_idx
        self.snapshot = snapshot
        self.max_depth = max_depth
        self.active = False

    # -- helpers
    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def _mark(self, phase):
        """Close a phase: its time and (GPU) peak and current memory"""
        self._sync()
        now = time.perf_counter()
        self.phases[phase] = {"ms": (now - self._last) * 1000}
        if self.cuda:
            self.phases[phase].update(
                peak_mib=torch.cuda.max_memory_allocated() / MIB,
                allocated_end_mib=torch.cuda.memory_allocated() / MIB,
            )
            torch.cuda.reset_peak_memory_stats()
        self._last = time.perf_counter()

    def _write_kernel_table(self):
        """Operations of the update sorted by own GPU (or CPU) time"""
        self.kernel_profile.__exit__(None, None, None)
        keys = (
            ["self_device_time_total", "self_cuda_time_total"]
            if self.cuda
            else []
        ) + ["self_cpu_time_total"]
        for key in keys:
            try:
                table = self.kernel_profile.key_averages().table(
                    sort_by=key, row_limit=30, max_name_column_width=70
                )
                break
            except Exception:  # pylint: disable=broad-except
                continue
        (self.out_dir / "kernels.txt").write_text(table)

    # -- Lightning hooks
    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if batch_idx != self.batch_idx:
            return
        self.active = True
        self.cuda = any(p.is_cuda for p in model.parameters())
        self.phases = {}
        self._sync()
        if self.cuda:
            self.static_mib = torch.cuda.memory_allocated() / MIB
            torch.cuda.reset_peak_memory_stats()
            if self.snapshot:
                torch.cuda.memory._record_memory_history(max_entries=500000)
        if self.kernels:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.kernel_profile = torch.profiler.profile(activities=activities)
            self.kernel_profile.__enter__()
        self.profiler = RegionProfiler(model, self.max_depth)
        self.profiler.start(batch)
        self._last = time.perf_counter()

    def on_before_backward(self, trainer, model, loss):
        if self.active:
            self.profiler.stop_saving()
            self._mark("forward")

    def on_after_backward(self, trainer, model):
        if self.active:
            self._mark("backward")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if not self.active:
            return
        self.active = False
        self._mark("optimizer")
        self.profiler.stop()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.kernels:
            self._write_kernel_table()
        if self.cuda and self.snapshot:
            torch.cuda.memory._dump_snapshot(
                str(self.out_dir / "memory_snapshot.pickle")
            )
            torch.cuda.memory._record_memory_history(enabled=None)
        args = getattr(model, "args", None)
        result = {
            "meta": {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name()
                if self.cuda
                else "cpu",
                "policy": getattr(model, "ckpt_policy", None),
                "ckpt_unroll_steps": getattr(model, "ckpt_unroll_steps", -1),
                "ar_steps_train": getattr(args, "ar_steps_train", None),
                "hidden_dim": getattr(args, "hidden_dim", None),
                "batch_idx": self.batch_idx,
                "regions_saved_valid": getattr(model, "ckpt_policy", "")
                == "off",
            },
            "static_mib": self.static_mib if self.cuda else None,
            "phases": self.phases,
            "saved_by_dtype_mib": self.profiler.dtype_summary(),
            "regions": self.profiler.summary(),
        }
        path = self.out_dir / "profile_step.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"Step profile written to {path}", flush=True)
