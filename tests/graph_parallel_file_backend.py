"""TEST ONLY: CPU FileStore transport for socket-restricted test sandboxes.

This exercises the actual distributed runtime/autograd functions in separate
processes without replacing the model or its MLPs. It is a transport
simulation, not Gloo/NCCL validation and never a performance benchmark.
Normal tests use Gloo. Opt in explicitly with NJORD_TEST_BACKEND=filetest.
Only default-process-group collectives used by the graph tests are provided.
"""

# Standard library
from collections import Counter
from io import BytesIO

# Third-party
import torch
import torch.distributed as dist


def _serialize(tensors):
    stream = BytesIO()
    torch.save([tensor.detach().cpu().clone() for tensor in tensors], stream)
    return stream.getvalue()


def _deserialize(payload):
    return torch.load(BytesIO(payload), map_location="cpu", weights_only=True)


def _copy_tensors(targets, values):
    if len(targets) != len(values):
        raise ValueError("File-test transport tensor count mismatch")
    with torch.no_grad():
        for target, value in zip(targets, values):
            if target.device.type != "cpu":
                raise ValueError("The file-test transport only supports CPU")
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("File-test transport shape/dtype mismatch")
            target.copy_(value)


class _FileWork(dist.Work):
    def __init__(self, complete=None):
        super().__init__()
        self._complete = complete
        self._done = False

    def wait(self, timeout=None):
        if not self._done:
            if self._complete is not None:
                self._complete()
            self._done = True
        return True

    def is_completed(self):
        return self._done

    def is_success(self):
        return self._done


class _FileProcessGroup(dist.ProcessGroup):
    def __init__(self, store, rank, world_size, timeout):
        super().__init__(rank, world_size)
        self.store = store
        self.store.set_timeout(timeout)
        self.rank_id = rank
        self.world_size = world_size
        self.counts = Counter()

    def getBackendName(self):
        return "filetest"

    def _get_backend_name(self):
        return "filetest"

    def _key(self, kind):
        index = self.counts[kind]
        self.counts[kind] += 1
        return f"{kind}/{index}"

    def allreduce(self, tensors, opts=None):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        operation = getattr(opts, "reduceOp", opts)
        operation = getattr(operation, "op", operation)
        if operation is None:
            operation = dist.ReduceOp.SUM
        if operation not in (
            dist.ReduceOp.SUM,
            dist.ReduceOp.MIN,
            dist.ReduceOp.MAX,
        ):
            raise NotImplementedError("Unsupported file-test reduction")
        key = self._key("allreduce")
        self.store.set(f"{key}/{self.rank_id}", _serialize(tensors))

        def complete():
            values = [
                _deserialize(self.store.get(f"{key}/{rank}"))
                for rank in range(self.world_size)
            ]
            result = []
            for parts in zip(*values):
                if operation == dist.ReduceOp.SUM:
                    reduced = torch.stack(parts).sum(dim=0)
                elif operation == dist.ReduceOp.MIN:
                    reduced = torch.stack(parts).amin(dim=0)
                else:
                    reduced = torch.stack(parts).amax(dim=0)
                result.append(reduced)
            _copy_tensors(tensors, result)

        return _FileWork(complete)

    def broadcast(self, tensors, opts=None):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        source = int(getattr(opts, "rootRank", opts if opts is not None else 0))
        key = self._key("broadcast")
        if self.rank_id == source:
            self.store.set(key, _serialize(tensors))
        return _FileWork(
            lambda: _copy_tensors(tensors, _deserialize(self.store.get(key)))
        )

    def send(self, tensors, dstRank, tag):
        kind = f"p2p/{self.rank_id}/{dstRank}/{tag}"
        key = self._key(kind)
        # Publish immediately so the peer can post all receives before waits.
        self.store.set(key, _serialize(tensors))
        return _FileWork()

    def recv(self, tensors, srcRank, tag):
        kind = f"p2p/{srcRank}/{self.rank_id}/{tag}"
        key = self._key(kind)
        # Waiting here would deadlock batch_isend_irecv's posting loop.
        return _FileWork(
            lambda: _copy_tensors(tensors, _deserialize(self.store.get(key)))
        )

    def allgather(self, output_tensors, input_tensors, opts=None):
        key = self._key("allgather")
        self.store.set(f"{key}/{self.rank_id}", _serialize(input_tensors))

        def complete():
            values = [
                _deserialize(self.store.get(f"{key}/{rank}"))
                for rank in range(self.world_size)
            ]
            for tensor_index, outputs in enumerate(output_tensors):
                _copy_tensors(outputs, [row[tensor_index] for row in values])

        return _FileWork(complete)

    def barrier(self, opts=None):
        key = self._key("barrier")
        self.store.set(f"{key}/{self.rank_id}", b"1")

        def complete():
            self.store.wait(
                [f"{key}/{rank}" for rank in range(self.world_size)]
            )

        return _FileWork(complete)


def register_file_test_backend():
    """Register the explicit test transport; never called by production."""
    if "filetest" not in dist.Backend.backend_list:
        dist.Backend.register_backend(
            "filetest", _FileProcessGroup, devices=["cpu"]
        )
