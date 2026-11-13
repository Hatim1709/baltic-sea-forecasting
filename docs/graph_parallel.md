# Experimental two-rank GraphEFM training

This path runs the existing `GraphEFM.training_step` on real configured data
with two cooperating processes. It includes the probabilistic encoder,
sampled latent representation, decoder, likelihood, optional learned prior
and KL term, backward pass and AdamW update. It does not use Lightning DDP or
divide different examples between processes.

## Scope of the boundary bridge

Each interaction layer computes receiver-owned node/edge rows and exchanges
the remote sender features it needs. Its outputs are differentiably assembled
in global order before the next existing model operation. The same parameters
are shared by both ranks. One explicit SUM reduction combines their parameter
gradients after backpropagation.

**Full node/edge representations at layer boundaries and the original
embedders remain replicated.** This is an integration bridge, not persistent
sharding of every model tensor. Memory reductions and useful maximum widths
must be measured; a successful run alone is not a speedup or capacity claim.

The initial runner accepts one training forecast step, float32, AdamW,
checkpointing disabled and CRPS disabled. It supports the common global
configuration where the input and output grid have the same node namespace.
Separate atmospheric/boundary input grids, checkpoint restoration, validation,
plotting and other precision/optimizer modes are rejected or outside this
entry point. Existing `neural_lam.train_model` behavior is unchanged.

## Prepare owners for the same graph used by the model

For a first integration smoke run, prepare a deterministic geometric baseline
using the existing PyTorch environment. Generated owners and results belong
outside the source checkout. From the repository:

```bash
python neural_lam/partition_owners.py \
  --graph-dir /data/sample_data/graphs/global_hierarchical_graph \
  --geometric \
  --output-dir /runs/full-graph
```

An existing fine-mesh ownership vector can be supplied with `--mesh0-owner`
instead of `--geometric`. The preparation
records the graph fingerprint and separate owners for grid and mesh levels.
The training command validates the owners against its actual graph files.
For example, replace `--geometric` with
`--mesh0-owner /runs/selected-mesh-partition/mesh0_owner.pt` to reuse an exact
previous candidate.
Known metadata formats have their original graph guards checked on import.
A bare owner tensor contains no original graph identity; its lineage remains
recorded as unverified, while the newly prepared complete plan is fingerprinted
against the current graph.

Alternatively, compute a new METIS partition in a preparation environment
with Python 3.10 or newer, PyTorch and PyMetis:

```bash
python -m pip install "pymetis==2025.2.2"
python neural_lam/partition_owners.py \
  --graph-dir /data/sample_data/graphs/global_hierarchical_graph \
  --metis-seed 9 \
  --output-dir /runs/full-graph-metis9
```

A seed alone does not guarantee identical partitions across METIS builds or
architectures; record the build and preserve owner vectors when comparing
machines. **The training runner does not import or require PyMetis.**

Copy `owners.pt` to the run directory on the training machine, preserving the
same downloaded graph files. This avoids rebuilding METIS partitions in the
training environment. The full model dependencies are needed for training,
whereas the standalone preparation script only needs PyTorch and, when
computing a new METIS fine-mesh partition, PyMetis.

## First run on two GPUs

Use the same configuration and graph that were used for owner preparation:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m neural_lam.train_graph_parallel \
  --config_path /data/sample_data/global_ocean_1.yaml \
  --graph global_hierarchical_graph \
  --partition-dir /runs/full-graph \
  --gp-output-dir /runs/two-gpu-first-step \
  --gp-device cuda
```

The defaults are a batch of one, one optimizer step and widths 16/8/8/4 for
the main/grid/fine-mesh/edge representations. Every rank checks that its
arguments, configuration, ownership arrays, initial parameters and actual
batch values agree. There is no DistributedSampler. Model RNG states are
synchronized, so full-shaped latent sampling is shared between graph ranks.
The receiver-owned wrappers must not consume random numbers. Standard MLP
MC-dropout must remain disabled for this mode.

Before the first distributed step, each rank executes a disposable serial
reference from identical weights, batch and RNG state. It then restores the
RNG and runs the partitioned step. The check compares loss, predicted means
and standard deviations, every parameter gradient and one AdamW parameter
update. The serial reference is released before recording the distributed
step's GPU peaks. This check temporarily requires enough memory for a serial
reference as well as the original model; start with the small defaults.

Loss, predictions and updated parameters use pointwise `assert_close` with
`rtol=1e-4`, `atol=1e-5`. Partitioning changes the summation order of shared
parameter gradients, and cancellation can affect individual small entries.
Each gradient tensor instead has to satisfy **both** bounds, with `n` entries:

```text
L2(g - reference) <= atol * sqrt(n) + rtol * L2(reference)
max(abs(g - reference)) <= atol + rtol * max(abs(reference))
```

Gradient shapes, dtypes, finite values and unused (`None`) status must also
agree. The report retains per-tensor absolute errors, relative L2 errors,
scaled bounds and pointwise mismatch counts. CUDA correctness runs disable
TF32. These checks do not imply bitwise identity of float32 reductions.

The distributed loss is divided by two before backward. Differentiable
boundary assembly accumulates replica derivative contributions, while
`sum_parameter_gradients` combines the two local contributions exactly once.
Do not wrap this runner in DDP or add another gradient averaging mechanism.

CPU/Gloo can exercise the same entry point when the real data dependencies
and configured dataset are available:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m neural_lam.train_graph_parallel \
  --config_path /data/sample_data/global_ocean_1.yaml \
  --graph global_hierarchical_graph \
  --partition-dir /runs/full-graph \
  --gp-output-dir /runs/two-cpu-first-step \
  --gp-device cpu
```

## Results and subsequent experiments

The runner prints per-rank loss, step time and CUDA allocated/reserved peaks
and writes `graph_parallel_run.json` in the requested output directory. It
records the implementation revision, working-tree status, source hashes,
arguments, relation installation counts
and first-step equivalence status. The values are smoke-run records, not a
warmed-up repeated benchmark.

After the first small correctness run passes, `--gp-max-steps` controls the
number of optimizer steps; `--epochs` remains an upper epoch bound. The
usual width arguments can be changed deliberately. For capacity experiments
whose serial counterpart does not fit, `--gp-skip-first-step-check` explicitly
skips that reference and records the omission. Such a run does not supply new
equivalence evidence. Keep the passed small-run configuration and results.

The distributed-layer tests separately compare node/edge input gradients,
repeated layers and operator residual/aggregation variants. A real-data CUDA
run is still required to establish CUDA/NCCL operation and model-level memory
behavior. Persistent node/edge sharding, mixed precision, multi-step
checkpointing and performance tuning are further integration stages.

## Local validation and its limits

The focused owner and CLI checks need the project's installed dependencies:

```bash
python -m pytest --noconftest \
  tests/test_partition_owners.py tests/test_graph_parallel_cli.py -q
```

Observed result: 9 owner checks and 20 CLI/comparison checks passed. The
numerical suite normally uses Gloo. A socket-restricted environment can
explicitly substitute a **test-only** CPU FileStore transport:

```bash
NJORD_TEST_BACKEND=filetest python -m pytest --noconftest \
  tests/test_graph_parallel.py -q
```

Observed result: four CPU cases passed, two CUDA cases skipped. They cover
the real interaction operators and an actual GraphEFM training-method fixture,
including two autoregressive steps with checkpointing disabled.

To exercise the real configuration, datastores, DataLoader and model with
the same test transport, use the manual harness and a prepared owner plan:

```bash
NJORD_TEST_BACKEND=filetest python tests/run_graph_parallel_real_data.py \
  --config_path /data/sample_data/global_ocean_1.yaml \
  --graph global_hierarchical_graph \
  --partition-dir /runs/full-graph \
  --gp-output-dir /runs/real-data-fp32-validation \
  --gp-device cpu
```

The optional `--diagnostic-float64 --gp-rtol 1e-8 --gp-atol 1e-9` arguments
cast the model and floating batch tensors for a separate precision diagnostic.
This is a test harness override; the production entry point still accepts
only float32. Its report labels the override explicitly.

Both real-data checks passed with the small default model and geometric
ownership. In float32, the loss was identical, the largest prediction error
was `4.77e-7`, and the largest AdamW parameter error was `3.73e-9`.
The worst per-tensor gradient relative L2 error was `2.47e-5`; three of
41,620 gradient entries exceeded the pointwise tolerance, while every tensor
satisfied both documented bounds. The float64 diagnostic at `rtol=1e-8`,
`atol=1e-9` had a worst gradient relative L2 error of `4.64e-14` and no
pointwise gradient mismatches. This precision check supports the diagnosis
that float32 reduction order and cancellation explain those small entries.

The real-data checks use the public sample's 42,348 grid nodes and mesh levels
of 7,193 and 1,812 nodes, with all six edge relations. The transport substitution
does not validate Gloo/NCCL or CUDA, and none of these runs establishes
hardware scaling or memory savings.
