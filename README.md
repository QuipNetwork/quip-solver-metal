# quip-solver-metal

Metal Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-metal-sa`), multi-spin coded SA
(`quip-metal-msa`), heat-bath Gibbs (`quip-metal-gibbs`) and the Otsuka
cluster anneal (`quip-metal-otsuka`), shipped as separate binaries.

**macOS arm64 only (Apple Silicon).** Metal and IOKit exist on no other
platform, so this crate does not build anywhere else and offers no stub or CPU
fallback. A non-macOS build fails while compiling the Apple-only dependencies.
Build and run on Apple Silicon.

Energies match consensus `quip_solver_core::quip_protocol::scoring::energy_milli`.
The multi-spin kernel computes each read's energy on the device in integer
milli units when every coefficient is a whole number in `i8` range, and the
host rescores 1 job in 1,000 to check it. The host scores the reads of the
other kernels, and of graphs with other coefficients.

## Binaries

| binary | algorithm |
|--------|-----------|
| `quip-metal-sa` | simulated annealing (Metropolis) |
| `quip-metal-msa` | multi-spin coded simulated annealing (32 replicas per word) |
| `quip-metal-gibbs` | heat-bath Gibbs |
| `quip-metal-otsuka` | Otsuka cluster anneal on `quip/twisted-ice-dimer` revision 1 jobs, simulated annealing on every other job |

Each
[Release](https://gitlab.com/quip.network/quip-solver-metal/-/releases)
has prebuilt `arm64` binaries. A macOS CI runner builds them best-effort
(see [`.gitlab-ci.yml`](.gitlab-ci.yml)).

## Build

```sh
cargo build --release
```

The solver contract (`quip-proto`, `quip-protocol`, `quip-solver-core`) comes
from crates.io at version 0.0.11, published from
[quip-solver-core](https://gitlab.com/quip.network/quip-solver-core). It
speaks protocol version 2, so the coordinator must also speak version 2.

The miners accept `ISING_GENERATE` salt leases. The session draws each salt's
problem from the lease and runs it through the same stream as a plain job.
It sends a `Result` only for a salt whose reads meet the target. Each lease ends
with one `LeaseDone` summary and a credit refund.

The SA, Gibbs and MSA binaries declare the generic problem-family tier. Each
advertises no `solver_id` and no generated or accepted family, because these
sessions do not expand family leases. The session rejects a family lease with
`FamilyUnsupported` and runs no fallback. For an explicit job with a
`FamilyBinding`, the miner solves the plain model and the coordinator decides
acceptance.
`tests/family_compatibility.rs` checks the declaration, each kernel's limits,
and both job paths.

## Otsuka binary (`quip-metal-otsuka`)

`quip-metal-otsuka` has the `solver_id` `quip-metal-otsuka/1`. It
lists `quip/twisted-ice-dimer` revision 1 in its generated and accepted
families. The session expands family leases of that revision and checks
the `FamilyBinding` of explicit jobs.

- A checked twisted-ice-dimer job runs on the Otsuka cluster kernel
  (`kernels/cluster.metal`). Each read is one anneal, and a job has at most 256
  reads. The schedule follows the job, as in the CUDA miner: the job's
  `num_sweeps`, `sweeps_per_beta`, beta range, and beta schedule give the
  beta ladder, and each command buffer runs inside one rung at that rung's
  beta. Every sweep has 2 rounds and a single-spin sweep. The job seed seeds
  the anneal. The host scores each read on the original model.
- Each other job runs on the Metal SA engine of `quip-metal-sa`. For the
  same seed, a plain job returns the same reads on both binaries.
- The cluster kernel and Metal SA never share the GPU. Each plans its
  command buffers against the 500 ms ceiling as if it owns the GPU. A family
  job waits until every plain job ahead of it answers, and no plain job starts
  while a family job runs. With one sweep per beta, the ladder runs one
  command buffer per sweep. One family job at 256 reads and 2,048 sweeps on
  the 4,577-node Advantage2 instance took 4.05 to 4.28 s in three runs on an
  Apple M4 Max: 2,050 command buffers, about 2 ms each, of which about
  1.8 ms is GPU time (`a_production_family_job_time` in
  `tests/cluster_device.rs`).
- A family job that does not fit the cluster layout, the memory budget, or the
  single-buffer limit gets a `Capacity` reject with one refund, and the
  session goes on. The miner refuses the job before it allocates a buffer.
- When the GPU load of other processes passes the `--utilization` ceiling
  with `--yielding` on, a family job stops before its next command buffer,
  and the miner rejects it as busy. The load of the family job itself does
  not count, because the job reports its GPU time to the governor.

`tests/cluster_session.rs` checks the routing, the cancellation, the
refusals, and the job order. `tests/family_compatibility.rs` checks that
the binary passes the family gate for twisted-ice-dimer revision 1.

## Running

**Connect to a coordinator** (production):

```sh
quip-metal-sa --quip-coordinator unix:///run/quip/coord.sock
```

**Driver / fixed-input (run in isolation).** Use the coordinator's
`drive` harness pointed at the binary — `--source random` for golden-drawn
problems, `--source list <jsonl>` for a fixed replay:

```sh
quip-coordinator drive --miner ./quip-metal-sa \
  --source random --topology-preset advantage2-system1 \
  --count 8 --num-reads 16 --num-sweeps 1030 --report out.jsonl
```

**Introspection:**

```sh
quip-metal-sa --capabilities   # capabilities JSON
quip-metal-sa --check          # probe the backend is runnable
```

## Advertised limits

Each binary derives `maxNodes` and `maxEdges` at startup from the unified
memory of the host and from the one-sweep limit of its kernels. The session
admission applies the same capacity model and the same one-sweep rule, so
`--capabilities` prints the limits that the session enforces.

- `maxEdges` is the coupler count of the largest complete model that the
  session admits.
- `maxNodes` is the largest node count that the session admits for a model
  with `maxEdges` couplers. The reference degree of that model is
  `2 * maxEdges / maxNodes`.
- Both corners hold at every read count from 32 to 256, at 64 betas, and on
  every coefficient layout that the kernel runs.
- The SA, Gibbs and MSA binaries advertise `maxDegree` 62, the sparse degree
  limit of the weighted multi-spin kernel (`sampler::MSA_SPARSE_MAX_DEG`).
- `quip-metal-otsuka` advertises `maxDegree` 128. Its plain jobs run on SA.
  The SA one-sweep limit refuses some models with the SA `maxEdges` and a
  mean degree between 62 and 128. For that reason, the Otsuka `maxEdges` is
  the coupler count of the largest regular degree-128 model that the session
  admits. Its `maxNodes` is the largest node count that the session admits
  for a model with that many couplers.
- `--max-nodes N` lowers `maxNodes`. A value of 0 or past the limit is a
  configuration error (exit code 64), and the binary opens nothing.

On the calibration host, an Apple M4 Max with a 40-core GPU and 128 GB of
memory, the limits are:

| Binary | `maxNodes` | `maxEdges` | Reference degree | `maxDegree` |
|---|---:|---:|---:|---:|
| `quip-metal-sa` | 41,647 | 998,991 | 48.0 | 62 |
| `quip-metal-gibbs` | 5,684,919 | 1,398,628 | 0.49 | 62 |
| `quip-metal-msa` | 13,724,198 | 11,992,753 | 1.75 | 62 |
| `quip-metal-otsuka` | 60,743 | 952,896 | 31.4 | 128 |

The session admits every model inside `maxNodes`, `maxEdges` and
`maxDegree`. The test `every_shape_in_the_advertised_box_is_admitted`
checks a grid over that box for all four binaries. For the SA, Gibbs, and MSA
binaries, the harness refuses a model with a node past degree 62, which
removes two kinds of refusal from inside the box:

- A model past degree 62 runs only on dense-matrix storage. The one-sweep
  limit holds a dense model to 1,414 nodes on SA, 1,673 on Gibbs and 4,898
  on MSA, so the session refused a larger model past that degree.
- The SA CSR one-sweep estimate refuses high mean degrees well below the
  dense size: it refuses 7,025 nodes at a mean degree of 284.

The memory model can still refuse a job at a corner while the multi-spin
slot pools hold memory. That refusal is a non-fatal `Capacity` reject with
one refund.

The complete models that set `maxEdges` pass degree 62. They run only
through the library API (`sample_ising`), not through a session.

This table gives the largest admitted node count at a fixed regular degree.
The capacity model computed it in 1 percent steps for an Apple M4 Max with a
40-core GPU and 128 GB:

| Binary | Degree 4 | Degree 20 | Degree 62 | Degree 200 |
|---|---:|---:|---:|---:|
| `quip-metal-sa` | 101,532 | 88,572 | 32,906 | 9,322 |
| `quip-metal-gibbs` | 5,684,919 | 978,887 | 358,494 | 91,959 |
| `quip-metal-msa` | 13,724,198 | 6,932,120 | 1,433,223 | 4,898 (dense) |

The degree-200 column passes `maxDegree`, so it applies to the library API
only. The Gibbs and MSA node limits extrapolate from the degree-4 points of
the sweep-time tables. No measurement exists at 5,684,919 or 13,724,198
nodes.

The sweep-time tables come from that one GPU
(`docs/perf/2026-11-metal-capacity.md`). On another chip, the limits scale
only with the GPU core count. The scaling is conservative for a GPU with
fewer cores, but not for a GPU whose cores are slower. The planner plans
each chunk against 500 ms (`TARGET_DISPATCH_MS`), and the one-sweep limit
keeps a measured chunk below 400 ms. The `TARGET_DISPATCH_MS` comment in
`src/sampler.rs` records the macOS GPU watchdog: MLX measured watchdog
kills at about 1.2 s per operation with a display attached.

## Engine selection

All binaries accept `enable_ane` and `enable_metal` in `backend_toml`. Both default to `true`.
Only MSA can run on the ANE. See [engine selection](docs/combined-engines.md) for modes, limits, and invalid settings.

## Probe cascade (`quip-metal-msa`)

Metal MSA streams always use the resident runner. Jobs stay in GPU slots
between probe checkpoints, and kept jobs continue to the full sweep budget.
Four workers prepare jobs before admission. The sampler saves controller
state between streams for the process lifetime.

No configuration turns the cascade off. A leftover `cascade` key logs the
unknown-field warning. Jobs that are not unit models (couplings and fields
other than −1, 0, and 1) run once at full budget through the batch path after
live slots drain. The batch path runs them on the weighted MSA kernel.
Other kernels and the Apple Neural Engine use separate paths.

| key | default | purpose |
|-----|---------|---------|
| `cascade_stages` | `[32, 256]` | sweep budget of each probe anneal |
| `cascade_keep` | `2000` | probe-to-full denominator |
| `cascade_keep_min` | `1000` | floor for the probe-to-full keep |
| `cascade_keep_max` | `30000` | ceiling for the probe-to-full keep |
| `cascade_audit` | `200` | audit lane denominator |
| `cascade_yield_per_million` | none | expected nonces per million below target |
| `cascade_screen` | `true` | screen chain-gated lease salts eight models to a word |
| `cascade_prefilter` | `true` | reject chain lease salts on the frustration score (stage F) before the screen |
| `cascade_open_gates` | `false` | keep every job at every checkpoint, for gate measurement only |

These keys enter through `backend_toml`.
The target energy comes from the session target, not from a key.
See [probe cascade](docs/cascade.md) for slots, schedules, controller lifetime,
key validation, and model checks.

## Multi-spin kernel (`quip-metal-msa`)

`kernels/msa.metal` is a Metal port of the multi-spin coded simulated
annealing in `quip-solver-cuda`'s `quip-cuda-msa` and `quip-solver-cpu`'s
`quip-cpu-msa` (Isakov, Zintchenko, Rønnow, Troyer, *Optimised simulated
annealing for Ising spin glasses*, Comput. Phys. Commun. 192, 2015). 32
replicas share one 32-bit word per spin. The Metropolis test is an integer
comparison against an inline geometric threshold draw. The kernel drives
spins one color class at a time, using the Advantage2 four-coloring by
default and falling back to the greedy coloring the Gibbs kernel uses when
the graph is not Advantage2. Set `QUIP_METAL_MSA_FOUR_COLOR` to `0`, `false`,
or `off` (case-insensitive) to force the greedy coloring. One threadgroup
anneals one 32-replica word of one problem in
threadgroup memory. A 128-read job dispatches four threadgroups per problem.

The CUDA kernel packs 64 replicas per `u64` in 99 KB of shared memory. Apple
GPUs cap threadgroup memory at 32 KB per threadgroup, which one 64-bit word
per spin exceeds for Advantage2's 4577 spins, so this port uses 32-bit
words. See `docs/metal-msa-design.md` for the
full comparison.

Unit models, with couplings and fields in `{-1, 0, 1}`, run on the unit MSA
kernel. It computes on-device energies, and the host rescores 1 job in 1,000
with `energy_milli` to check them. Other models whose coefficients are whole
milli values run on the weighted MSA kernel, and the host scores every read.
The weighted kernel holds a local field of at most 268,435,000 milli. The CSR
degree must be at most 20. The miner rejects any other model as over capacity
so the coordinator routes it elsewhere.

The 2026-09-15 runs used Apple M4 Max with 40 GPU cores and
`tests/fixtures/advantage2-system1.edges`, the Advantage2 System 1 working
graph. The fixture has 4577 nodes, 41515 edges, and eight greedy classes.
At one threadgroup per core, 40 jobs, 128 reads, and 16384 sweeps, the
multi-spin kernel reached 7.27 jobs/s. The `quip-metal-sa` reference
reached 1.02 jobs/s at 2048 sweeps and 256 reads. Those runs used
safety 0.2. The current 64-read envelope uses safety 0.4.
The multi-spin adapt envelope is 4096 to 14336 sweeps at 64
reads, set by the 2026-09-18 testnet studies in `docs/perf/`. On 60
regenerated qblocks, 64 reads at 14336 sweeps gives 1.67 times the valid
proofs per second of 128 reads at 16384 sweeps. The chance per job is flat
from 14336 to 16384 sweeps.
Three rounds at 7392 sweeps covered 1, 2, 5, 10, and 40 jobs.
Their largest `max_chunk_ms` was 261. The sweep-range runs peaked at 258 ms.
See `tests/msa_bench.rs` for the commands to repeat each run.

## Tests

```sh
cargo test --release
```

Conformance/golden and handshake tests drive the binary in isolation via
`quip-solver-conformance`'s scripted driver and check energies against
`conformance/golden_vectors.json`. The whole suite runs on a Mac with a Metal
device. CI runs it on a macOS runner, which is the only configuration this
crate builds in.

Two harnesses carry `#[ignore]` because they take minutes to hours of device
time. `tests/msa_bench.rs` times one problem through both kernels at a job
shape the environment sets. `tests/probe_screen.rs` draws fresh nonces on the
testnet's own instance distribution and runs each nonce through a list of
read and sweep shapes. That measures what a short probe says about the full
job. Each file's module comment carries its commands and its environment
knobs.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
