//! Launch the v0.2 Metal SA / Gibbs kernels and score with consensus energy.
//!
//! SA and Gibbs select int8 or integer-milli CSR by model representation.
//! Both use D-Wave incremental delta-energy SA or color-block Gibbs, with
//! bit-packed thread-local state on their sequential paths. The multi-spin kernel computes
//! energies on the device for whole-unit graphs, with one job in 1,000 audited
//! on the host. SA, Gibbs, and non-integer graphs are scored on the host with
//! [`quip_solver_core::quip_protocol::scoring::energy_milli`] (f64 consensus).
//!
//! # Batched dispatch (throughput)
//!
//! Int8 sequential SA and Gibbs use one threadgroup per problem and one thread
//! per read. Wide sequential SA and Gibbs pack reads into groups sized for the
//! pipeline, so one problem can occupy several groups. Chromatic Gibbs uses
//! one group per sample, while multi-spin SA uses one group per 32-read word.
//! The streaming loop batches problems to keep the GPU busy
//! ([`crate::streaming`]).
//!
//! [`encode_batch`] builds one batch's buffers and encodes the dispatch without
//! committing; [`harvest_batch`] reads the bit-packed samples per problem. The
//! synchronous [`sample_ising`] runs a single-problem batch and waits.

use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
use thiserror::Error;

/// Which GPU kernel a binary drives.
///
/// Distinct from [`quip_solver_core::Algorithm`], which does not have a
/// variant for every kernel. The crate keys pipelines, node caps, chunk
/// rates, and threadgroup budgets on `Kernel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Metropolis simulated annealing, one thread per read
    /// (`kernels/sa.metal`, `pure_simulated_annealing`).
    Sa,
    /// Multi-spin coded simulated annealing: 32 replicas per `uint` word, one
    /// threadgroup per (problem, word) (`kernels/msa.metal`, `msa_anneal`).
    Msa,
    /// Chromatic heat-bath Gibbs, one threadgroup per sample
    /// (`kernels/gibbs.metal`, `block_gibbs_parallel` / `block_gibbs_sampler`).
    Gibbs,
}

use crate::capacity::{host_limits, DeviceLimits, ModelShape, Storage};
use crate::representation::{batch_wide_i8_scale, classify, classify_msa, Representation};
use crate::topology::{fill_h_j, fill_h_j_float, fill_h_j_wide, SelfFeedingTopology};
use quip_solver_core::beta::default_ising_beta_range;
pub(crate) use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::BetaSchedule;

/// Failure from a Metal sample attempt: capacity refusal or driver fault.
#[derive(Debug, Error)]
pub enum SampleError {
    /// Device open, kernel compile, or pipeline construction failed.
    #[error(transparent)]
    Metal(#[from] crate::metal_device::MetalError),
    /// Command buffer or buffer readback failed after a dispatch.
    #[error("Metal driver: {0}")]
    Driver(String),
    /// The job exceeds a fixed capacity of this backend (kernel node cap,
    /// [`MAX_SWEEPS`]). Distinct from [`Self::Driver`] because it maps to a
    /// different harness condition: `Capacity` tells the coordinator to route
    /// the job elsewhere, while [`Self::Driver`] and [`Self::Metal`] describe
    /// the device itself, not this one job's size.
    #[error("job exceeds Metal backend capacity: {0}")]
    TooLarge(String),
}

impl SampleError {
    /// Map to the harness's device-condition report
    /// ([`quip_solver_core::SampleError`], "Gotchas" in `MIGRATING.md`): a
    /// capacity refusal maps to `Capacity`. Everything else here — a kernel
    /// compile failure, a device reset, a GPU watchdog timeout, or an
    /// internal invariant violation — is a state this backend will not
    /// recover from without a restart, so it maps to `DeviceFault` rather
    /// than `DeviceBusy`. Reporting these as transient load (`OVERLOADED`
    /// pre-migration) is exactly the "wedged GPU forever" failure mode
    /// `MIGRATING.md` warns against: `DeviceBusy` invites the coordinator to
    /// keep sending jobs a broken device can never serve, where `DeviceFault`
    /// ends the session for a supervisor restart. Arms are listed explicitly
    /// so a new variant forces a decision rather than silently inheriting one.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_solver_metal::sampler::SampleError;
    /// use quip_solver_core::SampleError as HarnessSampleError;
    ///
    /// let err = SampleError::TooLarge("nodes > SA cap".into());
    /// assert_eq!(err.to_sample_error(), HarnessSampleError::Capacity);
    ///
    /// let err = SampleError::Driver("command buffer failed".into());
    /// assert!(matches!(err.to_sample_error(), HarnessSampleError::DeviceFault(_)));
    /// ```
    pub fn to_sample_error(&self) -> quip_solver_core::SampleError {
        match self {
            Self::TooLarge(_) => quip_solver_core::SampleError::Capacity,
            Self::Driver(msg) => quip_solver_core::SampleError::DeviceFault(msg.clone()),
            Self::Metal(e) => quip_solver_core::SampleError::DeviceFault(e.to_string()),
        }
    }
}

/// Largest `num_reads` a dispatch allocates for (mirrors CUDA). Also the
/// per-threadgroup thread count, well under `maxTotalThreadsPerThreadgroup`.
pub(crate) const MAX_READS: usize = 256;

/// Apple GPU SIMD width (threads per simdgroup).
///
/// `num_reads` becomes the thread count of a threadgroup on the SA and
/// sequential-Gibbs kernels, so a count that is not a multiple of this leaves
/// lanes of the final simdgroup permanently idle — they are issued either way.
/// Measured at a constant total sample count, `reads = 16` (half a simdgroup)
/// costs ~20% against `reads = 64`.
pub(crate) const SIMD_WIDTH: usize = 32;

/// Round a job's `num_reads` up to a full simdgroup, capped at [`MAX_READS`].
///
/// The dispatch computes the rounded count; the extra samples are discarded
/// when the result is assembled — `streaming::finish_batch` on the batched path
/// and [`sample_ising`] on the synchronous one both truncate to the count the
/// job asked for — so this never changes what the coordinator sees.
/// It costs nothing: those lanes execute regardless of whether we use them.
/// `MAX_READS` is itself a multiple of `SIMD_WIDTH`, so the cap cannot round
/// back down below the request.
pub(crate) fn simd_rounded_reads(num_reads: usize) -> usize {
    num_reads
        .clamp(1, MAX_READS)
        .div_ceil(SIMD_WIDTH)
        .saturating_mul(SIMD_WIDTH)
        .min(MAX_READS)
}

/// SA local-build `N` cap: `thread int8_t delta_energy[4593]` in
/// `kernels/sa.metal`. A larger model runs the `QUIP_SA_GLOBAL` build on the
/// batch path, with each read's state in device memory.
///
/// `crate::METAL_SA_IDENTITY` advertises this same cap, so the identity const
/// and the kernel array have one source.
pub(crate) const SA_MAX_NODES: usize = 4593;
/// Gibbs threadgroup-build `N` cap: `threadgroup int8_t shared_state[4800]` of
/// the chromatic kernel and `thread int8_t packed_state[600]` (600*8) of the
/// sequential kernel in `kernels/gibbs.metal`. The batch path runs a larger
/// model on the chromatic kernel's spill build, which keeps the spins past
/// this prefix in device memory.
///
/// Single source of truth with the identity const, as for [`SA_MAX_NODES`].
pub(crate) const GIBBS_MAX_NODES: usize = 4800;

/// Replicas per 32-bit word in `kernels/msa.metal`.
///
/// Equal to [`SIMD_WIDTH`], so [`simd_rounded_reads`] always yields whole
/// words and `words = num_reads / MSA_LANES` is exact.
pub(crate) const MSA_LANES: usize = 32;
const _: () = assert!(
    MSA_LANES == SIMD_WIDTH,
    "simd_rounded_reads must produce whole 32-lane multi-spin words"
);
/// 1,024 threads split each colour class, bounded at dispatch by the pipeline's
/// `max_total_threads_per_threadgroup`. The persistent RNG buffer is sized by
/// that clamped width, not by this constant.
pub(crate) const MSA_THREADS: usize = 1024;
/// Fixed group width for wide node-parallel Gibbs RNG streams.
pub(crate) const WIDE_GIBBS_GROUP: usize = 128;

/// Wide SA work per spin update for chunk sizing, measured in
/// `docs/perf/2026-10-11-wide-path.md`.
const WIDE_SA_COST_FACTOR: f64 = 1.4;
/// Wide Gibbs work per spin update for chunk sizing, measured in
/// `docs/perf/2026-10-11-wide-path.md`.
const WIDE_GIBBS_COST_FACTOR: f64 = 2.0;
/// Weighted MSA work per word update for chunk sizing, measured in
/// `docs/perf/2026-11-09-weighted-msa.md`.
const WEIGHTED_MSA_COST_FACTOR: f64 = 3.5;
/// Extra SA work per spin update on the global build, which keeps each
/// read's state in device memory, for chunk sizing. It multiplies the
/// representation's factor. Measured in `docs/perf/2026-11-12-sa-global.md`.
const SA_GLOBAL_COST_FACTOR: f64 = 2.0;
/// Extra Gibbs work per spin update on the spill build, which reads and
/// writes the spins past its threadgroup prefix in device memory, for chunk
/// sizing. It multiplies the representation's factor. It covers the measured
/// ratio at 4,000 nodes (at most 2.53). The degree-20 ratios at 60,000 and
/// 200,000 nodes (3.50 and 4.61) exceed it, so those models rely on the
/// pessimistic Gibbs base rate; see `docs/perf/2026-11-12-gibbs-spill.md`.
const GIBBS_SPILL_COST_FACTOR: f64 = 3.0;
/// Float32 SA work per spin update for chunk sizing, measured in
/// `docs/perf/2026-11-11-float-kernels.md`.
const FLOAT_SA_COST_FACTOR: f64 = 1.3;
/// Float32 Gibbs work per spin update for chunk sizing, measured in
/// `docs/perf/2026-11-11-float-kernels.md`.
const FLOAT_GIBBS_COST_FACTOR: f64 = 1.3;
/// Float32 MSA work per word update for chunk sizing, measured in
/// `docs/perf/2026-11-11-float-kernels.md`: 1.5 times the weighted lane's
/// [`WEIGHTED_MSA_COST_FACTOR`], rounded up.
const FLOAT_MSA_COST_FACTOR: f64 = 5.3;
/// Weighted MSA work per word update and row chunk on the dense pipeline
/// (`int` storage, a row past one chunk), for chunk sizing. Measured against
/// the Float32 lane on degree-62 circulants: see
/// `weighted_msa_high_degree_chunks_stay_under_the_ceiling`.
const WEIGHTED_MSA_DENSE_COST_FACTOR: f64 = 6.8;

/// Matrix columns one threadgroup of the dense-matrix multi-spin kernel
/// walks per second: a word update of an `n`-node model walks `n` columns.
/// The slowest single chunk measured on a quiet GPU in
/// `docs/perf/2026-11-12-msa-dense.md` (M4 Max, 1 to 8 threadgroups) ran at
/// 68.2 million columns per second, one sweep of an Int32 model of 6,016
/// nodes. This value keeps a margin under it for contention from other GPU
/// work, which slowed a 5,099-node sweep to 408 and 467 ms. The planner and
/// the one-sweep admission limit both use it. Each threadgroup is one SIMD
/// group whose column walk is bound by latency, so the rate scales with the
/// threadgroups and not with the GPU cores, unlike
/// [`MSA_OCCUPANCY_CURVE`].
const MSA_MATRIX_COLUMNS_PER_SEC: f64 = 6.0e7;
/// Margin on [`MSA_MATRIX_COLUMNS_PER_SEC`], so a chunk planned at the
/// 500 ms ceiling runs about 200 ms on the slowest measured shape.
const MSA_MATRIX_THROUGHPUT_SAFETY: f64 = 0.4;
/// Threadgroups the dense-matrix rate was measured at, at most: one problem
/// of [`MAX_READS`] reads, the only batch the dense path runs. More
/// threadgroups at once can saturate the GPU, which the linear rate does not
/// model.
const MSA_MATRIX_MEASURED_GROUPS: usize = MAX_READS / MSA_LANES;
/// Longest single sweep a dense-matrix batch may plan, in milliseconds. A
/// chunk is at least one sweep, so the planner cannot split a longer sweep,
/// and a measured chunk must stay below 400 ms.
const MSA_MATRIX_MAX_SWEEP_MS: f64 = 400.0;

/// Planned column updates per second of `groups` dense-matrix threadgroups
/// that share the GPU with `in_flight` batches of the same shape.
fn msa_matrix_rate(groups: usize, in_flight: usize) -> f64 {
    debug_assert!(
        groups <= MSA_MATRIX_MEASURED_GROUPS,
        "{groups} dense-matrix threadgroups exceed the measured {MSA_MATRIX_MEASURED_GROUPS}"
    );
    MSA_MATRIX_COLUMNS_PER_SEC * MSA_MATRIX_THROUGHPUT_SAFETY * groups.max(1) as f64
        / in_flight.max(1) as f64
}

/// Matrix columns the reads of one dense-matrix SA dispatch walk per
/// second, at most [`MAX_READS`] reads of one problem: a flip walks the `n`
/// columns of its row, and the SIMD groups of a dispatch run side by side,
/// so one sweep of every read costs `n * n` columns of the one rate, at
/// every read count. The slowest single chunk measured on a quiet GPU in
/// `docs/perf/2026-11-12-sa-dense.md` (M4 Max, hot start of the schedule,
/// where every read flips most spins) ran at 5.76 million columns per
/// second, one sweep of an Int8 model of 4,898 nodes. This value keeps a
/// margin under it for contention from other GPU work. The planner and the
/// one-sweep admission limit both use it.
const SA_MATRIX_COLUMNS_PER_SEC: f64 = 5.0e6;

/// Planned column walks per second of one dense-matrix SA dispatch that
/// shares the GPU with `in_flight` batches of the same shape.
fn sa_matrix_rate(in_flight: usize) -> f64 {
    SA_MATRIX_COLUMNS_PER_SEC * MSA_MATRIX_THROUGHPUT_SAFETY / in_flight.max(1) as f64
}

/// The estimated GPU time of one sweep of an `n`-node dense-matrix batch, in
/// milliseconds: the planner's column count at the kernel's rate, without
/// the planning margin. Every threadgroup (multi-spin) or SIMD group (SA)
/// walks its `n * n` columns at once, so the estimate does not depend on the
/// read count.
fn matrix_sweep_ms(kernel: Kernel, n: usize) -> f64 {
    let rate = match kernel {
        Kernel::Sa => SA_MATRIX_COLUMNS_PER_SEC,
        Kernel::Msa | Kernel::Gibbs => MSA_MATRIX_COLUMNS_PER_SEC,
    };
    n as f64 * n as f64 / rate * 1000.0
}

/// Fill whole SIMD groups while keeping each wide sequential read's global ID.
fn wide_sa_group(num_reads: usize, pipeline_max: usize) -> usize {
    if pipeline_max < SIMD_WIDTH {
        return pipeline_max.max(1);
    }
    num_reads.min(pipeline_max / SIMD_WIDTH * SIMD_WIDTH)
}
/// Static threadgroup bytes `msa_anneal` declares: 32 atomic lane totals.
/// `msa_pipeline_compiles_and_admits_at_least_256_threads` checks the compiled figure
/// against a bound using the literal 144 bytes (16 bytes of alignment headroom),
/// rather than reading this constant.
pub(crate) const MSA_STATIC_TG_BYTES: usize = MSA_LANES * 4;
/// Threadgroup memory every Apple GPU family offers per threadgroup, bytes.
/// There is no opt-in above it (CUDA's `MAX_DYNAMIC_SHARED_SIZE_BYTES` has no
/// counterpart), which is why the kernel uses 32-bit words: one `u64` per
/// spin does not fit Advantage2's 4577 spins.
const APPLE_TG_MEMORY_BYTES: usize = 32 * 1024;
/// Multi-spin kernel `N` cap: `N * 4` bytes of spin words must fit beside the
/// lane totals under [`APPLE_TG_MEMORY_BYTES`] (6016 * 4 + 128 = 24,192).
///
/// `crate::METAL_MSA_IDENTITY` advertises this same cap, so the identity
/// const and the dispatch guard have one source.
pub(crate) const MSA_MAX_NODES: usize = 6016;
const _: () = assert!(
    MSA_MAX_NODES * 4 + MSA_STATIC_TG_BYTES <= APPLE_TG_MEMORY_BYTES,
    "MSA_MAX_NODES spin words must fit in Apple threadgroup memory"
);
/// Spin words the weighted multi-spin kernel keeps in threadgroup memory,
/// `K`: at most [`MSA_MAX_NODES`], at most what `tg_bytes` of threadgroup
/// memory holds beside the lane totals, and at most `cap`. A model of more
/// than `K` nodes keeps words `K..N` in device memory (`SplitSpins` in
/// `kernels/msa.metal`).
pub(crate) fn msa_shared_spins(tg_bytes: usize, cap: Option<usize>) -> usize {
    let fits = tg_bytes.saturating_sub(MSA_STATIC_TG_BYTES) / 16 * 4;
    MSA_MAX_NODES.min(fits).min(cap.unwrap_or(usize::MAX))
}

/// Largest CSR degree the unit kernel's unrolled 20-neighbour prefetch
/// admits, and the neighbours per row chunk of the weighted lanes' walk.
/// Zephyr (Advantage2) is degree 20. A nonzero field is a 21st input, which
/// the kernel's `popcount21` covers.
pub(crate) const MSA_UNIT_MAX_DEG: usize = 20;
/// Largest CSR degree the weighted multi-spin lanes (milli and Float32) walk,
/// in row chunks of [`MSA_UNIT_MAX_DEG`]: CUDA's sparse `MSA_MAX_DEG`. No
/// multi-spin kernel admits a denser model (`capacity::CapacityError::Degree`).
pub const MSA_SPARSE_MAX_DEG: usize = 62;

/// Nodes one colour class held per threadgroup step when
/// [`MSA_OCCUPANCY_CURVE`] was measured: 4577 Advantage2 nodes in eight
/// classes, each class one step of a 1,024-thread group.
const MSA_CURVE_NODES_PER_STEP: f64 = 4577.0 / 8.0;

/// How much a sweep of this colouring costs beyond `n` word updates, at least
/// one. Each colour class is `ceil(count / threads)` threadgroup steps closed
/// by a barrier, and a step takes about as long with one node as with the
/// calibration's 572 (`docs/perf/2026-11-11-float-kernels.md`), so a dense
/// model of few nodes in many classes is planned by its steps, not its nodes.
fn msa_colour_step_factor(counts: &[i32], threads_per_group: usize, n: usize) -> f64 {
    let steps: usize = counts
        .iter()
        .map(|&count| {
            usize::try_from(count)
                .unwrap_or(0)
                .div_ceil(threads_per_group.max(1))
                .max(1)
        })
        .sum();
    (steps as f64 * MSA_CURVE_NODES_PER_STEP / n.max(1) as f64).max(1.0)
}

/// Row chunks of [`MSA_UNIT_MAX_DEG`] neighbours the weighted lanes walk for
/// a node of `degree`; at least one.
fn msa_row_chunks(degree: usize) -> usize {
    degree.div_ceil(MSA_UNIT_MAX_DEG).max(1)
}

/// Largest CSR degree the multi-spin kernel admits for a model of `rep`.
/// The weighted lanes (milli and Float32) walk a row in chunks of
/// [`MSA_UNIT_MAX_DEG`] up to [`MSA_SPARSE_MAX_DEG`]; the unit kernel
/// unrolls one chunk.
pub(crate) const fn msa_max_degree(rep: Representation) -> usize {
    match rep {
        Representation::Wide | Representation::Float32 => MSA_SPARSE_MAX_DEG,
        Representation::Binary | Representation::Int8 => MSA_UNIT_MAX_DEG,
    }
}

/// The MSA lane a model of `rep` runs on when its CSR rows reach `degree`
/// over `nodes` nodes. A unit model denser than the unit kernel's
/// [`MSA_UNIT_MAX_DEG`], or with more nodes than threadgroup memory holds
/// ([`MSA_MAX_NODES`]), is a whole-milli model, so it runs on the weighted
/// lane, which walks rows up to [`MSA_SPARSE_MAX_DEG`] and spills spin words
/// past threadgroup memory.
const fn msa_lane(rep: Representation, degree: usize, nodes: usize) -> Representation {
    match rep {
        Representation::Binary if degree > MSA_UNIT_MAX_DEG || nodes > MSA_MAX_NODES => {
            Representation::Wide
        }
        other => other,
    }
}

/// Largest `num_sweeps` a dispatch accepts.
///
/// `num_sweeps` arrives from the coordinator and nothing upstream bounds it:
/// `quip_solver_core`'s `pick_param` returns the job's value verbatim when
/// non-zero, and unlike `num_reads` (rejected `TooLarge` against `max_reads`)
/// there is no identity-const gate for it. Unbounded, it sizes the beta
/// schedule — `geometric_beta_schedule` collects `num_sweeps / sweeps_per_beta`
/// `f64`s, so `u32::MAX` sweeps is a ~34 GB allocation — and then drives that
/// many kernel sweeps, i.e. both an OOM and a GPU-watchdog denial of service.
///
/// 1,048,576 admits the deep final stage of the chain-gated cascade, where the
/// catch rate on recent winners still rises past 131,072 sweeps. It bounds the
/// schedule to 1Mi `f64` + 1Mi `f32` (~12 MiB). It must stay at least
/// `2 * METAL_ADAPT.max_sweeps`, which a `const _: () = assert!(..)` in
/// `lib.rs` enforces at compile time.
pub(crate) const MAX_SWEEPS: usize = 1_048_576;

/// Target GPU time for one command buffer, in milliseconds.
///
/// macOS runs a non-configurable GPU watchdog that aborts a command buffer
/// running more than a few seconds, and a hang freezes the whole machine for
/// ~10 s while the GPU resets. Apple's guidance is to split long compute into
/// tiles well under ~100-250 ms; MLX measured watchdog kills at ~1.2 s per
/// operation with a display attached. Before chunking, this backend ran
/// 23-33 s command buffers at mining settings (82 s at max effort) and did
/// crash the host.
///
/// 500 ms is above Apple's conservative ~100-250 ms tiling guidance, trading
/// margin for efficiency: chunk overhead is ~2.5 ms of dispatch setup per
/// command buffer, so a longer chunk amortizes it to well under 1%. It still
/// keeps ~2x margin under the ~1.2 s at which watchdog kills were actually
/// observed, and is ~50-160x shorter than the 23-82 s this backend ran before
/// chunking — the regime that crashed the host.
///
/// This is a *ceiling*, not a goal: [`estimated_updates_per_sec`] is
/// deliberately pessimistic, so real chunks usually land well under it. Lower
/// this if a machine still stutters; the cost is throughput, not correctness.
const TARGET_DISPATCH_MS: f64 = 500.0;

/// Measured SA spin-update rate as a function of occupancy, in Gupd/s.
///
/// A single constant cannot size chunks correctly: throughput varies ~6x with
/// how full the GPU is, so an estimate tuned for a full batch produces
/// dangerously long chunks for a small one. (Measured: a 20-problem SA batch
/// runs at 0.65 Gupd/s where a 320-problem batch reaches 2.10.)
///
/// Points are `(threadgroups per core, Gupd/s)` from the occupancy sweep on an
/// M4 Max, full Advantage2 topology. SA only — chromatic Gibbs has a different
/// shape and gets its own estimate ([`GIBBS_UPDATES_PER_SEC`]).
const OCCUPANCY_CURVE: [(f64, f64); 8] = [
    (0.2, 0.32),
    (0.5, 0.49),
    (1.0, 0.83),
    (2.0, 1.40),
    (3.0, 1.54),
    (4.0, 1.83),
    (6.0, 2.03),
    (8.0, 2.10),
];

/// Margin applied to [`OCCUPANCY_CURVE`]. Overestimating throughput produces
/// chunks that run *longer* than the target — the failure mode that crashes the
/// host — so the estimate is deliberately pessimistic. Underestimating only
/// costs a few extra command buffers at ~2.5 ms each.
///
/// Measured at 0.7: SA adapt chunks peaked at 360 ms against a 500 ms ceiling.
const SA_THROUGHPUT_SAFETY: f64 = 0.7;

/// Chromatic Gibbs spin-update rate, flat rather than a curve.
///
/// Gibbs is saturated across the whole occupancy range we run it at (2.06-2.12
/// Gupd/s from 16 to 512 threadgroups/core), so occupancy is not the variable
/// that predicts it — the sampling shape is. The same kernel measured 2.21
/// Gupd/s at 64 reads / 256 sweeps but only 1.41 at 160 reads / 2308 sweeps.
/// This is the low end of that range, so chunk sizing is bounded by the
/// slowest configuration rather than the average.
///
/// Kept separate from SA's curve deliberately: sharing one estimate meant
/// tightening it for Gibbs also shrank SA's chunks, which were already well
/// inside the ceiling, and paid for it in extra command buffers.
const GIBBS_UPDATES_PER_SEC: f64 = 1.2e9;

/// See [`SA_THROUGHPUT_SAFETY`]. Gibbs carries its own margin because its
/// per-chunk cost includes work SA's does not — rebuilding the threadgroup spin
/// array from device memory on resume, and recomputing energies every chunk —
/// which the aggregate rate above does not separate out.
const GIBBS_THROUGHPUT_SAFETY: f64 = 0.7;

/// Multi-spin occupancy curve: `(threadgroups per core, word-updates/s)`.
/// One word update advances 32 replicas of one spin.
///
/// Measured 2026-09-15 on Apple M4 Max (40 GPU cores), using
/// `tests/fixtures/advantage2-system1.edges`: 4577 nodes, 41515 edges, eight
/// greedy colour classes. At T=1 and 7392 sweeps, calibration used 1, 2, 5,
/// 10, 40 jobs at 128 reads and 1, 2 jobs at 256 reads. Each point takes the
/// slowest batch at that occupancy, rounded to two significant figures.
const MSA_OCCUPANCY_CURVE: [(f64, f64); 5] = [
    (0.1, 2.6e8),
    (0.2, 5.2e8),
    (0.4, 1.0e9),
    (0.5, 1.2e9),
    (1.0, 1.2e9),
];
/// Margin for per-chunk timing variation after accounting for shared occupancy.
/// The streaming planner reserves two batches; synchronous sampling reserves one.
/// Measured 2026-09-16 on M4 Max: all 25 combinations of 1, 2, 5, 10, 40 jobs
/// and 2048, 4096, 7392, 8192, 16384 sweeps at 128 reads stayed below 400 ms.
/// The largest chunk was 374 ms. A 0.7 margin reached 627 ms with two batches.
const MSA_THROUGHPUT_SAFETY: f64 = 0.4;

/// Expected updates per second for a dispatch of `groups` threadgroups: spin
/// updates for SA and Gibbs, word updates (32 replicas each) for the
/// multi-spin kernel.
///
/// Below the first measured point the curve is extrapolated linearly toward the
/// origin (a nearly-empty GPU really is proportionally slow); above the last it
/// is held flat, since throughput has saturated by then.
fn estimated_updates_per_sec(kernel: Kernel, groups: usize) -> f64 {
    let (curve, safety, unit): (&[(f64, f64)], f64, f64) = match kernel {
        Kernel::Sa => (&OCCUPANCY_CURVE, SA_THROUGHPUT_SAFETY, 1e9),
        Kernel::Msa => (&MSA_OCCUPANCY_CURVE, MSA_THROUGHPUT_SAFETY, 1.0),
        Kernel::Gibbs => return GIBBS_UPDATES_PER_SEC * GIBBS_THROUGHPUT_SAFETY,
    };
    let cores = crate::iokit_gov::gpu_core_count().unwrap_or(10).max(1);
    let x = groups as f64 / cores as f64;
    let (first_x, first_y) = curve[0];
    let (last_x, last_y) = curve[curve.len() - 1];
    let rate = if x <= first_x {
        // Straight line through the origin, so a tiny batch is never credited
        // with more throughput than it can reach.
        first_y * x / first_x
    } else if x >= last_x {
        last_y
    } else {
        let hi = curve
            .iter()
            .position(|&(px, _)| px >= x)
            .unwrap_or(curve.len() - 1);
        let (x0, y0) = curve[hi - 1];
        let (x1, y1) = curve[hi];
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    };
    rate * unit * safety
}

/// One command buffer of an anneal: sweeps `sweep_start..sweep_start +
/// sweep_count` of each rung in `beta_start..beta_start + beta_count`. A
/// chunk is either whole rungs (`sweep_start == 0`, `sweep_count ==
/// sweeps_per_beta`) or a window inside one rung (`beta_count == 1`); the
/// SA and weighted multi-spin kernels take windows.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// First rung.
    pub beta_start: i32,
    /// Rungs.
    pub beta_count: i32,
    /// First sweep of each rung.
    pub sweep_start: i32,
    /// Sweeps of each rung.
    pub sweep_count: i32,
}

impl Chunk {
    /// Whole rungs `beta_start..beta_start + beta_count` of `sweeps_per`
    /// sweeps each.
    fn rungs(beta_start: i32, beta_count: i32, sweeps_per: usize) -> Self {
        Self {
            beta_start,
            beta_count,
            sweep_start: 0,
            sweep_count: i32::try_from(sweeps_per).unwrap_or(i32::MAX),
        }
    }
}

/// Test settings of the batch kernels. The default is production behaviour.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tuning {
    /// Caps `K`, the spin words the weighted multi-spin kernel keeps in
    /// threadgroup memory ([`msa_shared_spins`]).
    pub shared_spins_cap: Option<usize>,
    /// Plans chunks at this many updates per second instead of the measured
    /// rate.
    pub planning_rate: Option<f64>,
    /// Runs an SA model of at most [`SA_MAX_NODES`] nodes on the global
    /// build, which a larger model always takes.
    pub sa_global: bool,
    /// Refuses a model of more nodes, as a node cap below the kernel's own
    /// limit does.
    pub node_cap: Option<usize>,
    /// Caps `K`, the spins the chromatic Gibbs kernel keeps in threadgroup
    /// memory, at most [`GIBBS_MAX_NODES`]. A model of more nodes runs the
    /// spill build.
    pub gibbs_prefix_cap: Option<usize>,
    /// Runs a multi-spin model on dense-matrix storage
    /// ([`crate::capacity::Storage::Dense`]) whatever tier the capacity
    /// model routes it to. A model with a repeated coupler or a self-loop
    /// is refused.
    pub force_dense: bool,
    /// Keeps a batch on CSR storage whatever tier the capacity model routes
    /// it to, so tests and measurements can compare the two storages.
    pub force_sparse: bool,
}

/// Split a beta schedule into chunks whose dispatches each land near
/// [`TARGET_DISPATCH_MS`].
///
/// Returns chunks covering every sweep of `0..num_betas` in order. One sweep
/// touches every spin of every sample, so a rung's cost is
/// `threads * sweeps_per_beta * N` spin updates. When `windows` is set and
/// one rung costs more than a dispatch's budget, each rung splits into
/// windows of whole sweeps instead. `forced_rate` replaces the measured
/// update rate, for tests.
///
/// All kernels can resume mid-schedule: the first chunk initializes, any
/// other restores the carry-over state the previous chunk wrote.
fn chunk_plan(
    kernel: Kernel,
    dims: &BatchDims,
    groups: usize,
    in_flight: usize,
    cost: f64,
    windows: bool,
    forced_rate: Option<f64>,
) -> Vec<Chunk> {
    let num_betas = dims.num_betas.max(1);
    let sweeps_per = dims.sweeps_per.max(1);
    let per_sweep = (dims.num_threads as f64) * (dims.n.max(1) as f64) * cost;
    let per_beta = per_sweep * sweeps_per as f64;
    let budget = dispatch_budget(kernel, groups, in_flight, forced_rate);

    let mut plan = Vec::new();
    if windows && sweeps_per > 1 && budget < per_beta.max(1.0) {
        let sweeps = i32::try_from(sweeps_per).unwrap_or(i32::MAX);
        let window = units_per_dispatch(budget, per_sweep, sweeps);
        for beta_start in 0..num_betas {
            let mut start = 0;
            while start < sweeps {
                let count = window.min(sweeps - start);
                plan.push(Chunk {
                    beta_start,
                    beta_count: 1,
                    sweep_start: start,
                    sweep_count: count,
                });
                start += count;
            }
        }
        return plan;
    }
    let betas_per_chunk = units_per_dispatch(budget, per_beta, num_betas);
    let mut start = 0;
    while start < num_betas {
        let count = betas_per_chunk.min(num_betas - start);
        plan.push(Chunk::rungs(start, count, sweeps_per));
        start += count;
    }
    plan
}

/// Updates one dispatch may run: [`TARGET_DISPATCH_MS`] at the estimated
/// rate, or at `forced_rate` when set.
fn dispatch_budget(
    kernel: Kernel,
    groups: usize,
    in_flight: usize,
    forced_rate: Option<f64>,
) -> f64 {
    // Concurrent batches share the aggregate rate at their combined occupancy.
    // Reserve the full stream window even while priming or draining it.
    let in_flight = in_flight.max(1);
    let rate = forced_rate.unwrap_or_else(|| {
        estimated_updates_per_sec(kernel, groups.saturating_mul(in_flight)) / in_flight as f64
    });
    TARGET_DISPATCH_MS / 1000.0 * rate
}

/// Whole units of `per_unit` updates that fit `budget`, in `1..=max`.
fn units_per_dispatch(budget: f64, per_unit: f64, max: i32) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to 1..=max immediately below"
    )]
    let limit =
        ((budget / per_unit.max(1.0)).floor() as i64).clamp(1, i64::from(max.max(1))) as i32;
    limit
}

fn dispatch_beta_limit(
    kernel: Kernel,
    per_beta: f64,
    groups: usize,
    in_flight: usize,
    num_betas: i32,
) -> i32 {
    units_per_dispatch(
        dispatch_budget(kernel, groups, in_flight, None),
        per_beta,
        num_betas,
    )
}

/// Bound slot commands with the batch estimate, reserving both resident pools.
pub(crate) fn msa_step_limit(n: usize, groups: usize, slice: usize) -> usize {
    dispatch_beta_limit(
        Kernel::Msa,
        groups as f64 * n.max(1) as f64,
        groups,
        2,
        slice.clamp(1, MAX_SWEEPS) as i32,
    ) as usize
}

/// Whether Gibbs uses the chromatic (node-parallel) kernel — the default.
///
/// `block_gibbs_parallel` puts one threadgroup per *sample* and lets its
/// threads split each color's nodes over `threadgroup`-shared state, exploiting
/// the chromatic independence of a sweep. `block_gibbs_sampler` instead runs
/// one thread per read and walks all N nodes serially. Measured on an M4 Max
/// over the full Advantage2 topology (4577 nodes): 36.5 vs 13.2 jobs/s — 2.75x
/// — at equal solution quality (mean best energy within 0.01%, same diversity).
///
/// v0.2 shipped the sequential kernel as its default (`parallel=False`), so it
/// has more field exposure; `QUIP_METAL_GIBBS_SEQUENTIAL=1` forces it as an
/// escape hatch. SA is unaffected — its incremental delta-energy chain is
/// inherently serial, so it has no node-parallel variant.
pub(crate) fn gibbs_node_parallel() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("QUIP_METAL_GIBBS_SEQUENTIAL").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Parses `QUIP_METAL_MSA_FOUR_COLOR`: on unless `0`, `false`, or `off`.
fn four_color_from(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Whether the multi-spin MSA kernel uses the Advantage2 four-colouring — the
/// default. Set `QUIP_METAL_MSA_FOUR_COLOR` to `0`, `false`, or `off`
/// (case-insensitive) to disable it and fall back to the greedy colouring.
/// `build_with_advantage2_coloring` already falls back to the greedy colouring
/// when the graph is not Advantage2.
///
/// Read once per process so a streaming session keeps the same update order.
/// Other kernels always use their usual coloring.
fn msa_four_color() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| four_color_from(std::env::var("QUIP_METAL_MSA_FOUR_COLOR").ok().as_deref()))
}

/// Largest `N` the kernel's fixed-size arrays admit. For MSA this is the
/// threadgroup cap that the resident, streaming, and combined paths route
/// by; the batch path admits more nodes on the weighted lane, which spills.
pub(crate) fn kernel_max_nodes(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Sa => SA_MAX_NODES,
        Kernel::Msa => MSA_MAX_NODES,
        Kernel::Gibbs => GIBBS_MAX_NODES,
    }
}

fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Largest `sum |h| + sum |J|` in whole units for which the kernel's `int`
/// energy cannot overflow: every term is at most 1000 in magnitude per unit.
pub(crate) const DEVICE_ENERGY_MAX_UNITS: f64 = (i32::MAX / 1000) as f64;

/// Whether the multi-spin kernel's energy equals consensus `energy_milli` for
/// `graph`. The kernel truncates each coefficient to `i8` and sums in `int`,
/// so it matches only when every coefficient is a whole number in `i8` range
/// and the total magnitude cannot overflow. Consensus instances always pass.
pub(crate) fn device_energy_exact(graph: &IsingGraph) -> bool {
    let whole = |v: f64| v.is_finite() && v.fract() == 0.0 && (-128.0..=127.0).contains(&v);
    let mut units = 0.0;
    for &v in graph.h.iter().chain(graph.j.iter()) {
        if !whole(v) {
            return false;
        }
        units += v.abs();
    }
    units <= DEVICE_ENERGY_MAX_UNITS
}

/// Jobs harvested with device energies so far, for the 1-in-1,000 audit.
static ENERGY_AUDIT_JOBS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// One device-energy job in this many is rescored on the host.
const ENERGY_AUDIT_EVERY: u64 = 1000;

/// Beta schedule of the kind `schedule` names, cast to f32 for kernel upload,
/// plus sweeps-per-beta.
///
/// Uses the shared f64 schedule and casts each element to f32 — for a
/// geometric ladder, bit-identical to the prior in-crate f32 schedule.
pub(crate) fn build_beta_schedule(
    graph: &IsingGraph,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
    schedule: BetaSchedule,
) -> (Vec<f32>, usize) {
    let sweeps_per = sweeps_per_beta.max(1);
    let num_betas = (num_sweeps / sweeps_per).max(1);
    let (hot, cold) = beta_range.unwrap_or_else(|| default_ising_beta_range(graph));
    let sched: Vec<f32> = schedule
        .build(hot, cold, num_betas)
        .iter()
        .map(|&b| b as f32)
        .collect();
    (sched, sweeps_per)
}

/// The kernels' `uint` seed argument, mixed from all 64 bits of a host seed.
///
/// The kernel RNG state is 32-bit, so the seed must be narrowed. Keeping the
/// low half made seeds that differ only in their high 32 bits run identical
/// trajectories. The `SplitMix64` finalizer spreads every input bit over the
/// high word it returns, so any change to the seed changes the kernel seed,
/// short of a 2^-32 collision.
pub(crate) fn kernel_seed(seed: u64) -> u32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // The high word always fits, so the cast never truncates.
    (z >> 32) as u32
}

/// Unpack one read's bit-packed spins (LSB-first per byte, bit=1 -> -1,
/// bit=0 -> +1; matches the kernel's `set_spin_packed`). Identical to the
/// CUDA crate's unpacker — both kernels share the v0.2 packing convention.
///
/// `packed` is expected to hold `n.div_ceil(8)` bytes. A short slice reads as
/// zero past its end (i.e. `+1`, the initialized value) instead of panicking:
/// the caller sizes both the allocation and the harvest count from the same
/// `packed_size`, so a mismatch is a bug, not a reason to abort the miner.
pub(crate) fn unpack_spins(packed: &[i8], n: usize) -> Vec<i8> {
    let mut spins = vec![1i8; n];
    for (i, s) in spins.iter_mut().enumerate() {
        let byte = packed.get(i >> 3).copied().unwrap_or(0) as u8;
        let bit = (byte >> (i & 7)) & 1;
        *s = if bit == 1 { -1 } else { 1 };
    }
    spins
}

/// One built topology and the device buffers that depend only on its edges.
#[derive(Debug)]
pub(crate) struct CachedTopology {
    pub(crate) n: usize,
    pub(crate) edges: std::sync::Arc<[(usize, usize)]>,
    pub(crate) four_color: bool,
    /// What `topo` holds: CSR and colours, or less for dense-matrix storage.
    pub(crate) kind: TopologyKind,
    pub(crate) topo: SelfFeedingTopology,
    pub(crate) colors: Vec<metal::Buffer>,
    pub(crate) row: metal::Buffer,
    pub(crate) col: metal::Buffer,
}

/// Single cached topology. A miss replaces the entry.
#[derive(Debug, Default)]
pub(crate) struct TopologyCache {
    slot: std::sync::Mutex<Option<std::sync::Arc<CachedTopology>>>,
}

impl TopologyCache {
    pub(crate) fn get_or_build(
        &self,
        device: &crate::metal_device::MetalDevice,
        graph: &IsingGraph,
        four_color: bool,
    ) -> Result<std::sync::Arc<CachedTopology>, SampleError> {
        self.get_or_build_as(device, graph, four_color, TopologyKind::Csr)
    }

    /// [`Self::get_or_build`] for dense-matrix storage: the colouring alone,
    /// with no CSR on the host or the device.
    pub(crate) fn get_or_build_colors(
        &self,
        device: &crate::metal_device::MetalDevice,
        graph: &IsingGraph,
        four_color: bool,
    ) -> Result<std::sync::Arc<CachedTopology>, SampleError> {
        self.get_or_build_as(device, graph, four_color, TopologyKind::Colors)
    }

    /// [`Self::get_or_build`] for the dense-matrix SA kernel: no CSR and no
    /// colouring.
    pub(crate) fn get_or_build_bare(
        &self,
        device: &crate::metal_device::MetalDevice,
        graph: &IsingGraph,
    ) -> Result<std::sync::Arc<CachedTopology>, SampleError> {
        self.get_or_build_as(device, graph, false, TopologyKind::Bare)
    }

    fn get_or_build_as(
        &self,
        device: &crate::metal_device::MetalDevice,
        graph: &IsingGraph,
        four_color: bool,
        kind: TopologyKind,
    ) -> Result<std::sync::Arc<CachedTopology>, SampleError> {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(entry) = slot.as_ref() {
            if entry.n == graph.num_nodes()
                && entry.four_color == four_color
                && entry.kind == kind
                && entry.edges == graph.edges
            {
                return Ok(std::sync::Arc::clone(entry));
            }
        }
        let cached = std::sync::Arc::new(build_cached_topology(device, graph, four_color, kind)?);
        *slot = Some(std::sync::Arc::clone(&cached));
        Ok(cached)
    }
}

fn build_cached_topology(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    four_color: bool,
    kind: TopologyKind,
) -> Result<CachedTopology, SampleError> {
    let topo = match kind {
        TopologyKind::Bare => SelfFeedingTopology::bare(graph.num_nodes()),
        TopologyKind::Colors => SelfFeedingTopology::colors_only(graph, four_color),
        TopologyKind::Csr if four_color => {
            SelfFeedingTopology::build_with_advantage2_coloring(graph)
        }
        TopologyKind::Csr => SelfFeedingTopology::build(graph),
    };
    let colors = [&topo.colors.starts, &topo.colors.counts, &topo.colors.nodes]
        .into_iter()
        .map(|values| checked_buffer(device.new_buffer_from_slice(&pad_i32(values))))
        .collect::<Result<_, _>>()?;
    let zero = [0i32];
    let base_row: &[i32] = if topo.row_ptr.is_empty() {
        &zero
    } else {
        &topo.row_ptr
    };
    let base_col: &[i32] = if topo.nnz == 0 { &zero } else { &topo.col_ind };
    let row = checked_buffer(device.new_buffer_from_slice(base_row))?;
    let col = checked_buffer(device.new_buffer_from_slice(base_col))?;
    Ok(CachedTopology {
        n: graph.num_nodes(),
        edges: std::sync::Arc::clone(&graph.edges),
        four_color,
        kind,
        topo,
        colors,
        row,
        col,
    })
}

/// What a cached topology holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TopologyKind {
    /// The CSR arrays and the colouring.
    Csr,
    /// The colouring alone, for the dense-matrix multi-spin kernel: `topo`
    /// has no CSR arrays, and `row` and `col` are four-byte stubs.
    Colors,
    /// Neither, for the dense-matrix SA kernel: `topo` has no CSR arrays and
    /// no colours.
    Bare,
}

/// Free shared-storage buffers for later multi-spin and SA batches. At most 64.
#[derive(Debug, Default)]
pub(crate) struct BufferPool {
    free: std::sync::Mutex<Vec<metal::Buffer>>,
}

impl BufferPool {
    /// Smallest free buffer with `len <= length <= 2 * len`, else a new
    /// shared buffer, refusing as a capacity error when the device cannot
    /// allocate one.
    pub(crate) fn take(
        &self,
        device: &metal::DeviceRef,
        len: u64,
    ) -> Result<metal::Buffer, SampleError> {
        match self.take_free(len) {
            Some(buf) => Ok(buf),
            None => {
                checked_buffer(device.new_buffer(len, metal::MTLResourceOptions::StorageModeShared))
            }
        }
    }

    /// The smallest free buffer that fits `len`, if any.
    fn take_free(&self, len: u64) -> Option<metal::Buffer> {
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut best_index = None;
        let mut best_len = u64::MAX;
        for (index, buf) in free.iter().enumerate() {
            let length = buf.length();
            if length >= len && length <= len.saturating_mul(2) && length < best_len {
                best_index = Some(index);
                best_len = length;
            }
        }
        best_index.map(|index| free.swap_remove(index))
    }

    /// Keep `buf` if the pool holds fewer than 64 buffers.
    pub(crate) fn give(&self, buf: metal::Buffer) {
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if free.len() < 64 {
            free.push(buf);
        }
    }
}

/// Copy `data` into a pooled shared buffer.
///
/// An empty slice becomes a 4-byte zero stub, matching
/// [`crate::metal_device::MetalDevice::new_buffer_from_slice`].
fn pooled_from_slice<T: Copy>(
    pool: &BufferPool,
    device: &metal::DeviceRef,
    data: &[T],
) -> Result<metal::Buffer, SampleError> {
    let bytes = std::mem::size_of_val(data);
    if bytes == 0 {
        let buf = pool.take(device, 4)?;
        // SAFETY: `take(4)` returns StorageModeShared storage of at least 4
        // bytes, so `contents()` maps those bytes on the host. The write stays
        // inside that mapping.
        unsafe {
            std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, 4);
        }
        return Ok(buf);
    }
    let buf = pool.take(device, bytes as u64)?;
    // SAFETY: `take` returns StorageModeShared storage of at least `bytes`
    // bytes, so `contents()` is a non-null host mapping of that many bytes.
    // `data` is a separate allocation and cannot overlap the mapping.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            buf.contents().cast::<u8>(),
            bytes,
        );
    }
    Ok(buf)
}

/// The buffer slot of the `int8_t` storage scale: the next free slot after
/// each kernel's threshold rows (`kernels/sa.metal`, `gibbs.metal`, `msa.metal`).
fn coef_scale_slot(kernel: Kernel) -> u64 {
    match kernel {
        Kernel::Sa => 23,
        Kernel::Gibbs | Kernel::Msa => 26,
    }
}

/// A prepared batch retaining its inputs and carry-over state between chunks.
/// Only `commit_next` submits work, with at most one unfinished chunk per batch.
pub(crate) struct EncodedBatch {
    /// Energies come from the kernel's `final_energies` buffer instead of the host rescore.
    device_energy: bool,
    /// Command buffers in submission order. A long anneal is split across
    /// several so no single one trips the macOS GPU watchdog; the last carries
    /// the final state. See [`TARGET_DISPATCH_MS`].
    pub(crate) cmds: Vec<metal::CommandBuffer>,
    d_samples: metal::Buffer,
    n: usize,
    num_reads: usize,
    num_problems: usize,
    packed_size: usize,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    inputs: InputBuffers,
    out: DispatchBuffers,
    dims: BatchDims,
    kstate: KernelEncode,
    wide_rows: Option<(u64, metal::Buffer)>,
    /// Buffer slot and milli value of one stored unit, for `int8_t` storage.
    coef_scale: Option<(u64, i32)>,
    colors: Vec<metal::Buffer>,
    num_colors: i32,
    groups: usize,
    threads_per_group: usize,
    plan: Vec<Chunk>,
    /// Retains the topology buffers after a later miss replaces the device slot.
    #[expect(
        dead_code,
        reason = "keeps cached Metal buffers alive until the batch drops"
    )]
    cached: std::sync::Arc<CachedTopology>,
    pool: std::sync::Arc<BufferPool>,
    /// Multi-spin and SA carry-over buffers rented from `pool`.
    pooled: Vec<metal::Buffer>,
    /// The storage tier the batch runs on.
    pub(crate) storage: crate::capacity::Storage,
}

impl EncodedBatch {
    /// Submit one chunk after its predecessor retires, checking cancellation
    /// immediately before commit. Preparing all command buffers up front can
    /// deadlock at Metal's queue limit of 64 outstanding command buffers.
    pub(crate) fn commit_next(&mut self, mut cancelled: impl FnMut() -> bool) -> bool {
        if cancelled()
            || self
                .cmds
                .last()
                .is_some_and(|cmd| cmd.status() != metal::MTLCommandBufferStatus::Completed)
        {
            return false;
        }
        let Some(&chunk) = self.plan.get(self.cmds.len()) else {
            return false;
        };
        let cmd = self.queue.new_command_buffer().to_owned();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_chunk(encoder, chunk);
        encoder.end_encoding();
        if cancelled() {
            return false;
        }
        cmd.commit();
        self.cmds.push(cmd);
        true
    }

    fn encode_chunk(&self, encoder: &metal::ComputeCommandEncoderRef, chunk: Chunk) {
        let Chunk {
            beta_start,
            beta_count,
            sweep_start,
            sweep_count,
        } = chunk;
        encoder.set_compute_pipeline_state(&self.pipeline);
        bind_shared_args(encoder, &self.inputs, &self.out, &self.dims);
        match &self.kstate {
            KernelEncode::Sa { persist } => {
                bind_sa_chunk(encoder, persist, chunk);
            }
            KernelEncode::Gibbs {
                persist,
                shared_spins,
            } => {
                self.bind_colors(encoder, 0);
                bind_color_kernel_chunk(encoder, persist, beta_start, beta_count);
                // Only the chromatic kernel declares these; the sequential
                // kernel ignores them and its plans never window a rung.
                set_bytes_i32(encoder, 27, sweep_start);
                set_bytes_i32(encoder, 28, sweep_count);
                if let Some(shared) = shared_spins {
                    set_bytes_i32(encoder, 29, *shared);
                }
            }
            KernelEncode::Msa {
                persist,
                words,
                state_bytes,
                shared_spins,
                virtual_threads,
            } => {
                self.bind_colors(encoder, *words);
                bind_color_kernel_chunk(encoder, persist, beta_start, beta_count);
                encoder.set_threadgroup_memory_length(0, *state_bytes);
                if let Some(shared) = shared_spins {
                    set_bytes_i32(encoder, 27, *shared);
                    set_bytes_i32(encoder, 28, sweep_start);
                    set_bytes_i32(encoder, 29, sweep_count);
                }
                if let Some(width) = virtual_threads {
                    set_bytes_i32(encoder, 30, *width);
                }
            }
        }
        if let Some((index, rows)) = &self.wide_rows {
            encoder.set_buffer(*index, Some(rows), 0);
        }
        if let Some((index, scale)) = self.coef_scale {
            set_bytes_i32(encoder, index, scale);
        }
        encoder.dispatch_thread_groups(
            mtl_size_1d(self.groups),
            mtl_size_1d(self.threads_per_group),
        );
    }

    fn bind_colors(&self, encoder: &metal::ComputeCommandEncoderRef, slot19: i32) {
        for (index, buffer) in self.colors.iter().enumerate() {
            encoder.set_buffer((16 + index) as u64, Some(buffer), 0);
        }
        set_bytes_i32(encoder, 19, slot19);
        set_bytes_i32(encoder, 20, self.num_colors);
    }

    /// Block until the last chunk retires. Earlier chunks are ordered ahead of
    /// it on the same queue, so waiting on the tail waits for all of them.
    pub(crate) fn wait_until_completed(&self) {
        if let Some(last) = self.cmds.last() {
            last.wait_until_completed();
        }
    }

    /// The first chunk that did not complete, if any. A mid-sequence failure
    /// matters as much as the last one: a watchdog abort or fault partway
    /// through leaves the persistent state garbage, so the whole batch's
    /// samples are suspect.
    pub(crate) fn failed_status(&self) -> Option<metal::MTLCommandBufferStatus> {
        self.cmds
            .iter()
            .map(|c| c.status())
            .find(|s| *s != metal::MTLCommandBufferStatus::Completed)
    }

    /// Total GPU execution time across chunks, microseconds.
    pub(crate) fn gpu_time_us(&self) -> u64 {
        self.cmds.iter().map(|c| gpu_time_us(c)).sum()
    }

    /// Longest single chunk, microseconds — the quantity the macOS GPU watchdog
    /// actually judges. The sum above says how much work ran; this says whether
    /// any one command buffer got close to the abort threshold.
    pub(crate) fn max_chunk_us(&self) -> u64 {
        self.cmds.iter().map(|c| gpu_time_us(c)).max().unwrap_or(0)
    }

    /// Number of command buffers this batch was split into.
    pub(crate) fn chunk_count(&self) -> usize {
        self.plan.len()
    }
}

impl Drop for EncodedBatch {
    fn drop(&mut self) {
        // Give every pooled buffer back when `cmds` is empty or every command
        // buffer has status `Completed` or `Error`. Otherwise drop them: the
        // GPU may still be using them.
        let retired = self.cmds.iter().all(|cmd| {
            matches!(
                cmd.status(),
                metal::MTLCommandBufferStatus::Completed | metal::MTLCommandBufferStatus::Error
            )
        });
        if retired {
            for buf in self.pooled.drain(..) {
                self.pool.give(buf);
            }
        }
    }
}

/// True GPU execution time of a completed command buffer, in microseconds,
/// from `GPUEndTime - GPUStartTime` (`CFTimeInterval` seconds). Returns 0 if
/// the timestamps are unavailable / non-positive.
pub(crate) fn gpu_time_us(cmd: &metal::CommandBufferRef) -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "guarded finite and positive; GPU spans are far below u64::MAX microseconds"
    )]
    gpu_span(cmd).map_or(0, |(start, end)| ((end - start) * 1_000_000.0) as u64)
}

/// Host time in seconds, on the clock `MTLCommandBuffer.GPUStartTime` and
/// `GPUEndTime` report.
pub(crate) fn host_seconds() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let scale = *SCALE.get_or_init(|| {
        let mut info = Timebase { numer: 0, denom: 0 };
        // SAFETY: `info` is a valid, writable `mach_timebase_info_data_t`.
        let status = unsafe { mach_timebase_info(&mut info) };
        if status == 0 && info.denom != 0 {
            f64::from(info.numer) / f64::from(info.denom) * 1e-9
        } else {
            1e-9
        }
    });
    // SAFETY: `mach_absolute_time` takes no arguments and cannot fail.
    unsafe { mach_absolute_time() as f64 * scale }
}

/// `(GPUStartTime, GPUEndTime)` of a completed command buffer, in seconds on
/// the host's absolute clock, or `None` if unavailable or non-positive.
/// metal-rs 0.33 exposes no accessor, so read the properties via `objc`.
#[expect(
    unexpected_cfgs,
    reason = "objc 0.2 msg_send! expands to cfg(cargo-clippy) the compiler no longer recognizes"
)]
pub(crate) fn gpu_span(cmd: &metal::CommandBufferRef) -> Option<(f64, f64)> {
    use objc::{msg_send, sel, sel_impl};
    // SAFETY: `GPUStartTime`/`GPUEndTime` are `CFTimeInterval` (f64) properties
    // on a completed `MTLCommandBuffer`; `cmd` implements `objc::Message`.
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
    let dur = end - start;
    (dur.is_finite() && dur > 0.0).then_some((start, end))
}

fn set_bytes_i32(enc: &metal::ComputeCommandEncoderRef, index: u64, val: i32) {
    enc.set_bytes(
        index as metal::NSUInteger,
        4,
        &val as *const i32 as *const std::ffi::c_void,
    );
}

fn set_bytes_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, val: u32) {
    enc.set_bytes(
        index as metal::NSUInteger,
        4,
        &val as *const u32 as *const std::ffi::c_void,
    );
}

/// Tile a slice `times` times into one contiguous `Vec`.
fn tile_i32(src: &[i32], times: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(src.len() * times);
    for _ in 0..times {
        out.extend_from_slice(src);
    }
    out
}

/// A 1-D `MTLSize` — every dispatch here is one-dimensional.
fn mtl_size_1d(width: usize) -> metal::MTLSize {
    metal::MTLSize {
        width: width as metal::NSUInteger,
        height: 1,
        depth: 1,
    }
}

/// Scalar kernel arguments plus the sizes the algorithm-specific setup needs.
struct BatchDims {
    n: usize,
    num_betas: i32,
    sweeps_per: usize,
    base_seed: u32,
    num_threads: usize,
    num_problems: usize,
    num_reads: usize,
    packed_size: usize,
}

/// Per-batch input buffers: shared CSR structure plus per-problem `J` / `h`.
struct InputBuffers {
    row: metal::Buffer,
    col: metal::Buffer,
    j: metal::Buffer,
    h: metal::Buffer,
    row_off: metal::Buffer,
    col_off: metal::Buffer,
}

/// Beta ladder plus the kernel's two output buffers.
struct DispatchBuffers {
    beta: metal::Buffer,
    samples: metal::Buffer,
    energies: metal::Buffer,
}

/// Largest CSR row length `SelfFeedingTopology::build` would produce for
/// `graph`: an out-of-range endpoint skips the edge and a self-loop counts
/// once, the same rules the builder applies.
fn max_csr_degree(graph: &IsingGraph) -> usize {
    let n = graph.num_nodes();
    let mut degree = vec![0usize; n];
    for &(u, v) in graph.edges.iter() {
        if u >= n || v >= n {
            continue;
        }
        degree[u] += 1;
        if u != v {
            degree[v] += 1;
        }
    }
    degree.into_iter().max().unwrap_or(0)
}

/// The int8 SA and Gibbs build runs `Binary` and `Int8` models alike, so the
/// two tiers are one batch group there. MSA keeps `Binary` for its unit kernel.
pub(crate) fn sa_tier(rep: Representation, kernel: Kernel) -> Representation {
    if kernel != Kernel::Msa && rep == Representation::Binary {
        Representation::Int8
    } else {
        rep
    }
}

/// Whether `kernel` has a `QUIP_FLOAT` build. A kernel gains Float32 models
/// by its arm here, its pipeline in `MetalDevice`, its cost factor in
/// `encode_batch_inner`, and its coefficient upload in `upload_inputs`.
pub(crate) fn has_float32_kernel(kernel: Kernel) -> bool {
    match kernel {
        Kernel::Sa | Kernel::Gibbs | Kernel::Msa => true,
    }
}

/// The coefficient tier `kernel` runs `graph` on, or the capacity reason it
/// refuses the model: a classifier refusal, or a Float32 model on a kernel
/// with no Float32 build.
pub(crate) fn kernel_representation(
    graph: &IsingGraph,
    kernel: Kernel,
) -> Result<Representation, String> {
    let rep = if kernel == Kernel::Msa {
        classify_msa(graph)
    } else {
        classify(graph)
    }?;
    if rep == Representation::Float32 && !has_float32_kernel(kernel) {
        return Err(format!(
            "the model needs Float32 coefficients, and the {kernel:?} kernel has no Float32 build"
        ));
    }
    Ok(sa_tier(rep, kernel))
}

/// Refuse a Float32 model that the `float` kernel arithmetic cannot hold:
/// a beta past the finite `f32` range, or a rounded local field whose delta
/// `2 * field` overflows.
///
/// The field rule is CUDA's `check_real_fields` (commit `4bc82bb`) in its
/// order-independent form: the `f64` sum of the `f32` magnitudes of `h` and
/// the row's couplings, widened by the rounding bound of `degree + 1` `f32`
/// additions. The beta rule is CUDA's `build_planned_beta_schedule` (commit
/// `7d97e41`).
fn check_float_schedule(graph: &IsingGraph, beta: &[f32]) -> Result<(), SampleError> {
    if let Some(index) = beta.iter().position(|b| !b.is_finite()) {
        return Err(SampleError::TooLarge(format!(
            "beta schedule value at index {index} is outside the finite f32 range"
        )));
    }
    // Metal's fast math may compute `2 * beta` before it multiplies the field,
    // and a zero field then gives `inf * 0`. Bounding beta by half of
    // `f32::MAX` keeps every order of the product free of NaN.
    if let Some(index) = beta.iter().position(|b| b.abs() > f32::MAX / 2.0) {
        return Err(SampleError::TooLarge(format!(
            "beta schedule value at index {index} exceeds half of f32::MAX, so its double overflows f32"
        )));
    }
    let n = graph.num_nodes();
    let mut field: Vec<f64> = graph
        .h
        .iter()
        .map(|&v| f64::from((v as f32).abs()))
        .collect();
    let mut degree = vec![0_u32; n];
    for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
        let magnitude = f64::from((j as f32).abs());
        for node in [u, v] {
            field[node] += magnitude;
            degree[node] += 1;
        }
    }
    let max_delta = field
        .iter()
        .zip(&degree)
        .map(|(&f, &d)| {
            let error = (f64::from(d) + 1.0) * f64::from(f32::EPSILON);
            if error >= 1.0 {
                f64::INFINITY
            } else {
                2.0 * f / (1.0 - error)
            }
        })
        .fold(0.0, f64::max);
    if max_delta > f64::from(f32::MAX) {
        return Err(SampleError::TooLarge(
            "a local field exceeds the finite f32 delta range after rounding".into(),
        ));
    }
    Ok(())
}

/// The one representation every graph in a batch shares, or a capacity
/// refusal. `BatchKey` keeps representations apart, so a mix is a driver bug.
fn batch_representation(
    graphs: &[&IsingGraph],
    kernel: Kernel,
) -> Result<Representation, SampleError> {
    let mut shared = None;
    for graph in graphs {
        let rep = kernel_representation(graph, kernel).map_err(SampleError::TooLarge)?;
        if shared.is_some_and(|prev| prev != rep) {
            return Err(SampleError::Driver(
                "a batch mixes int8 and wide models".into(),
            ));
        }
        shared = Some(rep);
    }
    shared.ok_or_else(|| SampleError::Driver("encode_batch needs at least one graph".into()))
}

/// Validate a batch's shared inputs, returning the establishing graph, `N`,
/// and coefficient representation.
///
/// Split out of [`encode_batch`] so both rejections are reachable without a
/// Metal device.
pub(crate) fn validate_batch<'a>(
    graphs: &[&'a IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
) -> Result<(&'a IsingGraph, usize, Representation), SampleError> {
    validate_batch_within(
        graphs,
        params,
        kernel,
        host_device_limits(),
        false,
        None,
        DenseMode::Off,
    )
    .map(|batch| (batch.first, batch.n, batch.rep))
}

/// A batch [`validate_batch_within`] admits.
#[derive(Debug)]
struct ValidBatch<'a> {
    /// The establishing graph; every graph shares its topology.
    first: &'a IsingGraph,
    n: usize,
    rep: Representation,
    /// The `int8_t` storage scale, when the batch takes that storage.
    i8_scale: Option<i32>,
    /// An MSA batch with a CSR row past one chunk of [`MSA_UNIT_MAX_DEG`]:
    /// on CSR storage it runs on the dense-rows weighted pipeline.
    dense_rows: bool,
    /// The storage tier the admission priced the batch on.
    storage: Storage,
    /// Every coupling and field is in {-1, 0, 1}, so a dense-matrix batch
    /// holds a bit matrix.
    unit: bool,
}

/// Whether a batch may take dense-matrix storage ([`Storage::Dense`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseMode {
    /// CSR storage only: the resident, streaming, and slot gates, whose
    /// paths run no dense kernel.
    Off,
    /// The tier the capacity model routes the batch to (the batch path).
    Auto,
    /// Dense storage, refusing a model that is not a dense candidate
    /// ([`Tuning::force_dense`]).
    Forced,
}

/// Memory limits of this host, computed once.
pub(crate) fn host_device_limits() -> &'static DeviceLimits {
    static LIMITS: std::sync::OnceLock<DeviceLimits> = std::sync::OnceLock::new();
    LIMITS.get_or_init(host_limits)
}

/// [`validate_batch`] against explicit memory `limits`. `allow_i8` sizes a wide
/// batch that [`batch_i8_scale`] accepts at one byte per coefficient, for the
/// batch path that uploads it so. A dense MSA batch always takes `int`
/// storage, the faster one on the dense pipeline.
///
/// The footprint check comes last and runs before any buffer exists, so a
/// model that does not fit is refused without allocating. `dense` says
/// whether the batch may take dense-matrix storage.
fn validate_batch_within<'a>(
    graphs: &[&'a IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
    limits: &DeviceLimits,
    allow_i8: bool,
    node_cap: Option<usize>,
    dense: DenseMode,
) -> Result<ValidBatch<'a>, SampleError> {
    let Some(&first) = graphs.first() else {
        // Was a `debug_assert!`, which is compiled out in release — an empty
        // batch would have indexed `graphs[0]` and panicked the miner.
        return Err(SampleError::Driver(
            "encode_batch needs at least one graph".into(),
        ));
    };
    let n = first.num_nodes();
    // The weighted multi-spin lane spills spin words past threadgroup
    // memory, the SA global build keeps each read's state in device memory
    // past its thread-private frame, and the chromatic Gibbs kernel spills
    // spins past its threadgroup array, so the capacity footprint below
    // bounds every kernel's model instead. An explicit node cap is the limit
    // itself, even below a kernel's frame.
    let cap = crate::capacity::MAX_NODES.min(node_cap.unwrap_or(usize::MAX));
    if n > cap {
        // Defense in depth: the harness rejects N > max_nodes (identity const)
        // before the sampler. This applies the memory-scale bound or an
        // explicit node cap.
        return Err(SampleError::TooLarge(format!(
            "graph N={n} exceeds {kernel:?} kernel limit {cap}"
        )));
    }
    if params.num_sweeps > MAX_SWEEPS {
        // No defense in depth here: nothing upstream bounds `num_sweeps` at
        // all (see `MAX_SWEEPS`), so this is the only gate between a hostile
        // coordinator and an unbounded beta-schedule allocation.
        return Err(SampleError::TooLarge(format!(
            "num_sweeps={} exceeds Metal sampler limit {MAX_SWEEPS}",
            params.num_sweeps
        )));
    }
    let mut rep = batch_representation(graphs, kernel)?;
    let unit = rep == Representation::Binary;
    let mut dense_rows = false;
    let mut over_budget = None;
    if kernel == Kernel::Msa {
        // The unit kernel unrolls a fixed 20-neighbour prefetch, so a denser
        // unit model moves to the weighted lane, and the weighted lanes walk
        // at most `msa_max_degree` neighbours. A denser node is capacity,
        // not a driver fault: the batch path runs it on dense-matrix storage
        // when the capacity model routes it there, and otherwise the
        // coordinator routes the job to a backend that walks CSR rows of any
        // length.
        let max_deg = max_csr_degree(first);
        rep = msa_lane(rep, max_deg, n);
        let budget = msa_max_degree(rep);
        if max_deg > budget {
            over_budget = Some(format!(
                "graph max degree {max_deg} exceeds the multi-spin kernel's \
                 {budget}-neighbour budget"
            ));
        }
        dense_rows = max_deg > MSA_UNIT_MAX_DEG;
    }
    if dense == DenseMode::Off {
        if let Some(message) = over_budget.take() {
            return Err(SampleError::TooLarge(message));
        }
    }
    let i8_scale = batch_i8_scale(graphs, rep, allow_i8 && !dense_rows);
    let shape = model_shape(first, params, rep, i8_scale.is_some());
    let too_large = |e: crate::capacity::CapacityError| {
        SampleError::TooLarge(format!("{kernel:?} model of {n} nodes: {e}"))
    };
    let footprint = match dense {
        DenseMode::Off => crate::capacity::footprint_in(
            kernel,
            &shape,
            limits,
            &[Storage::Threadgroup, Storage::Spill],
        ),
        DenseMode::Auto => {
            crate::capacity::routed_runnable_footprint(kernel, first, &shape, limits)
        }
        DenseMode::Forced => {
            if !crate::capacity::runnable_storages(kernel).contains(&Storage::Dense) {
                return Err(SampleError::TooLarge(format!(
                    "the {kernel:?} kernel has no dense-matrix build"
                )));
            }
            if !crate::topology::dense_candidate(first) {
                return Err(SampleError::TooLarge(
                    "dense-matrix storage needs a model without a repeated coupler or a self-loop"
                        .into(),
                ));
            }
            crate::capacity::footprint_in(kernel, &shape, limits, &[Storage::Dense])
        }
    }
    .map_err(too_large)?;
    if footprint.storage == Storage::Dense {
        // A chunk is at least one sweep, so a dense batch whose one sweep
        // would pass the limit is refused before any buffer exists, under
        // Auto and Forced alike.
        let sweep_ms = matrix_sweep_ms(kernel, n);
        if sweep_ms > MSA_MATRIX_MAX_SWEEP_MS {
            return Err(SampleError::TooLarge(format!(
                "{kernel:?} model of {n} nodes: one dense-matrix sweep is estimated at \
                 {sweep_ms:.3} ms, past the {MSA_MATRIX_MAX_SWEEP_MS:.0} ms limit of one chunk"
            )));
        }
    }
    if footprint.storage != Storage::Dense {
        if let Some(message) = over_budget {
            return Err(SampleError::TooLarge(message));
        }
    }
    Ok(ValidBatch {
        first,
        n,
        rep,
        i8_scale,
        dense_rows,
        storage: footprint.storage,
        unit,
    })
}

/// The scale of a wide batch stored as `int8_t` coefficients, or `None` when
/// the batch runs on plain `int` storage: `allow` is false, the batch is not
/// wide, or its coefficients do not all fit `i8` after division by their gcd.
fn batch_i8_scale(graphs: &[&IsingGraph], rep: Representation, allow: bool) -> Option<i32> {
    if allow && rep == Representation::Wide {
        batch_wide_i8_scale(graphs)
    } else {
        None
    }
}

/// The size of one model of a batch for the capacity model. `i8_storage` is
/// a wide model held as `int8_t` coefficients with `int` deltas and fields.
///
/// A unit model costs what Int8 does on every tier. On CSR storage that is
/// its exact width. A dense-matrix unit model holds two bit planes, a
/// quarter of the Int8 bytes, so the dense tier overstates it.
fn model_shape(
    graph: &IsingGraph,
    params: &SampleParams,
    rep: Representation,
    i8_storage: bool,
) -> ModelShape {
    ModelShape {
        nodes: graph.num_nodes(),
        edges: graph.edges.len(),
        max_degree: max_csr_degree(graph),
        precision: if rep == Representation::Binary {
            Representation::Int8.precision()
        } else if i8_storage {
            crate::capacity::Precision::Int8Scaled
        } else {
            rep.precision()
        },
        // `num_sweeps` bounds the beta schedule length from above.
        betas: params.num_sweeps,
        reads: params.num_reads,
    }
}

/// The footprint `encode_batch` admits `graph` at, for tests of the storage
/// choice: a wide model whose coefficients fit `i8` after division by their
/// gcd counts one byte per coefficient.
///
/// # Errors
///
/// A capacity refusal, as [`sample_ising`] reports it.
#[doc(hidden)]
pub fn model_footprint(
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
) -> Result<crate::capacity::Footprint, SampleError> {
    let mut rep = kernel_representation(graph, kernel).map_err(SampleError::TooLarge)?;
    let mut dense_rows = false;
    if kernel == Kernel::Msa {
        let degree = max_csr_degree(graph);
        rep = msa_lane(rep, degree, graph.num_nodes());
        dense_rows = degree > MSA_UNIT_MAX_DEG;
    }
    // A model past the default `K` runs the spill build on `int` storage.
    let past_k = kernel == Kernel::Msa && graph.num_nodes() > MSA_MAX_NODES;
    let i8_scale = batch_i8_scale(&[graph], rep, !dense_rows && !past_k);
    // The resident gate runs CSR storage only.
    crate::capacity::footprint_in(
        kernel,
        &model_shape(graph, params, rep, i8_scale.is_some()),
        host_device_limits(),
        &[Storage::Threadgroup, Storage::Spill],
    )
    .map_err(|e| SampleError::TooLarge(format!("{kernel:?} model: {e}")))
}

/// A buffer, or a capacity refusal when the allocation returned a null
/// handle. Metal reports a failed allocation as nil, and metal-rs hands that
/// back as a `Buffer` rather than an `Option`.
pub(crate) fn checked_buffer(buffer: metal::Buffer) -> Result<metal::Buffer, SampleError> {
    use metal::foreign_types::ForeignType;
    require_buffer((!buffer.as_ptr().is_null()).then_some(buffer))
}

/// `buffer`, or a capacity refusal when it is `None`.
fn require_buffer(buffer: Option<metal::Buffer>) -> Result<metal::Buffer, SampleError> {
    buffer.ok_or_else(|| {
        SampleError::TooLarge("the device could not allocate a buffer for this model".into())
    })
}

/// Build and upload one batch's CSR structure and per-problem `h` / `J`.
///
/// The CSR structure is shared by every problem in the batch (same topology),
/// so the multi-spin kernel reads the one untiled copy in `cached`. The SA and
/// Gibbs kernels read a tiled copy through the offset arrays.
fn upload_inputs(
    device: &crate::metal_device::MetalDevice,
    cached: &CachedTopology,
    graphs: &[&IsingGraph],
    kernel: Kernel,
    rep: Representation,
    i8_scale: Option<i32>,
) -> Result<InputBuffers, SampleError> {
    let topo = &cached.topo;
    let num_problems = graphs.len();
    let n = topo.n;
    let nnz_alloc = topo.nnz.max(1);
    let rp_len = topo.row_ptr.len().max(1);

    // Shared CSR structure for MSA, tiled for SA / Gibbs; per-problem J / h values.
    let (row, col) = if kernel == Kernel::Msa {
        (cached.row.clone(), cached.col.clone())
    } else {
        let zero = [0i32];
        let base_row: &[i32] = if topo.row_ptr.is_empty() {
            &zero
        } else {
            &topo.row_ptr
        };
        let base_col: &[i32] = if topo.nnz == 0 { &zero } else { &topo.col_ind };
        (
            checked_buffer(device.new_buffer_from_slice(&tile_i32(base_row, num_problems)))?,
            checked_buffer(device.new_buffer_from_slice(&tile_i32(base_col, num_problems)))?,
        )
    };
    let row_ptr_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * rp_len) as i32).collect();
    let col_ind_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * nnz_alloc) as i32).collect();
    let (j, h) = match rep {
        Representation::Wide => {
            let mut all_j = vec![0_i32; num_problems * nnz_alloc];
            let mut all_h = vec![0_i32; num_problems * n];
            for (p, graph) in graphs.iter().enumerate() {
                let (j_csr, h_milli) = fill_h_j_wide(topo, graph);
                all_j[p * nnz_alloc..p * nnz_alloc + j_csr.len()].copy_from_slice(&j_csr);
                all_h[p * n..p * n + h_milli.len()].copy_from_slice(&h_milli);
            }
            if let Some(scale) = i8_scale {
                // `batch_wide_i8_scale` checked every quotient against `i8`.
                let to_i8 = |milli: &[i32]| -> Vec<i8> {
                    milli.iter().map(|&m| (m / scale) as i8).collect()
                };
                let (all_j, all_h) = (to_i8(&all_j), to_i8(&all_h));
                if kernel == Kernel::Msa {
                    (
                        pooled_from_slice(&device.buffer_pool, &device.device, &all_j)?,
                        pooled_from_slice(&device.buffer_pool, &device.device, &all_h)?,
                    )
                } else {
                    (
                        checked_buffer(device.new_buffer_from_slice(&all_j))?,
                        checked_buffer(device.new_buffer_from_slice(&all_h))?,
                    )
                }
            } else if kernel == Kernel::Msa {
                (
                    pooled_from_slice(&device.buffer_pool, &device.device, &all_j)?,
                    pooled_from_slice(&device.buffer_pool, &device.device, &all_h)?,
                )
            } else {
                (
                    checked_buffer(device.new_buffer_from_slice(&all_j))?,
                    checked_buffer(device.new_buffer_from_slice(&all_h))?,
                )
            }
        }
        Representation::Float32 => {
            let mut all_j = vec![0_f32; num_problems * nnz_alloc];
            let mut all_h = vec![0_f32; num_problems * n];
            for (p, graph) in graphs.iter().enumerate() {
                let (j_csr, h_f32) = fill_h_j_float(topo, graph);
                all_j[p * nnz_alloc..p * nnz_alloc + j_csr.len()].copy_from_slice(&j_csr);
                all_h[p * n..p * n + h_f32.len()].copy_from_slice(&h_f32);
            }
            if kernel == Kernel::Msa {
                (
                    pooled_from_slice(&device.buffer_pool, &device.device, &all_j)?,
                    pooled_from_slice(&device.buffer_pool, &device.device, &all_h)?,
                )
            } else {
                (
                    checked_buffer(device.new_buffer_from_slice(&all_j))?,
                    checked_buffer(device.new_buffer_from_slice(&all_h))?,
                )
            }
        }
        Representation::Binary | Representation::Int8 => {
            let mut all_j = vec![0i8; num_problems * nnz_alloc];
            let mut all_h = vec![0i8; num_problems * n];
            for (p, graph) in graphs.iter().enumerate() {
                let (j_csr, h_i8) = fill_h_j(topo, graph);
                all_j[p * nnz_alloc..p * nnz_alloc + j_csr.len()].copy_from_slice(&j_csr);
                all_h[p * n..p * n + h_i8.len()].copy_from_slice(&h_i8);
            }
            if kernel == Kernel::Msa {
                (
                    pooled_from_slice(&device.buffer_pool, &device.device, &all_j)?,
                    pooled_from_slice(&device.buffer_pool, &device.device, &all_h)?,
                )
            } else {
                (
                    checked_buffer(device.new_buffer_from_slice(&all_j))?,
                    checked_buffer(device.new_buffer_from_slice(&all_h))?,
                )
            }
        }
    };

    Ok(InputBuffers {
        row,
        col,
        j,
        h,
        row_off: checked_buffer(device.new_buffer_from_slice(&row_ptr_offsets))?,
        col_off: checked_buffer(device.new_buffer_from_slice(&col_ind_offsets))?,
    })
}

/// The element type of a dense coupling matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixKind {
    /// Two bit planes per 32 columns: the negative and the nonzero
    /// couplings of a unit model.
    Bits,
    /// `int8_t` milli coefficients divided by this scale.
    Int8(i32),
    /// Milli `int` coefficients.
    Int32,
    /// `float` coefficients.
    Float32,
}

/// Elements of one row of an `n`-node dense matrix: `uint` words of two bit
/// planes for a unit model, else one coefficient per column.
fn dense_row_elems(n: usize, unit: bool) -> usize {
    if unit {
        2 * n.div_ceil(32)
    } else {
        n
    }
}

/// One problem's matrix and fields in the layout `kernels/msa_dense.metal`
/// reads: row `u` holds the coupling of every column `v`, both halves of
/// each edge, and a zero diagonal. The caller routed only dense candidates
/// here, so every pair appears once.
fn fill_dense<T: Copy>(
    graph: &IsingGraph,
    n: usize,
    matrix: &mut [T],
    h: &mut [T],
    value: impl Fn(f64) -> T,
) {
    for (&(u, v), &j) in graph.edges.iter().zip(graph.j.iter()) {
        if u >= n || v >= n || u == v {
            continue;
        }
        matrix[u * n + v] = value(j);
        matrix[v * n + u] = value(j);
    }
    for (slot, &field) in h.iter_mut().zip(graph.h.iter()) {
        *slot = value(field);
    }
}

/// Upload a dense-matrix batch: each problem's matrix at buffer 2 and its
/// fields at buffer 15, in pooled buffers. The CSR and offset slots get
/// four-byte stubs the kernel never reads.
fn upload_dense_inputs(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    n: usize,
    kind: MatrixKind,
) -> Result<InputBuffers, SampleError> {
    use crate::representation::to_milli;
    let pool = &device.buffer_pool;
    let cells = n * n;
    let (j, h) = match kind {
        MatrixKind::Bits => {
            let row = dense_row_elems(n, true);
            let mut matrix = vec![0_u32; graphs.len() * n * row];
            let mut fields = vec![0_i8; graphs.len() * n];
            for (p, graph) in graphs.iter().enumerate() {
                let rows = &mut matrix[p * n * row..(p + 1) * n * row];
                for (&(u, v), &value) in graph.edges.iter().zip(graph.j.iter()) {
                    if u >= n || v >= n || u == v || value == 0.0 {
                        continue;
                    }
                    for (a, b) in [(u, v), (v, u)] {
                        let word = a * row + 2 * (b / 32);
                        let bit = 1_u32 << (b % 32);
                        if value < 0.0 {
                            rows[word] |= bit;
                        }
                        rows[word + 1] |= bit;
                    }
                }
                for (slot, &field) in fields[p * n..(p + 1) * n].iter_mut().zip(graph.h.iter()) {
                    // A unit field is -1, 0, or 1.
                    *slot = field as i8;
                }
            }
            (
                pooled_from_slice(pool, &device.device, &matrix)?,
                pooled_from_slice(pool, &device.device, &fields)?,
            )
        }
        MatrixKind::Int8(scale) => {
            let mut matrix = vec![0_i8; graphs.len() * cells];
            let mut fields = vec![0_i8; graphs.len() * n];
            for (p, graph) in graphs.iter().enumerate() {
                // `batch_wide_i8_scale` checked every quotient against `i8`.
                fill_dense(
                    graph,
                    n,
                    &mut matrix[p * cells..(p + 1) * cells],
                    &mut fields[p * n..(p + 1) * n],
                    |v| (to_milli(v).unwrap_or(0) / scale) as i8,
                );
            }
            (
                pooled_from_slice(pool, &device.device, &matrix)?,
                pooled_from_slice(pool, &device.device, &fields)?,
            )
        }
        MatrixKind::Int32 => {
            let mut matrix = vec![0_i32; graphs.len() * cells];
            let mut fields = vec![0_i32; graphs.len() * n];
            for (p, graph) in graphs.iter().enumerate() {
                fill_dense(
                    graph,
                    n,
                    &mut matrix[p * cells..(p + 1) * cells],
                    &mut fields[p * n..(p + 1) * n],
                    |v| to_milli(v).unwrap_or(0),
                );
            }
            (
                pooled_from_slice(pool, &device.device, &matrix)?,
                pooled_from_slice(pool, &device.device, &fields)?,
            )
        }
        MatrixKind::Float32 => {
            let mut matrix = vec![0_f32; graphs.len() * cells];
            let mut fields = vec![0_f32; graphs.len() * n];
            for (p, graph) in graphs.iter().enumerate() {
                fill_dense(
                    graph,
                    n,
                    &mut matrix[p * cells..(p + 1) * cells],
                    &mut fields[p * n..(p + 1) * n],
                    |v| v as f32,
                );
            }
            (
                pooled_from_slice(pool, &device.device, &matrix)?,
                pooled_from_slice(pool, &device.device, &fields)?,
            )
        }
    };
    let stub = || checked_buffer(device.new_zeroed_buffer(4));
    Ok(InputBuffers {
        row: stub()?,
        col: stub()?,
        j,
        h,
        row_off: stub()?,
        col_off: stub()?,
    })
}

/// Bind the shared buffer layout (indices 0..15) — identical in both kernels.
fn bind_shared_args(
    enc: &metal::ComputeCommandEncoderRef,
    inputs: &InputBuffers,
    out: &DispatchBuffers,
    dims: &BatchDims,
) {
    enc.set_buffer(0, Some(&inputs.row), 0);
    enc.set_buffer(1, Some(&inputs.col), 0);
    enc.set_buffer(2, Some(&inputs.j), 0);
    enc.set_buffer(3, Some(&inputs.row_off), 0);
    enc.set_buffer(4, Some(&inputs.col_off), 0);
    set_bytes_i32(enc, 5, dims.n as i32);
    set_bytes_i32(enc, 6, dims.num_betas);
    set_bytes_i32(enc, 7, dims.sweeps_per as i32);
    set_bytes_u32(enc, 8, dims.base_seed);
    enc.set_buffer(9, Some(&out.beta), 0);
    enc.set_buffer(10, Some(&out.samples), 0);
    enc.set_buffer(11, Some(&out.energies), 0);
    set_bytes_i32(enc, 12, dims.num_threads as i32);
    set_bytes_i32(enc, 13, dims.num_problems as i32);
    set_bytes_i32(enc, 14, dims.num_reads as i32);
    enc.set_buffer(15, Some(&inputs.h), 0);
}

/// SA carry-over buffers (indices 18..21): packed spins, `delta_energy`,
/// the RNG word, and the running energy of every read, rented from `pool`.
///
/// The first chunk writes every read's state before any read of it, and
/// every chunk writes all four at its end, so pooled buffers need no
/// zeroing. The global build works in the first two in place.
fn new_sa_persistent(
    pool: &BufferPool,
    device: &metal::DeviceRef,
    dims: &BatchDims,
    delta_bytes: usize,
) -> Result<[metal::Buffer; 4], SampleError> {
    Ok([
        pool.take(device, (dims.num_threads * dims.packed_size) as u64)?,
        pool.take(
            device,
            (dims.num_threads * dims.n.max(1) * delta_bytes) as u64,
        )?,
        pool.take(device, (dims.num_threads * 4) as u64)?, // uint rng
        pool.take(device, (dims.num_threads * 4) as u64)?, // int energy
    ])
}

/// Gibbs carry-over state: spins plus the RNG stream, rented from `pool`.
///
/// Sizes follow the two kernels' differing layouts. The chromatic kernel keeps
/// one *sample* per threadgroup and stores spins **unpacked** (one byte each,
/// mirroring its threadgroup array) to avoid the bit-packing races its own
/// comments warn about; its RNG stream is per *thread*, so the buffer scales
/// with the threadgroup width. The sequential kernel is one thread per read,
/// with bit-packed spins and one stream per thread.
///
/// Unlike SA, Gibbs persists no `delta_energy` or running energy: it recomputes
/// the effective field from the current spins on every node update, and its
/// energy is a pure function of the spins it already carries.
///
/// The first chunk writes every read's spins and every stream before any read
/// of them, and every chunk writes them at its end, so pooled buffers need no
/// zeroing. The spill build keeps the spins past its threadgroup prefix in
/// the first buffer for the whole anneal, one slice per read, and writes each
/// of them in the first chunk before any thread reads it.
fn new_gibbs_persistent(
    pool: &BufferPool,
    device: &metal::DeviceRef,
    dims: &BatchDims,
    node_parallel: bool,
    threads_per_group: usize,
) -> Result<[metal::Buffer; 2], SampleError> {
    // 4 x u32 of xoshiro128** state per stream.
    const RNG_BYTES_PER_STREAM: usize = 16;
    Ok(if node_parallel {
        [
            pool.take(device, (dims.num_threads * dims.n.max(1)) as u64)?,
            pool.take(
                device,
                (dims.num_threads * threads_per_group.max(1) * RNG_BYTES_PER_STREAM) as u64,
            )?,
        ]
    } else {
        [
            pool.take(device, (dims.num_threads * dims.packed_size) as u64)?,
            pool.take(device, (dims.num_threads * RNG_BYTES_PER_STREAM) as u64)?,
        ]
    })
}

/// Multi-spin carry-over state: one `uint` word per (threadgroup, spin) plus
/// one xoshiro128** stream (four `uint`) per (threadgroup, thread). Only spin
/// words and RNG streams persist across a chunk boundary, so the sizes do not
/// depend on the coefficient width: the int8, milli, and float lanes share
/// them.
fn new_msa_persistent(
    pool: &BufferPool,
    device: &metal::DeviceRef,
    num_streams: usize,
    n: usize,
    threads_per_group: usize,
) -> Result<[metal::Buffer; 2], SampleError> {
    const RNG_BYTES_PER_STREAM: usize = 16;
    Ok([
        pool.take(device, (num_streams * n.max(1) * 4) as u64)?,
        pool.take(
            device,
            (num_streams * threads_per_group.max(1) * RNG_BYTES_PER_STREAM) as u64,
        )?,
    ])
}

/// Per-kernel encode state: the carry-over buffers every chunk binds, plus
/// what the colour-block kernels need at bind time. Replaces the pair of
/// `Option`s the loop used to match on.
enum KernelEncode {
    Sa {
        persist: [metal::Buffer; 4],
    },
    Gibbs {
        persist: [metal::Buffer; 2],
        /// `K` of the spill build, bound at slot 29 of the chromatic kernel;
        /// `None` on the threadgroup build, which takes no such argument.
        shared_spins: Option<i32>,
    },
    Msa {
        persist: [metal::Buffer; 2],
        /// Words per problem, bound at slot 19.
        words: i32,
        /// Bytes of `threadgroup uint state[min(N, K)]` at threadgroup
        /// index 0, rounded up to Metal's 16-byte granularity.
        state_bytes: u64,
        /// `K` of the weighted entry (`msa_anneal_weighted`), bound at slot
        /// 27 with the sweep window at 28 and 29; `None` for the unit entry,
        /// which takes neither.
        shared_spins: Option<i32>,
        /// `V`, the sparse dispatch width the dense-matrix kernel emulates,
        /// bound at slot 30; `None` on CSR storage.
        virtual_threads: Option<i32>,
    },
}

/// Bind the chunk window and carry-over state of a colour-block kernel (Gibbs
/// or multi-spin). Indices start at 21 because the colour-block bindings
/// occupy 16..20.
fn bind_color_kernel_chunk(
    enc: &metal::ComputeCommandEncoderRef,
    persist: &[metal::Buffer; 2],
    beta_start: i32,
    beta_count: i32,
) {
    set_bytes_i32(enc, 21, beta_start);
    set_bytes_i32(enc, 22, beta_count);
    enc.set_buffer(23, Some(&persist[0]), 0);
    enc.set_buffer(24, Some(&persist[1]), 0);
}

/// Bind SA's chunk window and its carry-over state.
///
/// `beta_start == 0` tells the kernel to initialize; any other value makes it
/// resume from the persistent buffers, which it rewrites at the end of every
/// chunk. The buffers are shared across a batch's chunks — that carry-over is
/// what lets one anneal span several command buffers.
fn bind_sa_chunk(
    enc: &metal::ComputeCommandEncoderRef,
    persist: &[metal::Buffer; 4],
    chunk: Chunk,
) {
    set_bytes_i32(enc, 16, chunk.beta_start);
    set_bytes_i32(enc, 17, chunk.beta_count);
    enc.set_buffer(18, Some(&persist[0]), 0);
    enc.set_buffer(19, Some(&persist[1]), 0);
    enc.set_buffer(20, Some(&persist[2]), 0);
    enc.set_buffer(21, Some(&persist[3]), 0);
    set_bytes_i32(enc, 24, chunk.sweep_start);
    set_bytes_i32(enc, 25, chunk.sweep_count);
}

/// Build one batch's device buffers and plan its chunks. Does **not** commit.
///
/// `graphs` must be non-empty and share a topology (same `N` and `edges`) — the
/// caller ([`crate::streaming`]) guarantees this by batch key; the topology is
/// built from `graphs[0]`. `params` (reads, sweeps, beta) is shared across the
/// batch, matching v0.2 (`compute_beta_schedule(h[0], J[0], ...)`).
pub(crate) fn encode_batch(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
    in_flight: usize,
) -> Result<EncodedBatch, SampleError> {
    encode_batch_inner(device, graphs, params, kernel, in_flight, None, None)
}

/// [`encode_batch`] with optional chunk-plan and beta-schedule overrides.
/// A beta override supplies one rung per sweep, including reheating schedules.
/// Production passes `None` for both overrides.
fn encode_batch_inner(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
    in_flight: usize,
    plan_override: Option<&[(i32, i32)]>,
    beta_override: Option<Vec<f32>>,
) -> Result<EncodedBatch, SampleError> {
    encode_batch_storage(
        device,
        graphs,
        params,
        kernel,
        in_flight,
        plan_override,
        beta_override,
        true,
        Tuning::default(),
        DenseMode::Off,
    )
}

/// [`encode_batch_inner`] with `wide_i8` false forcing every wide batch onto
/// plain `int` coefficients, so tests compare the two storages, with the
/// batch kernels' test `tuning`, and with `dense` saying whether the batch
/// may take dense-matrix storage.
#[expect(
    clippy::too_many_arguments,
    reason = "encode_batch_inner's arguments plus the storage switches and tuning"
)]
fn encode_batch_storage(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
    in_flight: usize,
    plan_override: Option<&[(i32, i32)]>,
    beta_override: Option<Vec<f32>>,
    wide_i8: bool,
    tuning: Tuning,
    dense: DenseMode,
) -> Result<EncodedBatch, SampleError> {
    // `K`, the spin words the weighted multi-spin kernel keeps in threadgroup
    // memory. A model past it runs the spill build, which holds `int`
    // coefficients only, as the dense build does.
    let tg_limit = device.device.max_threadgroup_memory_length() as usize;
    let k = msa_shared_spins(tg_limit, tuning.shared_spins_cap);
    let past_k = kernel == Kernel::Msa && graphs.first().is_some_and(|g| g.num_nodes() > k);
    debug_assert!(
        !(tuning.force_dense && tuning.force_sparse),
        "force_dense and force_sparse exclude each other"
    );
    let dense = match dense {
        DenseMode::Auto if tuning.force_dense => DenseMode::Forced,
        DenseMode::Auto if tuning.force_sparse => DenseMode::Off,
        other => other,
    };
    let ValidBatch {
        first,
        n,
        rep,
        i8_scale,
        dense_rows,
        storage,
        unit,
    } = validate_batch_within(
        graphs,
        params,
        kernel,
        host_device_limits(),
        wide_i8 && !past_k,
        tuning.node_cap,
        dense,
    )?;
    let matrix = storage == Storage::Dense;
    // A model past the SA thread-private frame runs the global build, which
    // keeps each read's spins and `delta_energy` in the persistent buffers.
    // The dense-matrix SA kernel always keeps its state there.
    let sa_global = kernel == Kernel::Sa && (tuning.sa_global || n > SA_MAX_NODES || matrix);
    // `K`, the spins the chromatic Gibbs kernel keeps in threadgroup memory.
    // A model past it runs the spill build, which keeps the rest in each
    // read's slice of the persistent buffer.
    let gibbs_prefix = tuning
        .gibbs_prefix_cap
        .map_or(GIBBS_MAX_NODES, |cap| cap.min(GIBBS_MAX_NODES));
    let gibbs_spill = kernel == Kernel::Gibbs && n > gibbs_prefix;
    let wide = rep == Representation::Wide;
    let float = rep == Representation::Float32;
    // Wide and float SA and Gibbs hold 4-byte deltas and share a dispatch
    // geometry.
    let four_byte = wide || float;

    let num_problems = graphs.len();
    let num_reads = simd_rounded_reads(params.num_reads);
    let num_samples = num_problems * num_reads;
    let packed_size = n.div_ceil(8).max(1);

    let (beta, sweeps_per) = match beta_override {
        Some(beta) => {
            if beta.is_empty() || beta.len() > MAX_SWEEPS {
                return Err(SampleError::TooLarge(format!(
                    "beta override length {} must be in 1..={MAX_SWEEPS}",
                    beta.len()
                )));
            }
            (beta, 1)
        }
        None => build_beta_schedule(
            first,
            params.num_sweeps,
            params.sweeps_per_beta,
            params.beta_range,
            params.schedule,
        ),
    };
    if float {
        for graph in graphs {
            check_float_schedule(graph, &beta)?;
        }
    }
    let wide_rows = if wide {
        if !crate::wide_tables::wide_table_fits(beta.len()) {
            return Err(SampleError::TooLarge(format!(
                "{} beta rungs exceed the wide threshold table cap",
                beta.len()
            )));
        }
        let (index, rows) = match kernel {
            Kernel::Sa => (22, crate::wide_tables::metropolis_rows(&beta, params.seed)),
            Kernel::Gibbs => (25, crate::wide_tables::heat_bath_rows(&beta, params.seed)),
            Kernel::Msa => (25, crate::wide_tables::metropolis_rows(&beta, params.seed)),
        };
        Some((index, checked_buffer(device.new_buffer_from_slice(&rows))?))
    } else {
        None
    };

    // Chromatic Gibbs uses a different pipeline but the *same* buffer layout —
    // only the dispatch geometry below differs.
    // Only the chromatic kernel has a spill build, so a model past its
    // threadgroup array runs it whatever `QUIP_METAL_GIBBS_SEQUENTIAL` says.
    let node_parallel = kernel == Kernel::Gibbs && (gibbs_node_parallel() || gibbs_spill);
    // A milli model with a row past one chunk takes the weighted lane's dense
    // build, on `int` storage; the default build keeps the one-chunk code
    // alone. A weighted model past `K` takes the spill build, whose milli
    // variant also walks dense rows.
    let sparse_pipeline = match kernel {
        Kernel::Msa if wide && past_k => &device.msa_weighted_spill,
        Kernel::Msa if float && past_k => &device.msa_float_spill,
        Kernel::Sa if sa_global && i8_scale.is_some() => &device.sa_wide_i8_global,
        Kernel::Sa if sa_global && wide => &device.sa_wide_global,
        Kernel::Sa if sa_global && float => &device.sa_float_global,
        Kernel::Sa if sa_global => &device.sa_global,
        Kernel::Sa if i8_scale.is_some() => &device.sa_wide_i8,
        Kernel::Msa if i8_scale.is_some() => &device.msa_weighted_i8,
        Kernel::Gibbs if gibbs_spill && i8_scale.is_some() => &device.gibbs_parallel_wide_i8_spill,
        Kernel::Gibbs if gibbs_spill && wide => &device.gibbs_parallel_wide_spill,
        Kernel::Gibbs if gibbs_spill && float => &device.gibbs_parallel_float_spill,
        Kernel::Gibbs if gibbs_spill => &device.gibbs_parallel_spill,
        Kernel::Gibbs if node_parallel && i8_scale.is_some() => &device.gibbs_parallel_wide_i8,
        Kernel::Gibbs if i8_scale.is_some() => &device.gibbs_wide_i8,
        Kernel::Sa if wide => &device.sa_wide,
        Kernel::Sa if float => &device.sa_float,
        Kernel::Sa => &device.sa,
        Kernel::Msa if wide && dense_rows => &device.msa_weighted_dense,
        Kernel::Msa if wide => &device.msa_weighted,
        Kernel::Msa if float => &device.msa_float,
        Kernel::Msa => &device.msa,
        Kernel::Gibbs if node_parallel && wide => &device.gibbs_parallel_wide,
        Kernel::Gibbs if node_parallel && float => &device.gibbs_parallel_float,
        Kernel::Gibbs if node_parallel => &device.gibbs_parallel,
        Kernel::Gibbs if wide => &device.gibbs_wide,
        Kernel::Gibbs if float => &device.gibbs_float,
        Kernel::Gibbs => &device.gibbs,
    };
    // A dense-matrix multi-spin batch keeps the RNG streams of the sparse
    // pipeline the model would take, so it emulates that pipeline's dispatch
    // width. A dense-matrix batch holds a bit matrix for a unit model, and
    // otherwise the narrowest matrix its coefficients fit: `int8_t` storage
    // needs no CSR rule here.
    let (pipeline, i8_scale) = if matrix {
        let dense_i8 = if unit {
            None
        } else {
            batch_i8_scale(graphs, rep, wide_i8)
        };
        let pipeline = match kernel {
            Kernel::Sa if dense_i8.is_some() => &device.sa_matrix_wide_i8,
            Kernel::Sa if wide => &device.sa_matrix_int,
            Kernel::Sa if float => &device.sa_matrix_float,
            Kernel::Sa => &device.sa_matrix_i8,
            _ if unit && rep == Representation::Binary => &device.msa_matrix_bits_unit,
            _ if unit => &device.msa_matrix_bits,
            _ if dense_i8.is_some() => &device.msa_matrix_i8,
            _ if wide => &device.msa_matrix_int,
            _ => &device.msa_matrix_float,
        };
        (pipeline, dense_i8)
    } else {
        (sparse_pipeline, i8_scale)
    };

    // Dispatch geometry is constant across a batch's chunks, and must be: the
    // chromatic and multi-spin kernels' RNG stream identity is
    // `(threadgroup, thread_in_group)`, so varying the threadgroup width
    // mid-anneal would resume the wrong streams.
    //
    // Sequential kernels: one threadgroup per problem, `num_reads` threads (one
    // per read) each — `problem_id = thread_id / num_reads` = the threadgroup
    // index, so a model maps to a core and its reads are the threads inside.
    //
    // Chromatic Gibbs: one threadgroup per *sample* (`sample_id =
    // problem*num_reads + read`), and its threads split each color's nodes. The
    // group is capped at 256 by the kernel's `threadgroup int
    // partial_energies[256]` reduction array.
    //
    // Multi-spin: one threadgroup per (problem, 32-replica word), `words`
    // threadgroups per problem, 1,024 threads splitting each colour class.
    let words = num_reads.div_ceil(MSA_LANES);
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    // The sparse multi-spin width: the dense-matrix kernel's RNG streams.
    let msa_width = MSA_THREADS
        .min(sparse_pipeline.max_total_threads_per_threadgroup() as usize)
        .max(1);
    if matrix
        && kernel == Kernel::Msa
        && (max_threads < MSA_LANES || pipeline.thread_execution_width() as usize != MSA_LANES)
    {
        return Err(SampleError::TooLarge(format!(
            "the dense-matrix multi-spin kernel needs one {MSA_LANES}-lane SIMD group per threadgroup; \
             this device admits {max_threads} threads of width {}",
            pipeline.thread_execution_width()
        )));
    }
    let (groups, threads_per_group) = match kernel {
        // One SIMD group per (problem, word) on dense-matrix storage.
        Kernel::Msa if matrix => (num_problems * words, MSA_LANES),
        Kernel::Msa => (num_problems * words, msa_width),
        Kernel::Gibbs if node_parallel && wide => {
            if max_threads < WIDE_GIBBS_GROUP {
                return Err(SampleError::TooLarge(format!(
                    "the wide Gibbs kernel needs {WIDE_GIBBS_GROUP} threads per group; this device admits {max_threads}"
                )));
            }
            (num_samples, WIDE_GIBBS_GROUP)
        }
        Kernel::Gibbs if node_parallel => (num_samples, 256.min(max_threads).max(1)),
        Kernel::Sa | Kernel::Gibbs if four_byte => {
            let group = wide_sa_group(num_reads, max_threads);
            (num_samples.div_ceil(group), group)
        }
        Kernel::Sa | Kernel::Gibbs => (num_problems, num_reads),
    };
    if kernel == Kernel::Msa {
        tracing::debug!(threads_per_group, "msa dispatch width");
    }
    // `buffer(12)`: threads for SA / sequential Gibbs (one per read),
    // threadgroups for chromatic Gibbs and multi-spin. Also the count
    // `chunk_plan` multiplies: spin updates per thread, or word updates per
    // threadgroup.
    let num_threads = match kernel {
        Kernel::Msa => groups,
        Kernel::Sa | Kernel::Gibbs => num_samples,
    };
    let dims = BatchDims {
        n,
        num_betas: beta.len() as i32,
        sweeps_per,
        // Every batch kernel maps a zero seed to 1 itself.
        base_seed: kernel_seed(params.seed),
        num_threads,
        num_problems,
        num_reads,
        packed_size,
    };

    // The multi-spin kernel's spin words live in threadgroup memory sized per
    // dispatch; the weighted entry keeps the first `K` there and the rest in
    // its persistent state. Refuse a job the opened device cannot hold rather
    // than let Metal fail the command buffer, which would end the session as
    // a DeviceFault for a per-job size problem.
    let weighted_msa = kernel == Kernel::Msa && (wide || float);
    // The dense-matrix kernel keeps the spill layout at every precision.
    let shared_spins = (weighted_msa || matrix).then_some(k);
    let state_bytes = (shared_spins.map_or(n, |k| k.min(n)).max(1) * 4).div_ceil(16) * 16;
    if sa_global
        && num_threads
            .checked_mul(n.max(1))
            .is_none_or(|entries| entries > u32::MAX as usize)
    {
        // The global build indexes entry `i` of thread `t` as
        // `i * num_threads + t` in 32 bits.
        return Err(SampleError::TooLarge(format!(
            "{num_threads} SA reads of {n} spins exceed 32-bit indexing"
        )));
    }
    if kernel == Kernel::Msa {
        // The kernel indexes each threadgroup's persistent words as
        // `threadgroup * N` in 32 bits.
        if groups
            .checked_mul(n.max(1))
            .is_none_or(|words| words > u32::MAX as usize)
        {
            return Err(SampleError::TooLarge(format!(
                "{groups} multi-spin threadgroups of {n} spin words exceed 32-bit indexing"
            )));
        }
        let need = state_bytes + MSA_STATIC_TG_BYTES;
        let have = tg_limit;
        if need > have {
            return Err(SampleError::TooLarge(format!(
                "graph N={n} needs {need} B of threadgroup memory; device offers {have} B"
            )));
        }
    }
    // The kernel indexes a row of one problem's matrix in 32 bits.
    if matrix
        && n.checked_mul(dense_row_elems(n, unit))
            .is_none_or(|cells| cells > u32::MAX as usize)
    {
        return Err(SampleError::TooLarge(format!(
            "a dense matrix of {n} nodes exceeds 32-bit row indexing"
        )));
    }

    let four_color = kernel == Kernel::Msa && msa_four_color();
    let (cached, inputs) = if matrix {
        let cached = if kernel == Kernel::Sa {
            device.topology_cache.get_or_build_bare(device, first)?
        } else {
            device
                .topology_cache
                .get_or_build_colors(device, first, four_color)?
        };
        let kind = if unit {
            MatrixKind::Bits
        } else if let Some(scale) = i8_scale {
            MatrixKind::Int8(scale)
        } else if rep == Representation::Int8 {
            // Whole units: the milli value divided by 1,000.
            MatrixKind::Int8(1_000)
        } else if wide {
            MatrixKind::Int32
        } else {
            MatrixKind::Float32
        };
        let inputs = upload_dense_inputs(device, graphs, n, kind)?;
        (cached, inputs)
    } else {
        let cached = device
            .topology_cache
            .get_or_build(device, first, four_color)?;
        let inputs = upload_inputs(device, &cached, graphs, kernel, rep, i8_scale)?;
        (cached, inputs)
    };

    let cost = match (rep, kernel) {
        // Every word update of the dense-matrix kernel walks all `n`
        // columns, planned at `msa_matrix_rate` column updates per second.
        (_, Kernel::Msa) if matrix => n.max(1) as f64,
        // A flip of the SA kernel walks the `n` columns of its row, and the
        // SIMD groups of a dispatch run side by side, so one sweep of the
        // dispatch walks `n * n` columns (`num_threads * n * cost`), at
        // `sa_matrix_rate` columns per second.
        (_, Kernel::Sa) if matrix => n.max(1) as f64 / dims.num_threads.max(1) as f64,
        (Representation::Wide, Kernel::Sa) => WIDE_SA_COST_FACTOR,
        (Representation::Wide, Kernel::Gibbs) => WIDE_GIBBS_COST_FACTOR,
        (Representation::Float32, Kernel::Sa) => FLOAT_SA_COST_FACTOR,
        (Representation::Float32, Kernel::Gibbs) => FLOAT_GIBBS_COST_FACTOR,
        // The weighted factors are per word update of one row chunk. A
        // denser row is walked in chunks of MSA_UNIT_MAX_DEG, so the cost
        // grows with the chunk count, and a dense model of few nodes is
        // planned by its colour steps. The milli lane keeps its one-chunk
        // plan up to MSA_UNIT_MAX_DEG, so Advantage2 plans as it did. Every
        // graph of a batch shares `first`'s topology, so its degree is the
        // batch's.
        (Representation::Wide, Kernel::Msa) if dense_rows => {
            WEIGHTED_MSA_DENSE_COST_FACTOR
                * msa_row_chunks(max_csr_degree(first)) as f64
                * msa_colour_step_factor(&cached.topo.colors.counts, threads_per_group, n)
        }
        (Representation::Wide, Kernel::Msa) => WEIGHTED_MSA_COST_FACTOR,
        (Representation::Float32, Kernel::Msa) => {
            FLOAT_MSA_COST_FACTOR
                * msa_row_chunks(max_csr_degree(first)) as f64
                * msa_colour_step_factor(&cached.topo.colors.counts, threads_per_group, n)
        }
        (Representation::Binary | Representation::Int8, _) => 1.0,
    } * if sa_global && !matrix {
        SA_GLOBAL_COST_FACTOR
    } else if gibbs_spill {
        GIBBS_SPILL_COST_FACTOR
    } else {
        1.0
    };
    let plan = match plan_override {
        Some(p) => p
            .iter()
            .map(|&(start, count)| Chunk::rungs(start, count, sweeps_per))
            .collect(),
        None => chunk_plan(
            kernel,
            &dims,
            groups,
            in_flight,
            cost,
            weighted_msa
                || matrix
                || kernel == Kernel::Sa
                || (kernel == Kernel::Gibbs && node_parallel),
            if matrix {
                Some(tuning.planning_rate.unwrap_or_else(|| match kernel {
                    Kernel::Sa => sa_matrix_rate(in_flight),
                    _ => msa_matrix_rate(groups, in_flight),
                }))
            } else {
                tuning.planning_rate
            },
        ),
    };
    let mut pooled = Vec::new();
    let (out, kstate) = match kernel {
        Kernel::Sa => (
            DispatchBuffers {
                beta: checked_buffer(device.new_buffer_from_slice(&beta))?,
                samples: checked_buffer(
                    device.new_zeroed_buffer((num_samples * packed_size) as u64),
                )?,
                energies: checked_buffer(device.new_zeroed_buffer((num_samples * 4) as u64))?, // i32
            },
            KernelEncode::Sa {
                persist: {
                    let persist = new_sa_persistent(
                        &device.buffer_pool,
                        &device.device,
                        &dims,
                        if four_byte { 4 } else { 1 },
                    )?;
                    pooled.extend(persist.iter().cloned());
                    if matrix {
                        // The dense inputs are rented from the pool too.
                        pooled.extend([inputs.j.clone(), inputs.h.clone()]);
                    }
                    persist
                },
            },
        ),
        Kernel::Gibbs => (
            DispatchBuffers {
                beta: checked_buffer(device.new_buffer_from_slice(&beta))?,
                samples: checked_buffer(
                    device.new_zeroed_buffer((num_samples * packed_size) as u64),
                )?,
                energies: checked_buffer(device.new_zeroed_buffer((num_samples * 4) as u64))?, // i32
            },
            KernelEncode::Gibbs {
                persist: {
                    let persist = new_gibbs_persistent(
                        &device.buffer_pool,
                        &device.device,
                        &dims,
                        node_parallel,
                        threads_per_group,
                    )?;
                    pooled.extend(persist.iter().cloned());
                    persist
                },
                shared_spins: gibbs_spill.then(|| i32::try_from(gibbs_prefix).unwrap_or(i32::MAX)),
            },
        ),
        Kernel::Msa => {
            let beta_buf = pooled_from_slice(&device.buffer_pool, &device.device, &beta)?;
            // The kernel writes `samples`, `energies`, and the persistent buffers
            // in full before any read (first chunk initializes state and RNG, and
            // every read's bytes are packed), so pooled buffers need no zeroing.
            let samples = device
                .buffer_pool
                .take(&device.device, (num_samples * packed_size) as u64)?;
            let energies = device
                .buffer_pool
                .take(&device.device, (num_samples * 4) as u64)?;
            // The dense-matrix kernel holds the sparse width of streams.
            let streams_per_group = if matrix { msa_width } else { threads_per_group };
            let persist = new_msa_persistent(
                &device.buffer_pool,
                &device.device,
                groups,
                n,
                streams_per_group,
            )?;
            pooled.extend([
                inputs.j.clone(),
                inputs.h.clone(),
                beta_buf.clone(),
                samples.clone(),
                energies.clone(),
                persist[0].clone(),
                persist[1].clone(),
            ]);
            (
                DispatchBuffers {
                    beta: beta_buf,
                    samples,
                    energies,
                },
                KernelEncode::Msa {
                    persist,
                    words: words as i32,
                    state_bytes: state_bytes as u64,
                    shared_spins: shared_spins.map(|k| i32::try_from(k).unwrap_or(i32::MAX)),
                    virtual_threads: matrix.then(|| i32::try_from(msa_width).unwrap_or(i32::MAX)),
                },
            )
        }
    };

    let colors = cached.colors.clone();
    let num_colors = cached.topo.colors.num_colors;
    Ok(EncodedBatch {
        // The weighted entry and its float lane leave energies to the host.
        device_energy: kernel == Kernel::Msa
            && !matrix
            && !wide
            && !float
            && graphs.iter().all(|g| device_energy_exact(g)),
        cmds: Vec::with_capacity(plan.len()),
        d_samples: out.samples.clone(),
        n,
        num_reads,
        num_problems,
        packed_size,
        queue: device.queue.clone(),
        pipeline: pipeline.clone(),
        inputs,
        out,
        dims,
        kstate,
        wide_rows,
        coef_scale: i8_scale.map(|scale| (coef_scale_slot(kernel), scale)),
        colors,
        num_colors,
        groups,
        threads_per_group,
        plan,
        cached,
        pool: std::sync::Arc::clone(&device.buffer_pool),
        pooled,
        storage: if matrix {
            Storage::Dense
        } else if (weighted_msa && past_k) || sa_global || gibbs_spill {
            Storage::Spill
        } else {
            Storage::Threadgroup
        },
    })
}

fn pad_i32(v: &[i32]) -> Vec<i32> {
    if v.is_empty() {
        vec![0i32]
    } else {
        v.to_vec()
    }
}

/// Read a completed batch's bit-packed samples and return results in `graphs` order.
/// Whole-unit multi-spin graphs use device energies, with one job in 1,000
/// audited on the host. SA, Gibbs, and non-integer graphs use host scoring.
///
/// `graphs` must be the exact slice passed to [`encode_batch`] (same order and
/// length) so each problem's spins are scored against its own `h`/`J`.
/// Buffer reads run on the caller's thread. Unpacking and host scoring run
/// on a rayon pool, with one task per problem.
pub(crate) fn harvest_batch(
    batch: &EncodedBatch,
    graphs: &[&IsingGraph],
) -> Result<Vec<Vec<SamplerResult>>, SampleError> {
    // Was a `debug_assert_eq!`, which is compiled out in release — a length
    // mismatch would index `packed` from `graphs` while sizing it from
    // `batch.num_problems` and panic the miner.
    if graphs.len() != batch.num_problems {
        return Err(SampleError::Driver(format!(
            "harvest_batch graphs.len()={} != batch.num_problems={}",
            graphs.len(),
            batch.num_problems
        )));
    }
    let count = batch.num_problems * batch.num_reads * batch.packed_size;
    let packed = read_i8_buffer(&batch.d_samples, count)?;
    let (num_reads, packed_size, n) = (batch.num_reads, batch.packed_size, batch.n);
    let energies = if batch.device_energy {
        Some(read_i32_buffer(
            &batch.out.energies,
            batch.num_problems * num_reads,
        )?)
    } else {
        None
    };
    decode_packed_reads(
        &packed,
        energies.as_deref(),
        graphs,
        num_reads,
        packed_size,
        n,
    )
}

/// Decode dense job regions in graph order, then audit device energies.
/// Callers provide initialized storage for graphs.len() * num_reads reads.
pub(crate) fn decode_packed_reads(
    packed: &[i8],
    energies: Option<&[i32]>,
    graphs: &[&IsingGraph],
    num_reads: usize,
    packed_size: usize,
    n: usize,
) -> Result<Vec<Vec<SamplerResult>>, SampleError> {
    use rayon::prelude::*;

    let out: Vec<Vec<SamplerResult>> = graphs
        .par_iter()
        .enumerate()
        .map(|(p, graph)| {
            (0..num_reads)
                .map(|r| {
                    let start = (p * num_reads + r) * packed_size;
                    let spins = unpack_spins(&packed[start..start + packed_size], n);
                    match energies {
                        Some(e) => SamplerResult {
                            spins,
                            energy_milli: i64::from(e[p * num_reads + r]),
                        },
                        None => score_spins(&spins, graph),
                    }
                })
                .collect()
        })
        .collect();
    if energies.is_some() {
        audit_device_energies(&out, graphs)?;
    }
    Ok(out)
}

/// Rescore one job in [`ENERGY_AUDIT_EVERY`] on the host. A mismatch means the
/// kernel's reduction is wrong, which would turn into rejected proofs, so it
/// is a device fault rather than a per-job error.
pub(crate) fn audit_device_energies(
    out: &[Vec<SamplerResult>],
    graphs: &[&IsingGraph],
) -> Result<(), SampleError> {
    use std::sync::atomic::Ordering;
    let first = ENERGY_AUDIT_JOBS.fetch_add(out.len() as u64, Ordering::Relaxed);
    for (p, (reads, graph)) in out.iter().zip(graphs).enumerate() {
        if !(first + p as u64).is_multiple_of(ENERGY_AUDIT_EVERY) {
            continue;
        }
        for (r, read) in reads.iter().enumerate() {
            let want = energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges);
            if read.energy_milli != want {
                return Err(SampleError::Driver(format!(
                    "device energy {} != host energy {want} for problem {p} read {r}",
                    read.energy_milli
                )));
            }
        }
    }
    Ok(())
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem
/// (synchronous single-problem batch; used by [`crate::MetalSampler::sample`]
/// and the golden-parity tests).
///
/// A zero-node graph short-circuits to `num_reads` empty results without
/// touching the GPU.
///
/// # Errors
///
/// Returns [`SampleError::TooLarge`] (harness `SampleError::Capacity`) when:
/// - the model does not fit the device's memory
///   (`capacity::routed_runnable_footprint`, which runs dense-matrix storage
///   for a multi-spin model past the sparse degree),
///   which bounds every kernel's model;
/// - `params.num_sweeps` exceeds `MAX_SWEEPS`, the sampler's guard against an
///   unbounded coordinator-supplied beta schedule.
///
/// Returns [`SampleError::Driver`] (harness `SampleError::DeviceFault`) when:
/// - the command buffer finished in a non-`Completed` state (device reset,
///   kernel fault, or GPU watchdog timeout) — the zeroed sample buffer would
///   otherwise score as a real all-`+1` solution;
/// - the samples buffer reads back null or shorter than the harvest expects.
///
/// [`SampleError::Metal`] is not produced by this path — device and pipeline
/// construction errors surface earlier, from [`crate::metal_device`].
///
/// # Examples
///
/// ```no_run
/// use quip_solver_metal::{sample_ising, Kernel, IsingGraph, SampleParams, metal_device::MetalDevice};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = MetalDevice::open(0)?;
/// let graph = IsingGraph::new(
///     vec![1.0, -1.0, 0.0, 1.0],
///     vec![1.0, -1.0, 1.0, -1.0],
///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
/// );
/// let params = SampleParams {
///     num_reads: 4,
///     num_sweeps: 64,
///     ..Default::default()
/// };
/// let samples = sample_ising(&device, &graph, &params, Kernel::Sa)?;
/// assert_eq!(samples.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn sample_ising(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
) -> Result<Vec<SamplerResult>, SampleError> {
    sample_planned(device, graph, params, kernel, None, true)
}

/// [`sample_ising`] with the beta schedule split into chunks of
/// `rungs_per_chunk` rungs, so tests can compare a chunked anneal with an
/// unchunked one of the same seed.
///
/// # Errors
///
/// The same as [`sample_ising`].
#[doc(hidden)]
pub fn sample_ising_chunked(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
    rungs_per_chunk: usize,
) -> Result<Vec<SamplerResult>, SampleError> {
    let num_betas = (params.num_sweeps / params.sweeps_per_beta.max(1)).max(1);
    let step = rungs_per_chunk.clamp(1, num_betas);
    let plan: Vec<(i32, i32)> = (0..num_betas)
        .step_by(step)
        .map(|start| (start as i32, step.min(num_betas - start) as i32))
        .collect();
    sample_planned(device, graph, params, kernel, Some(&plan), true)
}

/// The reads of [`sample_ising_tuned`] and how its anneal was split.
#[doc(hidden)]
#[derive(Debug)]
pub struct TunedRun {
    /// The reads, as [`sample_ising`] returns them.
    pub reads: Vec<SamplerResult>,
    /// The command buffers, in order.
    pub plan: Vec<Chunk>,
    /// GPU time of the longest command buffer, microseconds.
    pub max_chunk_us: u64,
    /// GPU time of every command buffer together, microseconds.
    pub gpu_time_us: u64,
    /// The storage tier the anneal ran on.
    pub storage: crate::capacity::Storage,
}

/// [`sample_ising`] with the batch kernels' test `tuning`, so tests can
/// compare a spilled, global-state, or windowed anneal with a plain one of
/// the same seed.
///
/// # Errors
///
/// The same as [`sample_ising`].
#[doc(hidden)]
pub fn sample_ising_tuned(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
    tuning: Tuning,
) -> Result<TunedRun, SampleError> {
    run_planned(device, graph, params, kernel, None, true, tuning)
}

/// One problem through [`encode_batch_inner`], committed chunk by chunk and
/// harvested. A zero-node graph returns empty reads without the GPU.
fn sample_planned(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
    plan: Option<&[(i32, i32)]>,
    wide_i8: bool,
) -> Result<Vec<SamplerResult>, SampleError> {
    run_planned(
        device,
        graph,
        params,
        kernel,
        plan,
        wide_i8,
        Tuning::default(),
    )
    .map(|run| run.reads)
}

/// [`sample_planned`] with `tuning`, returning the chunk plan and timings.
fn run_planned(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
    plan: Option<&[(i32, i32)]>,
    wide_i8: bool,
    tuning: Tuning,
) -> Result<TunedRun, SampleError> {
    if graph.num_nodes() == 0 {
        let reads = params.num_reads.max(1);
        return Ok(TunedRun {
            reads: (0..reads)
                .map(|_| SamplerResult {
                    spins: vec![],
                    energy_milli: 0,
                })
                .collect(),
            plan: Vec::new(),
            max_chunk_us: 0,
            gpu_time_us: 0,
            storage: crate::capacity::Storage::Threadgroup,
        });
    }

    let mut batch = encode_batch_storage(
        device,
        &[graph],
        params,
        kernel,
        1,
        plan,
        None,
        wide_i8,
        tuning,
        DenseMode::Auto,
    )?;
    while batch.commit_next(|| false) {
        batch.wait_until_completed();
    }

    // A GPU-side failure (device reset, kernel fault, timeout) leaves d_samples
    // in its allocated-zero state; without this check the unpack would turn
    // those zeros into an all-`+1` config and score it as a real solution.
    if let Some(status) = batch.failed_status() {
        return Err(SampleError::Driver(format!(
            "metal command buffer did not complete: status {status:?}"
        )));
    }

    let mut per_problem = harvest_batch(&batch, &[graph])?;
    let mut samples = per_problem.remove(0);
    // The dispatch computes `simd_rounded_reads(num_reads)` samples to fill the
    // final simdgroup; the caller asked for `num_reads`. Truncate here as
    // `streaming::finish_batch` does on the batched path, so neither path can
    // return more solutions than the job requested.
    samples.truncate(params.num_reads.max(1));
    Ok(TunedRun {
        reads: samples,
        plan: batch.plan.clone(),
        max_chunk_us: batch.max_chunk_us(),
        gpu_time_us: batch.gpu_time_us(),
        storage: batch.storage,
    })
}

fn read_i8_buffer(buf: &metal::Buffer, count: usize) -> Result<Vec<i8>, SampleError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let len = buf.length() as usize;
    if len < count {
        // The `copy_nonoverlapping` below reads `count` bytes out of this
        // buffer; a short buffer would read past its allocation (UB). Sizing
        // is consistent today (allocation and harvest count share
        // `packed_size`), so this only fires on drift.
        return Err(SampleError::Driver(format!(
            "Metal buffer holds {len} bytes, need {count}"
        )));
    }
    let ptr = buf.contents() as *const i8;
    if ptr.is_null() {
        return Err(SampleError::Driver(
            "Metal buffer contents() returned null".into(),
        ));
    }
    let mut out = vec![0i8; count];
    // SAFETY: buffer is StorageModeShared; `contents()` is non-null and
    // `length() >= count` are both *checked* above (not merely assumed); `out`
    // owns `count` initialized bytes and cannot overlap the device allocation;
    // and the command buffer that wrote it completed before this call.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
    }
    Ok(out)
}

fn read_i32_buffer(buf: &metal::Buffer, count: usize) -> Result<Vec<i32>, SampleError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let len = buf.length() as usize;
    if len < count * 4 {
        return Err(SampleError::Driver(format!(
            "Metal buffer holds {len} bytes, need {}",
            count * 4
        )));
    }
    let ptr = buf.contents() as *const i32;
    if ptr.is_null() {
        return Err(SampleError::Driver(
            "Metal buffer contents() returned null".into(),
        ));
    }
    let mut out = vec![0i32; count];
    // SAFETY: buffer is StorageModeShared; `contents()` is non-null and
    // `length() >= count * 4` are checked above. Metal buffers are aligned
    // for i32; `out` owns `count` initialized i32 values and cannot overlap
    // the device allocation. The writing command buffer completed before this call.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
    }
    Ok(out)
}

// Host-side (GPU-free) logic only: everything under test here is `cfg(macos)`,
// so the module carries the same gate. The dispatch itself is covered by
// `tests/stream_contract.rs`, which needs a real device.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::Representation;

    #[test]
    fn packed_decode_preserves_jobs_reads_and_host_scoring() {
        let a = IsingGraph::new(vec![1.0, -1.0, 0.0], vec![], vec![]);
        let b = IsingGraph::new(vec![-1.0, 1.0, 0.0], vec![], vec![]);
        let graphs = [&a, &b];
        let packed = [0, 5, 7, 2];
        let expected = [[1, 1, 1], [-1, 1, -1], [-1, -1, -1], [1, -1, 1]];
        let energies = [0, -2000, 0, -2000];
        for device_energies in [Some(energies.as_slice()), None] {
            let results = decode_packed_reads(&packed, device_energies, &graphs, 2, 1, 3).unwrap();
            assert_eq!(results.len(), 2);
            for (p, reads) in results.iter().enumerate() {
                assert_eq!(reads.len(), 2);
                for (r, read) in reads.iter().enumerate() {
                    assert_eq!(read.spins, expected[p * 2 + r]);
                    assert_eq!(read.energy_milli, i64::from(energies[p * 2 + r]));
                }
            }
        }
    }

    #[test]
    fn four_color_is_on_unless_disabled() {
        assert!(four_color_from(None));
        assert!(four_color_from(Some("1")));
        assert!(!four_color_from(Some("0")));
        assert!(!four_color_from(Some("false")));
        assert!(!four_color_from(Some("OFF")));
    }

    #[test]
    fn device_energy_is_exact_only_for_whole_i8_coefficients() {
        let unit = IsingGraph::new(vec![1.0, -1.0, 0.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]);
        assert!(device_energy_exact(&unit));
        let half = IsingGraph::new(vec![0.5, 0.0, 0.0], vec![1.0, 1.0], vec![(0, 1), (1, 2)]);
        assert!(!device_energy_exact(&half));
        let big = IsingGraph::new(vec![0.0; 3], vec![200.0, 1.0], vec![(0, 1), (1, 2)]);
        assert!(!device_energy_exact(&big));
        let nan = IsingGraph::new(
            vec![f64::NAN, 0.0, 0.0],
            vec![1.0, 1.0],
            vec![(0, 1), (1, 2)],
        );
        assert!(!device_energy_exact(&nan));
        // 127 units per node: 16,909 nodes stay under i32::MAX / 1000 units,
        // 16,910 nodes would let the kernel's int sum wrap.
        let fits = IsingGraph::new(vec![127.0; 16_909], vec![], vec![]);
        assert!(device_energy_exact(&fits));
        let wraps = IsingGraph::new(vec![127.0; 16_910], vec![], vec![]);
        assert!(!device_energy_exact(&wraps));
    }

    #[test]
    fn msa_device_energies_equal_consensus_energy() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        // 50 reads: not a whole number of 32-lane words.
        for (graph, sweeps) in [(chain(64), 32), (ring(), 256), (chain(300), 1)] {
            let mut p = params(sweeps);
            p.num_reads = 50;
            let results = sample_ising(&dev, &graph, &p, Kernel::Msa).unwrap();
            assert_eq!(results.len(), 50);
            for r in &results {
                let want = energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges);
                assert_eq!(r.energy_milli, want);
            }
        }
    }

    /// Largest command buffer of Float32 MSA batches past one row chunk.
    /// `docs/perf/2026-11-11-float-kernels.md` records a run:
    /// `cargo test --release --lib float_msa_high_degree_chunks -- --ignored --nocapture`.
    #[test]
    #[ignore = "device timing; see the doc comment"]
    #[expect(clippy::print_stdout, reason = "report the measured chunk times")]
    fn float_msa_high_degree_chunks_stay_under_the_ceiling() {
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let portfolio = crate::model_file::read_model(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/portfolio/family-b-draw000-factor.qmodel"),
        )
        .unwrap()
        .graph;
        // A circulant of degree 62 on MSA_MAX_NODES nodes: four row chunks.
        let n = MSA_MAX_NODES;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| (1..=31).map(move |k| (u, (u + k) % n)))
            .collect();
        let circulant = |seed: usize| {
            let j = (0..edges.len())
                .map(|e| {
                    if (e * 7 + seed).is_multiple_of(3) {
                        0.0011
                    } else {
                        -0.0007
                    }
                })
                .collect();
            IsingGraph::new(vec![0.0003; n], j, edges.clone())
        };
        let circulants: Vec<IsingGraph> = (0..8).map(circulant).collect();
        let params = SampleParams {
            num_reads: 128,
            num_sweeps: 8192,
            sweeps_per_beta: 128,
            seed: 42,
            ..Default::default()
        };
        let batches: [(&str, Vec<&IsingGraph>); 3] = [
            ("portfolio x1", vec![&portfolio]),
            ("circulant x1", vec![&circulants[0]]),
            ("circulant x8", circulants.iter().collect()),
        ];
        for (name, graphs) in batches {
            let degree = max_csr_degree(graphs[0]);
            let mut batch =
                encode_batch_inner(&dev, &graphs, &params, Kernel::Msa, 1, None, None).unwrap();
            let chunks = batch.chunk_count();
            while batch.commit_next(|| false) {
                batch.wait_until_completed();
            }
            assert!(batch.failed_status().is_none(), "{name}");
            let max_ms = batch.max_chunk_us() as f64 / 1000.0;
            let total_ms = batch.gpu_time_us() as f64 / 1000.0;
            println!(
                "{{\"batch\":\"{name}\",\"degree\":{degree},\"chunks\":{chunks},\"max_chunk_ms\":{max_ms:.1},\"total_gpu_ms\":{total_ms:.1}}}"
            );
            assert!(max_ms < TARGET_DISPATCH_MS, "{name}: {max_ms} ms");
        }
    }

    /// Largest command buffer of milli MSA batches past one row chunk: a
    /// weighted and a unit circulant of degree 62 on MSA_MAX_NODES nodes,
    /// which take the dense weighted pipeline on `int` storage, beside a
    /// Float32 circulant of the same shape for reference. At 128 sweeps per
    /// beta the plan is one rung per chunk, its floor, so the kernel sets the
    /// chunk time; at one sweep per beta the plan groups rungs, so
    /// [`WEIGHTED_MSA_DENSE_COST_FACTOR`] sets it. Run:
    /// `cargo test --release --lib weighted_msa_high_degree_chunks -- --ignored --nocapture`.
    #[test]
    #[ignore = "device timing; see the doc comment"]
    #[expect(clippy::print_stdout, reason = "report the measured chunk times")]
    fn weighted_msa_high_degree_chunks_stay_under_the_ceiling() {
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let n = MSA_MAX_NODES;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| (1..=31).map(move |k| (u, (u + k) % n)))
            .collect();
        let circulant = |seed: usize, values: [f64; 3]| {
            let [plus, minus, h] = values;
            let j = (0..edges.len())
                .map(|e| {
                    if (e * 7 + seed).is_multiple_of(3) {
                        plus
                    } else {
                        minus
                    }
                })
                .collect();
            IsingGraph::new(vec![h; n], j, edges.clone())
        };
        let family = |values: [f64; 3]| -> Vec<IsingGraph> {
            (0..8).map(|seed| circulant(seed, values)).collect()
        };
        let weighted = family([1.5, -0.75, 0.25]);
        let unit = family([1.0, -1.0, 0.0]);
        let float = family([0.0011, -0.0007, 0.0003]);
        let batches: [(&str, Vec<&IsingGraph>); 5] = [
            ("weighted x1", vec![&weighted[0]]),
            ("weighted x8", weighted.iter().collect()),
            ("unit x1", vec![&unit[0]]),
            ("unit x8", unit.iter().collect()),
            ("float x8", float.iter().collect()),
        ];
        for sweeps_per_beta in [128, 1] {
            let params = SampleParams {
                num_reads: 128,
                num_sweeps: 8192,
                sweeps_per_beta,
                seed: 42,
                ..Default::default()
            };
            for (name, graphs) in &batches {
                let mut batch =
                    encode_batch_inner(&dev, graphs, &params, Kernel::Msa, 1, None, None).unwrap();
                assert!(batch.coef_scale.is_none(), "{name}: int storage");
                let chunks = batch.chunk_count();
                while batch.commit_next(|| false) {
                    batch.wait_until_completed();
                }
                assert!(batch.failed_status().is_none(), "{name}");
                let max_ms = batch.max_chunk_us() as f64 / 1000.0;
                let total_ms = batch.gpu_time_us() as f64 / 1000.0;
                println!(
                    "{{\"batch\":\"{name}\",\"sweeps_per_beta\":{sweeps_per_beta},\"chunks\":{chunks},\"max_chunk_ms\":{max_ms:.1},\"total_gpu_ms\":{total_ms:.1}}}"
                );
                assert!(max_ms < TARGET_DISPATCH_MS, "{name}: {max_ms} ms");
            }
        }
    }

    #[test]
    fn msa_non_unit_graph_falls_back_to_host_scoring() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let mut graph = chain(64);
        graph.h[3] = 0.5;
        let results = sample_ising(&dev, &graph, &params(32), Kernel::Msa).unwrap();
        for r in &results {
            assert_eq!(
                r.energy_milli,
                energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }

    #[test]
    fn msa_chunked_run_writes_energies_on_the_last_chunk_only() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = chain(64);
        let p = params(64);
        let graphs = [&graph];
        let mut batch = encode_batch_inner(
            &dev,
            &graphs,
            &p,
            Kernel::Msa,
            1,
            Some(&[(0, 20), (20, 44)]),
            None,
        )
        .unwrap();
        while batch.commit_next(|| false) {
            batch.wait_until_completed();
        }
        assert!(batch.failed_status().is_none());
        let results = harvest_batch(&batch, &graphs).unwrap();
        for r in &results[0] {
            assert_eq!(
                r.energy_milli,
                energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }

    /// Small ring: 0-1-2-3-0, unit J, ternary h (same fixture as `topology`).
    fn ring() -> IsingGraph {
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    fn params(num_sweeps: usize) -> SampleParams {
        SampleParams {
            num_sweeps,
            ..Default::default()
        }
    }

    fn driver_msg(err: SampleError) -> String {
        match err {
            SampleError::Driver(m) => m,
            other => panic!("expected Driver, got {other:?}"),
        }
    }

    /// Message of a capacity refusal, asserting the variant on the way — the
    /// variant is what picks the harness condition (`Capacity`, not
    /// `DeviceFault`), so a size gate degrading to `Driver` is a protocol
    /// regression: `Capacity` is a per-job reject, `DeviceFault` ends the
    /// session.
    fn too_large_msg(err: SampleError) -> String {
        match err {
            SampleError::TooLarge(m) => m,
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn unpack_spins_is_lsb_first_and_one_means_minus() {
        // 0b0000_0101 -> bits 0 and 2 set -> spins 0 and 2 are -1.
        let spins = unpack_spins(&[0b0000_0101i8], 8);
        assert_eq!(spins, vec![-1, 1, -1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn unpack_spins_reads_the_high_bit_of_a_byte() {
        // 0b1000_0000 is -128 as i8; the cast back to u8 must recover bit 7.
        let spins = unpack_spins(&[-128i8], 8);
        assert_eq!(spins, vec![1, 1, 1, 1, 1, 1, 1, -1]);
    }

    #[test]
    fn unpack_spins_crosses_byte_boundaries() {
        // Byte 0 = 0, byte 1 = bit 0 set -> node 8 is -1, nothing else.
        let spins = unpack_spins(&[0i8, 1i8], 12);
        assert_eq!(spins[..8], [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(spins[8..], [-1, 1, 1, 1]);
    }

    #[test]
    fn unpack_spins_all_ones_byte_is_all_minus() {
        assert_eq!(unpack_spins(&[-1i8], 8), vec![-1i8; 8]);
    }

    #[test]
    fn unpack_spins_tolerates_a_short_buffer() {
        // Nodes past the slice read as bit 0 -> +1, rather than panicking.
        let spins = unpack_spins(&[0b0000_0011i8], 16);
        assert_eq!(spins.len(), 16);
        assert_eq!(spins[..2], [-1, -1]);
        assert!(spins[2..].iter().all(|&s| s == 1));
    }

    #[test]
    fn unpack_spins_of_zero_nodes_is_empty() {
        assert!(unpack_spins(&[], 0).is_empty());
    }

    #[test]
    fn a_linear_schedule_is_evenly_spaced() {
        let (sched, _) = build_beta_schedule(&ring(), 5, 1, Some((1.0, 5.0)), BetaSchedule::Linear);
        assert_eq!(sched, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// `--solve` seeds that differ only in their high 32 bits used to give the
    /// same kernel seed, and so the same samples.
    #[test]
    fn the_kernel_seed_depends_on_every_bit_of_the_host_seed() {
        let base = 0x0000_0000_dead_beef_u64;
        assert_eq!(kernel_seed(base), kernel_seed(base), "deterministic");
        for bit in 0..64 {
            assert_ne!(
                kernel_seed(base),
                kernel_seed(base ^ (1 << bit)),
                "flipping bit {bit} must change the kernel seed"
            );
        }
    }

    #[test]
    fn beta_schedule_length_is_sweeps_over_sweeps_per_beta() {
        let (sched, sweeps_per) =
            build_beta_schedule(&ring(), 100, 10, Some((0.1, 10.0)), BetaSchedule::Geometric);
        assert_eq!(sweeps_per, 10);
        assert_eq!(sched.len(), 10);
    }

    #[test]
    fn wide_gibbs_group_is_a_device_independent_constant() {
        assert_eq!(WIDE_GIBBS_GROUP, 128);
        assert_eq!(wide_sa_group(64, 1024), 64);
        assert_eq!(wide_sa_group(256, 96), 96);
        assert_eq!(wide_sa_group(256, 40), 32);
    }

    #[test]
    fn a_wide_schedule_past_the_table_cap_is_refused() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = IsingGraph::new(vec![0.25, 0.0], vec![1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 8193,
            sweeps_per_beta: 1,
            ..Default::default()
        };
        let err = encode_batch(&dev, &[&graph], &params, Kernel::Sa, 1)
            .err()
            .unwrap();
        assert!(matches!(err, SampleError::TooLarge(_)), "{err:?}");
    }

    #[test]
    fn beta_schedule_runs_hot_to_cold() {
        let (sched, _) =
            build_beta_schedule(&ring(), 64, 1, Some((0.1, 10.0)), BetaSchedule::Geometric);
        assert_eq!(sched.len(), 64);
        assert!((sched[0] - 0.1).abs() < 1e-6, "starts hot: {}", sched[0]);
        let last = sched[sched.len() - 1];
        assert!((last - 10.0).abs() < 1e-4, "ends cold: {last}");
        assert!(sched.windows(2).all(|w| w[1] >= w[0]), "monotonic");
    }

    #[test]
    fn beta_schedule_floors_at_one_beta() {
        // Zero sweeps and zero sweeps-per-beta must not divide by zero or
        // produce an empty ladder the kernel would read as num_betas = 0.
        let (sched, sweeps_per) =
            build_beta_schedule(&ring(), 0, 0, Some((0.1, 10.0)), BetaSchedule::Geometric);
        assert_eq!(sweeps_per, 1);
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn beta_schedule_at_the_sweep_cap_is_bounded() {
        // The whole point of MAX_SWEEPS: the largest accepted job still
        // allocates a schedule of a size we can name.
        let (sched, _) = build_beta_schedule(
            &ring(),
            MAX_SWEEPS,
            1,
            Some((0.1, 10.0)),
            BetaSchedule::Geometric,
        );
        assert_eq!(sched.len(), MAX_SWEEPS);
    }

    #[test]
    fn validate_batch_rejects_an_empty_batch() {
        // Release builds compiled the old `debug_assert!` out and panicked on
        // `graphs[0]`.
        let err = validate_batch(&[], &params(64), Kernel::Sa).unwrap_err();
        assert!(driver_msg(err).contains("at least one graph"));
    }

    /// A complete Int32 model on `n` nodes: couplings 1.3 and -0.127 have
    /// milli gcd 1, so no `int8_t` scale holds them.
    fn complete_int32(n: usize) -> IsingGraph {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .collect();
        let j = (0..edges.len())
            .map(|k| if k % 2 == 0 { 1.3 } else { -0.127 })
            .collect();
        IsingGraph::new(vec![0.0; n], j, edges)
    }

    /// One dense-matrix sweep of 4,898 nodes is estimated at 399.84 ms and
    /// one of 4,899 nodes at 400.003 ms, at any read count. The longer sweep is
    /// refused before any buffer exists; the shorter one is admitted dense.
    #[test]
    fn a_dense_sweep_past_one_chunk_is_refused_before_allocation() {
        assert!(matrix_sweep_ms(Kernel::Msa, 4_898) <= MSA_MATRIX_MAX_SWEEP_MS);
        assert!(matrix_sweep_ms(Kernel::Msa, 4_899) > MSA_MATRIX_MAX_SWEEP_MS);
        let p = SampleParams {
            num_reads: 256,
            num_sweeps: 1,
            ..Default::default()
        };
        let below = complete_int32(4_898);
        let admitted = validate_batch_within(
            &[&below],
            &p,
            Kernel::Msa,
            host_device_limits(),
            true,
            None,
            DenseMode::Auto,
        )
        .unwrap();
        assert_eq!(admitted.storage, Storage::Dense);
        assert_eq!(admitted.rep, Representation::Wide);
        assert!(admitted.i8_scale.is_none());

        // `encode_batch_storage` validates before it touches the device, so
        // a refusal here allocates nothing. Device allocations of the
        // parallel tests make a process-wide byte count unusable as a check.
        let above = complete_int32(4_899);
        let err = validate_batch_within(
            &[&above],
            &p,
            Kernel::Msa,
            host_device_limits(),
            true,
            None,
            DenseMode::Auto,
        )
        .unwrap_err();
        assert_eq!(
            err.to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        let message = too_large_msg(err);
        assert!(message.contains("one dense-matrix sweep"), "{message}");
        assert!(message.contains("400.003 ms"), "{message}");
        // A forced dense run is refused the same way.
        let err = validate_batch_within(
            &[&above],
            &p,
            Kernel::Msa,
            host_device_limits(),
            true,
            None,
            DenseMode::Forced,
        )
        .unwrap_err();
        assert!(too_large_msg(err).contains("one dense-matrix sweep"));
    }

    /// One dense-matrix SA sweep of 1,414 nodes is estimated at 399.88 ms and
    /// one of 1,415 nodes at 400.445 ms, at any read count. The longer sweep
    /// is refused before any buffer exists, under Auto and Forced; the shorter
    /// one is admitted dense.
    #[test]
    fn a_dense_sa_sweep_past_one_chunk_is_refused_before_allocation() {
        assert!(matrix_sweep_ms(Kernel::Sa, 1_414) <= MSA_MATRIX_MAX_SWEEP_MS);
        assert!(matrix_sweep_ms(Kernel::Sa, 1_415) > MSA_MATRIX_MAX_SWEEP_MS);
        let p = SampleParams {
            num_reads: 256,
            num_sweeps: 1,
            ..Default::default()
        };
        let admit = |graph: &'static IsingGraph, dense| {
            validate_batch_within(
                &[graph],
                &p,
                Kernel::Sa,
                host_device_limits(),
                true,
                None,
                dense,
            )
        };
        let below: &'static IsingGraph = Box::leak(Box::new(complete_int32(1_414)));
        let admitted = admit(below, DenseMode::Auto).unwrap();
        assert_eq!(admitted.storage, Storage::Dense);
        assert_eq!(admitted.rep, Representation::Wide);
        assert!(admitted.i8_scale.is_none());

        // `encode_batch_storage` validates before it touches the device, so
        // a refusal here allocates nothing.
        let above: &'static IsingGraph = Box::leak(Box::new(complete_int32(1_415)));
        // Auto and Forced both refuse it as Capacity.
        for mode in [DenseMode::Auto, DenseMode::Forced] {
            let err = admit(above, mode).unwrap_err();
            assert_eq!(
                err.to_sample_error(),
                quip_solver_core::SampleError::Capacity
            );
            let message = too_large_msg(err);
            assert!(message.contains("one dense-matrix sweep"), "{message}");
            assert!(message.contains("400.445 ms"), "{message}");
        }
        // The CSR gates never take dense storage, so the limit does not
        // reach them.
        assert_ne!(
            admit(above, DenseMode::Off).unwrap().storage,
            Storage::Dense
        );
    }

    #[test]
    fn validate_batch_refuses_before_allocation() {
        // Zero usable bytes refuse every model, so the footprint check is the
        // only possible cause of the refusal.
        let limits = crate::capacity::DeviceLimits {
            memory_bytes: 0,
            tg_memory_bytes: 32_768,
            max_buffer_bytes: 0,
        };
        let device = (crate::metal_device::MetalDevice::device_count() > 0)
            .then(|| crate::metal_device::MetalDevice::open(0).unwrap());
        let before = device.as_ref().map(|d| d.device.current_allocated_size());
        let g = ring();
        let err = validate_batch_within(
            &[&g],
            &params(64),
            Kernel::Sa,
            &limits,
            false,
            None,
            DenseMode::Auto,
        )
        .unwrap_err();
        let SampleError::TooLarge(message) = &err else {
            panic!("expected TooLarge, got {err:?}");
        };
        assert!(message.contains("bytes needed"), "{message}");
        assert!(message.contains("usable"), "{message}");
        assert_eq!(
            err.to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        let after = device.as_ref().map(|d| d.device.current_allocated_size());
        assert_eq!(before, after);
    }

    #[test]
    fn a_failed_buffer_allocation_is_a_capacity_error() {
        let err = require_buffer(None).unwrap_err();
        assert!(matches!(err, SampleError::TooLarge(_)), "{err:?}");
        assert_eq!(
            err.to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
    }

    #[test]
    fn validate_batch_accepts_a_normal_job() {
        let g = ring();
        let (first, n, _) = validate_batch(&[&g], &params(64), Kernel::Sa).unwrap();
        assert_eq!(n, 4);
        assert_eq!(first.num_nodes(), 4);
    }

    #[test]
    fn validate_batch_refuses_a_non_representable_model_as_capacity() {
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 8,
            ..Default::default()
        };
        let graph = IsingGraph::new(vec![f64::NAN, 0.0], vec![1.0], vec![(0, 1)]);
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let err = validate_batch(&[&graph], &params, kernel).unwrap_err();
            assert!(
                matches!(err, SampleError::TooLarge(_)),
                "{kernel:?}: {err:?}"
            );
        }
    }

    #[test]
    fn validate_batch_reports_the_representation() {
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 8,
            ..Default::default()
        };
        let pm1 = IsingGraph::new(vec![1.0, -1.0], vec![-1.0], vec![(0, 1)]);
        let quarter = IsingGraph::new(vec![0.25, 0.0], vec![-0.75], vec![(0, 1)]);
        let (_, _, rep) = validate_batch(&[&pm1], &params, Kernel::Sa).unwrap();
        assert_eq!(rep, Representation::Int8);
        let (_, _, rep) = validate_batch(&[&pm1], &params, Kernel::Msa).unwrap();
        assert_eq!(rep, Representation::Binary);
        let (_, _, rep) = validate_batch(&[&quarter], &params, Kernel::Sa).unwrap();
        assert_eq!(rep, Representation::Wide);
    }

    #[test]
    fn validate_batch_runs_float32_models_on_every_kernel() {
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 8,
            ..Default::default()
        };
        let off_grid = IsingGraph::new(vec![0.1 + 0.2, 0.0], vec![1.0], vec![(0, 1)]);
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let (_, _, rep) = validate_batch(&[&off_grid], &params, kernel).unwrap();
            assert_eq!(rep, Representation::Float32, "{kernel:?}");
        }
    }

    fn too_large(result: Result<(), SampleError>) -> String {
        match result {
            Err(SampleError::TooLarge(m)) => m,
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn a_large_coupling_runs_float32_under_its_default_schedule() {
        let bond = IsingGraph::new(vec![0.0; 2], vec![1e30], vec![(0, 1)]);
        let (beta, _) = build_beta_schedule(&bond, 64, 1, None, BetaSchedule::Geometric);
        check_float_schedule(&bond, &beta).unwrap();
    }

    #[test]
    fn a_large_beta_beside_a_large_coupling_is_admitted() {
        // The kernel skips a flip when `de >= 22.18 / beta`, so `de * beta`
        // stays small wherever `exp` runs. CUDA admits these models too.
        let graph = IsingGraph::new(vec![0.0, 0.0, 1e-12], vec![1e30], vec![(0, 1)]);
        let (beta, _) = build_beta_schedule(&graph, 64, 1, None, BetaSchedule::Geometric);
        check_float_schedule(&graph, &beta).unwrap();
        let bond = IsingGraph::new(vec![0.0; 2], vec![1e30], vec![(0, 1)]);
        check_float_schedule(&bond, &[1.0, 1e9]).unwrap();
    }

    #[test]
    fn a_beta_past_the_f32_range_is_refused() {
        let graph = IsingGraph::new(vec![1.2e-38; 64], vec![], vec![]);
        let (beta, _) = build_beta_schedule(&graph, 64, 1, None, BetaSchedule::Geometric);
        let message = too_large(check_float_schedule(&graph, &beta));
        assert!(message.contains("finite f32"), "{message}");
        too_large(check_float_schedule(&graph, &[1.0, f32::NAN]));
    }

    #[test]
    fn a_beta_past_half_f32_max_is_refused() {
        let graph = IsingGraph::new(vec![1.0; 2], vec![], vec![]);
        let half = f32::MAX / 2.0;
        check_float_schedule(&graph, &[half]).unwrap();
        check_float_schedule(&graph, &[-half]).unwrap();
        let above = f32::from_bits(half.to_bits() + 1);
        for beta in [above, -above] {
            let message = too_large(check_float_schedule(&graph, &[1.0, beta]));
            assert!(message.contains("half of f32::MAX"), "{message}");
        }
    }

    #[test]
    fn a_rounded_field_past_half_f32_max_is_refused() {
        // A single bond of 1e38 has a delta of 2e38, inside f32. Two at one
        // node double past f32::MAX.
        let bond = IsingGraph::new(vec![0.0; 2], vec![1e38], vec![(0, 1)]);
        check_float_schedule(&bond, &[0.0]).unwrap();
        let chain = IsingGraph::new(vec![0.0; 3], vec![1e38; 2], vec![(0, 1), (1, 2)]);
        let message = too_large(check_float_schedule(&chain, &[0.0]));
        assert!(message.contains("delta range"), "{message}");
        // CUDA's rounding case: h just under half of f32::MAX plus three
        // couplings whose exact sum is below half, but whose bound is not.
        let half_max = f32::MAX / 2.0;
        let h = f32::from_bits(half_max.to_bits() - 2);
        let coupling = f64::from((half_max - h) * 0.3);
        let star = IsingGraph::new(
            vec![f64::from(h), 0.0, 0.0, 0.0],
            vec![coupling; 3],
            vec![(0, 1), (0, 2), (0, 3)],
        );
        too_large(check_float_schedule(&star, &[0.0]));
    }

    #[test]
    fn validate_batch_rejects_n_over_an_explicit_cap() {
        let big = IsingGraph::new(vec![0.0; GIBBS_MAX_NODES + 1], vec![], vec![]);
        let err = validate_batch_within(
            &[&big],
            &params(64),
            Kernel::Gibbs,
            host_device_limits(),
            false,
            Some(GIBBS_MAX_NODES),
            DenseMode::Auto,
        )
        .map(drop)
        .unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("exceeds"), "{msg}");
        assert!(msg.contains(&GIBBS_MAX_NODES.to_string()), "{msg}");
    }

    /// The Gibbs spill build takes a model past the threadgroup array, so the
    /// memory footprint bounds it instead of the array.
    #[test]
    fn validate_batch_admits_gibbs_past_the_threadgroup_array() {
        let big = IsingGraph::new(vec![0.0; GIBBS_MAX_NODES + 1], vec![], vec![]);
        let (_, n, _) = validate_batch(&[&big], &params(64), Kernel::Gibbs).unwrap();
        assert_eq!(n, GIBBS_MAX_NODES + 1);
    }

    /// The SA global build takes a model past the thread-private frame, so
    /// the memory footprint bounds it instead of the frame.
    #[test]
    fn validate_batch_admits_sa_past_the_local_frame() {
        let big = IsingGraph::new(vec![0.0; SA_MAX_NODES + 1], vec![], vec![]);
        let (_, n, _) = validate_batch(&[&big], &params(64), Kernel::Sa).unwrap();
        assert_eq!(n, SA_MAX_NODES + 1);
    }

    #[test]
    fn validate_batch_rejects_num_sweeps_over_the_cap() {
        let g = ring();
        let err = validate_batch(&[&g], &params(MAX_SWEEPS + 1), Kernel::Sa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("num_sweeps"), "{msg}");
    }

    #[test]
    fn validate_batch_rejects_the_unbounded_coordinator_value() {
        // The reported DoS: `num_sweeps = u32::MAX` sized a ~34 GB Vec<f64>.
        let g = ring();
        let sweeps = u32::MAX as usize;
        let err = validate_batch(&[&g], &params(sweeps), Kernel::Sa).unwrap_err();
        assert!(too_large_msg(err).contains("num_sweeps"));
    }

    #[test]
    fn to_sample_error_separates_capacity_refusals_from_device_faults() {
        // A capacity refusal is permanent for this backend: the coordinator
        // must re-route, not retry here.
        assert_eq!(
            SampleError::TooLarge("n".into()).to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        // A driver failure is a state this backend cannot recover from on its
        // own — the session must end for a supervisor restart, not reject a
        // job and keep accepting more work a wedged device can never serve.
        assert_eq!(
            SampleError::Driver("device reset".into()).to_sample_error(),
            quip_solver_core::SampleError::DeviceFault("device reset".into())
        );
    }

    #[test]
    fn validate_batch_accepts_the_cap_exactly() {
        let g = ring();
        validate_batch(&[&g], &params(MAX_SWEEPS), Kernel::Sa).unwrap();
    }

    #[test]
    fn sweep_cap_clears_the_gibbs_doubled_adapt_maximum() {
        // `METAL_ADAPT.max_sweeps` is 2048 and quip-solver-core doubles it for
        // Gibbs, so no legitimate adapt-driven job may be rejected. Update
        // this alongside `lib.rs` if either bound moves.
        let adapt_max_sweeps = 2048usize;
        assert!(MAX_SWEEPS >= 2 * adapt_max_sweeps);
        let g = ring();
        let sweeps = 2 * adapt_max_sweeps;
        validate_batch(&[&g], &params(sweeps), Kernel::Gibbs).unwrap();
    }

    #[test]
    fn kernel_max_nodes_matches_the_kernel_arrays() {
        // `delta_energy[4593]` in sa.metal, `shared_state[4800]` and
        // `packed_state[600]` (600*8) in gibbs.metal.
        assert_eq!(kernel_max_nodes(Kernel::Sa), 4593);
        assert_eq!(kernel_max_nodes(Kernel::Gibbs), 4800);
        assert_eq!(kernel_max_nodes(Kernel::Sa), SA_MAX_NODES);
        assert_eq!(kernel_max_nodes(Kernel::Gibbs), GIBBS_MAX_NODES);
        assert_eq!(kernel_max_nodes(Kernel::Msa), MSA_MAX_NODES);
        assert_eq!(kernel_max_nodes(Kernel::Msa), 6016);
    }

    /// A star whose hub has `leaves` neighbours.
    fn star(leaves: usize) -> IsingGraph {
        let n = leaves + 1;
        let edges: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
        IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges)
    }

    #[test]
    fn validate_batch_rejects_msa_degree_over_the_sparse_budget() {
        // The weighted lanes walk at most 62 neighbours; a denser node is
        // capacity, not DeviceFault: the coordinator routes the job to
        // another backend.
        let g = star(MSA_SPARSE_MAX_DEG + 1);
        let err = validate_batch(&[&g], &params(64), Kernel::Msa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("degree 63"), "{msg}");
        // The other kernels walk CSR rows of any length.
        validate_batch(&[&g], &params(64), Kernel::Sa).unwrap();
        validate_batch(&[&g], &params(64), Kernel::Gibbs).unwrap();
    }

    #[test]
    fn validate_batch_accepts_msa_degree_at_the_budget() {
        let g = star(MSA_SPARSE_MAX_DEG);
        validate_batch(&[&g], &params(64), Kernel::Msa).unwrap();
    }

    #[test]
    fn msa_shared_spins_is_at_most_the_threadgroup_cap() {
        assert_eq!(msa_shared_spins(32 * 1024, None), MSA_MAX_NODES);
        assert_eq!(msa_shared_spins(64 * 1024, None), 6016);
        assert_eq!(msa_shared_spins(32 * 1024, Some(1_000)), 1_000);
        assert_eq!(msa_shared_spins(32 * 1024, Some(0)), 0);
        // 16 KB holds (16,384 - 128) / 4 = 4,064 words, a multiple of four.
        assert_eq!(msa_shared_spins(16 * 1024, None), 4_064);
        assert_eq!(msa_shared_spins(0, None), 0);
    }

    #[test]
    fn a_unit_model_past_threadgroup_memory_takes_the_weighted_lane() {
        let unit = Representation::Binary;
        assert_eq!(msa_lane(unit, 20, MSA_MAX_NODES), unit);
        assert_eq!(msa_lane(unit, 20, MSA_MAX_NODES + 1), Representation::Wide);
        assert_eq!(msa_lane(unit, 21, 100), Representation::Wide);
        assert_eq!(
            msa_lane(Representation::Float32, 20, 200_000),
            Representation::Float32
        );
    }

    #[test]
    fn validate_batch_admits_a_weighted_msa_model_past_threadgroup_memory() {
        let n = MSA_MAX_NODES + 1;
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let unit = IsingGraph::new(vec![0.0; n], vec![1.0; n], edges.clone());
        let (_, _, rep) = validate_batch(&[&unit], &params(64), Kernel::Msa).unwrap();
        assert_eq!(rep, Representation::Wide);
        let weighted = IsingGraph::new(vec![0.0; n], vec![1.5; n], edges);
        validate_batch(&[&weighted], &params(64), Kernel::Msa).unwrap();
        // SA and Gibbs run past their frames too, on their global and spill
        // builds.
        validate_batch(&[&unit], &params(64), Kernel::Gibbs).unwrap();
        validate_batch(&[&unit], &params(64), Kernel::Sa).unwrap();
    }

    #[test]
    fn float_msa_plans_by_row_chunks_and_colour_steps() {
        assert_eq!(msa_row_chunks(0), 1);
        assert_eq!(msa_row_chunks(MSA_UNIT_MAX_DEG), 1);
        assert_eq!(msa_row_chunks(43), 3);
        assert_eq!(msa_row_chunks(62), 4);
        // The calibration shape plans by its nodes.
        let advantage2 = [573, 572, 572, 572, 572, 572, 572, 572];
        assert!((msa_colour_step_factor(&advantage2, 1024, 4577) - 1.0).abs() < 1e-3);
        // A complete graph on 44 nodes: 44 one-node classes, 44 steps.
        let factor = msa_colour_step_factor(&[1; 44], 1024, 44);
        assert!((factor - MSA_CURVE_NODES_PER_STEP).abs() < 1e-9, "{factor}");
    }

    #[test]
    fn the_weighted_lanes_admit_msa_degree_up_to_the_sparse_limit() {
        let scaled = |leaves: usize, j: f64| {
            let edges: Vec<(usize, usize)> = (1..=leaves).map(|i| (0, i)).collect();
            IsingGraph::new(vec![0.0; leaves + 1], vec![j; leaves], edges)
        };
        let budget = MSA_SPARSE_MAX_DEG;
        for (j, want) in [
            (0.0005, Representation::Float32),
            (1.5, Representation::Wide),
        ] {
            let (_, _, rep) =
                validate_batch(&[&scaled(budget, j)], &params(64), Kernel::Msa).unwrap();
            assert_eq!(rep, want);
            let msg = too_large_msg(
                validate_batch(&[&scaled(budget + 1, j)], &params(64), Kernel::Msa).unwrap_err(),
            );
            assert!(msg.contains(&format!("degree {}", budget + 1)), "{msg}");
        }
    }

    #[test]
    fn msa_runs_denser_unit_models_on_the_weighted_lane() {
        let (_, _, rep) = validate_batch(&[&star(20)], &params(64), Kernel::Msa).unwrap();
        assert_eq!(rep, Representation::Binary);
        for leaves in [21, 62] {
            let (_, _, rep) = validate_batch(&[&star(leaves)], &params(64), Kernel::Msa).unwrap();
            assert_eq!(rep, Representation::Wide, "star of {leaves}");
        }
        let msg =
            too_large_msg(validate_batch(&[&star(63)], &params(64), Kernel::Msa).unwrap_err());
        assert!(msg.contains("degree 63"), "{msg}");
    }

    /// K_40 has degree 39, two row chunks. Integer fields are exact in any
    /// order, and every carry-over resumes from the persistent buffers, so a
    /// split plan returns the reads of one whole chunk. Both models fit
    /// `int8_t` storage, but a dense batch takes `int` storage whatever the
    /// request.
    #[test]
    fn the_weighted_lane_resumes_rows_past_one_chunk_without_changing_reads() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let n = 40;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .collect();
        let weighted = IsingGraph::new(
            (0..n)
                .map(|i| f64::from((i % 3) as i32 - 1) * 0.5)
                .collect(),
            edges
                .iter()
                .map(|&(u, v)| f64::from(((u * 7 + v * 3) % 5) as i32 - 2) * 0.75)
                .collect(),
            edges.clone(),
        );
        let unit = IsingGraph::new(
            vec![0.0; n],
            edges
                .iter()
                .map(|&(u, v)| if (u + v) % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            edges,
        );
        let p = SampleParams {
            num_reads: 64,
            num_sweeps: 256,
            seed: 7,
            ..Default::default()
        };
        let whole = [(0, 256)];
        let split = [(0, 100), (100, 156)];
        for graph in [&weighted, &unit] {
            let (_, _, rep) = validate_batch(&[graph], &p, Kernel::Msa).unwrap();
            assert_eq!(rep, Representation::Wide);
            assert!(batch_wide_i8_scale(&[graph]).is_some());
            let batch = encode_batch_storage(
                &dev,
                &[graph],
                &p,
                Kernel::Msa,
                1,
                None,
                None,
                true,
                Tuning::default(),
                DenseMode::Auto,
            )
            .unwrap();
            assert!(
                batch.coef_scale.is_none(),
                "a dense batch takes int storage"
            );
            let mut storages = Vec::new();
            for wide_i8 in [true, false] {
                let a =
                    sample_planned(&dev, graph, &p, Kernel::Msa, Some(&whole), wide_i8).unwrap();
                let b =
                    sample_planned(&dev, graph, &p, Kernel::Msa, Some(&split), wide_i8).unwrap();
                assert_eq!(a, b, "wide_i8 {wide_i8}");
                for r in &a {
                    assert_eq!(
                        r.energy_milli,
                        energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
                    );
                }
                storages.push(a);
            }
            assert_eq!(storages[0], storages[1], "both requests run int storage");
        }
    }

    #[test]
    fn max_csr_degree_follows_the_topology_builder_rules() {
        // A self-loop counts once; an out-of-range endpoint skips the edge.
        let g = IsingGraph::new(
            vec![0.0; 3],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![(0, 1), (1, 1), (1, 2), (2, 9)],
        );
        assert_eq!(max_csr_degree(&g), 3);
        let t = SelfFeedingTopology::build(&g);
        let builder_max = t
            .row_ptr
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap();
        assert_eq!(max_csr_degree(&g), builder_max);
    }

    #[test]
    fn msa_static_threadgroup_budget_is_the_lane_totals_only() {
        assert_eq!(MSA_STATIC_TG_BYTES, 32 * 4);
    }

    #[test]
    fn msa_node_cap_fits_apple_threadgroup_memory() {
        assert_eq!(kernel_max_nodes(Kernel::Msa), MSA_MAX_NODES);
        const {
            assert!(MSA_MAX_NODES * 4 + MSA_STATIC_TG_BYTES <= APPLE_TG_MEMORY_BYTES);
            // Larger than either existing cap: the kernel holds 4 bytes per spin
            // where SA holds a delta-energy byte plus a packed bit per spin per
            // thread.
            assert!(MSA_MAX_NODES > SA_MAX_NODES);
            assert!(MSA_MAX_NODES > GIBBS_MAX_NODES);
        }
    }

    #[test]
    fn simd_rounded_reads_are_whole_msa_words() {
        for reads in [1usize, 31, 32, 33, 64, 100, 255, 256, 1000] {
            let r = simd_rounded_reads(reads);
            assert_eq!(r % MSA_LANES, 0, "reads={reads} rounded to {r}");
            assert!((1..=MAX_READS / MSA_LANES).contains(&(r / MSA_LANES)));
        }
    }

    /// Ferromagnetic chain of `n` spins with ternary fields, so every degree
    /// (1 at the ends, 2 inside) and every field value gets exercised.
    fn chain(n: usize) -> IsingGraph {
        let h = (0..n).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let j = vec![-1.0; edges.len()];
        IsingGraph::new(h, j, edges)
    }

    /// 48 nodes on a ring with chords. Every coefficient is a multiple of 250
    /// milli with quotients up to 100, so the model is wide (not the int8
    /// tier) and `int8_t` storage takes it at scale 250.
    fn scaled_i8_model() -> IsingGraph {
        let n = 48;
        let mut edges: Vec<(usize, usize)> = (0..n).map(|v| (v, (v + 1) % n)).collect();
        edges.extend((0..n).map(|v| (v, (v + 7) % n)).filter(|&(u, v)| u < v));
        let j = (0..edges.len())
            .map(|k| f64::from(((k * 37) % 201) as i32 - 100) * 0.25)
            .collect();
        let h = (0..n)
            .map(|i| f64::from(((i * 11) % 9) as i32 - 4) * 0.25)
            .collect();
        let mut graph = IsingGraph::new(h, j, edges);
        graph.j[0] = 25.0; // 100 units of 250 milli
        graph.j[1] = -0.25;
        graph
    }

    #[test]
    fn a_scaled_i8_model_matches_the_wide_path() {
        let graph = scaled_i8_model();
        assert_eq!(classify(&graph), Ok(Representation::Wide));
        assert_eq!(
            batch_i8_scale(&[&graph], Representation::Wide, true),
            Some(250)
        );
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let p = SampleParams {
            num_reads: 64,
            num_sweeps: 256,
            seed: 7,
            ..Default::default()
        };
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            // The i8 side binds the scale and the i8 pipeline; the forced
            // side binds neither.
            let narrow = encode_batch_storage(
                &dev,
                &[&graph],
                &p,
                kernel,
                1,
                None,
                None,
                true,
                Tuning::default(),
                DenseMode::Auto,
            )
            .unwrap();
            let plain = encode_batch_storage(
                &dev,
                &[&graph],
                &p,
                kernel,
                1,
                None,
                None,
                false,
                Tuning::default(),
                DenseMode::Auto,
            )
            .unwrap();
            assert_eq!(
                narrow.coef_scale.map(|(_, scale)| scale),
                Some(250),
                "{kernel:?}"
            );
            assert!(plain.coef_scale.is_none(), "{kernel:?}");
            let plan = [(0, 100), (100, 156)];
            for plan in [None, Some(&plan[..])] {
                let i8_reads = sample_planned(&dev, &graph, &p, kernel, plan, true).unwrap();
                let wide_reads = sample_planned(&dev, &graph, &p, kernel, plan, false).unwrap();
                assert_eq!(i8_reads, wide_reads, "{kernel:?} plan {plan:?}");
            }
        }
    }

    #[test]
    fn topology_cache_hits_on_the_same_edges_and_misses_on_new_ones() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let a = chain(50);
        let mut a2 = chain(50);
        a2.j[0] = -a2.j[0]; // same topology, different couplings
        let b = ring();
        let first = dev.topology_cache.get_or_build(&dev, &a, false).unwrap();
        let again = dev.topology_cache.get_or_build(&dev, &a2, false).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        // Same node count, one edge moved: the cache must rebuild.
        let mut moved = chain(50);
        let mut edges = moved.edges.to_vec();
        edges[0] = (0, 2);
        moved.edges = edges.into();
        let rewired = dev
            .topology_cache
            .get_or_build(&dev, &moved, false)
            .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &rewired));
        let other = dev.topology_cache.get_or_build(&dev, &b, false).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        assert_eq!(other.n, b.num_nodes());
        let colored = dev.topology_cache.get_or_build(&dev, &b, true).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&other, &colored));
    }

    #[test]
    fn buffer_pool_reuses_a_returned_buffer_of_fitting_size() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let pool = BufferPool::default();
        let a = pool.take(&dev.device, 4096).unwrap();
        let ptr = a.contents();
        pool.give(a);
        let b = pool.take(&dev.device, 3000).unwrap();
        assert_eq!(b.contents(), ptr);
        let c = pool.take(&dev.device, 100_000).unwrap();
        assert!(c.length() >= 100_000);
    }

    #[test]
    fn consecutive_batches_on_one_topology_give_consensus_energies() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        for graph in [chain(80), chain(80), ring(), chain(80)] {
            let results = sample_ising(&dev, &graph, &params(32), Kernel::Msa).unwrap();
            for r in &results {
                assert_eq!(
                    r.energy_milli,
                    energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
                );
            }
        }
    }

    #[test]
    fn preparing_a_batch_does_not_submit_its_chunks() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = chain(32);
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 4,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: Some((0.1, 1.0)),
            seed: 7,
        };
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let batch = encode_batch_inner(
                &device,
                &[&graph],
                &params,
                kernel,
                1,
                Some(&[(0, 1), (1, 1), (2, 1), (3, 1)]),
                None,
            )
            .unwrap();
            batch.wait_until_completed();
            assert!(
                batch.cmds.is_empty(),
                "{kernel:?} submitted work before a cancellation checkpoint"
            );
        }
    }

    #[test]
    fn cancellation_stops_after_one_committed_chunk_for_every_kernel() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = chain(32);
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 128,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: Some((0.1, 1.0)),
            seed: 7,
        };
        let plan: Vec<_> = (0..128).map(|start| (start, 1)).collect();
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let mut batch =
                encode_batch_inner(&device, &[&graph], &params, kernel, 2, Some(&plan), None)
                    .unwrap();
            let cancel = quip_solver_core::CancelToken::default();
            assert!(batch.commit_next(|| cancel.is_cancelled(Some(1))));
            cancel.cancel_through(1);
            batch.wait_until_completed();
            assert!(!batch.commit_next(|| cancel.is_cancelled(Some(1))));
            assert_eq!(batch.cmds.len(), 1, "{kernel:?} ran cancelled chunks");
            assert!(batch.failed_status().is_none());
        }
    }

    #[test]
    fn cancellation_during_encoding_prevents_commit() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = chain(32);
        let params = SampleParams {
            num_reads: 32,
            num_sweeps: 4,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: Some((0.1, 1.0)),
            seed: 7,
        };
        let mut batch = encode_batch(&device, &[&graph], &params, Kernel::Msa, 2).unwrap();
        let mut checks = 0;
        assert!(!batch.commit_next(|| {
            checks += 1;
            checks > 1
        }));
        assert!(batch.cmds.is_empty());
    }

    #[test]
    fn overlapping_saturated_batches_share_the_chunk_budget() {
        let cores = crate::iokit_gov::gpu_core_count().unwrap_or(10).max(1);
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let dims = BatchDims {
                n: 4577,
                num_betas: 16384,
                sweeps_per: 1,
                base_seed: 1,
                num_threads: cores * 128,
                num_problems: cores,
                num_reads: 128,
                packed_size: 573,
            };
            let groups = cores * 1024;
            let solo = chunk_plan(kernel, &dims, groups, 1, 1.0, false, None)[0].beta_count;
            let shared = chunk_plan(kernel, &dims, groups, 2, 1.0, false, None)[0].beta_count;
            assert_eq!(shared, (solo / 2).max(1), "{kernel:?}");
        }
    }

    #[test]
    fn resident_step_limit_matches_two_in_flight_batch_chunks() {
        for n in [64, 4577, MSA_MAX_NODES] {
            for groups in [1, 2, 20, 80, 512] {
                for slice in [1, 32, MAX_SWEEPS, usize::MAX] {
                    let dims = BatchDims {
                        n,
                        num_betas: slice.min(MAX_SWEEPS) as i32,
                        sweeps_per: 1,
                        base_seed: 1,
                        num_threads: groups,
                        num_problems: groups,
                        num_reads: MSA_LANES,
                        packed_size: n.div_ceil(8),
                    };
                    let limit = msa_step_limit(n, groups, slice);
                    assert_eq!(
                        limit,
                        chunk_plan(Kernel::Msa, &dims, groups, 2, 1.0, false, None)[0].beta_count
                            as usize
                    );
                    assert!((1..=slice.min(MAX_SWEEPS)).contains(&limit));
                    let work = limit as f64 * groups as f64 * n as f64;
                    let budget = TARGET_DISPATCH_MS / 1000.0
                        * estimated_updates_per_sec(Kernel::Msa, groups * 2)
                        / 2.0;
                    assert!(work <= budget || limit == 1);
                }
            }
        }
        assert!(msa_step_limit(4577, 80, MAX_SWEEPS) < MAX_SWEEPS);
    }

    /// Chunk boundaries fall on rung boundaries, and every carry-over (the
    /// spin words and each thread's RNG stream) round-trips through the
    /// persistent buffers, so splitting an anneal into chunks must not change
    /// one bit of output. This is the resume path's only direct test: small
    /// graphs never plan more than one chunk on their own.
    #[test]
    fn all_kernels_resume_beyond_the_command_queue_limit_without_changing_samples() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = chain(96);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 128,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: Some((0.1, 4.0)),
            seed: 7,
        };
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let mut whole = encode_batch_inner(
                &device,
                &[&graph],
                &params,
                kernel,
                1,
                Some(&[(0, 128)]),
                None,
            )
            .unwrap();
            let plan: Vec<_> = (0..128).map(|start| (start, 1)).collect();
            let mut split =
                encode_batch_inner(&device, &[&graph], &params, kernel, 1, Some(&plan), None)
                    .unwrap();
            while whole.commit_next(|| false) {
                whole.wait_until_completed();
            }
            while split.commit_next(|| false) {
                split.wait_until_completed();
            }
            assert!(whole.failed_status().is_none());
            assert!(split.failed_status().is_none());
            let count = 64 * whole.packed_size;
            let a = read_i8_buffer(&whole.d_samples, count).unwrap();
            let b = read_i8_buffer(&split.d_samples, count).unwrap();
            assert_eq!(a, b);
            // The anneal did something: not every read is the all-+1 zero state.
            assert!(a.iter().any(|&byte| byte != 0));
            assert_eq!(split.cmds.len(), 128);
        }
    }

    #[test]
    fn msa_shared_csr_matches_tiled_results_bit_for_bit() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let a = chain(200);
        let mut b = chain(200);
        b.j.iter_mut().step_by(3).for_each(|v| *v = -*v);
        let mut c = chain(200);
        c.j.iter_mut().skip(1).step_by(5).for_each(|v| *v = -*v);
        let graphs = [&a, &b, &c];
        let p = params(64);
        let mut batch = encode_batch(&dev, &graphs, &p, Kernel::Msa, 1).unwrap();
        while batch.commit_next(|| false) {
            batch.wait_until_completed();
        }
        batch.wait_until_completed();
        let got = harvest_batch(&batch, &graphs).unwrap();
        // Every problem has its own J, so a problem reading another's slice
        // gives reads whose energies disagree with consensus. The row and
        // column structure is one untiled copy.
        for (reads, g) in got.iter().zip(graphs) {
            for r in reads {
                assert_eq!(r.energy_milli, energy_milli(&r.spins, &g.h, &g.j, &g.edges));
            }
        }
        assert_eq!(
            batch.inputs.row.length() as usize,
            (a.num_nodes() + 1) * std::mem::size_of::<i32>()
        );
        assert_eq!(
            batch.inputs.col.length() as usize,
            2 * a.edges.len() * std::mem::size_of::<i32>()
        );
    }

    /// The spins of every read of one batch run at `seed`: a 64-node ring with
    /// unit couplings of both signs (degree 2, so MSA takes it), two MSA words
    /// of reads, and a short anneal kept hot so each spin depends on the RNG.
    fn seeded_reads(
        dev: &crate::metal_device::MetalDevice,
        kernel: Kernel,
        seed: u64,
    ) -> Vec<Vec<i8>> {
        let n = 64;
        let h = (0..n).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let j = (0..n)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let graph = IsingGraph::new(h, j, edges);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 8,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: Some((0.1, 0.5)),
            seed,
        };
        let mut batch = encode_batch(dev, &[&graph], &params, kernel, 1).unwrap();
        while batch.commit_next(|| false) {
            batch.wait_until_completed();
        }
        assert!(batch.failed_status().is_none());
        let mut problems = harvest_batch(&batch, &[&graph]).unwrap();
        let reads = problems.remove(0);
        assert_eq!(reads.len(), 64);
        reads.into_iter().map(|r| r.spins).collect()
    }

    /// The batch path feeds the kernel `kernel_seed` of the whole host seed:
    /// a repeated seed repeats every read, and flipping only the top bit,
    /// which the low 32 bits cannot see, changes them.
    fn assert_reads_follow_the_full_seed(kernel: Kernel) {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let seed = 0x0123_4567_89ab_cdef_u64;
        let first = seeded_reads(&dev, kernel, seed);
        assert_eq!(
            first,
            seeded_reads(&dev, kernel, seed),
            "{kernel:?}: the same seed must give the same reads"
        );
        assert_ne!(
            first,
            seeded_reads(&dev, kernel, seed ^ (1 << 63)),
            "{kernel:?}: a seed differing in bit 63 must give other reads"
        );
    }

    #[test]
    fn sa_reads_follow_the_full_seed() {
        assert_reads_follow_the_full_seed(Kernel::Sa);
    }

    #[test]
    fn gibbs_reads_follow_the_full_seed() {
        assert_reads_follow_the_full_seed(Kernel::Gibbs);
    }

    #[test]
    fn msa_reads_follow_the_full_seed() {
        assert_reads_follow_the_full_seed(Kernel::Msa);
    }

    /// Two problems in one batch with two words each: every (problem, word)
    /// threadgroup packs into its own region and nothing overlaps.
    #[test]
    fn msa_batch_packs_each_problem_and_word_into_its_own_region() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let n = 128;
        let mut a = chain(n);
        a.h.fill(1.0);
        let mut b = chain(n);
        b.h.fill(-1.0);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 32,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            // Keep thermal variation so distinct words need not converge.
            beta_range: Some((0.1, 0.5)),
            seed: 3,
        };
        let mut batch = encode_batch(&device, &[&a, &b], &params, Kernel::Msa, 1).unwrap();
        while batch.commit_next(|| false) {
            batch.wait_until_completed();
        }
        assert!(batch.failed_status().is_none());
        let per_problem = harvest_batch(&batch, &[&a, &b]).unwrap();
        assert_eq!(per_problem.len(), 2);
        for (problem, reads) in per_problem.iter().enumerate() {
            assert_eq!(reads.len(), 64);
            for r in reads {
                assert_eq!(r.spins.len(), n);
                assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
                // E includes +h*s, so a spin opposite to its field lowers E.
                let preferred = if problem == 0 { -1 } else { 1 };
                assert!(
                    r.spins.iter().filter(|&&s| s == preferred).count() > n / 2,
                    "problem {problem} read must reflect its own fields"
                );
            }
            // Each problem has two words with independent random streams.
            let (w0, w1) = reads.split_at(32);
            assert!(
                w0.iter().zip(w1).any(|(x, y)| x.spins != y.spins),
                "problem {problem} words must differ"
            );
        }
    }

    #[test]
    fn msa_throughput_scales_with_occupancy() {
        assert_eq!(estimated_updates_per_sec(Kernel::Msa, 0), 0.0);
        let cores = crate::iokit_gov::gpu_core_count().unwrap_or(10).max(1);
        let (first_x, first_y) = MSA_OCCUPANCY_CURVE[0];
        let first_groups = first_x * cores as f64;
        for groups in 0..first_groups.ceil() as usize {
            let expected = first_y * MSA_THROUGHPUT_SAFETY * groups as f64 / first_groups;
            let actual = estimated_updates_per_sec(Kernel::Msa, groups);
            assert!((actual - expected).abs() <= expected.max(1.0) * 1e-12);
        }
        let mut previous = 0.0;
        for groups in 1..=cores * 2 {
            let rate = estimated_updates_per_sec(Kernel::Msa, groups);
            assert!(rate >= previous, "rate decreased at {groups} groups");
            previous = rate;
        }
        assert_eq!(
            estimated_updates_per_sec(Kernel::Msa, cores * 2),
            MSA_OCCUPANCY_CURVE.last().unwrap().1 * MSA_THROUGHPUT_SAFETY
        );
    }

    #[test]
    fn tile_i32_repeats_the_slice() {
        assert_eq!(tile_i32(&[1, 2, 3], 3), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn tile_i32_edge_cases() {
        assert!(tile_i32(&[1, 2], 0).is_empty());
        assert!(tile_i32(&[], 4).is_empty());
        assert_eq!(tile_i32(&[7], 1), vec![7]);
    }

    #[test]
    fn pad_i32_substitutes_one_zero_for_empty() {
        // Metal rejects a zero-length buffer, so empty color blocks upload a
        // single unread slot.
        assert_eq!(pad_i32(&[]), vec![0]);
        assert_eq!(pad_i32(&[4, 5]), vec![4, 5]);
    }

    // -----------------------------------------------------------------------
    // Property tests (proptest is a dev-dep; private fns are reachable here)
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    /// Reference packer for the kernel bit contract (LSB-first per byte;
    /// bit set → spin -1, bit clear → +1). Inverse of [`unpack_spins`].
    fn pack_spins(spins: &[i8]) -> Vec<i8> {
        let nbytes = spins.len().div_ceil(8);
        let mut packed = vec![0i8; nbytes];
        for (i, &s) in spins.iter().enumerate() {
            if s < 0 {
                let byte_i = i >> 3;
                let bit = (i & 7) as u8;
                packed[byte_i] = (packed[byte_i] as u8 | (1u8 << bit)) as i8;
            }
        }
        packed
    }

    /// ±1 spin vectors; n includes non-byte-aligned sizes (1, 7, 8, 9, 63, 65).
    fn arb_spins() -> impl Strategy<Value = Vec<i8>> {
        prop_oneof![
            Just(1usize),
            Just(7usize),
            Just(8usize),
            Just(9usize),
            Just(63usize),
            Just(65usize),
            0usize..=96,
        ]
        .prop_flat_map(|n| prop::collection::vec(prop_oneof![Just(-1i8), Just(1i8)], n))
    }

    /// Small graphs with consensus-range h/J for scoring properties.
    fn arb_score_input() -> impl Strategy<Value = (Vec<i8>, IsingGraph)> {
        (1usize..=24).prop_flat_map(|n| {
            let max_edges = 32.min(n.saturating_mul(2));
            let spins = prop::collection::vec(prop_oneof![Just(-1i8), Just(1i8)], n);
            let h = prop::collection::vec(prop_oneof![Just(-1.0f64), Just(0.0), Just(1.0)], n);
            let edges = prop::collection::vec((0usize..n, 0usize..n), 0..=max_edges);
            (spins, h, edges).prop_flat_map(move |(spins, h, edges)| {
                let m = edges.len();
                prop::collection::vec(prop_oneof![Just(-1.0f64), Just(1.0)], m).prop_map(move |j| {
                    let graph = IsingGraph::new(h.clone(), j, edges.clone());
                    (spins.clone(), graph)
                })
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        /// unpack(pack(s)) == s for any ±1 configuration.
        #[test]
        fn unpack_spins_round_trips(spins in arb_spins()) {
            let packed = pack_spins(&spins);
            let got = unpack_spins(&packed, spins.len());
            prop_assert_eq!(got, spins);
        }

        /// `score_spins` is bit-identical to consensus `energy_milli` and
        /// copies the input spin vector into `SamplerResult::spins`.
        #[test]
        fn score_spins_matches_energy_milli((spins, graph) in arb_score_input()) {
            let got = score_spins(&spins, &graph);
            let want = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
            prop_assert_eq!(got.energy_milli, want);
            prop_assert_eq!(got.spins, spins);
        }

        /// `chunk_plan` covers every sweep of `0..num_betas.max(1)` in order
        /// with positive counts and no gaps; chunk sizes never under/overflow
        /// the schedule, and a window stays inside one rung.
        ///
        /// Inputs are plain Debug scalars (BatchDims has no Debug derive; we
        /// must not change non-test code).
        #[test]
        fn chunk_plan_covers_schedule_without_zero_counts(
            kernel in prop_oneof![Just(Kernel::Sa), Just(Kernel::Msa), Just(Kernel::Gibbs)],
            num_betas in 0i32..=512,
            n in 0usize..=64,
            sweeps_per in 0usize..=256,
            num_threads in 0usize..=1024,
            groups in 1usize..=512,
            in_flight in 0usize..=4,
            windows in proptest::bool::ANY,
            forced_rate in prop_oneof![Just(None), (1.0f64..1e9).prop_map(Some)],
        ) {
            let dims = BatchDims {
                n,
                num_betas,
                sweeps_per,
                base_seed: 0,
                num_threads,
                num_problems: 1,
                num_reads: 1,
                packed_size: 0,
            };
            let plan = chunk_plan(kernel, &dims, groups, in_flight, 1.0, windows, forced_rate);
            let cover = dims.num_betas.max(1);
            let sweeps = sweeps_per.max(1) as i32;
            prop_assert!(!plan.is_empty());
            // (rung, sweep) of the next sweep the plan must start at.
            let mut cursor = (0i32, 0i32);
            for chunk in &plan {
                let (start, count) = (chunk.beta_start, chunk.beta_count);
                prop_assert_eq!((start, chunk.sweep_start), cursor, "gap or overlap at {:?}", chunk);
                prop_assert!(count > 0, "zero-length chunk at start={}", start);
                prop_assert!(chunk.sweep_count > 0, "zero-sweep chunk {:?}", chunk);
                prop_assert!(
                    start.checked_add(count).is_some(),
                    "start+count overflows i32: start={} count={}",
                    start,
                    count
                );
                let end = chunk.sweep_start + chunk.sweep_count;
                prop_assert!(end <= sweeps, "window past its rung: {:?}", chunk);
                if end < sweeps || chunk.sweep_start > 0 {
                    prop_assert!(windows, "a window without windows: {:?}", chunk);
                    prop_assert_eq!(count, 1, "a window spans rungs: {:?}", chunk);
                }
                cursor = if end == sweeps {
                    (start + count, 0)
                } else {
                    (start, end)
                };
            }
            prop_assert_eq!(cursor, (cover, 0), "plan does not cover full schedule");
        }
    }
}
