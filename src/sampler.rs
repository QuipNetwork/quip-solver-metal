//! Launch the v0.2 Metal SA / Gibbs kernels and score with consensus energy.
//!
//! Kernels are the original v0.2 Metal sources (`GPU/metal_kernels.metal` /
//! `GPU/metal_gibbs.metal`, copied verbatim into `kernels/`): int8-quantized
//! CSR, D-Wave incremental delta-energy SA / color-block Gibbs, bit-packed
//! thread-local state, one thread per read. The multi-spin kernel computes
//! energies on the device for whole-unit graphs, with one job in 1,000 audited
//! on the host. SA, Gibbs, and non-integer graphs are scored on the host with
//! [`quip_solver_core::quip_protocol::scoring::energy_milli`] (f64 consensus).
//!
//! # Batched dispatch (throughput)
//!
//! The kernel maps **one threadgroup per problem, one thread per read**
//! (`thread_id = threadgroup·num_reads + read`, `problem_id = thread_id /
//! num_reads`). A dispatch of `P` problems is
//! `dispatchThreadgroups(P, num_reads)` → `P` threadgroups occupy `P` GPU
//! cores, so a batch of ≈`gpu_cores` problems fills the whole GPU and hides
//! the kernel's thread-private memory latency. This mirrors v0.2
//! `metal_sa.py::_dispatch_batch`. Driving `P = 1` (one problem per command
//! buffer) leaves all but one core idle — a ~`gpu_cores`× slowdown — which is
//! why the streaming loop batches ([`crate::streaming`]).
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

use crate::topology::{fill_h_j, SelfFeedingTopology};
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

/// SA kernel `N` cap: `thread int8_t delta_energy[4593]` in `kernels/sa.metal`.
///
/// `crate::METAL_SA_IDENTITY` advertises this same cap, so the identity const
/// and the kernel array have one source.
pub(crate) const SA_MAX_NODES: usize = 4593;
/// Gibbs kernel `N` cap: `thread int8_t packed_state[600]` (600*8) in
/// `kernels/gibbs.metal`.
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
/// Largest CSR degree the kernel's unrolled 20-neighbour prefetch admits.
/// Zephyr (Advantage2) is degree 20. A nonzero field is a 21st input, which
/// the kernel's `popcount21` covers.
pub(crate) const MSA_MAX_DEG: usize = 20;

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

/// Split a beta schedule into chunks whose dispatches each land near
/// [`TARGET_DISPATCH_MS`].
///
/// Returns `(beta_start, beta_count)` pairs covering `0..num_betas`. One sweep
/// touches every spin of every sample, so a chunk's cost is
/// `threads * sweeps_per_beta * betas * N` spin updates.
///
/// All kernels can resume mid-schedule: `beta_start == 0` initializes, any
/// other value restores the carry-over state the previous chunk wrote.
fn chunk_plan(
    kernel: Kernel,
    dims: &BatchDims,
    groups: usize,
    in_flight: usize,
) -> Vec<(i32, i32)> {
    let num_betas = dims.num_betas.max(1);
    let per_beta =
        (dims.num_threads as f64) * (dims.sweeps_per.max(1) as f64) * (dims.n.max(1) as f64);
    let betas_per_chunk = dispatch_beta_limit(kernel, per_beta, groups, in_flight, num_betas);

    let mut plan = Vec::new();
    let mut start = 0;
    while start < num_betas {
        let count = betas_per_chunk.min(num_betas - start);
        plan.push((start, count));
        start += count;
    }
    plan
}

fn dispatch_beta_limit(
    kernel: Kernel,
    per_beta: f64,
    groups: usize,
    in_flight: usize,
    num_betas: i32,
) -> i32 {
    // Concurrent batches share the aggregate rate at their combined occupancy.
    // Reserve the full stream window even while priming or draining it.
    let in_flight = in_flight.max(1);
    let rate =
        estimated_updates_per_sec(kernel, groups.saturating_mul(in_flight)) / in_flight as f64;
    let budget = TARGET_DISPATCH_MS / 1000.0 * rate;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to 1..=num_betas immediately below"
    )]
    let limit = ((budget / per_beta.max(1.0)).floor() as i64).clamp(1, i64::from(num_betas)) as i32;
    limit
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

/// Largest `N` the kernel's fixed-size arrays admit.
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
    u32::try_from(z >> 32).unwrap_or(u32::MAX)
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
    ) -> std::sync::Arc<CachedTopology> {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(entry) = slot.as_ref() {
            if entry.n == graph.num_nodes()
                && entry.four_color == four_color
                && entry.edges == graph.edges
            {
                return std::sync::Arc::clone(entry);
            }
        }
        let cached = std::sync::Arc::new(build_cached_topology(device, graph, four_color));
        *slot = Some(std::sync::Arc::clone(&cached));
        cached
    }
}

fn build_cached_topology(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    four_color: bool,
) -> CachedTopology {
    let topo = if four_color {
        SelfFeedingTopology::build_with_advantage2_coloring(graph)
    } else {
        SelfFeedingTopology::build(graph)
    };
    let colors = [&topo.colors.starts, &topo.colors.counts, &topo.colors.nodes]
        .into_iter()
        .map(|values| device.new_buffer_from_slice(&pad_i32(values)))
        .collect();
    let zero = [0i32];
    let base_row: &[i32] = if topo.row_ptr.is_empty() {
        &zero
    } else {
        &topo.row_ptr
    };
    let base_col: &[i32] = if topo.nnz == 0 { &zero } else { &topo.col_ind };
    let row = device.new_buffer_from_slice(base_row);
    let col = device.new_buffer_from_slice(base_col);
    CachedTopology {
        n: graph.num_nodes(),
        edges: std::sync::Arc::clone(&graph.edges),
        four_color,
        topo,
        colors,
        row,
        col,
    }
}

/// Free shared-storage buffers for later multi-spin batches. At most 64.
#[derive(Debug, Default)]
pub(crate) struct BufferPool {
    free: std::sync::Mutex<Vec<metal::Buffer>>,
}

impl BufferPool {
    /// Smallest free buffer with `len <= length <= 2 * len`, else a new shared buffer.
    pub(crate) fn take(&self, device: &metal::DeviceRef, len: u64) -> metal::Buffer {
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
        if let Some(index) = best_index {
            return free.swap_remove(index);
        }
        drop(free);
        device.new_buffer(len, metal::MTLResourceOptions::StorageModeShared)
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
) -> metal::Buffer {
    let bytes = std::mem::size_of_val(data);
    if bytes == 0 {
        let buf = pool.take(device, 4);
        // SAFETY: `take(4)` returns StorageModeShared storage of at least 4
        // bytes, so `contents()` maps those bytes on the host. The write stays
        // inside that mapping.
        unsafe {
            std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, 4);
        }
        return buf;
    }
    let buf = pool.take(device, bytes as u64);
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
    buf
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
    colors: Vec<metal::Buffer>,
    num_colors: i32,
    groups: usize,
    threads_per_group: usize,
    plan: Vec<(i32, i32)>,
    /// Retains the topology buffers after a later miss replaces the device slot.
    #[expect(
        dead_code,
        reason = "keeps cached Metal buffers alive until the batch drops"
    )]
    cached: std::sync::Arc<CachedTopology>,
    pool: std::sync::Arc<BufferPool>,
    /// Multi-spin buffers rented from `pool`.
    pooled: Vec<metal::Buffer>,
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
        let Some(&(beta_start, beta_count)) = self.plan.get(self.cmds.len()) else {
            return false;
        };
        let cmd = self.queue.new_command_buffer().to_owned();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_chunk(encoder, beta_start, beta_count);
        encoder.end_encoding();
        if cancelled() {
            return false;
        }
        cmd.commit();
        self.cmds.push(cmd);
        true
    }

    fn encode_chunk(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        beta_start: i32,
        beta_count: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipeline);
        bind_shared_args(encoder, &self.inputs, &self.out, &self.dims);
        match &self.kstate {
            KernelEncode::Sa { persist } => {
                bind_sa_chunk(encoder, persist, beta_start, beta_count);
            }
            KernelEncode::Gibbs { persist } => {
                self.bind_colors(encoder, 0);
                bind_color_kernel_chunk(encoder, persist, beta_start, beta_count);
            }
            KernelEncode::Msa {
                persist,
                words,
                state_bytes,
            } => {
                self.bind_colors(encoder, *words);
                bind_color_kernel_chunk(encoder, persist, beta_start, beta_count);
                encoder.set_threadgroup_memory_length(0, *state_bytes);
            }
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

/// Validate a batch's shared inputs, returning the establishing graph and `N`.
///
/// Split out of [`encode_batch`] so both rejections are reachable without a
/// Metal device.
pub(crate) fn validate_batch<'a>(
    graphs: &[&'a IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
) -> Result<(&'a IsingGraph, usize), SampleError> {
    let Some(&first) = graphs.first() else {
        // Was a `debug_assert!`, which is compiled out in release — an empty
        // batch would have indexed `graphs[0]` and panicked the miner.
        return Err(SampleError::Driver(
            "encode_batch needs at least one graph".into(),
        ));
    };
    let n = first.num_nodes();
    let cap = kernel_max_nodes(kernel);
    if n > cap {
        // Defense in depth: the harness rejects N > max_nodes (identity const)
        // before the sampler; this catches drift before it overruns the
        // kernel's fixed-size thread-local arrays.
        return Err(SampleError::TooLarge(format!(
            "graph N={n} exceeds {kernel:?} kernel limit {cap}"
        )));
    }
    if kernel == Kernel::Msa {
        // The kernel unrolls a fixed 20-neighbour prefetch; a denser node
        // would read past it. Capacity, not a driver fault: the coordinator
        // routes the job to a backend that walks CSR rows of any length.
        let max_deg = max_csr_degree(first);
        if max_deg > MSA_MAX_DEG {
            return Err(SampleError::TooLarge(format!(
                "graph max degree {max_deg} exceeds the multi-spin kernel's \
                 {MSA_MAX_DEG}-neighbour budget"
            )));
        }
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
    Ok((first, n))
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
) -> InputBuffers {
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
            device.new_buffer_from_slice(&tile_i32(base_row, num_problems)),
            device.new_buffer_from_slice(&tile_i32(base_col, num_problems)),
        )
    };
    let mut all_j = vec![0i8; num_problems * nnz_alloc];
    let mut all_h = vec![0i8; num_problems * n];
    for (p, graph) in graphs.iter().enumerate() {
        let (j_csr, h_i8) = fill_h_j(topo, graph);
        // `j_csr` has length `topo.nnz`; pad region is the trailing slot when
        // nnz == 0. `h_i8` has length N.
        all_j[p * nnz_alloc..p * nnz_alloc + j_csr.len()].copy_from_slice(&j_csr);
        all_h[p * n..p * n + h_i8.len()].copy_from_slice(&h_i8);
    }
    let row_ptr_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * rp_len) as i32).collect();
    let col_ind_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * nnz_alloc) as i32).collect();
    let (j, h) = if kernel == Kernel::Msa {
        (
            pooled_from_slice(&device.buffer_pool, &device.device, &all_j),
            pooled_from_slice(&device.buffer_pool, &device.device, &all_h),
        )
    } else {
        (
            device.new_buffer_from_slice(&all_j),
            device.new_buffer_from_slice(&all_h),
        )
    };

    InputBuffers {
        row,
        col,
        j,
        h,
        row_off: device.new_buffer_from_slice(&row_ptr_offsets),
        col_off: device.new_buffer_from_slice(&col_ind_offsets),
    }
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

/// SA-only chunked-dispatch persistent buffers (indices 16..21).
///
/// Single-shot run: beta_start = 0, beta_count = num_betas, so these are
/// written but never re-read — allocated as zeroed scratch.
fn new_sa_persistent(
    device: &crate::metal_device::MetalDevice,
    dims: &BatchDims,
) -> [metal::Buffer; 4] {
    [
        device.new_zeroed_buffer((dims.num_threads * dims.packed_size) as u64),
        device.new_zeroed_buffer((dims.num_threads * dims.n.max(1)) as u64),
        device.new_zeroed_buffer((dims.num_threads * 4) as u64), // uint rng
        device.new_zeroed_buffer((dims.num_threads * 4) as u64), // int energy
    ]
}

/// Gibbs carry-over state: spins plus the RNG stream, allocated per anneal.
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
fn new_gibbs_persistent(
    device: &crate::metal_device::MetalDevice,
    dims: &BatchDims,
    node_parallel: bool,
    threads_per_group: usize,
) -> [metal::Buffer; 2] {
    // 4 x u32 of xoshiro128** state per stream.
    const RNG_BYTES_PER_STREAM: usize = 16;
    if node_parallel {
        [
            device.new_zeroed_buffer((dims.num_threads * dims.n.max(1)) as u64),
            device.new_zeroed_buffer(
                (dims.num_threads * threads_per_group.max(1) * RNG_BYTES_PER_STREAM) as u64,
            ),
        ]
    } else {
        [
            device.new_zeroed_buffer((dims.num_threads * dims.packed_size) as u64),
            device.new_zeroed_buffer((dims.num_threads * RNG_BYTES_PER_STREAM) as u64),
        ]
    }
}

/// Multi-spin carry-over state: one `uint` word per (threadgroup, spin) plus
/// one xoshiro128** stream (four `uint`) per (threadgroup, thread). Only spin
/// words and RNG streams persist across a chunk boundary.
fn new_msa_persistent(
    pool: &BufferPool,
    device: &metal::DeviceRef,
    num_streams: usize,
    n: usize,
    threads_per_group: usize,
) -> [metal::Buffer; 2] {
    const RNG_BYTES_PER_STREAM: usize = 16;
    [
        pool.take(device, (num_streams * n.max(1) * 4) as u64),
        pool.take(
            device,
            (num_streams * threads_per_group.max(1) * RNG_BYTES_PER_STREAM) as u64,
        ),
    ]
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
    },
    Msa {
        persist: [metal::Buffer; 2],
        /// Words per problem, bound at slot 19.
        words: i32,
        /// Bytes of `threadgroup uint state[N]` at threadgroup index 0,
        /// rounded up to Metal's 16-byte granularity.
        state_bytes: u64,
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
    beta_start: i32,
    beta_count: i32,
) {
    set_bytes_i32(enc, 16, beta_start);
    set_bytes_i32(enc, 17, beta_count);
    enc.set_buffer(18, Some(&persist[0]), 0);
    enc.set_buffer(19, Some(&persist[1]), 0);
    enc.set_buffer(20, Some(&persist[2]), 0);
    enc.set_buffer(21, Some(&persist[3]), 0);
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
    let (first, n) = validate_batch(graphs, params, kernel)?;

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

    // Chromatic Gibbs uses a different pipeline but the *same* buffer layout —
    // only the dispatch geometry below differs.
    let node_parallel = kernel == Kernel::Gibbs && gibbs_node_parallel();
    let pipeline = match kernel {
        Kernel::Sa => &device.sa,
        Kernel::Msa => &device.msa,
        Kernel::Gibbs if node_parallel => &device.gibbs_parallel,
        Kernel::Gibbs => &device.gibbs,
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
    let (groups, threads_per_group) = match kernel {
        Kernel::Msa => (num_problems * words, MSA_THREADS.min(max_threads).max(1)),
        Kernel::Gibbs if node_parallel => (num_samples, 256.min(max_threads).max(1)),
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
        base_seed: kernel_seed(params.seed).wrapping_add(1),
        num_threads,
        num_problems,
        num_reads,
        packed_size,
    };

    // The multi-spin kernel's spin words live in threadgroup memory sized per
    // dispatch. Refuse a job the opened device cannot hold rather than let
    // Metal fail the command buffer, which would end the session as a
    // DeviceFault for a per-job size problem.
    let state_bytes = (n.max(1) * 4).div_ceil(16) * 16;
    if kernel == Kernel::Msa {
        let need = state_bytes + MSA_STATIC_TG_BYTES;
        let have = device.device.max_threadgroup_memory_length() as usize;
        if need > have {
            return Err(SampleError::TooLarge(format!(
                "graph N={n} needs {need} B of threadgroup memory; device offers {have} B"
            )));
        }
    }

    let cached = device.topology_cache.get_or_build(
        device,
        first,
        kernel == Kernel::Msa && msa_four_color(),
    );
    let inputs = upload_inputs(device, &cached, graphs, kernel);

    let plan = match plan_override {
        Some(p) => p.to_vec(),
        None => chunk_plan(kernel, &dims, groups, in_flight),
    };
    let mut pooled = Vec::new();
    let (out, kstate) = match kernel {
        Kernel::Sa => (
            DispatchBuffers {
                beta: device.new_buffer_from_slice(&beta),
                samples: device.new_zeroed_buffer((num_samples * packed_size) as u64),
                energies: device.new_zeroed_buffer((num_samples * 4) as u64), // i32
            },
            KernelEncode::Sa {
                persist: new_sa_persistent(device, &dims),
            },
        ),
        Kernel::Gibbs => (
            DispatchBuffers {
                beta: device.new_buffer_from_slice(&beta),
                samples: device.new_zeroed_buffer((num_samples * packed_size) as u64),
                energies: device.new_zeroed_buffer((num_samples * 4) as u64), // i32
            },
            KernelEncode::Gibbs {
                persist: new_gibbs_persistent(device, &dims, node_parallel, threads_per_group),
            },
        ),
        Kernel::Msa => {
            let beta_buf = pooled_from_slice(&device.buffer_pool, &device.device, &beta);
            // The kernel writes `samples`, `energies`, and the persistent buffers
            // in full before any read (first chunk initializes state and RNG, and
            // every read's bytes are packed), so pooled buffers need no zeroing.
            let samples = device
                .buffer_pool
                .take(&device.device, (num_samples * packed_size) as u64);
            let energies = device
                .buffer_pool
                .take(&device.device, (num_samples * 4) as u64);
            let persist = new_msa_persistent(
                &device.buffer_pool,
                &device.device,
                groups,
                n,
                threads_per_group,
            );
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
                },
            )
        }
    };

    let colors = cached.colors.clone();
    let num_colors = cached.topo.colors.num_colors;
    Ok(EncodedBatch {
        device_energy: kernel == Kernel::Msa && graphs.iter().all(|g| device_energy_exact(g)),
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
        colors,
        num_colors,
        groups,
        threads_per_group,
        plan,
        cached,
        pool: std::sync::Arc::clone(&device.buffer_pool),
        pooled,
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
/// - `graph.num_nodes()` exceeds the algorithm's kernel node cap
///   (`SA_MAX_NODES` / `GIBBS_MAX_NODES`);
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
    let n = graph.num_nodes();
    if n == 0 {
        let reads = params.num_reads.max(1);
        return Ok((0..reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect());
    }

    let mut batch = encode_batch(device, &[graph], params, kernel, 1)?;
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
    Ok(samples)
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

    #[test]
    fn validate_batch_accepts_a_normal_job() {
        let g = ring();
        let (first, n) = validate_batch(&[&g], &params(64), Kernel::Sa).unwrap();
        assert_eq!(n, 4);
        assert_eq!(first.num_nodes(), 4);
    }

    #[test]
    fn validate_batch_rejects_n_over_the_kernel_cap() {
        let big = IsingGraph::new(vec![0.0; SA_MAX_NODES + 1], vec![], vec![]);
        let err = validate_batch(&[&big], &params(64), Kernel::Sa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("exceeds"), "{msg}");
        assert!(msg.contains(&SA_MAX_NODES.to_string()), "{msg}");
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
        // `delta_energy[4593]` in sa.metal, `packed_state[600]` (600*8) in
        // gibbs.metal.
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
    fn validate_batch_rejects_msa_degree_over_the_prefetch_budget() {
        // The kernel unrolls a fixed 20-neighbour prefetch; a denser node
        // would read past it. Capacity, not DeviceFault: the coordinator
        // routes the job to another backend.
        let g = star(21);
        let err = validate_batch(&[&g], &params(64), Kernel::Msa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("degree 21"), "{msg}");
        // The other kernels walk CSR rows of any length.
        validate_batch(&[&g], &params(64), Kernel::Sa).unwrap();
        validate_batch(&[&g], &params(64), Kernel::Gibbs).unwrap();
    }

    #[test]
    fn validate_batch_accepts_msa_degree_at_the_budget() {
        let g = star(MSA_MAX_DEG);
        validate_batch(&[&g], &params(64), Kernel::Msa).unwrap();
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
        let first = dev.topology_cache.get_or_build(&dev, &a, false);
        let again = dev.topology_cache.get_or_build(&dev, &a2, false);
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        // Same node count, one edge moved: the cache must rebuild.
        let mut moved = chain(50);
        let mut edges = moved.edges.to_vec();
        edges[0] = (0, 2);
        moved.edges = edges.into();
        let rewired = dev.topology_cache.get_or_build(&dev, &moved, false);
        assert!(!std::sync::Arc::ptr_eq(&first, &rewired));
        let other = dev.topology_cache.get_or_build(&dev, &b, false);
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        assert_eq!(other.n, b.num_nodes());
        let colored = dev.topology_cache.get_or_build(&dev, &b, true);
        assert!(!std::sync::Arc::ptr_eq(&other, &colored));
    }

    #[test]
    fn buffer_pool_reuses_a_returned_buffer_of_fitting_size() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let dev = crate::metal_device::MetalDevice::open(0).unwrap();
        let pool = BufferPool::default();
        let a = pool.take(&dev.device, 4096);
        let ptr = a.contents();
        pool.give(a);
        let b = pool.take(&dev.device, 3000);
        assert_eq!(b.contents(), ptr);
        let c = pool.take(&dev.device, 100_000);
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
            let solo = chunk_plan(kernel, &dims, groups, 1)[0].1;
            let shared = chunk_plan(kernel, &dims, groups, 2)[0].1;
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
                        chunk_plan(Kernel::Msa, &dims, groups, 2)[0].1 as usize
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

        /// `chunk_plan` covers `0..num_betas.max(1)` with positive counts and
        /// no gaps; chunk sizes never under/overflow the schedule.
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
            let plan = chunk_plan(kernel, &dims, groups, in_flight);
            let cover = dims.num_betas.max(1);
            prop_assert!(!plan.is_empty());
            let mut cursor = 0i32;
            for &(start, count) in &plan {
                prop_assert_eq!(start, cursor, "gap or overlap at start={}", start);
                prop_assert!(count > 0, "zero-length chunk at start={}", start);
                prop_assert!(
                    start.checked_add(count).is_some(),
                    "start+count overflows i32: start={} count={}",
                    start,
                    count
                );
                cursor = start + count;
            }
            prop_assert_eq!(cursor, cover, "plan does not cover full schedule");
        }
    }
}
