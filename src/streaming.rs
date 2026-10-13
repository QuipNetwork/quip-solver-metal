//! Host-driven batched Metal streaming.
//!
//! Mirrors v0.2 `GPU/metal_sa.py::stream_read_split_batches` /
//! `_dispatch_batch`: collect up to [`stream_width`] queued jobs that share a
//! topology, sampling params, and cancellation watermark. Each batch shares
//! device buffers across chunks, then host-scores and emits one result per job.
//! Two batches can run concurrently, with one committed chunk each.
//! SA and sequential Gibbs use one threadgroup per problem; chromatic Gibbs
//! uses one per read, and MSA uses one per 32-replica word.
//!
//! This is the throughput mechanism: on Metal the GPU is filled *inside* one
//! dispatch (many threadgroups). Chunk submission checks cancellation before
//! committing more work, while the next batch overlaps host preparation.
//!
//! # Threading
//!
//! `run_stream` runs on the single blocking thread the harness gives
//! `Sampler::sample_stream`; every Metal object stays on that thread.

use crate::metal_device::MetalDevice;
use crate::sampler::{self, kernel_max_nodes, Kernel};
use quip_solver_core::{
    CancelToken, IsingGraph, SampleError, SamplerResult, StreamJob, StreamOutcome, StreamResult,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

/// Fallback GPU core count when IOKit can't report `gpu-core-count`. Sizes the
/// batch (problems per dispatch), so a miss costs throughput tuning, not
/// correctness.
const DEFAULT_GPU_CORES: usize = 10;

/// Backend-facing read cap for [`crate::MetalSampler::max_reads`] (mirrors CUDA).
///
/// # Examples
///
/// ```
/// use quip_solver_metal::{streaming, Kernel};
///
/// assert_eq!(streaming::max_reads(Kernel::Sa), 256);
/// assert_eq!(streaming::max_reads(Kernel::Gibbs), 256);
/// ```
pub fn max_reads(_kernel: Kernel) -> u32 {
    sampler::MAX_READS as u32
}

/// Threadgroup budget per GPU core, per kernel — the measured definition of
/// "fully loaded" for this backend.
///
/// The three kernels map to hardware differently, so a single problems-per-core
/// constant means three different occupancies:
///
/// ```text
/// SA / sequential Gibbs:  threadgroups = P           (one per problem, R threads each)
/// chromatic Gibbs:        threadgroups = P * R       (one per SAMPLE, 256 threads each)
/// multi-spin:             threadgroups = P * (R/32)  (one per 32-lane word)
/// ```
///
/// Budgeting in threadgroups instead makes one constant mean one thing. The SA
/// and Gibbs values are from the occupancy sweep on an M4 Max (40 cores), full
/// Advantage2 topology, 64 reads / 128 sweeps, measured in spin-updates/s from
/// per-dispatch GPU time. Multi-spin measurements are in [`MSA_TG_PER_CORE`].
///
/// ```text
/// SA      tg/core:  0.2   0.5   1     2     3     4     6     8
///         Gupd/s:   0.32  0.49  0.83  1.40  1.54  1.83  2.03  2.10
///         dispatch: 1.11  1.46  1.75  2.16  2.57  2.90  4.17  5.33 s
///
/// Gibbs   tg/core:  16    32    64    128   182   256   384   512
///         Gupd/s:   2.06  2.05  2.12  2.09  2.05  2.00  1.98  1.98
///         dispatch: 0.36  0.73  1.43  2.82  4.10  5.95  8.88  11.87 s
/// ```
///
/// SA climbs to 8 and is still gaining; 6 is the knee (+45% over the old
/// 2-problems-per-core default) and is chosen over 8 because dispatch length —
/// the GPU lock-up a foreground app feels, and the latency a `Cancel` waits on —
/// grows faster than throughput past it.
///
/// Chromatic Gibbs is flat from 16 upward: it is saturated at the *bottom* of
/// the range, and everything above merely lengthens dispatches. The old default
/// put it at 128 tg/core, paying 8x the dispatch length for nothing.
///
/// Run-to-run variance is ~13%, so treat neighbouring points as ties.
/// `QUIP_METAL_TG_PER_CORE` overrides for GPUs where the optimum differs.
const SA_TG_PER_CORE: f64 = 6.0;
/// See [`SA_TG_PER_CORE`]. Chromatic Gibbs saturates here; higher only costs
/// dispatch length.
const GIBBS_TG_PER_CORE: f64 = 16.0;
/// Multi-spin threadgroup budget per GPU core.
///
/// Measured 2026-09-15 on Apple M4 Max (40 GPU cores), using
/// `tests/fixtures/advantage2-system1.edges`: 4577 nodes, 41515 edges, eight
/// greedy colour classes. Each run used 80 jobs, 128 reads and 7392 sweeps.
///
/// ```text
/// T, tg/core:  1      2      4      6      8
/// jobs/s:     13.64  14.23  12.16  11.57  10.85
/// wall, s:     5.9    5.6    6.6    6.9    7.4
/// ```
///
/// T=1 is the smallest within 10% of the best: 13.64 >= 0.9 * 14.23.
/// Each group uses 256 threads and 26,756 bytes of memory on this fixture.
const MSA_TG_PER_CORE: f64 = 1.0;

/// Nominal reads used to size [`stream_width`] before any job has arrived:
/// each kernel's adapt envelope `min_reads`, the smallest count the adapt
/// path issues. Read from the envelopes themselves so the width follows a
/// change to them.
fn nominal_reads(kernel: Kernel) -> usize {
    let bounds = match kernel {
        Kernel::Sa | Kernel::Gibbs => crate::METAL_ADAPT,
        Kernel::Msa => crate::METAL_MSA_ADAPT,
    };
    bounds.min_reads as usize
}

/// Threadgroups this dispatch aims to have in flight.
fn tg_budget(kernel: Kernel) -> usize {
    let cores = crate::iokit_gov::gpu_core_count()
        .unwrap_or(DEFAULT_GPU_CORES)
        .max(1);
    let default = match kernel {
        Kernel::Sa => SA_TG_PER_CORE,
        Kernel::Msa => MSA_TG_PER_CORE,
        Kernel::Gibbs => GIBBS_TG_PER_CORE,
    };
    let per_core = std::env::var("QUIP_METAL_TG_PER_CORE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|m| m.is_finite() && *m > 0.0)
        .unwrap_or(default);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "core count times a small positive multiplier is far below usize::MAX"
    )]
    let budget = (cores as f64 * per_core).round() as usize;
    budget.max(1)
}

/// Problems per dispatch for a job with `num_reads` reads.
///
/// Converts the threadgroup budget into a problem count using the kernel's own
/// mapping: chromatic Gibbs spends `num_reads` threadgroups per problem and
/// multi-spin spends `num_reads / 32`, so their batches shrink as reads grow.
/// SA spends one.
pub(crate) fn batch_size_for_reads(kernel: Kernel, num_reads: usize) -> usize {
    let budget = tg_budget(kernel);
    let per_problem = match kernel {
        Kernel::Gibbs if sampler::gibbs_node_parallel() => {
            sampler::simd_rounded_reads(num_reads).max(1)
        }
        Kernel::Msa => sampler::simd_rounded_reads(num_reads)
            .div_ceil(sampler::MSA_LANES)
            .max(1),
        Kernel::Sa | Kernel::Gibbs => 1,
    };
    budget.div_ceil(per_problem).max(1)
}

/// Apply the governor's budget scale to a nominal problem count.
///
/// Never returns 0: one problem per dispatch is the floor, because a dispatch
/// of nothing makes no progress and would never free the coordinator's credit.
pub(crate) fn scale_budget(nominal: usize, scale: f64) -> usize {
    if !scale.is_finite() || scale >= 1.0 {
        return nominal.max(1);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "batch sizes are small positive counts; scale is clamped to 0..1"
    )]
    let scaled = (nominal as f64 * scale.max(0.0)).round() as usize;
    scaled.max(1)
}

/// The width [`stream_width`] resolves to for `kernel`, with no device.
///
/// A pure function of the kernel — the device does not participate in the
/// Metal width — split out so `Sampler::declared_stream_width` can advertise
/// the same number without opening a device (`--capabilities` must not).
/// For the multi-spin kernel the width also covers the resident runner's
/// preparation queue, `resident::PREP_BOUND`.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::{streaming, Kernel};
///
/// assert!(streaming::declared_stream_width(Kernel::Sa) >= 1);
/// ```
#[must_use]
pub fn declared_stream_width(kernel: Kernel) -> usize {
    let batches = batch_size_for_reads(kernel, nominal_reads(kernel)) * 2;
    if kernel == Kernel::Msa {
        // Two live slot pools plus the resident preparation queue. The
        // session's lease expander and the coordinator's credits both stop at
        // this width, and each salt needs a host draw before it can prepare.
        batches + crate::resident::PREP_BOUND
    } else {
        batches.max(1)
    }
}

/// `Sampler::stream_width`: how many models the backend keeps in flight.
///
/// Sized from [`nominal_reads`] because it is fixed at startup, before any job
/// reveals its read count. Two batches' worth, so the harness buffers the next
/// batch while one dispatches.
///
/// The `_device` parameter is unused today (width is core-count driven via
/// IOKit), but kept so the signature matches the harness and stays ready for
/// per-device overrides. Until such an override exists, this and
/// [`declared_stream_width`] are the same number by construction.
///
/// # Examples
///
/// ```no_run
/// use quip_solver_metal::{streaming, Kernel, metal_device::MetalDevice};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = MetalDevice::open(0)?;
/// let width = streaming::stream_width(&device, Kernel::Sa);
/// assert!(width >= 1);
/// # Ok(())
/// # }
/// ```
pub fn stream_width(_device: &MetalDevice, kernel: Kernel) -> usize {
    declared_stream_width(kernel)
}

/// Structural + sampling identity a single dispatch batches over: same topology
/// (so the shared CSR/coloring stay valid) and same reads/beta shape (one
/// `num_reads`, `num_betas`, `beta_schedule` per dispatch).
/// Jobs also share a watermark, so cancellation can stop their carry-over state
/// together without removing a live problem from the device buffer layout.
///
/// `edges` is borrowed from the seed job rather than cloned: at Advantage2
/// scale that list is ~40k pairs (~640 KB), and an owned copy existed only to
/// free the seed for `batch.push` while `fill_batch` still needed the key.
/// Callers keep the seed live across `fill_batch`, then assemble `batch`.
struct BatchKey<'a> {
    watermark: Option<u64>,
    n: usize,
    edges: &'a [(usize, usize)],
    num_reads: usize,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
    schedule: quip_solver_core::BetaSchedule,
}

impl<'a> BatchKey<'a> {
    fn from_job(job: &'a StreamJob) -> Self {
        Self {
            watermark: job.watermark,
            n: job.graph.h.len(),
            edges: &job.graph.edges,
            num_reads: job.params.num_reads.clamp(1, sampler::MAX_READS),
            num_sweeps: job.params.num_sweeps,
            sweeps_per_beta: job.params.sweeps_per_beta.max(1),
            beta_range: job.params.beta_range,
            schedule: job.params.schedule,
        }
    }

    fn matches(&self, job: &StreamJob) -> bool {
        self.watermark == job.watermark
            && self.n == job.graph.h.len()
            && self.num_reads == job.params.num_reads.clamp(1, sampler::MAX_READS)
            && self.num_sweeps == job.params.num_sweeps
            && self.sweeps_per_beta == job.params.sweeps_per_beta.max(1)
            && self.beta_range == job.params.beta_range
            && self.schedule == job.params.schedule
            && self.edges == &*job.graph.edges
    }
}

/// Whether [`next_seed`] / [`form_and_commit`] may block waiting for a job.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seed {
    /// Wait for a job; returns `None` only on channel close.
    Blocking,
    /// Return `None` immediately if nothing is queued (overlap path).
    NonBlocking,
}

/// What the streaming loop needs from the GPU governor.
///
/// A trait rather than the closure this used to take, because dispatch sizing
/// is now a feedback loop: the loop reports how much device time it consumed,
/// and reads back a budget scale derived from it. Those two halves have to see
/// the same state.
pub trait GpuGovernor {
    /// True while the miner should hold off dispatching entirely.
    fn should_throttle(&self) -> bool;
    /// Fraction of the nominal threadgroup budget to dispatch (0 < s <= 1).
    fn budget_scale(&self) -> f64;
    /// Report a completed batch's GPU-busy microseconds (device time).
    fn record_gpu_busy_us(&self, us: u64);
}

impl GpuGovernor for crate::iokit_gov::UtilGovernor {
    fn should_throttle(&self) -> bool {
        Self::should_throttle(self)
    }
    fn budget_scale(&self) -> f64 {
        Self::budget_scale(self)
    }
    fn record_gpu_busy_us(&self, us: u64) {
        Self::record_gpu_busy_us(self, us);
    }
}

/// How long to pause per throttle check. Shorter than the governor's 250 ms
/// sensor poll, so a cleared ceiling is noticed within roughly one poll rather
/// than one sleep.
const THROTTLE_PAUSE: Duration = Duration::from_millis(50);

/// Upper bound on one throttle gate. The miner is itself the main GPU load, so
/// `should_throttle` stays true until *we* stop — without a cap, a ceiling
/// below the miner's own steady-state utilization would stall it forever
/// instead of duty-cycling it. Capping the pause turns the ceiling into a duty
/// cycle: work, yield, re-measure.
const THROTTLE_MAX_PAUSE: Duration = Duration::from_millis(500);

/// Shared borrow cluster threaded through the streaming helpers.
struct StreamCtx<'a> {
    jobs: &'a mut Receiver<StreamJob>,
    out: &'a Sender<StreamResult>,
    pending: &'a mut Option<StreamJob>,
    kernel: Kernel,
    /// Utilization ceiling, external-load accounting, and dispatch backpressure.
    gov: &'a dyn GpuGovernor,
    /// Reseed watermark: job watermarks at or below it were abandoned by the
    /// coordinator and must not consume GPU time.
    cancel: &'a CancelToken,
}

impl StreamCtx<'_> {
    /// Hold off the next dispatch while the governor asks us to yield, bounded
    /// by [`THROTTLE_MAX_PAUSE`].
    ///
    /// Called only from the batch-forming path, which is the one point where no
    /// GPU work of ours is queued behind us, so pausing here actually leaves the
    /// device idle rather than merely delaying our own enqueue.
    ///
    /// The pause is the coarse, immediate lever. Sustained sharing is
    /// [`GpuGovernor::budget_scale`]'s job — it shrinks the dispatch itself, so
    /// the miner keeps running at a smaller size instead of stopping and
    /// starting.
    fn yield_gate(&self) {
        yield_gate(self.out, self.gov);
    }
}

pub(crate) fn yield_gate(out: &Sender<StreamResult>, gov: &dyn GpuGovernor) {
    if !gov.should_throttle() {
        return;
    }
    tracing::debug!("yield gate: pausing");
    let deadline = Instant::now() + THROTTLE_MAX_PAUSE;
    while !out.is_closed() && Instant::now() < deadline && gov.should_throttle() {
        std::thread::sleep(THROTTLE_PAUSE);
    }
}

/// Emit an empty-graph job's answer directly (no GPU work needed).
pub(crate) fn answer_empty(out: &Sender<StreamResult>, job: StreamJob) {
    let reads = job.params.num_reads.max(1);
    if out
        .blocking_send(StreamResult {
            job_id: job.job_id,
            outcome: StreamOutcome::Completed(Ok((0..reads)
                .map(|_| SamplerResult {
                    spins: vec![],
                    energy_milli: 0,
                })
                .collect())),
            device_access_time_us: 0,
        })
        .is_err()
    {
        // consumer gone — nobody to tell
    }
}

/// Report a job abandoned because its watermark was cancelled. Produces no
/// `Result` upstream — the harness only refunds its credit — so the pipeline
/// keeps its depth for the live round.
fn send_cancelled(out: &Sender<StreamResult>, job: StreamJob) {
    if out
        .blocking_send(StreamResult {
            job_id: job.job_id,
            outcome: StreamOutcome::Cancelled,
            device_access_time_us: 0,
        })
        .is_err()
    {
        // consumer gone — nobody to tell
    }
}

pub(crate) fn send_reject(out: &Sender<StreamResult>, job: StreamJob, err: SampleError) {
    if out
        .blocking_send(StreamResult {
            job_id: job.job_id,
            outcome: StreamOutcome::Completed(Err(err)),
            device_access_time_us: 0,
        })
        .is_err()
    {
        // consumer gone — nobody to tell
    }
}

/// Pull the next non-empty, in-range job to seed a batch. Empty-graph jobs are
/// answered inline and oversized jobs rejected without occupying a batch slot.
///
/// [`Seed::Blocking`] waits for a job (returns `None` only on channel close);
/// [`Seed::NonBlocking`] returns `None` the moment no job is immediately
/// queued — the overlap path uses this so it never blocks while an in-flight
/// batch is un-harvested.
fn next_seed(ctx: &mut StreamCtx<'_>, seed: Seed) -> Option<StreamJob> {
    loop {
        let job = match ctx.pending.take() {
            Some(j) => j,
            None if seed == Seed::Blocking => ctx.jobs.blocking_recv()?,
            None => ctx.jobs.try_recv().ok()?,
        };
        // Drop abandoned watermarks before they reach the GPU: a reseed can
        // leave a full prefetch window of stale nonces queued, and computing
        // them would burn a dispatch on work the coordinator has moved past.
        if ctx.cancel.is_cancelled(job.watermark) {
            send_cancelled(ctx.out, job);
            continue;
        }
        if job.graph.num_nodes() == 0 {
            answer_empty(ctx.out, job);
            continue;
        }
        if job.graph.num_nodes() > kernel_max_nodes(ctx.kernel) {
            send_reject(ctx.out, job, SampleError::Capacity);
            continue;
        }
        return Some(job);
    }
}

/// Collect jobs matching `key` into `matches` up to `cap - already` more
/// slots (the seed already occupies `already` of the batch). Drains what's
/// queued and briefly waits for a fuller batch. A topology/param mismatch is
/// stashed in `pending` to seed the next batch. Returns `false` once the
/// channel has closed.
///
/// `matches` does not include the seed: the seed stays borrowed by `key`
/// (edge list) for the duration of this call, then is prepended by the caller.
fn fill_batch(
    ctx: &mut StreamCtx<'_>,
    key: &BatchKey<'_>,
    matches: &mut Vec<StreamJob>,
    already: usize,
    cap: usize,
) -> bool {
    // Steady state: the previous batch's GPU time already filled the channel,
    // so try_recv drains a near-full batch immediately. The short idle wait
    // only matters at cold start.
    let hard_cap = Instant::now() + Duration::from_secs(2);
    let idle_timeout = Duration::from_millis(50);
    let mut last_arrival = Instant::now();
    while matches.len() + already < cap && Instant::now() < hard_cap {
        match ctx.jobs.try_recv() {
            Ok(job) if ctx.cancel.is_cancelled(job.watermark) => send_cancelled(ctx.out, job),
            Ok(job) if job.graph.num_nodes() == 0 => answer_empty(ctx.out, job),
            Ok(job) if job.graph.num_nodes() > kernel_max_nodes(ctx.kernel) => {
                send_reject(ctx.out, job, SampleError::Capacity)
            }
            Ok(job) if key.matches(&job) => {
                matches.push(job);
                last_arrival = Instant::now();
            }
            Ok(job) => {
                *ctx.pending = Some(job); // different topology/params → next batch
                return true;
            }
            Err(TryRecvError::Empty) => {
                if last_arrival.elapsed() > idle_timeout {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(TryRecvError::Disconnected) => return false,
        }
    }
    true
}

/// One committed batch awaiting completion + harvest.
struct InFlight {
    encoded: sampler::EncodedBatch,
    jobs: Vec<StreamJob>,
}

impl InFlight {
    fn commit_next(&mut self, cancel: &CancelToken) -> bool {
        self.encoded.commit_next(|| {
            self.jobs
                .iter()
                .any(|job| cancel.is_cancelled(job.watermark))
        })
    }
}

/// Drive the batched streaming loop for the lifetime of `jobs`.
///
/// Keep two batches in flight, with one committed chunk per batch. After each
/// chunk completes, check cancellation before submitting its successor. Batch
/// preparation and host scoring overlap the other batch's GPU execution.
pub fn run_stream(
    device: &MetalDevice,
    kernel: Kernel,
    mut jobs: Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    gov: &dyn GpuGovernor,
    cancel: &CancelToken,
) {
    let mut pending: Option<StreamJob> = None;
    let mut ctx = StreamCtx {
        jobs: &mut jobs,
        out,
        pending: &mut pending,
        kernel,
        gov,
        cancel,
    };

    // Each batch owns at most one committed chunk. Alternate batches so host
    // submission and scoring overlap the other batch's GPU execution.
    let mut inflight = std::collections::VecDeque::with_capacity(2);
    if let Some(batch) = form_and_commit(device, &mut ctx, Seed::Blocking) {
        inflight.push_back(batch);
    }
    while let Some(mut current) = inflight.pop_front() {
        // Never wait for another seed while a batch holds coordinator credits.
        // During throttling, drain both batches before the next yield gate.
        if inflight.is_empty() && !ctx.gov.should_throttle() {
            if let Some(next) = form_and_commit(device, &mut ctx, Seed::NonBlocking) {
                inflight.push_back(next);
            }
        }
        current.encoded.wait_until_completed();
        if ctx.out.is_closed() {
            ctx.gov.record_gpu_busy_us(current.encoded.gpu_time_us());
            for batch in inflight {
                batch.encoded.wait_until_completed();
                ctx.gov.record_gpu_busy_us(batch.encoded.gpu_time_us());
            }
            return;
        }
        if current.commit_next(ctx.cancel) {
            inflight.push_back(current);
        } else {
            ctx.gov
                .record_gpu_busy_us(finish_batch(current, ctx.out, ctx.cancel));
        }
        if inflight.is_empty() {
            if let Some(next) = form_and_commit(device, &mut ctx, Seed::Blocking) {
                inflight.push_back(next);
            }
        }
    }
}

/// Collect the next batch and commit it to the GPU without waiting. With
/// [`Seed::Blocking`], waits for a seed and retries after rejecting a batch
/// or cancelling all its jobs. Returns `None` only when the job channel closes.
/// The non-blocking overlap path also returns `None` if no job is queued,
/// encoding fails, or cancellation empties the batch. The caller then finishes
/// the in-flight batch before retrying with a blocking seed.
fn form_and_commit(device: &MetalDevice, ctx: &mut StreamCtx<'_>, seed: Seed) -> Option<InFlight> {
    loop {
        if ctx.out.is_closed() {
            return None;
        }
        // Yield before taking a seed, not after: once a job is dequeued it is ours
        // to answer, and holding it through a pause would stall the coordinator's
        // credit for no benefit.
        ctx.yield_gate();
        if ctx.out.is_closed() {
            return None;
        }
        let seed_job = next_seed(ctx, seed)?;
        // Batch size follows the seed's read count: chromatic Gibbs spends
        // `num_reads` threadgroups per problem, so the same threadgroup budget is a
        // different number of problems at 64 reads than at 256.
        //
        // Scaled by the governor: the utilization ceiling, less any external load
        // while yielding. Sizing the dispatch is what actually shares the GPU —
        // a smaller grid leaves cores free for whoever else wants them, for the
        // whole duration of the dispatch rather than only in the gaps.
        let nominal = batch_size_for_reads(ctx.kernel, seed_job.params.num_reads);
        let cap = scale_budget(nominal, ctx.gov.budget_scale());
        // Keep `seed_job` live while `key` borrows its edge list; collect further
        // matches into a side vec, then assemble the full batch.
        let mut matches = Vec::with_capacity(cap.saturating_sub(1));
        {
            let key = BatchKey::from_job(&seed_job);
            fill_batch(ctx, &key, &mut matches, 1, cap);
        }
        let mut batch = Vec::with_capacity(matches.len() + 1);
        batch.push(seed_job);
        batch.append(&mut matches);

        // Last checkpoint before the GPU commits: filling a batch can take up to
        // `hard_cap`, and a `Cancel` arriving in that window would otherwise buy a
        // full dispatch of abandoned work. Re-check every job now that the batch is
        // final. Each chunk also checks cancellation immediately before commit.
        if batch.iter().any(|j| ctx.cancel.is_cancelled(j.watermark)) {
            let (live, stale): (Vec<StreamJob>, Vec<StreamJob>) = batch
                .into_iter()
                .partition(|j| !ctx.cancel.is_cancelled(j.watermark));
            for job in stale {
                send_cancelled(ctx.out, job);
            }
            batch = live;
            if batch.is_empty() {
                if seed == Seed::NonBlocking {
                    return None;
                }
                continue;
            }
        }

        // Scope `graphs` so its borrow of `batch` ends before `batch` moves.
        if ctx.out.is_closed() {
            return None;
        }
        let encoded = {
            let graphs: Vec<&IsingGraph> = batch.iter().map(|j| &j.graph).collect();
            sampler::encode_batch(device, &graphs, &batch[0].params, ctx.kernel, 2)
        };
        match encoded {
            Ok(enc) => {
                if ctx.out.is_closed() {
                    return None;
                }
                tracing::debug!(batch = batch.len(), cap, chunks = enc.chunk_count(), seed = ?seed, "prepared batch");
                let mut inflight = InFlight {
                    encoded: enc,
                    jobs: batch,
                };
                inflight.commit_next(ctx.cancel);
                return Some(inflight);
            }
            Err(e) => {
                // Every job in a batch shares the encode inputs that can be
                // refused for size (`num_sweeps` is part of the batch key, and `N`
                // is pre-filtered by `next_seed`), so a capacity refusal applies to
                // all of them alike — reject the batch with the condition the
                // failure actually carries rather than a blanket `DeviceFault`.
                let err = e.to_sample_error();
                tracing::error!(error = %e, ?err, "metal batch encode failed");
                for job in batch {
                    send_reject(ctx.out, job, err.clone());
                }
            }
        }
        if seed == Seed::NonBlocking {
            return None;
        }
    }
}

/// Wait on a committed batch, then host-score (rayon) and emit one result per
/// job. `device_access_time_us` is the true GPU execution time
/// (`GPUEndTime - GPUStartTime`), not the wall clock — the wall includes this
/// batch's overlap with host work on either side.
fn finish_batch(inflight: InFlight, out: &Sender<StreamResult>, cancel: &CancelToken) -> u64 {
    let InFlight { encoded, jobs } = inflight;

    encoded.wait_until_completed();
    let device_access_time_us = encoded.gpu_time_us();
    // The watchdog judges a single command buffer, not the batch, so the max is
    // the number that matters for staying alive.
    tracing::debug!(
        chunks = encoded.cmds.len(),
        max_chunk_ms = encoded.max_chunk_us() / 1000,
        total_ms = device_access_time_us / 1000,
        "batch complete"
    );

    if let Some(status) = encoded.failed_status() {
        tracing::error!(
            ?status,
            chunks = encoded.chunk_count(),
            "metal batch command buffer did not complete"
        );
        // A command buffer that did not complete (device reset, kernel fault,
        // GPU watchdog timeout) is a state this backend will not recover from
        // on its own — `DeviceFault`, not a transient per-job reject, so the
        // session ends for a supervisor restart. Mirrors `sample_ising`'s
        // identical check on the synchronous path.
        let err = SampleError::DeviceFault(format!(
            "metal command buffer did not complete: status {status:?}"
        ));
        for job in jobs {
            send_reject(out, job, err.clone());
        }
        // A failed batch still occupied the device; report it or the governor
        // would read the failure as idle time and size the next batch up.
        return device_access_time_us;
    }

    if jobs.iter().any(|job| cancel.is_cancelled(job.watermark)) {
        for job in jobs {
            // Every job has the same watermark, so partial state is discarded
            // without changing the layout or RNG identity of a live job.
            let _ = out.blocking_send(StreamResult {
                job_id: job.job_id,
                outcome: StreamOutcome::Cancelled,
                device_access_time_us,
            });
        }
        return device_access_time_us;
    }

    let per_problem = {
        let graphs: Vec<&IsingGraph> = jobs.iter().map(|j| &j.graph).collect();
        sampler::harvest_batch(&encoded, &graphs)
    };
    let per_problem = match per_problem {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "metal batch harvest failed");
            let err = e.to_sample_error();
            for job in jobs {
                send_reject(out, job, err.clone());
            }
            return device_access_time_us;
        }
    };

    for (job, results) in jobs.into_iter().zip(per_problem) {
        let take = job.params.num_reads.max(1);
        let outcome = StreamOutcome::Completed(Ok(results
            .into_iter()
            .take(take)
            .collect::<Vec<SamplerResult>>()));
        if out
            .blocking_send(StreamResult {
                job_id: job.job_id,
                outcome,
                device_access_time_us,
            })
            .is_err()
        {
            break; // consumer gone
        }
    }
    device_access_time_us
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use quip_solver_core::SampleParams;

    fn params(num_reads: usize, num_sweeps: usize, sweeps_per_beta: usize) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            sweeps_per_beta,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: None,
            seed: 0,
        }
    }

    fn job(
        job_id: &[u8],
        graph: IsingGraph,
        num_reads: usize,
        num_sweeps: usize,
        sweeps_per_beta: usize,
    ) -> StreamJob {
        StreamJob {
            job_id: job_id.to_vec(),
            graph,
            params: params(num_reads, num_sweeps, sweeps_per_beta),
            // generation 0 maps to watermark None (never cancelled), the same
            // rule `quip-solver-core`'s `prepare_job` applies on the real path.
            watermark: None,
        }
    }

    /// A linear job must not share a geometric batch's beta buffer.
    #[test]
    fn batch_key_separates_schedule_kinds() {
        let graph = IsingGraph::new(vec![0.0], vec![], vec![]);
        let first = job(b"first", graph.clone(), 32, 4, 1);
        let mut second = job(b"second", graph, 32, 4, 1);
        assert!(BatchKey::from_job(&first).matches(&second));
        second.params.schedule = quip_solver_core::BetaSchedule::Linear;
        assert!(!BatchKey::from_job(&first).matches(&second));
    }

    #[test]
    fn batch_key_separates_cancellation_watermarks() {
        let graph = IsingGraph::new(vec![0.0], vec![], vec![]);
        let mut first = job(b"first", graph.clone(), 32, 4, 1);
        first.watermark = Some(1);
        let mut second = job(b"second", graph, 32, 4, 1);
        second.watermark = Some(2);
        assert!(!BatchKey::from_job(&first).matches(&second));
        second.watermark = None;
        assert!(!BatchKey::from_job(&first).matches(&second));
        second.watermark = Some(1);
        assert!(BatchKey::from_job(&first).matches(&second));
    }

    #[test]
    fn cancelling_inflight_batch_refunds_each_job_and_accounts_for_executed_work() {
        let device = MetalDevice::open(0).unwrap();
        let graph = IsingGraph::new(vec![1.0; 1024], vec![], vec![]);
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            for committed in [false, true] {
                let mut first = job(b"first", graph.clone(), 128, 65536, 1);
                first.watermark = Some(1);
                let mut second = job(b"second", graph.clone(), 128, 65536, 1);
                second.watermark = Some(1);
                let encoded = sampler::encode_batch(
                    &device,
                    &[&first.graph, &second.graph],
                    &first.params,
                    kernel,
                    2,
                )
                .unwrap();
                assert!(encoded.chunk_count() > 1);
                let mut inflight = InFlight {
                    encoded,
                    jobs: vec![first, second],
                };
                let cancel = CancelToken::default();
                if committed {
                    assert!(inflight.commit_next(&cancel));
                }
                cancel.cancel_through(1);
                inflight.encoded.wait_until_completed();
                assert!(!inflight.commit_next(&cancel));
                assert_eq!(inflight.encoded.cmds.len(), usize::from(committed));
                let expected_time = inflight.encoded.gpu_time_us();
                assert_eq!(expected_time > 0, committed);
                let (out, mut results) = tokio::sync::mpsc::channel(4);
                assert_eq!(finish_batch(inflight, &out, &cancel), expected_time);
                for id in [b"first".as_slice(), b"second".as_slice()] {
                    let result = results.try_recv().unwrap();
                    assert_eq!(result.job_id, id);
                    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
                    assert_eq!(result.device_access_time_us, expected_time);
                }
                assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
            }
        }
    }

    #[test]
    fn run_stream_cancels_a_committed_batch_and_completes_the_live_watermark() {
        struct CancelOnSecondBatch {
            cancel: CancelToken,
            batches: std::cell::Cell<usize>,
            busy_us: std::cell::Cell<u64>,
        }
        impl GpuGovernor for CancelOnSecondBatch {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                self.batches.set(self.batches.get() + 1);
                if self.batches.get() == 2 {
                    // The stream commits the first chunk before collecting
                    // this batch, so cancellation exercises the resume path.
                    self.cancel.cancel_through(1);
                }
                1.0
            }
            fn record_gpu_busy_us(&self, us: u64) {
                self.busy_us.set(self.busy_us.get() + us);
            }
        }
        let device = MetalDevice::open(0).unwrap();
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let graph = IsingGraph::new(vec![1.0; 1024], vec![], vec![]);
            let mut stale = job(b"stale", graph.clone(), 128, 65536, 1);
            stale.watermark = Some(1);
            let mut live = job(b"live", graph, 32, 1, 1);
            live.watermark = Some(2);
            let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
            job_tx.blocking_send(stale).unwrap();
            job_tx.blocking_send(live).unwrap();
            drop(job_tx);
            let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(4);
            let cancel = CancelToken::default();
            let gov = CancelOnSecondBatch {
                cancel: cancel.clone(),
                batches: std::cell::Cell::new(0),
                busy_us: std::cell::Cell::new(0),
            };
            run_stream(&device, kernel, job_rx, &out_tx, &gov, &cancel);
            let stale = out_rx.try_recv().unwrap();
            assert_eq!(stale.job_id, b"stale");
            assert!(matches!(stale.outcome, StreamOutcome::Cancelled));
            assert!(stale.device_access_time_us > 0);
            let live = out_rx.try_recv().unwrap();
            assert_eq!(live.job_id, b"live");
            let StreamOutcome::Completed(Ok(samples)) = live.outcome else {
                panic!("live watermark did not complete");
            };
            assert_eq!(samples.len(), 32);
            assert!(samples.iter().all(|sample| sample.spins.len() == 1024));
            assert_eq!(
                gov.busy_us.get(),
                stale.device_access_time_us + live.device_access_time_us
            );
            assert!(matches!(out_rx.try_recv(), Err(TryRecvError::Empty)));
        }
    }

    #[test]
    fn closed_result_channel_does_not_dispatch_queued_jobs() {
        struct NoDispatch;
        impl GpuGovernor for NoDispatch {
            fn should_throttle(&self) -> bool {
                panic!("closed result channel must stop before the dispatch gate");
            }
            fn budget_scale(&self) -> f64 {
                panic!("closed result channel must not form a batch");
            }
            fn record_gpu_busy_us(&self, _: u64) {
                panic!("no GPU work should run");
            }
        }
        let device = MetalDevice::open(0).unwrap();
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(1);
        job_tx
            .blocking_send(job(
                b"queued",
                IsingGraph::new(vec![1.0], vec![], vec![]),
                128,
                16384,
                1,
            ))
            .unwrap();
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(1);
        drop(out_rx);
        run_stream(
            &device,
            Kernel::Msa,
            job_rx,
            &out_tx,
            &NoDispatch,
            &CancelToken::default(),
        );
        assert!(job_tx.is_closed());
    }

    /// A dispatch must never scale to zero problems: an empty batch makes no
    /// progress and never frees the coordinator's credit.
    #[test]
    fn scale_budget_never_reaches_zero() {
        assert_eq!(scale_budget(4, 0.0), 1);
        assert_eq!(scale_budget(1, 0.5), 1);
        assert_eq!(scale_budget(0, 1.0), 1);
    }

    #[test]
    fn scale_budget_halves_and_saturates() {
        assert_eq!(scale_budget(240, 0.5), 120);
        assert_eq!(scale_budget(240, 0.8), 192);
        // At or above 1.0 the nominal budget passes through untouched.
        assert_eq!(scale_budget(240, 1.0), 240);
        assert_eq!(scale_budget(240, 2.0), 240);
        // A NaN scale (impossible from the governor, but the cast would be UB)
        // degrades to the nominal budget rather than poisoning the batch size.
        assert_eq!(scale_budget(240, f64::NAN), 240);
    }

    /// Governor stub: never throttles, never scales. Keeps the batch-forming
    /// tests measuring batching logic rather than governor state.
    struct NoGovernor;

    impl GpuGovernor for NoGovernor {
        fn should_throttle(&self) -> bool {
            false
        }
        fn budget_scale(&self) -> f64 {
            1.0
        }
        fn record_gpu_busy_us(&self, _us: u64) {}
    }

    fn ring4() -> IsingGraph {
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    #[test]
    fn batch_key_matches_identical_job() {
        let j = job(b"a", ring4(), 16, 64, 1);
        let key = BatchKey::from_job(&j);
        assert!(key.matches(&j));
    }

    #[test]
    fn batch_key_rejects_different_edges() {
        let j = job(b"a", ring4(), 16, 64, 1);
        let key = BatchKey::from_job(&j);
        let mut other = job(b"b", ring4(), 16, 64, 1);
        other.graph.edges = vec![(0, 1), (1, 2), (2, 3)].into(); // drop one edge
        assert!(!key.matches(&other));
    }

    #[test]
    fn batch_key_rejects_different_n() {
        let j = job(b"a", ring4(), 16, 64, 1);
        let key = BatchKey::from_job(&j);
        let other = job(
            b"b",
            IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]),
            16,
            64,
            1,
        );
        assert!(!key.matches(&other));
    }

    #[test]
    fn batch_key_rejects_different_num_sweeps() {
        let j = job(b"a", ring4(), 16, 64, 1);
        let key = BatchKey::from_job(&j);
        let other = job(b"b", ring4(), 16, 128, 1);
        assert!(!key.matches(&other));
    }

    #[test]
    fn batch_key_matches_out_of_range_num_reads_via_clamp() {
        // Both clamp to MAX_READS, so they share a batch key.
        let over = sampler::MAX_READS + 10;
        let more = sampler::MAX_READS + 50;
        let j = job(b"a", ring4(), over, 64, 1);
        let key = BatchKey::from_job(&j);
        assert_eq!(key.num_reads, sampler::MAX_READS);
        let other = job(b"b", ring4(), more, 64, 1);
        assert!(key.matches(&other));
    }

    #[test]
    fn batch_key_matches_zero_num_reads_via_clamp() {
        // 0 and 1 both clamp to 1.
        let j = job(b"a", ring4(), 0, 64, 1);
        let key = BatchKey::from_job(&j);
        assert_eq!(key.num_reads, 1);
        let other = job(b"b", ring4(), 1, 64, 1);
        assert!(key.matches(&other));
    }

    #[test]
    fn batch_key_matches_zero_sweeps_per_beta_via_max() {
        // 0 and 1 both become 1 via max(1).
        let j = job(b"a", ring4(), 16, 64, 0);
        let key = BatchKey::from_job(&j);
        assert_eq!(key.sweeps_per_beta, 1);
        let other = job(b"b", ring4(), 16, 64, 1);
        assert!(key.matches(&other));
    }

    #[test]
    fn threadgroup_budget_converts_to_problems_per_algorithm() {
        // Recompute the formula in-process rather than mutating the
        // environment (avoids cross-test races under `cargo test`).
        let cores = crate::iokit_gov::gpu_core_count()
            .unwrap_or(DEFAULT_GPU_CORES)
            .max(1);
        let env = std::env::var("QUIP_METAL_TG_PER_CORE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|m| m.is_finite() && *m > 0.0);

        // SA spends one threadgroup per problem, so problems == budget and the
        // batch does not shrink as reads grow.
        let sa_budget = ((cores as f64 * env.unwrap_or(SA_TG_PER_CORE)).round() as usize).max(1);
        assert_eq!(batch_size_for_reads(Kernel::Sa, 64), sa_budget);
        assert_eq!(batch_size_for_reads(Kernel::Sa, 256), sa_budget);

        // Chromatic Gibbs spends `num_reads` threadgroups per problem, so the
        // same budget buys 4x fewer problems at 4x the reads. This is the whole
        // point of budgeting in threadgroups: one constant, one meaning.
        if sampler::gibbs_node_parallel() {
            let g_budget =
                ((cores as f64 * env.unwrap_or(GIBBS_TG_PER_CORE)).round() as usize).max(1);
            assert_eq!(
                batch_size_for_reads(Kernel::Gibbs, 64),
                g_budget.div_ceil(64).max(1)
            );
            assert_eq!(
                batch_size_for_reads(Kernel::Gibbs, 256),
                g_budget.div_ceil(256).max(1)
            );
        }
        assert!(batch_size_for_reads(Kernel::Gibbs, 4096) >= 1, "never zero");

        // Multi-spin spends one threadgroup per 32-lane word, so 128 reads
        // cost four threadgroups per problem and 32 reads cost one.
        let m_budget = ((cores as f64 * env.unwrap_or(MSA_TG_PER_CORE)).round() as usize).max(1);
        assert_eq!(
            batch_size_for_reads(Kernel::Msa, 128),
            m_budget.div_ceil(4).max(1)
        );
        assert_eq!(batch_size_for_reads(Kernel::Msa, 32), m_budget);
        assert_eq!(
            batch_size_for_reads(Kernel::Msa, 256),
            m_budget.div_ceil(8).max(1)
        );
    }

    #[test]
    fn declared_width_uses_each_kernels_nominal_reads() {
        // The width is sized before any job arrives, from the smallest read
        // count the adapt envelope issues, so it follows the envelope.
        assert_eq!(
            nominal_reads(Kernel::Sa),
            crate::METAL_ADAPT.min_reads as usize
        );
        assert_eq!(
            nominal_reads(Kernel::Gibbs),
            crate::METAL_ADAPT.min_reads as usize
        );
        assert_eq!(
            nominal_reads(Kernel::Msa),
            crate::METAL_MSA_ADAPT.min_reads as usize
        );
        assert_eq!(nominal_reads(Kernel::Msa), 64);
        assert_eq!(
            declared_stream_width(Kernel::Msa),
            batch_size_for_reads(Kernel::Msa, 64) * 2 + crate::resident::PREP_BOUND
        );
    }

    /// The session keeps at most `stream_width` salts in flight. The resident
    /// runner holds two live slot pools plus `PREP_BOUND` jobs in preparation,
    /// so a narrower window leaves slots idle between salts.
    #[test]
    fn msa_width_covers_live_slots_and_preparation() {
        let batch = batch_size_for_reads(Kernel::Msa, nominal_reads(Kernel::Msa));
        assert_eq!(
            declared_stream_width(Kernel::Msa),
            2 * batch + crate::resident::PREP_BOUND
        );
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            assert_eq!(
                declared_stream_width(kernel),
                (batch_size_for_reads(kernel, nominal_reads(kernel)) * 2).max(1)
            );
        }
    }

    #[test]
    fn reads_round_up_to_a_full_simdgroup() {
        // Partial simdgroups leave lanes idle that are issued anyway.
        assert_eq!(sampler::simd_rounded_reads(1), 32);
        assert_eq!(sampler::simd_rounded_reads(16), 32);
        assert_eq!(sampler::simd_rounded_reads(33), 64);
        // Already aligned counts are untouched, and the cap is never exceeded.
        assert_eq!(sampler::simd_rounded_reads(64), 64);
        assert_eq!(
            sampler::simd_rounded_reads(sampler::MAX_READS),
            sampler::MAX_READS
        );
        assert_eq!(sampler::simd_rounded_reads(usize::MAX), sampler::MAX_READS);
    }

    #[test]
    fn next_seed_answers_empty_graph() {
        let (job_tx, mut job_rx) = tokio::sync::mpsc::channel(4);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(4);
        let empty = job(b"empty", IsingGraph::new(vec![], vec![], vec![]), 3, 64, 1);
        job_tx.blocking_send(empty).unwrap();
        drop(job_tx);

        let mut pending = None;
        let mut ctx = StreamCtx {
            jobs: &mut job_rx,
            out: &out_tx,
            pending: &mut pending,
            kernel: Kernel::Sa,
            gov: &NoGovernor,
            cancel: &CancelToken::default(),
        };
        // Empty job is answered inline; channel then closes → None.
        assert!(next_seed(&mut ctx, Seed::Blocking).is_none());

        let Some(r) = out_rx.blocking_recv() else {
            assert_eq!("got", "empty-graph answer on out channel");
            return;
        };
        assert_eq!(r.job_id, b"empty");
        let StreamOutcome::Completed(Ok(reads)) = r.outcome else {
            assert_eq!("got", "Ok empty reads");
            return;
        };
        assert_eq!(reads.len(), 3);
        assert!(reads
            .iter()
            .all(|s| s.spins.is_empty() && s.energy_milli == 0));
        assert_eq!(r.device_access_time_us, 0);
    }

    #[test]
    fn next_seed_rejects_too_large() {
        let n = kernel_max_nodes(Kernel::Sa) + 1;
        let (job_tx, mut job_rx) = tokio::sync::mpsc::channel(4);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(4);
        let huge = job(
            b"huge",
            IsingGraph::new(vec![0.0; n], vec![], vec![]),
            1,
            64,
            1,
        );
        job_tx.blocking_send(huge).unwrap();
        drop(job_tx);

        let mut pending = None;
        let mut ctx = StreamCtx {
            jobs: &mut job_rx,
            out: &out_tx,
            pending: &mut pending,
            kernel: Kernel::Sa,
            gov: &NoGovernor,
            cancel: &CancelToken::default(),
        };
        assert!(next_seed(&mut ctx, Seed::Blocking).is_none());

        let Some(r) = out_rx.blocking_recv() else {
            assert_eq!("got", "TooLarge reject on out channel");
            return;
        };
        assert_eq!(r.job_id, b"huge");
        let is_too_large = matches!(
            r.outcome,
            StreamOutcome::Completed(Err(SampleError::Capacity))
        );
        assert!(is_too_large, "expected Capacity, got unexpected reject/ok");
    }

    #[test]
    fn run_stream_continues_after_msa_degree_rejection() {
        let device = MetalDevice::open(0).unwrap();
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(2);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(2);
        let star = IsingGraph::new(
            vec![0.0; 22],
            vec![1.0; 21],
            (1..=21).map(|leaf| (0, leaf)).collect(),
        );
        job_tx.blocking_send(job(b"star", star, 64, 32, 1)).unwrap();
        job_tx
            .blocking_send(job(b"ring", ring4(), 64, 32, 1))
            .unwrap();
        drop(job_tx);

        run_stream(
            &device,
            Kernel::Msa,
            job_rx,
            &out_tx,
            &NoGovernor,
            &CancelToken::default(),
        );
        drop(out_tx);

        let rejected = out_rx.blocking_recv().expect("star rejection");
        assert_eq!(rejected.job_id, b"star");
        assert!(matches!(
            rejected.outcome,
            StreamOutcome::Completed(Err(SampleError::Capacity))
        ));
        let completed = out_rx.blocking_recv().expect("ring result after rejection");
        assert_eq!(completed.job_id, b"ring");
        let StreamOutcome::Completed(Ok(reads)) = completed.outcome else {
            panic!("expected successful ring result");
        };
        assert_eq!(reads.len(), 64);
        assert!(out_rx.blocking_recv().is_none());
    }

    #[test]
    fn next_seed_nonblocking_empty_channel() {
        let (_job_tx, mut job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, _out_rx) = tokio::sync::mpsc::channel(1);
        let mut pending = None;
        let mut ctx = StreamCtx {
            jobs: &mut job_rx,
            out: &out_tx,
            pending: &mut pending,
            kernel: Kernel::Sa,
            gov: &NoGovernor,
            cancel: &CancelToken::default(),
        };
        assert!(next_seed(&mut ctx, Seed::NonBlocking).is_none());
    }

    #[test]
    fn next_seed_returns_in_range_job() {
        let (job_tx, mut job_rx) = tokio::sync::mpsc::channel(4);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(4);
        let j = job(b"ok", ring4(), 8, 32, 1);
        job_tx.blocking_send(j).unwrap();
        drop(job_tx);

        let mut pending = None;
        let mut ctx = StreamCtx {
            jobs: &mut job_rx,
            out: &out_tx,
            pending: &mut pending,
            kernel: Kernel::Sa,
            gov: &NoGovernor,
            cancel: &CancelToken::default(),
        };
        let Some(got) = next_seed(&mut ctx, Seed::Blocking) else {
            assert_eq!("got", "in-range seed job");
            return;
        };
        assert_eq!(got.job_id, b"ok");
        assert_eq!(got.graph.num_nodes(), 4);
        // No reject / empty answer emitted.
        assert!(out_rx.try_recv().is_err());
    }

    // -----------------------------------------------------------------------
    // Property tests (proptest is a dev-dep; private fns are reachable here)
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        /// Floor of one problem: NaN, ±inf, 0, negatives, and all other scales.
        #[test]
        fn scale_budget_never_zero_for_any_f64(
            nominal in any::<usize>(),
            scale in proptest::num::f64::ANY
        ) {
            prop_assert!(
                scale_budget(nominal, scale) >= 1,
                "scale_budget({}, {:?}) returned 0",
                nominal,
                scale
            );
        }

        /// Problem batch size is always at least one, for any read count.
        #[test]
        fn batch_size_for_reads_never_zero(
            kernel in prop_oneof![Just(Kernel::Sa), Just(Kernel::Msa), Just(Kernel::Gibbs)],
            num_reads in any::<usize>()
        ) {
            prop_assert!(
                batch_size_for_reads(kernel, num_reads) >= 1,
                "batch_size_for_reads({:?}, {}) returned 0",
                kernel,
                num_reads
            );
        }
    }
}
