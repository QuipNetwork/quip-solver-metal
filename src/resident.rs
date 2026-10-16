// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Two alternating pools keep cascade jobs resident between checkpoints.

use crate::cascade::{
    topology_fingerprint, CascadeSettings, Controller, PreparedSchedule, ScheduleCache, Ticket,
    CHAIN_GATES, MAX_STAGES,
};
use crate::metal_device::MetalDevice;
use crate::sampler::{self, Kernel, SampleError};
use crate::screen::{self, Screener, Verdict};
use crate::slots::{Edges, PreparedInputs, SlotPool};
use crate::streaming::{batch_size_for_reads, scale_budget, GpuGovernor};
use crate::topology::SelfFeedingTopology;
use quip_solver_core::quip_protocol::lease::TopologyView;
use quip_solver_core::{
    CancelToken, IsingGraph, SampleParams, SamplerResult, StreamJob, StreamOutcome, StreamResult,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

/// Sentinel stored in [`LiveTarget`] for "no target". Real targets are
/// milli-energies; the benchmark and test fixtures span roughly
/// -15,000,000..=i64::MAX, never this value, so it is safe to reserve.
const NO_TARGET: i64 = i64::MIN;

/// A lease's session target, shared by every unit the lease has in flight.
///
/// `MetalSampler::sample_lease` (src/lib.rs) owns the write side: its
/// intake loop already runs continuously while the lease is open, so it
/// refreshes this cell from `LeaseSink::target_energy_milli` (the live
/// `watch` channel quip-solver-core keeps for the session) on every pass
/// instead of capturing the target once at enqueue time. The resident
/// runner (this module) only ever reads it, at each checkpoint/harvest, via
/// a relaxed atomic load — no lock, no polling thread of its own.
#[derive(Clone)]
pub(crate) struct LiveTarget(Arc<AtomicI64>);

impl LiveTarget {
    pub(crate) fn new(target_milli: Option<i64>) -> Self {
        Self(Arc::new(AtomicI64::new(target_milli.unwrap_or(NO_TARGET))))
    }

    /// Refresh the shared cell. Called from the lease's intake loop.
    pub(crate) fn set(&self, target_milli: Option<i64>) {
        self.0
            .store(target_milli.unwrap_or(NO_TARGET), Ordering::Relaxed);
    }

    /// The current live target. Called at each checkpoint/harvest.
    pub(crate) fn get(&self) -> Option<i64> {
        match self.0.load(Ordering::Relaxed) {
            NO_TARGET => None,
            value => Some(value),
        }
    }
}

/// How many of a lease's best-finishing salts are streamed back with reads
/// instead of merely screened. Streaming as units finish (rather than
/// holding the set to lease end) means a round cancel does not drop them.
/// A lease's total `Result` count is its final-checkpoint/target-hit
/// survivors (always pushed) plus the salts that ever entered this running
/// top-N as a screen: expected entrants are about `k + k*ln(n/k)` for
/// `k = LEASE_TOP_N` and `n` salts, roughly 125 for a 1,000,000-salt lease.
pub(crate) const LEASE_TOP_N: usize = 10;

/// A lease's running top-[`LEASE_TOP_N`] finished energies, shared by every
/// unit of that lease still in flight. Fixed-size and allocation-free: a new
/// lease starts empty via [`LeaseTopK::new`].
#[derive(Clone)]
pub(crate) struct LeaseTopK(Arc<Mutex<TopKState>>);

struct TopKState {
    /// Ascending (best/lowest first); only `len` entries are meaningful.
    energies: [i64; LEASE_TOP_N],
    len: usize,
}

impl LeaseTopK {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Mutex::new(TopKState {
            energies: [0; LEASE_TOP_N],
            len: 0,
        })))
    }

    /// Whether `energy_milli` would enter the running top
    /// [`LEASE_TOP_N`], without entering it.
    fn qualifies(&self, energy_milli: i64) -> bool {
        let state = self.0.lock().unwrap_or_else(|p| p.into_inner());
        state.len < LEASE_TOP_N || energy_milli < state.energies[LEASE_TOP_N - 1]
    }

    /// Offers a finished unit's best energy to the lease's running top
    /// [`LEASE_TOP_N`]. Returns whether it enters. A tie with the current
    /// worst kept entry does not enter. Inserts in sorted position when it
    /// does; no allocation.
    fn offer(&self, energy_milli: i64) -> bool {
        let mut state = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let TopKState { energies, len } = &mut *state;
        if *len < LEASE_TOP_N {
            let pos = energies[..*len].partition_point(|&e| e <= energy_milli);
            energies.copy_within(pos..*len, pos + 1);
            energies[pos] = energy_milli;
            *len += 1;
            true
        } else if energy_milli < energies[LEASE_TOP_N - 1] {
            let pos = energies[..*len].partition_point(|&e| e <= energy_milli);
            energies.copy_within(pos..LEASE_TOP_N - 1, pos + 1);
            energies[pos] = energy_milli;
            true
        } else {
            false
        }
    }
}

struct Live {
    job: StreamJob,
    origin: Origin,
    ticket: Ticket,
    accounted_us: u64,
    admitted: Instant,
    /// The target last seen for this unit, to log only on a change.
    last_target_milli: Option<i64>,
}

pub(crate) enum SaltOutcome {
    Screened {
        index: u64,
        energy_milli: i64,
    },
    Survived {
        index: u64,
        reads: Vec<SamplerResult>,
    },
    /// Cancel or shutdown ended the salt before it finished.
    Dropped {
        index: u64,
    },
    /// Drawing, preparing, or running the salt failed.
    Failed {
        index: u64,
        error: quip_solver_core::SampleError,
    },
}

pub(crate) struct Salt {
    pub(crate) topology: Arc<TopologyView>,
    pub(crate) nonce: [u8; 32],
    pub(crate) index: u64,
    /// The miner's lease round when this salt was fed; the cascade report
    /// covers one round at a time.
    pub(crate) round: u64,
    pub(crate) params: SampleParams,
    /// Set when the issuing lease ends. Its queued and live units then stop
    /// and answer `Dropped`.
    pub(crate) stop: Arc<AtomicBool>,
    /// Shared handle to the lease's live target; `MetalSampler::sample_lease`
    /// keeps it fresh for as long as the lease is open.
    pub(crate) target: LiveTarget,
    /// Shared handle to the lease's running top-[`LEASE_TOP_N`] finished
    /// energies; a new lease starts this empty.
    pub(crate) top10: LeaseTopK,
    pub(crate) reply: mpsc::Sender<SaltOutcome>,
}

pub(crate) enum Source {
    Job(StreamJob),
    Salt(Salt),
}

enum Origin {
    Stream,
    Salt {
        index: u64,
        round: u64,
        target: LiveTarget,
        top10: LeaseTopK,
        stop: Arc<AtomicBool>,
        reply: mpsc::Sender<SaltOutcome>,
    },
}

impl Origin {
    /// Whether the lease that issued this unit has ended.
    fn stopped(&self) -> bool {
        match self {
            Self::Stream => false,
            Self::Salt { stop, .. } => stop.load(Ordering::Acquire),
        }
    }

    /// Offers `energy_milli` to this unit's lease's running top
    /// [`LEASE_TOP_N`]. Always `false` for a plain stream job, which has no
    /// lease.
    fn offer_top10(&self, energy_milli: i64) -> bool {
        match self {
            Self::Stream => false,
            Self::Salt { top10, .. } => top10.offer(energy_milli),
        }
    }
}

/// Whether nobody wants the unit's answer: its generation was cancelled or
/// its lease ended.
fn abandoned(cancel: &CancelToken, job: &StreamJob, origin: &Origin) -> bool {
    cancel.is_cancelled(job.watermark) || origin.stopped()
}

enum Answer {
    Reads(Vec<SamplerResult>),
    Screened(i64),
    Cancelled,
    Failed(quip_solver_core::SampleError),
}

fn answer(
    out: &Sender<StreamResult>,
    origin: Origin,
    job: StreamJob,
    outcome: Answer,
    device_access_time_us: u64,
) -> bool {
    match origin {
        Origin::Stream => {
            let outcome = match outcome {
                Answer::Reads(reads) => StreamOutcome::Completed(Ok(reads)),
                Answer::Screened(_) => {
                    debug_assert!(false, "stream jobs cannot be screened");
                    StreamOutcome::Completed(Ok(Vec::new()))
                }
                Answer::Cancelled => StreamOutcome::Cancelled,
                Answer::Failed(error) => StreamOutcome::Completed(Err(error)),
            };
            out.blocking_send(StreamResult {
                job_id: job.job_id,
                outcome,
                device_access_time_us,
            })
            .is_ok()
        }
        Origin::Salt { index, reply, .. } => {
            let outcome = match outcome {
                Answer::Reads(reads) => SaltOutcome::Survived { index, reads },
                Answer::Screened(energy_milli) => SaltOutcome::Screened {
                    index,
                    energy_milli,
                },
                Answer::Cancelled => SaltOutcome::Dropped { index },
                Answer::Failed(error) => SaltOutcome::Failed { index, error },
            };
            reply.send(outcome).is_ok()
        }
    }
}

/// The stream job for a salt whose draw gave `h_milli` and `j_milli`.
fn drawn_job(salt: &Salt, h_milli: &[i32], j_milli: &[i32]) -> StreamJob {
    let to_units = |values: &[i32]| {
        values
            .iter()
            .map(|&value| f64::from(value) / 1000.0)
            .collect()
    };
    let mut params = salt.params.clone();
    params.seed = salt_seed(salt.nonce);
    StreamJob {
        job_id: salt.index.to_le_bytes().to_vec(),
        graph: IsingGraph::new(
            to_units(h_milli),
            to_units(j_milli),
            salt.topology.edges.clone(),
        ),
        params,
        watermark: None,
    }
}

/// The job a salt answers with when it has no model: a failed draw or a
/// panic during preparation.
fn modelless_job(salt: &Salt) -> StreamJob {
    StreamJob {
        job_id: salt.index.to_le_bytes().to_vec(),
        graph: IsingGraph::new(Vec::new(), Vec::new(), Vec::new()),
        params: salt.params.clone(),
        watermark: None,
    }
}

/// The nonce is unique to the lease and salt, so concurrent leases never share a seed.
fn salt_seed(nonce: [u8; 32]) -> u64 {
    use std::hash::BuildHasher;
    static KEYS: std::sync::OnceLock<std::collections::hash_map::RandomState> =
        std::sync::OnceLock::new();
    KEYS.get_or_init(Default::default).hash_one(nonce).max(1)
}

/// Most sweeps a slot in a fresh deep leg advances per step. Screening slots
/// keep the first stage's step, which sets the step's pace. On an M4 Max with
/// 20-slot pools (`slots::tests::deep_slice_bench`), 16 nearly doubles a deep
/// slot's rate (about 8k to 14k sweeps per second, a 2.1M-sweep unit in 145 s
/// instead of 265 s) for 9 to 12 percent of screen throughput with one to four
/// deep slots live. Larger steps gain little: a deep slot peaks near its own
/// threadgroup's rate, and every step then waits on it.
const DEEP_SLICE: usize = 16;

struct Pool {
    slots: SlotPool,
    live: Vec<Option<Live>>,
    reads: usize,
    stride: usize,
}

impl Pool {
    /// `stride` is the longest schedule a slot holds: a job's prepared
    /// schedule, whose fresh legs together run longer than its sweep budget.
    fn new(
        device: &MetalDevice,
        job: &StreamJob,
        capacity: usize,
        stride: usize,
    ) -> Result<Self, SampleError> {
        let reads = job.params.num_reads.clamp(1, sampler::MAX_READS);
        Ok(Self {
            slots: SlotPool::new(device, &job.graph, reads, capacity, stride)?,
            live: (0..capacity).map(|_| None).collect(),
            reads,
            stride,
        })
    }

    fn fits(&mut self, job: &StreamJob, edges: &Edges, schedule_len: usize) -> bool {
        self.slots.matches_prepared(
            edges,
            job.graph.num_nodes(),
            job.params.num_reads.clamp(1, sampler::MAX_READS),
        ) && schedule_len <= self.stride
    }

    fn harvest(
        &mut self,
        controller: &mut Controller,
        out: &Sender<StreamResult>,
        cancel: &CancelToken,
        gov: &dyn GpuGovernor,
    ) -> Result<u64, SampleError> {
        self.slots.wait();
        let checkpoints = self.slots.take_checkpoints()?;
        let mut busy_us = 0u64;
        for (slot, entry) in self.live.iter_mut().enumerate() {
            let Some(live) = entry else { continue };
            let us = self.slots.device_us(slot);
            busy_us = busy_us.saturating_add(us.saturating_sub(live.accounted_us));
            live.accounted_us = us;
            if abandoned(cancel, &live.job, &live.origin) {
                self.slots.release(slot)?;
                let Some(live) = entry.take() else { continue };
                let _ = answer(out, live.origin, live.job, Answer::Cancelled, us);
                controller.finish(&live.ticket, None, false);
            }
        }
        gov.record_gpu_busy_us(busy_us);
        let mut done = Vec::with_capacity(checkpoints.len());
        // One summary per distinct (old, new) pair per harvest, not one per
        // unit: several units of the same lease can each observe the same
        // target change in this batch, and that is one event. Concurrent
        // leases with different targets still each get their own line.
        let mut target_changes: Vec<(Option<i64>, Option<i64>, u32)> = Vec::new();
        for checkpoint in checkpoints {
            let Some(live) = self.live[checkpoint.slot].as_mut() else {
                continue;
            };
            let target_milli = match &live.origin {
                Origin::Stream => None,
                Origin::Salt { target, .. } => target.get(),
            };
            if target_milli != live.last_target_milli {
                match target_changes
                    .iter_mut()
                    .find(|(old, new, _)| *old == live.last_target_milli && *new == target_milli)
                {
                    Some((.., units)) => *units += 1,
                    None => target_changes.push((live.last_target_milli, target_milli, 1)),
                }
                live.last_target_milli = target_milli;
            }
            if checkpoint.observe {
                // An observe-only readback between real checkpoints: same
                // strict chain rule as a gate hit, but no gate, no stage
                // advance, and no controller stats — it is not a checkpoint.
                if target_milli.is_some_and(|target| checkpoint.best < target) {
                    live.origin.offer_top10(checkpoint.best);
                    let index = match &live.origin {
                        Origin::Salt { index, .. } => Some(*index),
                        Origin::Stream => None,
                    };
                    tracing::info!(
                        target: "quip_solver_metal::cascade_trace",
                        ?index,
                        sweeps = checkpoint.position,
                        best = checkpoint.best,
                        elapsed_ms = live.admitted.elapsed().as_millis() as u64,
                        at = "observe",
                        "salt finished on a target hit"
                    );
                    done.push(checkpoint.slot);
                }
                continue;
            }
            debug_assert_eq!(checkpoint.index, live.ticket.stage);
            tracing::debug!(
                target: "quip_solver_metal::cascade_trace",
                job = %String::from_utf8_lossy(&live.job.job_id),
                stage = checkpoint.index,
                best = checkpoint.best,
                "checkpoint"
            );
            if checkpoint.last {
                live.origin.offer_top10(checkpoint.best);
                done.push(checkpoint.slot);
            } else if !controller.checkpoint(&mut live.ticket, checkpoint.best, target_milli) {
                if let Some(sweeps) = live.ticket.finish_sweeps {
                    live.origin.offer_top10(checkpoint.best);
                    let index = match &live.origin {
                        Origin::Salt { index, .. } => Some(*index),
                        Origin::Stream => None,
                    };
                    tracing::info!(
                        target: "quip_solver_metal::cascade_trace",
                        ?index,
                        sweeps,
                        best = checkpoint.best,
                        elapsed_ms = live.admitted.elapsed().as_millis() as u64,
                        at = "gate",
                        "salt finished on a target hit"
                    );
                    done.push(checkpoint.slot);
                    continue;
                }
                // A gate-screened unit whose energy still enters the lease's
                // running top-N is read back and pushed like a survivor,
                // instead of released without read-back. `done` decodes and
                // answers it below through the same path as a real survivor.
                if live.origin.offer_top10(checkpoint.best) {
                    done.push(checkpoint.slot);
                    continue;
                }
                self.slots.release(checkpoint.slot)?;
                let Some(live) = self.live[checkpoint.slot].take() else {
                    continue;
                };
                let delivered = answer(
                    out,
                    live.origin,
                    live.job,
                    Answer::Screened(checkpoint.best),
                    live.accounted_us,
                );
                controller.finish(&live.ticket, Some(checkpoint.best), delivered);
            }
        }
        for (old, new, units) in target_changes {
            tracing::debug!(
                target: "quip_solver_metal::cascade_trace",
                ?old,
                ?new,
                units,
                "live target changed"
            );
        }
        // Decode completed slots and gate-screened slots that qualify for the
        // lease's running top-N; screened-out slots without a qualifying
        // energy are released above without read-back.
        let reads = self.slots.reads_many(&done, self.reads)?;
        for (slot, reads) in done.into_iter().zip(reads) {
            self.slots.release(slot)?;
            let Some(live) = self.live[slot].take() else {
                continue;
            };
            let cancelled = abandoned(cancel, &live.job, &live.origin);
            let best = if cancelled {
                None
            } else {
                reads.iter().map(|r| r.energy_milli).min()
            };
            // bh1.3.7: StreamResult needs sweeps_done to report the actual checkpoint position.
            let delivered = answer(
                out,
                live.origin,
                live.job,
                if cancelled {
                    Answer::Cancelled
                } else {
                    Answer::Reads(reads)
                },
                live.accounted_us,
            );
            controller.finish(&live.ticket, best, delivered && !cancelled);
        }
        Ok(busy_us)
    }
}

fn validate(job: &StreamJob) -> Result<(), SampleError> {
    if job.graph.num_nodes() > sampler::kernel_max_nodes(Kernel::Msa)
        || job.params.num_sweeps > sampler::MAX_SWEEPS
    {
        return Err(SampleError::TooLarge(
            "resident job exceeds node or sweep limit".into(),
        ));
    }
    if job.graph.j.len() != job.graph.edges.len() {
        return Err(SampleError::TooLarge(
            "resident jobs require one coefficient per edge".into(),
        ));
    }
    Ok(())
}

pub(crate) struct Prepared {
    pub(crate) job: StreamJob,
    origin: Origin,
    /// None selects the non-slot path, including direct empty-graph answers.
    pub(crate) data: Result<Option<PreparedData>, SampleError>,
}

pub(crate) struct PreparedData {
    pub(crate) schedule: PreparedSchedule,
    pub(crate) inputs: PreparedInputs,
}

/// The screen stages of one lease topology, shared by its salts.
pub(crate) struct ScreenPlan {
    /// The topology with unit couplings, to build the screener's buffers.
    graph: IsingGraph,
    /// One standard anneal per [`CHAIN_GATES`] screen stage.
    schedules: Vec<Vec<f32>>,
}

impl ScreenPlan {
    /// Each preparation worker builds its own plan; equal plans share a
    /// screener.
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.graph.edges, &other.graph.edges) && self.schedules == other.schedules
    }
}

/// A salt for the nonce-parallel screen: its edge sign bits only.
pub(crate) struct Candidate {
    salt: Salt,
    plan: Arc<ScreenPlan>,
    bits: Vec<u32>,
}

/// What preparation hands the runner.
#[expect(
    clippy::large_enum_variant,
    reason = "moved once per job through a channel; boxing adds an allocation per job"
)]
pub(crate) enum Ready {
    Unit(Prepared),
    Screen(Candidate),
}

#[cfg(test)]
impl Ready {
    fn unit(self) -> Prepared {
        match self {
            Self::Unit(prepared) => prepared,
            Self::Screen(_) => panic!("expected a slot unit, got a screen candidate"),
        }
    }
}

/// What [`Preparer::prepare_salt`] made of a salt.
#[expect(
    clippy::large_enum_variant,
    reason = "returned once per salt; boxing adds an allocation per salt"
)]
enum SaltPrep {
    Unit(StreamJob, Result<Option<PreparedData>, SampleError>),
    Screen(Arc<ScreenPlan>, Vec<u32>),
}

/// The last prepared topology. Workers share it, so every job on one topology
/// carries the same edge storage and the runner can skip its edge compares.
struct PreparedTopology {
    topology: SelfFeedingTopology,
    edges: Edges,
    /// [`crate::cascade::node_degrees`] of `edges`, for the beta range.
    degrees: Vec<u32>,
    /// Whether this is the chain topology.
    chain: bool,
}

type SharedTopology = Arc<Mutex<Option<Arc<PreparedTopology>>>>;

/// What one worker learned about the current lease topology from the
/// allowed values, so its salts skip the per-salt checks.
struct SaltTopology {
    /// Held, so pointer equality identifies the lease's view.
    view: Arc<TopologyView>,
    prepared: Arc<PreparedTopology>,
    /// The allowed values in whole device units, when every draw has exact
    /// device energies and the edges meet
    /// [`crate::topology::fill_couplings`]'s precondition.
    units: Option<(Vec<i8>, Vec<i8>)>,
    /// Every draw has couplings ±1 and fields 0 or ±1.
    unit: bool,
    /// Set when every draw has couplings ±1 and zero fields on the chain
    /// topology, which the screen takes.
    screen: Option<Arc<ScreenPlan>>,
}

impl SaltTopology {
    /// The screen plan for `salt` when the screen takes it.
    fn screen_for(
        &self,
        salt: &Salt,
        settings: CascadeSettings,
        screen: bool,
    ) -> Option<&Arc<ScreenPlan>> {
        self.screen.as_ref().filter(|_| {
            screen
                && settings.screen
                && !settings.deep
                && !settings.open_gates
                && settings.chain_gated(self.prepared.chain, &salt.params)
        })
    }

    /// `None` when `prepared` is not `view`'s topology.
    fn learn(view: &Arc<TopologyView>, prepared: Arc<PreparedTopology>) -> Option<Self> {
        let n = view.num_nodes;
        if n != prepared.topology.n || !view.edges.iter().eq(prepared.edges.iter()) {
            return None;
        }
        let whole = |values: &[i32]| {
            values
                .iter()
                .all(|&v| v % 1000 == 0 && (-128..=127).contains(&(v / 1000)))
        };
        let largest = |values: &[i32]| {
            values
                .iter()
                .map(|&v| f64::from(v.unsigned_abs() / 1000))
                .fold(0.0, f64::max)
        };
        let units = n as f64 * largest(&view.allowed_h_milli)
            + view.edges.len() as f64 * largest(&view.allowed_j_milli);
        let simple_edges = view.edges.iter().all(|&(u, v)| u != v && u < n && v < n);
        let exact = simple_edges
            && whole(&view.allowed_h_milli)
            && whole(&view.allowed_j_milli)
            && units <= sampler::DEVICE_ENERGY_MAX_UNITS;
        let to_units = |values: &[i32]| values.iter().map(|&v| (v / 1000) as i8).collect();
        let unit_j = view.allowed_j_milli.iter().all(|&j| j.abs() == 1000);
        let screen =
            (exact && unit_j && prepared.chain && view.allowed_h_milli == [0]).then(|| {
                let graph = IsingGraph {
                    h: vec![0.0; n],
                    j: vec![1.0; view.edges.len()],
                    edges: Arc::clone(&prepared.edges),
                };
                let range = crate::cascade::unit_beta_range(
                    std::iter::repeat_n(false, n),
                    &prepared.degrees,
                );
                let schedules = CHAIN_GATES
                    .screen
                    .iter()
                    .map(|&(sweeps, _)| {
                        let params = SampleParams {
                            num_reads: CHAIN_GATES.min_reads,
                            num_sweeps: sweeps,
                            sweeps_per_beta: 1,
                            schedule: quip_solver_core::BetaSchedule::Geometric,
                            beta_range: Some(range),
                            seed: 1,
                        };
                        crate::cascade::segment_schedule(&graph, &params, &[]).0
                    })
                    .collect();
                Arc::new(ScreenPlan { graph, schedules })
            });
        Some(Self {
            view: Arc::clone(view),
            prepared,
            units: exact.then(|| {
                (
                    to_units(&view.allowed_h_milli),
                    to_units(&view.allowed_j_milli),
                )
            }),
            screen,
            unit: unit_j
                && view
                    .allowed_h_milli
                    .iter()
                    .all(|&h| h == 0 || h.abs() == 1000),
        })
    }
}

#[derive(Default)]
struct Preparer {
    topology: SharedTopology,
    schedules: ScheduleCache,
    salt: Option<SaltTopology>,
}

impl Preparer {
    fn prepare(
        &mut self,
        job: &mut StreamJob,
        settings: CascadeSettings,
        screen: bool,
    ) -> Result<Option<PreparedData>, SampleError> {
        if job.graph.num_nodes() == 0 {
            return Ok(None);
        }
        validate(job)?;
        if job.params.num_sweeps == 0 || !sampler::device_energy_exact(&job.graph) {
            sampler::validate_batch(&[&job.graph], &job.params, Kernel::Msa)?;
            return Ok(None);
        }
        let lookup = |cached: Option<&Arc<PreparedTopology>>| {
            cached.and_then(|cached| {
                PreparedInputs::new(&job.graph, &cached.topology, &cached.edges)
                    .map(|inputs| (inputs, Arc::clone(cached)))
            })
        };
        let cached = self
            .topology
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let (inputs, prepared) = match lookup(cached.as_ref()) {
            Some(hit) => hit,
            None => {
                // Build under the lock, so workers that miss together share
                // one edge storage. A miss happens once per topology.
                let mut shared = self.topology.lock().unwrap_or_else(|p| p.into_inner());
                match lookup(shared.as_ref()) {
                    Some(hit) => hit,
                    None => {
                        // Degree depends only on nodes and ordered edges. The cache is
                        // established only after validate_batch, and coefficient filling
                        // checks exact topology equality on every hit. Recounting degree
                        // would allocate and walk the same 41,515 edges on every job.
                        sampler::validate_batch(&[&job.graph], &job.params, Kernel::Msa)?;
                        // This cache contains no Metal objects. Coloring does not
                        // change the CSR positions used for coefficient order.
                        let topology = SelfFeedingTopology::build(&job.graph);
                        let edges = Arc::clone(&job.graph.edges);
                        let inputs = PreparedInputs::new(&job.graph, &topology, &edges)
                            .ok_or_else(|| {
                                SampleError::Driver("preparation topology mismatch".into())
                            })?;
                        let chain = topology_fingerprint(job.graph.num_nodes(), &job.graph.edges)
                            == CHAIN_GATES.fingerprint;
                        tracing::info!(
                            chain,
                            gated = settings.chain_gated(chain, &job.params),
                            nodes = job.graph.num_nodes(),
                            edges = job.graph.edges.len(),
                            "cascade topology prepared"
                        );
                        let prepared = Arc::new(PreparedTopology {
                            topology,
                            degrees: crate::cascade::node_degrees(
                                job.graph.num_nodes(),
                                &job.graph.edges,
                            ),
                            edges,
                            chain,
                        });
                        *shared = Some(Arc::clone(&prepared));
                        (inputs, prepared)
                    }
                }
            }
        };
        let gated = screen && settings.chain_gated(prepared.chain, &job.params);
        // Open gates measure the budget the job asked for.
        if gated && !settings.open_gates {
            job.params.num_sweeps = CHAIN_GATES.full_sweeps;
        }
        // `inputs` verified the job's edges against `prepared.edges`, edge by
        // edge, so the cached degrees are this job's.
        let beta_range = job
            .params
            .beta_range
            .is_none()
            .then(|| crate::cascade::resident_beta_range_from(&job.graph, &prepared.degrees));
        let schedule = self
            .schedules
            .prepare(job, settings, gated, screen, beta_range)?;
        Ok(Some(PreparedData { schedule, inputs }))
    }

    /// Draw and prepare a lease salt. Always returns the salt's job, so the
    /// runner can answer it. The first salt of a lease draws milli values,
    /// takes [`Self::prepare`], and teaches this worker the topology. Later
    /// salts draw straight into device units and share the lease's edges.
    fn prepare_salt(&mut self, salt: &Salt, settings: CascadeSettings, screen: bool) -> SaltPrep {
        let view = &salt.topology;
        let known = self
            .salt
            .as_ref()
            .filter(|known| Arc::ptr_eq(&known.view, view) && salt.params.num_sweeps != 0);
        let Some((known, (allowed_h, allowed_j))) =
            known.and_then(|known| known.units.as_ref().map(|units| (known, units)))
        else {
            let (h_milli, j_milli) = match view.draw(salt.nonce) {
                Ok(drawn) => drawn,
                Err(error) => {
                    let error = SampleError::Driver(format!("lease draw: {error}"));
                    return SaltPrep::Unit(modelless_job(salt), Err(error));
                }
            };
            let mut job = drawn_job(salt, &h_milli, &j_milli);
            let data = self.prepare(&mut job, settings, screen);
            if matches!(data, Ok(Some(_)))
                && !self
                    .salt
                    .as_ref()
                    .is_some_and(|k| Arc::ptr_eq(&k.view, view))
            {
                let cached = self
                    .topology
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone();
                self.salt = cached.and_then(|prepared| SaltTopology::learn(view, prepared));
            }
            // The worker's first salt of a lease teaches it the topology;
            // screen it too, so a lease never mixes plans.
            let learned = self.salt.as_ref().filter(|k| Arc::ptr_eq(&k.view, view));
            if let Some(plan) = learned.and_then(|k| k.screen_for(salt, settings, screen)) {
                let signs: Vec<i8> = j_milli
                    .iter()
                    .map(|&j| if j < 0 { -1 } else { 1 })
                    .collect();
                return SaltPrep::Screen(Arc::clone(plan), screen::sign_bits(&signs));
            }
            return SaltPrep::Unit(job, data);
        };
        let prepared = &known.prepared;
        let screened = known.screen_for(salt, settings, screen);
        let (fields, j_units) = match crate::draw::draw_units(
            salt.nonce,
            view.num_nodes,
            view.edges.len(),
            allowed_h,
            allowed_j,
        ) {
            Ok(drawn) => drawn,
            Err(error) => {
                let error = SampleError::Driver(format!("lease draw: {error}"));
                return SaltPrep::Unit(modelless_job(salt), Err(error));
            }
        };
        if let Some(plan) = screened {
            return SaltPrep::Screen(Arc::clone(plan), screen::sign_bits(&j_units));
        }
        let to_f64 = |values: &[i8]| values.iter().map(|&v| f64::from(v)).collect();
        let mut params = salt.params.clone();
        params.seed = salt_seed(salt.nonce);
        let mut job = StreamJob {
            job_id: salt.index.to_le_bytes().to_vec(),
            graph: IsingGraph {
                h: to_f64(&fields),
                j: to_f64(&j_units),
                edges: Arc::clone(&prepared.edges),
            },
            params,
            watermark: None,
        };
        if let Err(error) = validate(&job) {
            return SaltPrep::Unit(job, Err(error));
        }
        let gated = screen && settings.chain_gated(prepared.chain, &job.params);
        if gated && !settings.open_gates {
            job.params.num_sweeps = CHAIN_GATES.full_sweeps;
        }
        let beta_range = job.params.beta_range.is_none().then(|| {
            if known.unit {
                crate::cascade::unit_beta_range(fields.iter().map(|&h| h != 0), &prepared.degrees)
            } else {
                crate::cascade::resident_beta_range_from(&job.graph, &prepared.degrees)
            }
        });
        let inputs =
            PreparedInputs::from_units(&prepared.topology, &prepared.edges, fields, &j_units);
        let data = self
            .schedules
            .prepare(&job, settings, gated, screen, beta_range)
            .map(|schedule| Some(PreparedData { schedule, inputs }));
        SaltPrep::Unit(job, data)
    }
}

struct Work {
    source: Source,
    settings: CascadeSettings,
    screen: bool,
    reply: mpsc::SyncSender<Ready>,
}

// Warm Advantage2 preparation measured 251 us/job on one M4 Max host thread
// (preparation_cost_advantage2), down from 436 us before the cache/beta fixes.
// Allow 3x that cost under concurrent load: four workers / 0.753 ms = 5,312
// jobs/s, 15% above the 4,615 jobs/s target. Four decode workers leave four
// of the 12 performance cores, plus four efficiency cores, for the runner,
// producers (~1.3 cores) and other work. Use std threads for explicit queue
// ownership and joining, without borrowing Rayon's decode workers.
// At most 40 jobs (two nominal 20-slot batches) are queued, preparing, or ready
// in total. Per-worker request queues remove shared-receiver contention;
// one bounded reply per job preserves admission order.
/// How often the runner logs the cascade report.
const REPORT_PERIOD: Duration = Duration::from_secs(60);
/// GPU utilization of the slot runner over a window.
///
/// `gpu_busy` is the share of wall time with a step command executing (the
/// union of command intervals). `occupancy` is the share of a pool's slots
/// those commands advanced, weighted by GPU time. `util` is slot-weighted GPU
/// time per wall second; it exceeds `gpu_busy` when the two pools' commands
/// overlap on the GPU. `runner_wait` is the share of wall time the runner
/// thread spent blocked on the GPU: near zero means the host, not the GPU,
/// sets the pace. `host_late` and `launch` split the GPU's idle time between
/// steps: `host_late` is time before the next step was committed, `launch`
/// is time after it was committed but before it started.
struct Utilization {
    started: Instant,
    busy_s: f64,
    slot_s: f64,
    span_s: f64,
    last_end: f64,
    late_s: f64,
    launch_s: f64,
    wait: Duration,
    steps: u64,
}

impl Utilization {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            busy_s: 0.0,
            slot_s: 0.0,
            span_s: 0.0,
            last_end: 0.0,
            late_s: 0.0,
            launch_s: 0.0,
            wait: Duration::ZERO,
            steps: 0,
        }
    }

    fn record(&mut self, span: crate::slots::StepSpan, capacity: usize) {
        let duration = span.end - span.start;
        if self.last_end > 0.0 && span.start > self.last_end {
            let gap = span.start - self.last_end;
            let late = (span.committed - self.last_end).clamp(0.0, gap);
            self.late_s += late;
            self.launch_s += gap - late;
        }
        let start = span.start.max(self.last_end);
        if span.end > start {
            self.busy_s += span.end - start;
        }
        self.last_end = self.last_end.max(span.end);
        self.span_s += duration;
        self.steps += 1;
        self.slot_s += duration * span.slots as f64 / capacity.max(1) as f64;
    }

    /// Log and restart the window once `period` has passed.
    fn report(&mut self, period: Duration, info: bool) {
        let wall = self.started.elapsed();
        if wall < period {
            return;
        }
        let wall_s = wall.as_secs_f64();
        let gpu_busy = self.busy_s / wall_s;
        let occupancy = self.slot_s / self.span_s.max(f64::EPSILON);
        let util = self.slot_s / wall_s;
        let runner_wait = self.wait.as_secs_f64() / wall_s;
        let steps_per_s = self.steps as f64 / wall_s;
        let step_us = 1e6 * self.span_s / self.steps.max(1) as f64;
        let host_late = self.late_s / wall_s;
        let launch = self.launch_s / wall_s;
        if info {
            tracing::info!(
                gpu_busy = format_args!("{gpu_busy:.3}"),
                occupancy = format_args!("{occupancy:.3}"),
                util = format_args!("{util:.3}"),
                runner_wait = format_args!("{runner_wait:.3}"),
                steps_per_s = format_args!("{steps_per_s:.0}"),
                step_us = format_args!("{step_us:.0}"),
                host_late = format_args!("{host_late:.3}"),
                launch = format_args!("{launch:.3}"),
                "gpu utilization"
            );
        } else {
            tracing::debug!(
                gpu_busy = format_args!("{gpu_busy:.3}"),
                occupancy = format_args!("{occupancy:.3}"),
                util = format_args!("{util:.3}"),
                runner_wait = format_args!("{runner_wait:.3}"),
                steps_per_s = format_args!("{steps_per_s:.0}"),
                step_us = format_args!("{step_us:.0}"),
                host_late = format_args!("{host_late:.3}"),
                launch = format_args!("{launch:.3}"),
                "gpu utilization"
            );
        }
        *self = Self::new();
    }
}

/// Host threads drawing and preparing lease salts. On an M4 Max, four kept
/// the runner waiting on preparation most of the time and capped the GPU near
/// 60 percent busy; eight keep it waiting on the GPU instead.
pub(crate) const PREP_WORKERS: usize = 8;
pub(crate) const PREP_BOUND: usize = 40;
/// Outstanding preparations while salts go to the screen. A candidate holds
/// 5 KB of sign bits, not a model, and the screen takes up to 2,560 per turn.
pub(crate) const SCREEN_PREP_BOUND: usize = 2_048;
/// Models the runner keeps queued in the screen before it stops taking more.
const SCREEN_QUEUE: usize = 2 * screen::BANK_WORDS * screen::GROUPS;
/// Most sweeps a deep slot advances per step beside a screen dispatch, which
/// takes about as long.
const SCREEN_DEEP_SLICE: usize = 64;

/// Run one preparation, turning a panic into an error so the worker still
/// returns its job to the runner exactly once.
fn unwind_safe(
    prepare: impl FnOnce() -> Result<Option<PreparedData>, SampleError>,
) -> Result<Option<PreparedData>, SampleError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(prepare))
        .unwrap_or_else(|_| Err(SampleError::Driver("job preparation panicked".into())))
}

pub(crate) struct Preparation {
    requests: Vec<mpsc::SyncSender<Work>>,
    next_worker: usize,
    ready: VecDeque<mpsc::Receiver<Ready>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// [`PREP_BOUND`], or [`SCREEN_PREP_BOUND`] while salts go to the screen.
    bound: usize,
}

impl Preparation {
    pub(crate) fn new() -> Result<Self, SampleError> {
        let mut preparation = Self {
            requests: Vec::new(),
            next_worker: 0,
            ready: VecDeque::new(),
            workers: Vec::new(),
            bound: PREP_BOUND,
        };
        let schedules = ScheduleCache::default();
        let topology = SharedTopology::default();
        for index in 0..PREP_WORKERS {
            // Each receiver has one owner. No worker holds a shared lock while
            // waiting for work. PREP_BOUND still bounds all outstanding replies.
            let (tx, rx) = mpsc::sync_channel::<Work>(SCREEN_PREP_BOUND);
            let schedules = schedules.clone();
            let topology = Arc::clone(&topology);
            let worker = std::thread::Builder::new()
                .name(format!("resident-prepare-{index}"))
                .spawn(move || {
                    let mut preparer = Preparer {
                        topology,
                        schedules,
                        salt: None,
                    };
                    loop {
                        let request = rx.recv();
                        let Ok(Work {
                            source,
                            settings,
                            screen: work_screen,
                            reply,
                        }) = request
                        else {
                            break;
                        };
                        // Keep ownership of the job outside unwinding so a failed
                        // worker still returns it to the runner exactly once.
                        let ready = match source {
                            Source::Job(mut job) => {
                                let data = unwind_safe(|| {
                                    preparer.prepare(&mut job, settings, work_screen)
                                });
                                Ready::Unit(Prepared {
                                    job,
                                    origin: Origin::Stream,
                                    data,
                                })
                            }
                            Source::Salt(salt) => {
                                let made =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        preparer.prepare_salt(&salt, settings, work_screen)
                                    }))
                                    .unwrap_or_else(|_| {
                                        SaltPrep::Unit(
                                            modelless_job(&salt),
                                            Err(SampleError::Driver(
                                                "job preparation panicked".into(),
                                            )),
                                        )
                                    });
                                match made {
                                    SaltPrep::Screen(plan, bits) => {
                                        Ready::Screen(Candidate { salt, plan, bits })
                                    }
                                    SaltPrep::Unit(job, data) => Ready::Unit(Prepared {
                                        job,
                                        origin: Origin::Salt {
                                            index: salt.index,
                                            round: salt.round,
                                            target: salt.target,
                                            top10: salt.top10,
                                            stop: salt.stop,
                                            reply: salt.reply,
                                        },
                                        data,
                                    }),
                                }
                            }
                        };
                        let _ = reply.send(ready);
                    }
                })
                .map_err(|error| {
                    SampleError::Driver(format!("start preparation worker: {error}"))
                })?;
            preparation.workers.push(worker);
            preparation.requests.push(tx);
        }
        Ok(preparation)
    }

    pub(crate) fn len(&self) -> usize {
        self.ready.len()
    }

    fn fill(
        &mut self,
        jobs: &mut Receiver<StreamJob>,
        salts: &mpsc::Receiver<Salt>,
        settings: CascadeSettings,
        eof: &mut bool,
        prefer_salt: &mut bool,
    ) {
        while self.len() < self.bound {
            let source = if *prefer_salt {
                salts.try_recv().ok().map(Source::Salt).or_else(|| {
                    if *eof {
                        None
                    } else {
                        match jobs.try_recv() {
                            Ok(job) => Some(Source::Job(job)),
                            Err(TryRecvError::Disconnected) => {
                                *eof = true;
                                None
                            }
                            Err(TryRecvError::Empty) => None,
                        }
                    }
                })
            } else if !*eof {
                match jobs.try_recv() {
                    Ok(job) => Some(Source::Job(job)),
                    Err(TryRecvError::Disconnected) => {
                        *eof = true;
                        salts.try_recv().ok().map(Source::Salt)
                    }
                    Err(TryRecvError::Empty) => salts.try_recv().ok().map(Source::Salt),
                }
            } else {
                salts.try_recv().ok().map(Source::Salt)
            };
            let Some(source) = source else { break };
            *prefer_salt = match source {
                Source::Job(_) => true,
                Source::Salt(_) => false,
            };
            self.submit(source, settings);
        }
    }

    pub(crate) fn submit(&mut self, source: Source, settings: CascadeSettings) {
        let screen = match &source {
            Source::Job(_) => false,
            Source::Salt(_) => true,
        };
        self.submit_with(source, settings, screen);
    }

    /// [`Self::submit`] with an explicit screen flag. Screen survivors come
    /// back through here with [`CascadeSettings::deep`] set.
    fn submit_with(&mut self, source: Source, settings: CascadeSettings, screen: bool) {
        let (reply, result) = mpsc::sync_channel(1);
        let work = Work {
            source,
            settings,
            screen,
            reply,
        };
        // The runner submits only below PREP_BOUND, so this queue cannot block.
        // A disconnected queue is returned through the same result owner.
        let tx = &self.requests[self.next_worker];
        self.next_worker = (self.next_worker + 1) % self.requests.len();
        if let Err(mpsc::SendError(work)) = tx.send(work) {
            match work.source {
                Source::Job(job) => {
                    let _ = work.reply.send(Ready::Unit(Prepared {
                        job,
                        origin: Origin::Stream,
                        data: Err(SampleError::Driver("preparation workers stopped".into())),
                    }));
                }
                Source::Salt(salt) => {
                    refuse_salt(
                        salt,
                        &SampleError::Driver("preparation workers stopped".into()),
                    );
                }
            }
        }
        self.ready.push_back(result);
    }

    pub(crate) fn next(&mut self) -> Option<Ready> {
        self.ready.pop_front().and_then(|result| result.recv().ok())
    }
}

impl Drop for Preparation {
    fn drop(&mut self) {
        self.requests.clear();
        // Replies have capacity one each. Joining never depends on the runner
        // draining results or on input EOF, including after output closes.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn reject_or_cancel(
    out: &Sender<StreamResult>,
    origin: Origin,
    job: StreamJob,
    error: &SampleError,
    cancel: &CancelToken,
) {
    let outcome = if abandoned(cancel, &job, &origin) {
        Answer::Cancelled
    } else {
        Answer::Failed(error.to_sample_error())
    };
    let _ = answer(out, origin, job, outcome, 0);
}

/// Answer a prepared item that will not run.
fn reject_ready(
    out: &Sender<StreamResult>,
    ready: Ready,
    error: &SampleError,
    cancel: &CancelToken,
) {
    match ready {
        Ready::Unit(prepared) => {
            reject_or_cancel(out, prepared.origin, prepared.job, error, cancel)
        }
        Ready::Screen(candidate) => refuse_salt(candidate.salt, error),
    }
}

fn reject_tail(
    pending: Option<Ready>,
    jobs: &mut Receiver<StreamJob>,
    salts: &mpsc::Receiver<Salt>,
    out: &Sender<StreamResult>,
    error: &SampleError,
    cancel: &CancelToken,
) {
    jobs.close();
    if let Some(ready) = pending {
        reject_ready(out, ready, error, cancel);
    }
    // close() revokes new reservations, but existing permits may still send.
    // None means both buffered jobs and outstanding permits have drained.
    while let Some(job) = jobs.blocking_recv() {
        reject_or_cancel(out, Origin::Stream, job, error, cancel);
    }
    while let Ok(salt) = salts.try_recv() {
        refuse_salt(salt, error);
    }
}

/// Answer a salt that will not run.
fn refuse_salt(salt: Salt, error: &SampleError) {
    let index = salt.index;
    let _ = salt.reply.send(if salt.stop.load(Ordering::Acquire) {
        SaltOutcome::Dropped { index }
    } else {
        SaltOutcome::Failed {
            index,
            error: error.to_sample_error(),
        }
    });
}

fn reject_preparation(
    preparation: &mut Preparation,
    pending: Option<Ready>,
    jobs: &mut Receiver<StreamJob>,
    salts: &mpsc::Receiver<Salt>,
    out: &Sender<StreamResult>,
    error: &SampleError,
    cancel: &CancelToken,
) {
    jobs.close();
    while let Some(ready) = preparation.next() {
        reject_ready(out, ready, error, cancel);
    }
    reject_tail(pending, jobs, salts, out, error, cancel);
}

fn run_fallback(
    device: &MetalDevice,
    job: StreamJob,
    origin: Origin,
    out: &Sender<StreamResult>,
    gov: &dyn GpuGovernor,
    cancel: &CancelToken,
) -> Result<(), SampleError> {
    let mut device_access_time_us = 0;
    let result = (|| {
        let mut batch = sampler::encode_batch(device, &[&job.graph], &job.params, Kernel::Msa, 1)?;
        for _ in 0..batch.chunk_count() {
            if out.is_closed() || abandoned(cancel, &job, &origin) {
                break;
            }
            crate::streaming::yield_gate(out, gov);
            if !batch.commit_next(|| out.is_closed() || abandoned(cancel, &job, &origin)) {
                break;
            }
            batch.wait_until_completed();
        }
        device_access_time_us = batch.gpu_time_us();
        gov.record_gpu_busy_us(device_access_time_us);
        if let Some(status) = batch.failed_status() {
            return Err(SampleError::Driver(format!(
                "metal command buffer did not complete: status {status:?}"
            )));
        }
        if out.is_closed() || abandoned(cancel, &job, &origin) {
            return Ok(Vec::new());
        }
        let mut reads = sampler::harvest_batch(&batch, &[&job.graph])?.remove(0);
        reads.truncate(job.params.num_reads.max(1));
        Ok(reads)
    })();
    let fault = match &result {
        Err(SampleError::TooLarge(_)) | Ok(_) => None,
        Err(SampleError::Driver(message)) => Some(SampleError::Driver(message.clone())),
        Err(SampleError::Metal(error)) => Some(SampleError::Driver(error.to_string())),
    };
    let outcome = if out.is_closed() || abandoned(cancel, &job, &origin) {
        Answer::Cancelled
    } else {
        match result {
            Ok(reads) => Answer::Reads(reads),
            Err(error) => Answer::Failed(error.to_sample_error()),
        }
    };
    let _ = answer(out, origin, job, outcome, device_access_time_us);
    match fault {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Answer or advance one screened salt. Survivors, and failures that a
/// lease still wants (a running top-N entry or a live-target hit), go back
/// to preparation for the deep plan.
fn screen_verdict(
    verdict: Verdict<Salt>,
    controller: &mut Controller,
    preparation: &mut Preparation,
    config: CascadeSettings,
) {
    let last = CHAIN_GATES.screen.len() - 1;
    let (salt, best, passed) = match verdict {
        Verdict::Kept { stage } => {
            controller.screened(stage, true);
            return;
        }
        Verdict::Failed { item, stage, best } => {
            controller.screened(stage, false);
            (item, best, false)
        }
        Verdict::Passed { item, best } => {
            controller.screened(last, true);
            (item, best, true)
        }
    };
    if salt.stop.load(Ordering::Acquire) {
        let _ = salt.reply.send(SaltOutcome::Dropped { index: salt.index });
        return;
    }
    let wanted = passed
        || salt.top10.qualifies(best)
        || salt.target.get().is_some_and(|target| best < target);
    if wanted {
        let settings = CascadeSettings {
            deep: true,
            ..config
        };
        preparation.submit_with(Source::Salt(salt), settings, true);
        return;
    }
    controller.screen_finished(salt.round, best);
    let _ = salt.reply.send(SaltOutcome::Screened {
        index: salt.index,
        energy_milli: best,
    });
}

/// Run on the sampler's blocking thread; no Metal object leaves this thread.
#[expect(
    clippy::too_many_arguments,
    reason = "the runner receives independent channels and device services"
)]
pub(crate) fn run(
    device: &MetalDevice,
    settings: &Mutex<CascadeSettings>,
    store: &Mutex<Option<Controller>>,
    mut jobs: Receiver<StreamJob>,
    salts: &mpsc::Receiver<Salt>,
    out: &Sender<StreamResult>,
    gov: &dyn GpuGovernor,
    cancel: &CancelToken,
) {
    let mut config = *settings.lock().unwrap_or_else(|p| p.into_inner());
    let mut controller = store
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .unwrap_or_else(|| Controller::new(config));
    controller.refresh(config);
    let mut preparation = match Preparation::new() {
        Ok(preparation) => preparation,
        Err(error) => {
            jobs.close();
            reject_tail(None, &mut jobs, salts, out, &error, cancel);
            *store.lock().unwrap_or_else(|p| p.into_inner()) = Some(controller);
            return;
        }
    };
    let mut pools: Option<[Pool; 2]> = None;
    // The screen of the current lease topology, created by its first candidate.
    let mut screener: Option<(Arc<ScreenPlan>, Screener<Salt>)> = None;
    let mut pending: Option<Ready> = None;
    let mut eof = false;
    let mut prefer_salt = true;
    let mut turn = 0;
    let mut window = Instant::now();
    let mut busy_us = 0u64;
    let (mut second, mut minute) = (Utilization::new(), Utilization::new());
    let mut fault = None;

    'run: loop {
        if out.is_closed() {
            break;
        }
        preparation.fill(&mut jobs, salts, config, &mut eof, &mut prefer_salt);
        if let Some(pools) = &mut pools {
            let waited = Instant::now();
            pools[turn].slots.wait();
            let waited = waited.elapsed();
            second.wait += waited;
            minute.wait += waited;
            let capacity = pools[turn].slots.capacity();
            match pools[turn].harvest(&mut controller, out, cancel, gov) {
                Ok(us) => {
                    busy_us = busy_us.saturating_add(us);
                    if let Some(span) = pools[turn].slots.take_span() {
                        tracing::trace!(
                            pool = turn,
                            committed = span.committed,
                            start = span.start,
                            end = span.end,
                            slots = span.slots,
                            "step span"
                        );
                        second.record(span, capacity);
                        minute.record(span, capacity);
                    }
                }
                Err(error) => {
                    fault = Some(error);
                    break;
                }
            }
            if window.elapsed() >= Duration::from_secs(1) {
                let mut live_per_stage = [0usize; MAX_STAGES + 1];
                for pool in pools.iter() {
                    for live in pool.live.iter().flatten() {
                        live_per_stage[live.ticket.stage] += 1;
                    }
                }
                controller.update_load(
                    busy_us as f64 / window.elapsed().as_micros().max(1) as f64,
                    &live_per_stage,
                    pools.iter().map(|p| p.slots.capacity()).sum(),
                );
                config = *settings.lock().unwrap_or_else(|p| p.into_inner());
                controller.refresh(config);
                controller.report(Instant::now(), REPORT_PERIOD);
                second.report(Duration::ZERO, false);
                minute.report(REPORT_PERIOD, true);
                window = Instant::now();
                busy_us = 0;
            }
        }

        if let Some((_, screen)) = &mut screener {
            match screen.harvest(turn) {
                Ok((verdicts, busy)) => {
                    gov.record_gpu_busy_us(busy);
                    busy_us = busy_us.saturating_add(busy);
                    for verdict in verdicts {
                        screen_verdict(verdict, &mut controller, &mut preparation, config);
                    }
                }
                Err(error) => {
                    fault = Some(error);
                    break;
                }
            }
        }

        loop {
            if out.is_closed() {
                break 'run;
            }
            let empty = pools
                .as_ref()
                .is_none_or(|ps| ps.iter().all(|p| p.slots.live() == 0));
            let full = pools.as_ref().is_some_and(|ps| {
                let pool = &ps[turn];
                pool.slots.live() >= scale_budget(pool.slots.capacity(), gov.budget_scale())
            });
            let screen_full = screener
                .as_ref()
                .is_some_and(|(_, screen)| screen.queued() >= SCREEN_QUEUE);
            if pending.is_none() {
                // Refill as admissions consume replies. The preparation bound
                // must not cap slot occupancy for small read counts or a larger
                // configured threadgroup budget.
                preparation.fill(&mut jobs, salts, config, &mut eof, &mut prefer_salt);
                if full && (screen_full || screener.is_none()) {
                    break;
                }
                pending = preparation.next();
            }
            let prepared = match pending.take() {
                None => break,
                Some(Ready::Screen(candidate)) => {
                    let stale = screener
                        .as_ref()
                        .is_none_or(|(plan, _)| !plan.same(&candidate.plan));
                    if stale {
                        if screener
                            .as_ref()
                            .is_some_and(|(_, screen)| !screen.is_empty())
                        {
                            // A new topology waits for the old screen to empty.
                            pending = Some(Ready::Screen(candidate));
                            break;
                        }
                        let plan = Arc::clone(&candidate.plan);
                        let schedules: Vec<&[f32]> =
                            plan.schedules.iter().map(Vec::as_slice).collect();
                        let gates: Vec<i64> = CHAIN_GATES.screen.iter().map(|&(_, g)| g).collect();
                        match Screener::new(device, &plan.graph, &schedules, &gates) {
                            Ok(screen) => screener = Some((plan, screen)),
                            Err(error) => {
                                screener = None;
                                refuse_salt(candidate.salt, &error);
                                continue;
                            }
                        }
                    }
                    let Some((_, screen)) = &mut screener else {
                        continue;
                    };
                    if screen.queued() >= SCREEN_QUEUE {
                        pending = Some(Ready::Screen(candidate));
                        break;
                    }
                    if candidate.salt.stop.load(Ordering::Acquire) || !screen.fits(&candidate.bits)
                    {
                        refuse_salt(
                            candidate.salt,
                            &SampleError::Driver("screen candidate does not fit".into()),
                        );
                        continue;
                    }
                    preparation.bound = SCREEN_PREP_BOUND;
                    screen.offer(candidate.salt, candidate.bits);
                    continue;
                }
                Some(Ready::Unit(prepared)) => {
                    if full {
                        pending = Some(Ready::Unit(prepared));
                        break;
                    }
                    prepared
                }
            };
            let Prepared {
                mut job,
                origin,
                data,
            } = prepared;
            if abandoned(cancel, &job, &origin) {
                let _ = answer(out, origin, job, Answer::Cancelled, 0);
                continue;
            }
            let mut data = match data {
                Ok(Some(data)) => data,
                Ok(None) => {
                    if job.graph.num_nodes() == 0 {
                        let reads = (0..job.params.num_reads.max(1))
                            .map(|_| SamplerResult {
                                spins: Vec::new(),
                                energy_milli: 0,
                            })
                            .collect();
                        let _ = answer(out, origin, job, Answer::Reads(reads), 0);
                        continue;
                    }
                    // Run on the Metal-owning runner only after live slots drain,
                    // so a rare full-budget fallback cannot stall slot stepping.
                    if !empty {
                        pending = Some(Ready::Unit(Prepared {
                            job,
                            origin,
                            data: Ok(None),
                        }));
                        break;
                    }
                    let started = Instant::now();
                    let result = run_fallback(device, job, origin, out, gov, cancel);
                    // Fallback time belongs to the governor, not cascade load.
                    window += started.elapsed();
                    if let Err(error) = result {
                        fault = Some(error);
                        break 'run;
                    }
                    continue;
                }
                Err(error) => {
                    reject_or_cancel(out, origin, job, &error, cancel);
                    continue;
                }
            };
            let edges = Arc::clone(data.inputs.edges());
            let schedule_len = data.schedule.betas.len();
            if pools
                .as_mut()
                .is_none_or(|ps| !ps[turn].fits(&job, &edges, schedule_len))
            {
                if !empty {
                    pending = Some(Ready::Unit(Prepared {
                        job,
                        origin,
                        data: Ok(Some(data)),
                    }));
                    break;
                }
                let capacity = batch_size_for_reads(
                    Kernel::Msa,
                    job.params.num_reads.clamp(1, sampler::MAX_READS),
                );
                // Both pools are empty, so changing storage cannot discard work.
                let rebuilt = Pool::new(device, &job, capacity, schedule_len)
                    .and_then(|a| Pool::new(device, &job, capacity, schedule_len).map(|b| [a, b]));
                match rebuilt {
                    Ok(rebuilt) => {
                        pools = Some(rebuilt);
                    }
                    Err(error) => {
                        reject_or_cancel(out, origin, job, &error, cancel);
                        continue;
                    }
                }
            }
            let Some(pools) = &mut pools else { continue };
            let pool = &mut pools[turn];
            // Plain jobs have no target and leave the yield check alone.
            let initial_target_milli = if let Origin::Salt { target, round, .. } = &origin {
                controller.enter_round(*round);
                let target_milli = target.get();
                controller.set_yield_target(target_milli);
                target_milli
            } else {
                None
            };
            let ticket = match controller.admit_prepared(&job, &mut data.schedule, &edges) {
                Ok(ticket) => ticket,
                Err(error) => {
                    reject_or_cancel(out, origin, job, &error, cancel);
                    continue;
                }
            };
            // Only salts carry a live target, so only they can finish early
            // at an observe point. Stream jobs never get observe points.
            let observe = matches!(origin, Origin::Salt { .. });
            // The slot keeps the graph for the device energy audit; the live
            // entry needs only the job's identity and parameters.
            let graph = std::mem::replace(
                &mut job.graph,
                IsingGraph::new(Vec::new(), Vec::new(), Vec::new()),
            );
            let admitted = pool.slots.admit_prepared(
                data.inputs,
                graph,
                data.schedule,
                job.params.seed,
                observe,
            );
            match admitted {
                Ok(slot) => {
                    pool.live[slot] = Some(Live {
                        job,
                        origin,
                        ticket,
                        accounted_us: 0,
                        admitted: Instant::now(),
                        last_target_milli: initial_target_milli,
                    })
                }
                Err(error) => {
                    controller.finish(&ticket, None, false);
                    reject_or_cancel(out, origin, job, &error, cancel);
                }
            }
        }

        let empty = pools
            .as_ref()
            .is_none_or(|ps| ps.iter().all(|p| p.slots.live() == 0));
        let screen_idle = screener
            .as_ref()
            .is_none_or(|(_, screen)| screen.is_empty());
        if empty && screen_idle && eof && pending.is_none() && preparation.len() == 0 {
            break;
        }
        // Flush partial screen words when no more salts are on the way.
        let flush = preparation.len() == 0 && pending.is_none();
        let words = match &mut screener {
            Some((_, screen)) => match screen.load(turn, flush) {
                Ok(words) => words,
                Err(error) => {
                    fault = Some(error);
                    break;
                }
            },
            None => 0,
        };
        let mut shared = false;
        if let Some(pools) = &mut pools {
            if gov.should_throttle() {
                // Drain submitted work before the pause so it frees GPU time.
                for pool in pools.iter() {
                    pool.slots.wait();
                }
                crate::streaming::yield_gate(out, gov);
            }
            if out.is_closed() {
                break;
            }
            let bank = screener
                .as_ref()
                .filter(|_| words > 0)
                .map(|(_, screen)| screen.bank(turn));
            let deep_slice = if screener.is_some() {
                SCREEN_DEEP_SLICE
            } else {
                DEEP_SLICE
            };
            match pools[turn]
                .slots
                .commit_step_with(config.stages[0].max(1), deep_slice, bank)
            {
                Ok(committed) => shared = committed && words > 0,
                Err(error) => {
                    fault = Some(error);
                    break;
                }
            }
            if shared {
                if let (Some((_, screen)), Some(command)) =
                    (&mut screener, pools[turn].slots.command())
                {
                    screen.attach(turn, command);
                }
            }
        }
        if words > 0 && !shared {
            if let Some((_, screen)) = &mut screener {
                if let Err(error) = screen.commit(turn, &device.queue) {
                    fault = Some(error);
                    break;
                }
            }
        }
        turn ^= 1;
        if empty && screen_idle && pending.is_none() {
            // blocking_recv cannot wake when only the output channel closes.
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    jobs.close();
    if let Some(pools) = &mut pools {
        for pool in pools {
            pool.slots.wait();
            for live in pool.live.iter_mut().filter_map(Option::take) {
                controller.finish(&live.ticket, None, false);
                if let Some(error) = &fault {
                    reject_or_cancel(out, live.origin, live.job, error, cancel);
                } else {
                    let _ = answer(
                        out,
                        live.origin,
                        live.job,
                        Answer::Cancelled,
                        live.accounted_us,
                    );
                }
            }
        }
    }
    if let Some((_, screen)) = &mut screener {
        for salt in screen.drain() {
            match &fault {
                Some(error) => refuse_salt(salt, error),
                None => {
                    let _ = salt.reply.send(SaltOutcome::Dropped { index: salt.index });
                }
            }
        }
    }
    if let Some(error) = &fault {
        tracing::error!(%error, "resident stream failed");
    }
    // At normal EOF these are empty. On a fault, reject every outstanding job.
    // A closed output cannot receive an answer, but each owned job still has
    // one terminal send attempt, and workers never send stream results.
    let error =
        fault.unwrap_or_else(|| SampleError::Driver("resident result receiver closed".into()));
    reject_preparation(
        &mut preparation,
        pending,
        &mut jobs,
        salts,
        out,
        &error,
        cancel,
    );
    *store.lock().unwrap_or_else(|p| p.into_inner()) = Some(controller);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool stride long enough for any stage plan of `job`: at most
    /// `MAX_STAGES` checkpoint legs plus the full-budget leg.
    fn any_plan(job: &StreamJob) -> usize {
        (crate::cascade::MAX_STAGES + 1) * job.params.num_sweeps
    }
    use crate::cascade::stage_array;
    use crate::slots::SlotJob;
    use crate::{IsingGraph, MetalSampler, SampleParams};
    use quip_solver_core::Sampler;

    #[test]
    fn mixed_intake_pushes_a_gate_screen_that_enters_the_empty_top10_preserves_survivors_and_fails_draw_errors(
    ) {
        let device = MetalDevice::open(0).unwrap();
        let gov = crate::iokit_gov::UtilGovernor::start(0, 100, false);
        let topology = Arc::new(quip_solver_core::quip_protocol::lease::TopologyView {
            num_nodes: 2,
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000],
        });
        let (reply, outcomes) = mpsc::channel();
        let (salts_tx, salts) = mpsc::sync_channel(PREP_BOUND);
        // Shared by every salt of this lease, as `MetalSampler::sample_lease`
        // shares one handle per lease. With fewer than LEASE_TOP_N finishers,
        // a gate screen (salt 0) still enters the running top-N and is
        // pushed with reads instead of merely screened.
        let top10 = LeaseTopK::new();
        for index in 0..3 {
            let topology = if index == 2 {
                let mut invalid = (*topology).clone();
                invalid.allowed_h_milli.clear();
                Arc::new(invalid)
            } else {
                Arc::clone(&topology)
            };
            salts_tx
                .send(Salt {
                    topology,
                    nonce: [0; 32],
                    index,
                    round: 0,
                    params: SampleParams {
                        num_reads: 4,
                        num_sweeps: 16,
                        ..Default::default()
                    },
                    stop: Arc::default(),
                    target: LiveTarget::new(Some(if index == 1 { i64::MAX } else { -1_000_000 })),
                    top10: top10.clone(),
                    reply: reply.clone(),
                })
                .unwrap();
        }
        let (tx, jobs) = tokio::sync::mpsc::channel(1);
        tx.try_send(job(100, 16)).unwrap();
        drop(tx);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        let mut settings = CascadeSettings {
            stages: stage_array(&[8]),
            ..Default::default()
        };
        settings.gates[0] = Some(-1_000_000);
        run(
            &device,
            &Mutex::new(settings),
            &Mutex::new(None),
            jobs,
            &salts,
            &out,
            &gov,
            &CancelToken::default(),
        );
        let mut seen = [false; 3];
        for _ in 0..3 {
            let index = match outcomes.try_recv().unwrap() {
                SaltOutcome::Screened { index, .. } => {
                    panic!("salt {index}: an empty lease's top-{LEASE_TOP_N} always has room")
                }
                SaltOutcome::Survived { index, reads } => {
                    assert!(index == 0 || index == 1, "unexpected survivor {index}");
                    assert_eq!(reads.len(), 4);
                    for read in reads {
                        assert_eq!(read.spins.len(), 2);
                        assert_eq!(
                            read.energy_milli,
                            -1000 * i64::from(read.spins[0]) * i64::from(read.spins[1])
                        );
                    }
                    index
                }
                SaltOutcome::Failed { index, error } => {
                    assert_eq!(index, 2);
                    assert!(error.is_fatal(), "{error}");
                    index
                }
                SaltOutcome::Dropped { index } => panic!("salt {index} dropped"),
            };
            assert!(!seen[index as usize]);
            seen[index as usize] = true;
        }
        assert_eq!(seen, [true; 3]);
        assert!(outcomes.try_recv().is_err());
        let result = results.try_recv().unwrap();
        assert_eq!(result.job_id, 100usize.to_le_bytes());
        match result.outcome {
            StreamOutcome::Completed(Ok(reads)) => assert_eq!(reads.len(), 4),
            StreamOutcome::Completed(Err(error)) => panic!("{error}"),
            StreamOutcome::Cancelled => panic!("plain job was cancelled"),
        }
        assert!(results.try_recv().is_err());
    }

    #[test]
    fn a_gate_screen_stays_screened_with_no_read_back_when_the_lease_top10_is_full() {
        let device = MetalDevice::open(0).unwrap();
        let gov = crate::iokit_gov::UtilGovernor::start(0, 100, false);
        let topology = Arc::new(quip_solver_core::quip_protocol::lease::TopologyView {
            num_nodes: 2,
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000],
        });
        let (reply, outcomes) = mpsc::channel();
        let (salts_tx, salts) = mpsc::sync_channel(PREP_BOUND);
        // Pre-fill the lease's shared top-LEASE_TOP_N with energies below
        // -1000: this 2-node, one-edge, unit-coupling graph can only ever
        // score -1000 or 1000 (see the `read.energy_milli` assertion in the
        // sibling empty-top10 test above), so no real salt from it can ever
        // beat this fabricated, already-full set.
        let top10 = LeaseTopK::new();
        for below_reach in (1010..=1100).step_by(10) {
            assert!(top10.offer(-below_reach), "filling the top-{LEASE_TOP_N}");
        }
        salts_tx
            .send(Salt {
                topology,
                nonce: [0; 32],
                index: 0,
                round: 0,
                params: SampleParams {
                    num_reads: 4,
                    num_sweeps: 16,
                    ..Default::default()
                },
                stop: Arc::default(),
                // Unreachable, so the salt never finishes via a target hit —
                // it must reach the gate-screening branch this test checks.
                target: LiveTarget::new(Some(-1_000_000)),
                top10,
                reply,
            })
            .unwrap();
        let (tx, jobs) = tokio::sync::mpsc::channel::<StreamJob>(1);
        drop(tx);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        let mut settings = CascadeSettings {
            stages: stage_array(&[8]),
            ..Default::default()
        };
        // Below the graph's reachable range, so the gate always screens.
        settings.gates[0] = Some(-1_000_000);
        run(
            &device,
            &Mutex::new(settings),
            &Mutex::new(None),
            jobs,
            &salts,
            &out,
            &gov,
            &CancelToken::default(),
        );
        // A full running top-N must screen the salt with no read-back: if
        // the harvest branch that checks `offer_top10` before releasing the
        // slot ever dropped its `continue` or released before pushing to
        // `done`, this would come back `Survived` (or the run would fault
        // on a double release, turning this into `Failed`/`Dropped`)
        // instead of `Screened`.
        match outcomes.try_recv().unwrap() {
            SaltOutcome::Screened {
                index,
                energy_milli,
            } => {
                assert_eq!(index, 0);
                assert!(
                    (-1000..=1000).contains(&energy_milli),
                    "energy {energy_milli} outside the graph's reachable range"
                );
            }
            other => panic!(
                "expected a screen with no read-back, got {}",
                match other {
                    SaltOutcome::Survived { .. } => "Survived",
                    SaltOutcome::Dropped { .. } => "Dropped",
                    SaltOutcome::Failed { .. } => "Failed",
                    SaltOutcome::Screened { .. } => unreachable!(),
                }
            ),
        }
        // Exactly one answer: no duplicate send from a dropped `continue`.
        assert!(outcomes.try_recv().is_err());
        assert!(results.try_recv().is_err());
    }

    #[test]
    fn lease_top_k_offer_admits_evicts_ties_and_a_new_lease_starts_empty() {
        let top10 = LeaseTopK::new();
        // Ten energies in increasing (worsening) order: the empty lease's
        // top-LEASE_TOP_N has room for each regardless of order, so every
        // `offer` call returns true (the harvest-level effect — pushed with
        // reads instead of merely screened — is exercised separately, in
        // `mixed_intake_pushes_a_gate_screen_that_enters_the_empty_top10_preserves_survivors_and_fails_draw_errors`
        // and its full-top-10 counterpart).
        let bests = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10];
        for &best in &bests {
            assert!(
                top10.offer(best),
                "energy {best} must enter the still-open top-{LEASE_TOP_N}"
            );
        }
        // An 11th unit no better than the current worst kept entry (-10):
        // it does not enter, so it is screened with no read-back.
        assert!(
            !top10.offer(0),
            "worse than every kept entry must not enter"
        );
        // A tie with the worst kept entry does not enter either.
        assert!(
            !top10.offer(-10),
            "a tie with the worst kept entry must not enter"
        );
        // Strictly better than the current worst kept entry evicts it and
        // enters: this unit is reported with reads.
        assert!(
            top10.offer(-15),
            "strictly better than the worst kept entry must enter"
        );

        // A second lease starts with an empty top-LEASE_TOP_N: an energy
        // (-10) the first lease's tightened set would now reject has room
        // here.
        let second_lease = LeaseTopK::new();
        assert!(
            second_lease.offer(-10),
            "a new lease's top-{LEASE_TOP_N} must start empty"
        );
    }

    #[test]
    fn a_target_hit_finishes_early_and_a_salt_without_a_target_runs_every_stage() {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping target-hit finish test: no Metal device");
            }
            return;
        }
        struct Governor;
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = MetalDevice::open(0).unwrap();
        let settings = CascadeSettings {
            stages: stage_array(&[8]),
            open_gates: true,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let hit_job = job(2000, 256);
        let mut pool = Pool::new(&device, &hit_job, 2, any_plan(&hit_job)).unwrap();

        let (hit_reply, hit_outcomes) = mpsc::channel();
        let (hit_ticket, hit_prepared) = controller.admit_schedule(&hit_job);
        let hit_slot = pool
            .slots
            .admit(SlotJob {
                graph: hit_job.graph.clone(),
                schedule: Arc::clone(&hit_prepared.betas),
                checkpoints: hit_prepared.checkpoints.clone(),
                fresh_from: hit_prepared.fresh_from,
                seed: hit_job.params.seed,
                observe: true,
            })
            .unwrap();
        pool.live[hit_slot] = Some(Live {
            job: hit_job,
            origin: Origin::Salt {
                index: 1,
                round: 0,
                // Easily beaten: any real energy is below it, so the unit
                // finishes at the first checkpoint (the chain rule, strict <).
                target: LiveTarget::new(Some(i64::MAX - 1)),
                top10: LeaseTopK::new(),
                stop: Arc::default(),
                reply: hit_reply,
            },
            ticket: hit_ticket,
            accounted_us: 0,
            admitted: Instant::now(),
            last_target_milli: None,
        });

        let plain_job = job(2001, 256);
        let (plain_reply, plain_outcomes) = mpsc::channel();
        let (plain_ticket, plain_prepared) = controller.admit_schedule(&plain_job);
        let plain_slot = pool
            .slots
            .admit(SlotJob {
                graph: plain_job.graph.clone(),
                schedule: Arc::clone(&plain_prepared.betas),
                checkpoints: plain_prepared.checkpoints.clone(),
                fresh_from: plain_prepared.fresh_from,
                seed: plain_job.params.seed,
                observe: true,
            })
            .unwrap();
        pool.live[plain_slot] = Some(Live {
            job: plain_job,
            origin: Origin::Salt {
                index: 2,
                round: 0,
                target: LiveTarget::new(None),
                top10: LeaseTopK::new(),
                stop: Arc::default(),
                reply: plain_reply,
            },
            ticket: plain_ticket,
            accounted_us: 0,
            admitted: Instant::now(),
            last_target_milli: None,
        });

        let (out, mut results) = tokio::sync::mpsc::channel(4);
        let cancel = CancelToken::default();
        pool.slots.commit_step(8, 8).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();

        // The target hit ends the unit at the first checkpoint: released now,
        // with the reads it already has, and the later stage never runs.
        assert!(pool.live[hit_slot].is_none());
        match hit_outcomes.try_recv().unwrap() {
            SaltOutcome::Survived { index, reads } => {
                assert_eq!(index, 1);
                assert_eq!(reads.len(), 4);
            }
            SaltOutcome::Screened { .. } => panic!("a target hit must not screen the unit out"),
            other => panic!("expected a survivor, got {}", debug_variant(&other)),
        }

        // The salt without a target is still mid-schedule: no report yet.
        assert!(pool.live[plain_slot].is_some());
        assert!(plain_outcomes.try_recv().is_err());

        pool.slots.commit_step(256, 256).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        match plain_outcomes.try_recv().unwrap() {
            SaltOutcome::Survived { index, reads } => {
                assert_eq!(index, 2);
                assert_eq!(reads.len(), 4);
            }
            other => panic!("expected a survivor, got {}", debug_variant(&other)),
        }
        assert_eq!(pool.slots.live(), 0);
        assert!(results.try_recv().is_err());

        fn debug_variant(outcome: &SaltOutcome) -> &'static str {
            match outcome {
                SaltOutcome::Screened { .. } => "Screened",
                SaltOutcome::Survived { .. } => "Survived",
                SaltOutcome::Dropped { .. } => "Dropped",
                SaltOutcome::Failed { .. } => "Failed",
            }
        }
    }

    #[test]
    fn easing_the_live_target_between_checkpoints_finishes_the_unit_early() {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping live-target finish test: no Metal device");
            }
            return;
        }
        struct Governor;
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = MetalDevice::open(0).unwrap();
        // Three checkpoints (8, 64, 256) so the second is not the last: the
        // unit must finish there on the eased target, not by running out
        // the schedule.
        let settings = CascadeSettings {
            stages: stage_array(&[8, 64]),
            open_gates: true,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let unit_job = job(3000, 256);
        let mut pool = Pool::new(&device, &unit_job, 1, any_plan(&unit_job)).unwrap();

        let (reply, outcomes) = mpsc::channel();
        let (ticket, prepared) = controller.admit_schedule(&unit_job);
        let slot = pool
            .slots
            .admit(SlotJob {
                graph: unit_job.graph.clone(),
                schedule: Arc::clone(&prepared.betas),
                checkpoints: prepared.checkpoints.clone(),
                fresh_from: prepared.fresh_from,
                seed: unit_job.params.seed,
                observe: true,
            })
            .unwrap();
        // Unreachable: this job's energy never goes below -1_000_000 milli,
        // so the first checkpoint cannot hit it and the unit keeps running.
        let target = LiveTarget::new(Some(-1_000_000));
        pool.live[slot] = Some(Live {
            job: unit_job,
            origin: Origin::Salt {
                index: 9,
                round: 0,
                target: target.clone(),
                top10: LeaseTopK::new(),
                stop: Arc::default(),
                reply,
            },
            ticket,
            accounted_us: 0,
            admitted: Instant::now(),
            last_target_milli: None,
        });

        let (out, mut results) = tokio::sync::mpsc::channel(2);
        let cancel = CancelToken::default();

        // First checkpoint (8 sweeps): the target is unreachable, so the
        // unit keeps running.
        pool.slots.commit_step(8, 8).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert!(pool.live[slot].is_some(), "unit ended before easing");
        assert!(outcomes.try_recv().is_err());

        // The chain's decay ratchet eases the session target above this
        // unit's best energy, live, between checkpoints.
        target.set(Some(i64::MAX - 1));

        // Second checkpoint (64 sweeps, not the schedule's last): the live
        // read now sees the eased target and finishes the unit here.
        pool.slots.commit_step(64, 64).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert!(
            pool.live[slot].is_none(),
            "the eased live target must finish the unit at the next checkpoint"
        );
        match outcomes.try_recv().unwrap() {
            SaltOutcome::Survived { index, reads } => {
                assert_eq!(index, 9);
                assert_eq!(reads.len(), 4);
            }
            other => panic!(
                "expected a survivor from the eased target, got {}",
                match other {
                    SaltOutcome::Screened { .. } => "Screened",
                    SaltOutcome::Dropped { .. } => "Dropped",
                    SaltOutcome::Failed { .. } => "Failed",
                    SaltOutcome::Survived { .. } => unreachable!(),
                }
            ),
        }
        assert_eq!(pool.slots.live(), 0);
        assert!(results.try_recv().is_err());
    }

    #[test]
    fn observe_point_finishes_a_unit_before_the_next_gate_checkpoint() {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping observe-point finish test: no Metal device");
            }
            return;
        }
        struct Governor;
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = MetalDevice::open(0).unwrap();
        // One gate at 8 sweeps, then nothing until the end of the schedule,
        // two OBSERVE_INTERVAL boundaries later: the kind of gap a deep
        // cascade stage leaves between real checkpoints.
        let settings = CascadeSettings {
            stages: stage_array(&[8]),
            open_gates: true,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let sweeps = 8 + 2 * crate::slots::OBSERVE_INTERVAL;
        let unit_job = job(4000, sweeps);
        let mut pool = Pool::new(&device, &unit_job, 1, any_plan(&unit_job)).unwrap();

        let (reply, outcomes) = mpsc::channel();
        let (ticket, prepared) = controller.admit_schedule(&unit_job);
        let checkpoints = prepared.checkpoints.clone();
        assert_eq!(checkpoints, vec![8, 8 + sweeps]);
        let slot = pool
            .slots
            .admit(SlotJob {
                graph: unit_job.graph.clone(),
                schedule: Arc::clone(&prepared.betas),
                checkpoints: prepared.checkpoints.clone(),
                fresh_from: prepared.fresh_from,
                seed: unit_job.params.seed,
                observe: true,
            })
            .unwrap();
        let target = LiveTarget::new(None);
        pool.live[slot] = Some(Live {
            job: unit_job,
            origin: Origin::Salt {
                index: 5,
                round: 0,
                target: target.clone(),
                top10: LeaseTopK::new(),
                stop: Arc::default(),
                reply,
            },
            ticket,
            accounted_us: 0,
            admitted: Instant::now(),
            last_target_milli: None,
        });

        let (out, mut results) = tokio::sync::mpsc::channel(2);
        let cancel = CancelToken::default();

        // Capture the "salt finished on a target hit" log so this test can
        // tell *where* the unit finished apart from just *that* it
        // finished: a broken observe point (e.g. never checked, or a no-op)
        // would let the unit run all the way to the real final checkpoint,
        // which also reports `Survived` but never emits this log line
        // (`checkpoint.last` delivers directly, with no target-hit log at
        // all) and would obviously never carry `at = "observe"`.
        #[derive(Clone)]
        struct LogWriter(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for LogWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let output = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(Arc::clone(&output));
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            // The gate at 8 sweeps: open_gates keeps it running regardless
            // of any target.
            pool.slots.commit_step(8, 8).unwrap();
            pool.harvest(&mut controller, &out, &cancel, &Governor)
                .unwrap();
            assert!(pool.live[slot].is_some());
            assert!(outcomes.try_recv().is_err());

            // Reach the first observe boundary with no target set: an
            // observe point with nothing to check against must do nothing.
            while pool.slots.position(slot) < crate::slots::OBSERVE_INTERVAL {
                pool.slots
                    .commit_step(
                        crate::slots::OBSERVE_INTERVAL,
                        crate::slots::OBSERVE_INTERVAL,
                    )
                    .unwrap();
                pool.harvest(&mut controller, &out, &cancel, &Governor)
                    .unwrap();
            }
            assert!(
                pool.live[slot].is_some(),
                "an observe point with no live target must not finish the unit"
            );
            assert!(outcomes.try_recv().is_err());

            // Set a target so easy that any real energy beats it.
            target.set(Some(i64::MAX - 1));

            // Reach the second observe boundary: still far short of the
            // final gate checkpoint at `sweeps`. The observe point must
            // finish the unit here, not at the schedule's end.
            while pool.live[slot].is_some() {
                pool.slots
                    .commit_step(
                        crate::slots::OBSERVE_INTERVAL,
                        crate::slots::OBSERVE_INTERVAL,
                    )
                    .unwrap();
                pool.harvest(&mut controller, &out, &cancel, &Governor)
                    .unwrap();
            }
        });

        match outcomes.try_recv().unwrap() {
            SaltOutcome::Survived { index, reads } => {
                assert_eq!(index, 5);
                assert_eq!(reads.len(), 4);
            }
            other => panic!(
                "expected a survivor from the observe point, got {}",
                match other {
                    SaltOutcome::Screened { .. } => "Screened",
                    SaltOutcome::Dropped { .. } => "Dropped",
                    SaltOutcome::Failed { .. } => "Failed",
                    SaltOutcome::Survived { .. } => unreachable!(),
                }
            ),
        }
        assert_eq!(pool.slots.live(), 0);
        assert!(results.try_recv().is_err());

        let log = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        let target_hit_lines: Vec<&str> = log
            .lines()
            .filter(|line| line.contains("salt finished on a target hit"))
            .collect();
        assert_eq!(
            target_hit_lines.len(),
            1,
            "expected exactly one target-hit log line, got:\n{log}"
        );
        assert!(
            target_hit_lines[0].contains("at=\"observe\""),
            "the unit must finish at the observe point, not the final gate checkpoint: {}",
            target_hit_lines[0]
        );
    }

    fn salt(index: u64, reply: &mpsc::Sender<SaltOutcome>) -> Salt {
        Salt {
            topology: Arc::new(quip_solver_core::quip_protocol::lease::TopologyView {
                num_nodes: 2,
                edges: vec![(0, 1)],
                allowed_h_milli: vec![0],
                allowed_j_milli: vec![-1000],
            }),
            nonce: [0; 32],
            index,
            round: 0,
            params: SampleParams {
                num_reads: 4,
                num_sweeps: 16,
                ..Default::default()
            },
            stop: Arc::default(),
            target: LiveTarget::new(None),
            top10: LeaseTopK::new(),
            reply: reply.clone(),
        }
    }

    #[test]
    fn the_same_salt_index_in_two_leases_gets_two_seeds() {
        let (reply, _outcomes) = mpsc::channel();
        let first = salt(3, &reply);
        let mut second = salt(3, &reply);
        second.nonce = [1; 32];
        assert_ne!(
            drawn_job(&first, &[0, 0], &[-1000]).params.seed,
            drawn_job(&second, &[0, 0], &[-1000]).params.seed
        );
    }

    #[test]
    fn a_fault_fails_queued_salts_with_its_error() {
        let (reply, outcomes) = mpsc::channel();
        let (salts_tx, salts) = mpsc::sync_channel(PREP_BOUND);
        salts_tx.send(salt(7, &reply)).unwrap();
        let (_tx, mut jobs) = tokio::sync::mpsc::channel(1);
        let (out, _results) = tokio::sync::mpsc::channel(1);
        reject_tail(
            None,
            &mut jobs,
            &salts,
            &out,
            &SampleError::Driver("injected fault".into()),
            &CancelToken::default(),
        );
        match outcomes.try_recv().unwrap() {
            SaltOutcome::Failed { index, error } => {
                assert_eq!(index, 7);
                assert!(error.to_string().contains("injected fault"), "{error}");
            }
            _ => panic!("a faulted salt must carry the fault"),
        }
        assert!(outcomes.try_recv().is_err());
    }

    #[test]
    fn shutdown_drains_a_permit_held_across_close() {
        let (tx, mut jobs) = tokio::sync::mpsc::channel(2);
        let permit = tx.clone().try_reserve_owned().unwrap();
        let (out, mut results) = tokio::sync::mpsc::channel(2);
        let (finished, completion) = mpsc::channel();
        let cleanup = std::thread::spawn(move || {
            jobs.close();
            let (_salts_tx, salts) = mpsc::channel();
            reject_tail(
                Some(Ready::Unit(Prepared {
                    job: job(0, 32),
                    origin: Origin::Stream,
                    data: Ok(None),
                })),
                &mut jobs,
                &salts,
                &out,
                &SampleError::Driver("injected fault".into()),
                &CancelToken::default(),
            );
            finished.send(()).unwrap();
        });
        assert_eq!(
            results.blocking_recv().unwrap().job_id,
            0usize.to_le_bytes()
        );
        // The first rejection proves that cleanup has closed the receiver.
        // Keep tx alive: draining must wait for permits, not sender EOF.
        let early_completion = completion.recv_timeout(Duration::from_millis(50));
        permit.send(job(1, 32));
        cleanup.join().unwrap();
        assert_eq!(early_completion, Err(mpsc::RecvTimeoutError::Timeout));
        assert_eq!(
            results.blocking_recv().expect("reserved job lost").job_id,
            1usize.to_le_bytes()
        );
        assert!(results.blocking_recv().is_none());
        assert!(tx.is_closed());
    }

    #[test]
    fn chain_jobs_get_the_deep_final_budget_unless_gates_are_open() {
        let edges: Vec<(usize, usize)> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().parse().unwrap(),
                    words.next().unwrap().parse().unwrap(),
                )
            })
            .filter(|&(u, v): &(usize, usize)| (u.min(v), u.max(v)) != (880, 2695))
            .collect();
        let mut chain = job(0, 14_336);
        chain.graph = IsingGraph::new(
            vec![0.0; 4577],
            (0..edges.len())
                .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                .collect(),
            edges,
        );
        chain.params.num_reads = 64;
        let mut preparer = Preparer::default();
        let data = preparer
            .prepare(&mut chain, CascadeSettings::default(), true)
            .unwrap()
            .unwrap();
        assert_eq!(chain.params.num_sweeps, CHAIN_GATES.full_sweeps);
        assert_eq!(
            data.schedule.checkpoints.last(),
            Some(
                &(CHAIN_GATES
                    .stages
                    .iter()
                    .filter(|&&s| s > crate::cascade::REHEAT_THROUGH)
                    .sum::<usize>()
                    + crate::cascade::REHEAT_THROUGH
                    + CHAIN_GATES.full_sweeps)
            )
        );
        let open = CascadeSettings {
            open_gates: true,
            ..CascadeSettings::default()
        };
        chain.params.num_sweeps = 14_336;
        preparer.prepare(&mut chain, open, true).unwrap().unwrap();
        assert_eq!(chain.params.num_sweeps, 14_336);
        let mut other = job(1, 14_336);
        preparer
            .prepare(&mut other, CascadeSettings::default(), true)
            .unwrap();
        assert_eq!(other.params.num_sweeps, 14_336);
    }

    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "host preparation cost informs the fixed worker budget"
    )]
    fn preparation_cost_advantage2() {
        let edges = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().parse().unwrap(),
                    words.next().unwrap().parse().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut job = job(0, 32);
        let mut rng = 12345u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        job.graph = IsingGraph::new(
            (0..4577)
                .map(|_| [-1.0, 0.0, 1.0][(draw() % 3) as usize])
                .collect(),
            (0..edges.len())
                .map(|_| if draw() & 1 == 0 { -1.0 } else { 1.0 })
                .collect(),
            edges,
        );
        job.params.num_reads = 64;
        let mut preparer = Preparer::default();
        let settings = CascadeSettings::default();
        preparer.prepare(&mut job, settings, true).unwrap();
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(
                preparer
                    .prepare(std::hint::black_box(&mut job), settings, true)
                    .unwrap(),
            );
        }
        eprintln!(
            "Advantage2 preparation: {:.3} us/job (1000 jobs, warm topology, one host thread)",
            start.elapsed().as_secs_f64() * 1000.0
        );
        let mut preparation = Preparation::new().unwrap();
        let start = Instant::now();
        let mut submitted = 0usize;
        let mut completed = 0usize;
        loop {
            while submitted < 4000 && preparation.len() < PREP_BOUND {
                preparation.submit(
                    Source::Job(StreamJob {
                        job_id: submitted.to_le_bytes().to_vec(),
                        graph: job.graph.clone(),
                        params: job.params.clone(),
                        watermark: None,
                    }),
                    settings,
                );
                submitted += 1;
            }
            let Some(prepared) = preparation.next().map(Ready::unit) else {
                break;
            };
            assert_eq!(prepared.job.job_id, completed.to_le_bytes());
            std::hint::black_box(prepared.data.unwrap());
            completed += 1usize;
        }
        assert_eq!(completed, 4000);
        eprintln!("Advantage2 preparation stage: {:.1} jobs/s ({PREP_WORKERS} workers, 4000 jobs, includes graph copies and cache startup)", completed as f64 / start.elapsed().as_secs_f64());
    }

    /// The Aglais lease topology: the Advantage2 fixture without edge (880, 2695).
    fn aglais_view() -> Arc<quip_solver_core::quip_protocol::lease::TopologyView> {
        let edges = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().parse().unwrap(),
                    words.next().unwrap().parse().unwrap(),
                )
            })
            .filter(|&edge| edge != (880, 2695))
            .collect::<Vec<_>>();
        assert_eq!(edges.len(), 41_514);
        Arc::new(quip_solver_core::quip_protocol::lease::TopologyView {
            num_nodes: 4577,
            edges,
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
        })
    }

    fn aglais_salt(
        topology: &Arc<quip_solver_core::quip_protocol::lease::TopologyView>,
        index: u64,
        reply: &mpsc::Sender<SaltOutcome>,
    ) -> Salt {
        let mut nonce = [7u8; 32];
        nonce[..8].copy_from_slice(&index.to_le_bytes());
        Salt {
            topology: Arc::clone(topology),
            nonce,
            index,
            round: 0,
            params: SampleParams {
                num_reads: 64,
                num_sweeps: CHAIN_GATES.full_sweeps,
                ..Default::default()
            },
            stop: Arc::default(),
            target: LiveTarget::new(None),
            top10: LeaseTopK::new(),
            reply: reply.clone(),
        }
    }

    /// Feed `total` Aglais salts sharing `top10` and `stop`, run the resident
    /// runner on them, and return it on a thread with the outcome receiver.
    fn run_aglais_salts(
        total: u64,
        top10: &LeaseTopK,
        stop: &Arc<AtomicBool>,
    ) -> (std::thread::JoinHandle<()>, mpsc::Receiver<SaltOutcome>) {
        let topology = aglais_view();
        let (reply, outcomes) = mpsc::channel();
        let (salts_tx, salts) = mpsc::sync_channel(total as usize);
        for index in 0..total {
            let mut salt = aglais_salt(&topology, index, &reply);
            salt.top10 = top10.clone();
            salt.stop = Arc::clone(stop);
            salts_tx.send(salt).unwrap();
        }
        let runner = std::thread::spawn(move || {
            let device = MetalDevice::open(0).unwrap();
            let gov = crate::iokit_gov::UtilGovernor::start(0, 100, false);
            let (tx, jobs) = tokio::sync::mpsc::channel(1);
            drop(tx);
            let (out, _results) = tokio::sync::mpsc::channel(1);
            run(
                &device,
                &Mutex::new(CascadeSettings::default()),
                &Mutex::new(None),
                jobs,
                &salts,
                &out,
                &gov,
                &CancelToken::default(),
            );
        });
        (runner, outcomes)
    }

    /// A lease top-N that no screened salt can enter.
    fn full_top10() -> LeaseTopK {
        let top10 = LeaseTopK::new();
        for _ in 0..LEASE_TOP_N {
            top10.offer(i64::MIN);
        }
        top10
    }

    /// Every chain salt goes through the nonce-parallel screen and gets
    /// exactly one answer; a rejected salt answers with its screen best.
    #[test]
    fn screen_answers_every_salt_once() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let total = 1_000;
        let stop = Arc::new(AtomicBool::new(false));
        let (runner, outcomes) = run_aglais_salts(total, &full_top10(), &stop);
        let mut seen = vec![false; total as usize];
        let (mut screened, mut second_stage) = (0, 0);
        for _ in 0..total {
            let outcome = outcomes
                .recv_timeout(Duration::from_secs(600))
                .expect("every salt answers");
            let index = match outcome {
                SaltOutcome::Screened {
                    index,
                    energy_milli,
                } => {
                    // 16 sweeps reach about -13,500; 128 about -14,060.
                    assert!(
                        (-14_600_000..-13_000_000).contains(&energy_milli),
                        "salt {index}: {energy_milli}"
                    );
                    screened += 1;
                    // Only the 128-sweep stage reaches this deep.
                    second_stage += usize::from(energy_milli <= -13_800_000);
                    index
                }
                SaltOutcome::Survived { index, reads } => {
                    assert_eq!(reads.len(), 64);
                    index
                }
                SaltOutcome::Dropped { index } => panic!("salt {index} dropped"),
                SaltOutcome::Failed { index, error } => panic!("salt {index}: {error}"),
            };
            assert!(
                !std::mem::replace(&mut seen[index as usize], true),
                "salt {index} twice"
            );
        }
        assert!(screened >= total - 2, "{screened} of {total} screened");
        // About 5 percent of salts pass the 16-sweep stage.
        assert!(
            (20..=100).contains(&second_stage),
            "{second_stage} second-stage screens"
        );
        runner.join().unwrap();
        assert!(outcomes.try_recv().is_err());
    }

    /// End-to-end screen rate on Aglais salts: preparation, screen and
    /// answers, with no coordinator. Reports salts per second between 10 and
    /// 90 percent of `QUIP_BENCH_SALTS` answers, then stops the lease so
    /// deep survivors drop.
    ///
    /// ```text
    /// cargo test --release --lib screen_runner_rate -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device; a benchmark"]
    #[expect(clippy::print_stderr, reason = "the benchmark reports on stderr")]
    fn screen_runner_rate() {
        let total: u64 = std::env::var("QUIP_BENCH_SALTS").map_or(200_000, |v| v.parse().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let (runner, outcomes) = run_aglais_salts(total, &full_top10(), &stop);
        let (mut answered, mut survived) = (0u64, 0u64);
        let (mut first, mut last) = (None, None);
        let started = Instant::now();
        while answered < total {
            let Ok(outcome) = outcomes.recv_timeout(Duration::from_secs(600)) else {
                break;
            };
            answered += 1;
            if answered.is_multiple_of(total / 20) {
                eprintln!(
                    "{answered} answered at {:.1} s",
                    started.elapsed().as_secs_f64()
                );
            }
            if matches!(outcome, SaltOutcome::Survived { .. }) {
                survived += 1;
            }
            if answered == total / 10 {
                first = Some(Instant::now());
            }
            if answered == total * 9 / 10 {
                last = Some(Instant::now());
                stop.store(true, Ordering::Release);
            }
        }
        while outcomes.recv_timeout(Duration::from_secs(60)).is_ok() {}
        runner.join().unwrap();
        let (first, last) = (first.unwrap(), last.unwrap());
        eprintln!(
            "{total} Aglais salts: {:.0} salts/s, {survived} survived before the stop",
            (total * 8 / 10) as f64 / (last - first).as_secs_f64()
        );
    }

    /// Prepare `salt` the way a worker does.
    fn prepare_drawn(
        preparer: &mut Preparer,
        salt: &Salt,
        settings: CascadeSettings,
    ) -> (StreamJob, Option<PreparedData>) {
        let SaltPrep::Unit(job, data) = preparer.prepare_salt(salt, settings, true) else {
            panic!("expected a slot unit, got a screen candidate");
        };
        (job, data.unwrap())
    }

    /// The device-unit fast path gives the generic path's coefficients,
    /// schedule, parameters and graph, on unit, non-unit and inexact
    /// topologies.
    #[test]
    fn salt_fast_path_matches_generic_preparation() {
        let (reply, _outcomes) = mpsc::channel();
        let settings = CascadeSettings::default();
        let ring = |allowed_h_milli: Vec<i32>, allowed_j_milli: Vec<i32>| {
            Arc::new(TopologyView {
                num_nodes: 64,
                edges: (0..64).map(|i| (i, (i + 1) % 64)).collect(),
                allowed_h_milli,
                allowed_j_milli,
            })
        };
        for (view, fast) in [
            (aglais_view(), true),
            (ring(vec![-1000, 0, 1000], vec![-1000, 1000]), true),
            (ring(vec![-2000, 3000], vec![-2000, 1000, 5000]), true),
            (ring(vec![0], vec![-1500, 1000]), false),
        ] {
            let mut fast_worker = Preparer::default();
            for index in 0..40 {
                let mut salt = aglais_salt(&view, index, &reply);
                salt.params.num_reads = 8;
                let (job, data) = prepare_drawn(&mut fast_worker, &salt, settings);
                let (h_milli, j_milli) = salt.topology.draw(salt.nonce).unwrap();
                let mut reference = drawn_job(&salt, &h_milli, &j_milli);
                let expected = Preparer::default()
                    .prepare(&mut reference, settings, true)
                    .unwrap();
                if !fast {
                    assert!(
                        expected.is_none() && data.is_none(),
                        "an inexact draw falls back"
                    );
                    continue;
                }
                let (data, expected) = (data.unwrap(), expected.unwrap());
                assert_eq!(data.inputs.coefficients(), expected.inputs.coefficients());
                assert_eq!(data.schedule.betas, expected.schedule.betas);
                assert_eq!(data.schedule.checkpoints, expected.schedule.checkpoints);
                assert_eq!(job.params.num_sweeps, reference.params.num_sweeps);
                assert_eq!(job.params.beta_range, reference.params.beta_range);
                assert_eq!(job.params.seed, reference.params.seed);
                assert_eq!(
                    (&job.graph.h, &job.graph.j, &job.graph.edges),
                    (
                        &reference.graph.h,
                        &reference.graph.j,
                        &reference.graph.edges
                    )
                );
            }
            let known = fast_worker.salt.as_ref();
            assert_eq!(known.is_some_and(|k| k.units.is_some()), fast);
        }
    }

    /// Per-salt host cost of each preparation step on the Aglais lease
    /// topology, one thread, warm caches, then the eight-worker stage rate.
    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture"]
    #[expect(clippy::print_stderr, reason = "the measurement is the output")]
    fn salt_preparation_profile() {
        const SALTS: u64 = 2_000;
        let topology = aglais_view();
        let (reply, _outcomes) = mpsc::channel();
        let settings = CascadeSettings::default();
        let mut preparer = Preparer::default();
        prepare_drawn(
            &mut preparer,
            &aglais_salt(&topology, u64::MAX, &reply),
            settings,
        );
        let known = preparer.salt.as_ref().unwrap();
        assert!(known.unit);
        let (allowed_h, allowed_j) = known.units.clone().unwrap();
        let cached = Arc::clone(&known.prepared);

        let steps = [
            "draw into i8 units (crate::draw)",
            "f64 graph, shared edges",
            "PreparedInputs::from_units",
            "schedule (unit beta range, cache hit)",
            "free per-salt buffers",
            "whole prepare_salt",
            "draw into i32 milli (generic path)",
            "whole generic Preparer::prepare",
        ];
        let mut spent = [Duration::ZERO; 8];
        for index in 0..SALTS {
            let salt = aglais_salt(&topology, index, &reply);
            let mut clock = Instant::now();
            let mut lap = |step: usize, clock: &mut Instant| {
                let now = Instant::now();
                spent[step] += now - *clock;
                *clock = now;
            };
            let (fields, j_units) = std::hint::black_box(crate::draw::draw_units(
                salt.nonce,
                topology.num_nodes,
                topology.edges.len(),
                &allowed_h,
                &allowed_j,
            ))
            .unwrap();
            lap(0, &mut clock);
            let to_f64 =
                |values: &[i8]| -> Vec<f64> { values.iter().map(|&v| f64::from(v)).collect() };
            let graph = std::hint::black_box(IsingGraph {
                h: to_f64(&fields),
                j: to_f64(&j_units),
                edges: Arc::clone(&cached.edges),
            });
            lap(1, &mut clock);
            let field_set = fields.iter().map(|&h| h != 0).collect::<Vec<_>>();
            let inputs = std::hint::black_box(PreparedInputs::from_units(
                &cached.topology,
                &cached.edges,
                fields,
                &j_units,
            ));
            lap(2, &mut clock);
            let job = StreamJob {
                job_id: Vec::new(),
                graph,
                params: salt.params.clone(),
                watermark: None,
            };
            let gated = settings.chain_gated(cached.chain, &job.params);
            let range = crate::cascade::unit_beta_range(field_set.into_iter(), &cached.degrees);
            std::hint::black_box(
                preparer
                    .schedules
                    .prepare(&job, settings, gated, true, Some(range))
                    .unwrap(),
            );
            lap(3, &mut clock);
            drop((inputs, job, j_units));
            lap(4, &mut clock);
            let prepared = std::hint::black_box(preparer.prepare_salt(&salt, settings, true));
            lap(5, &mut clock);
            drop(prepared);
            lap(4, &mut clock);
            let (h, j) = std::hint::black_box(salt.topology.draw(salt.nonce).unwrap());
            let mut job = drawn_job(&salt, &h, &j);
            lap(6, &mut clock);
            let prepared =
                std::hint::black_box(preparer.prepare(&mut job, settings, true).unwrap());
            lap(7, &mut clock);
            drop((prepared, job, h, j));
            lap(4, &mut clock);
        }
        let per_salt = |d: Duration| d.as_secs_f64() * 1e6 / SALTS as f64;
        let parts: f64 = spent[..5].iter().map(|&d| per_salt(d)).sum();
        eprintln!("one thread, {SALTS} Aglais salts (4,577 nodes, 41,514 edges):");
        for (step, &d) in steps.iter().zip(&spent).take(5) {
            eprintln!(
                "  {step:38} {:8.1} us/salt {:5.1}%",
                per_salt(d),
                100.0 * per_salt(d) / parts
            );
        }
        eprintln!("  sum of steps 1-5 {parts:.1} us/salt");
        for (step, &d) in steps.iter().zip(&spent).skip(5) {
            eprintln!("  {step:38} {:8.1} us/salt", per_salt(d));
        }

        let total = 20_000u64;
        let mut preparation = Preparation::new().unwrap();
        let start = Instant::now();
        let (mut submitted, mut completed) = (0u64, 0u64);
        loop {
            while submitted < total && preparation.len() < PREP_BOUND {
                preparation.submit(
                    Source::Salt(aglais_salt(&topology, submitted, &reply)),
                    settings,
                );
                submitted += 1;
            }
            let Some(ready) = preparation.next() else {
                break;
            };
            if let Ready::Unit(prepared) = ready {
                std::hint::black_box(prepared.data.unwrap());
            }
            completed += 1;
        }
        assert_eq!(completed, total);
        eprintln!(
            "{PREP_WORKERS} workers, {total} salts through Preparation: {:.0} salts/s",
            completed as f64 / start.elapsed().as_secs_f64()
        );
    }

    #[test]
    fn preparation_cache_never_bypasses_changed_topology_or_scalar_limits() {
        let mut preparer = Preparer::default();
        let settings = CascadeSettings::default();
        let mut input = job(0, 32);
        preparer.prepare(&mut input, settings, true).unwrap();
        input.graph.edges = (1..=sampler::MSA_MAX_DEG + 1).map(|v| (0, v)).collect();
        input.graph.j = vec![1.0; input.graph.edges.len()];
        assert!(matches!(
            preparer.prepare(&mut input, settings, true),
            Err(SampleError::TooLarge(_))
        ));
        let mut input = job(1, sampler::MAX_SWEEPS + 1);
        assert!(matches!(
            preparer.prepare(&mut input, settings, true),
            Err(SampleError::TooLarge(_))
        ));
        input.params.num_sweeps = 32;
        input.graph.j[0] = 0.5;
        assert!(preparer
            .prepare(&mut input, settings, true)
            .unwrap()
            .is_none());
    }

    fn job(id: usize, sweeps: usize) -> StreamJob {
        let n = 64;
        StreamJob {
            job_id: id.to_le_bytes().to_vec(),
            graph: IsingGraph::new(
                (0..n).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect(),
                vec![-1.0; n - 1],
                (0..n - 1).map(|i| (i, i + 1)).collect(),
            ),
            params: SampleParams {
                num_reads: 4,
                num_sweeps: sweeps,
                seed: id as u64 + 1,
                ..Default::default()
            },
            watermark: None,
        }
    }

    #[test]
    fn preparation_bounds_prefetch_and_drains_eof() {
        let count = PREP_BOUND * 3;
        let (tx, mut jobs) = tokio::sync::mpsc::channel(count);
        for id in 0..count {
            tx.try_send(job(id, 32)).unwrap();
        }
        drop(tx);
        let mut preparation = Preparation::new().unwrap();
        let mut eof = false;
        let (_salts_tx, salts) = mpsc::channel();
        let mut prefer_salt = false;
        preparation.fill(
            &mut jobs,
            &salts,
            CascadeSettings::default(),
            &mut eof,
            &mut prefer_salt,
        );
        assert_eq!(preparation.len(), PREP_BOUND);
        assert_eq!(jobs.len(), count - PREP_BOUND);
        for id in 0..count {
            let prepared = preparation.next().unwrap().unit();
            assert_eq!(prepared.job.job_id, id.to_le_bytes());
            preparation.fill(
                &mut jobs,
                &salts,
                CascadeSettings::default(),
                &mut eof,
                &mut prefer_salt,
            );
            assert!(preparation.len() <= PREP_BOUND);
        }
        assert!(eof);
        assert!(preparation.next().is_none());
    }

    #[test]
    fn closed_output_cleanup_drains_preparation_without_waiting_for_input_eof() {
        let (done, finished) = mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut preparation = Preparation::new().unwrap();
            for id in 0..PREP_BOUND {
                preparation.submit(Source::Job(job(id, 32)), CascadeSettings::default());
            }
            let (tx, mut jobs) = tokio::sync::mpsc::channel(1);
            tx.try_send(job(PREP_BOUND, 32)).unwrap();
            let (out, results) = tokio::sync::mpsc::channel(1);
            drop(results);
            let (_salts_tx, salts) = mpsc::channel();
            reject_preparation(
                &mut preparation,
                None,
                &mut jobs,
                &salts,
                &out,
                &SampleError::Driver("closed output".into()),
                &CancelToken::default(),
            );
            assert_eq!(preparation.len(), 0);
            assert_eq!(jobs.len(), 0);
            assert!(tx.is_closed());
            drop(preparation);
            done.send(()).unwrap();
        });
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("closed output cleanup hung");
        runner.join().unwrap();
    }

    #[test]
    fn preparation_returns_valid_and_invalid_jobs_once_in_input_order() {
        let mut preparation = Preparation::new().unwrap();
        for id in 0..PREP_BOUND {
            let mut job = job(
                id,
                if id % 3 == 0 {
                    sampler::MAX_SWEEPS + 1
                } else {
                    64
                },
            );
            if id % 2 == 0 {
                let mut edges = job.graph.edges.to_vec();
                edges.swap(0, 1);
                job.graph.edges = edges.into();
            }
            preparation.submit(Source::Job(job), CascadeSettings::default());
        }
        assert_eq!(preparation.len(), PREP_BOUND);
        for id in 0..PREP_BOUND {
            let prepared = preparation.next().unwrap().unit();
            assert_eq!(prepared.job.job_id, id.to_le_bytes());
            if id % 3 == 0 {
                assert!(matches!(prepared.data, Err(SampleError::TooLarge(_))));
            } else {
                let data = prepared.data.unwrap().unwrap();
                assert_eq!(data.schedule.betas.len(), 64);
                assert_eq!(data.schedule.checkpoints.last(), Some(&64));
            }
        }
        assert_eq!(preparation.len(), 0);
        assert!(preparation.next().is_none());
    }

    #[test]
    fn preparation_shutdown_does_not_need_result_consumption_or_input_eof() {
        let (finished, done) = mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut preparation = Preparation::new().unwrap();
            for id in 0..PREP_BOUND {
                preparation.submit(Source::Job(job(id, 64)), CascadeSettings::default());
            }
            drop(preparation);
            finished.send(()).unwrap();
        });
        done.recv_timeout(Duration::from_secs(5))
            .expect("preparation shutdown hung");
        runner.join().unwrap();
    }

    #[test]
    fn fault_drains_preparation_pending_and_input_once() {
        let mut preparation = Preparation::new().unwrap();
        let (tx, mut jobs) = tokio::sync::mpsc::channel(2);
        let (out, mut results) = tokio::sync::mpsc::channel(PREP_BOUND + 3);
        for id in 0..PREP_BOUND {
            preparation.submit(
                Source::Job(job(id, if id % 2 == 0 { 0 } else { 64 })),
                CascadeSettings::default(),
            );
        }
        tx.try_send(job(PREP_BOUND + 1, 64)).unwrap();
        tx.try_send(job(PREP_BOUND + 2, 64)).unwrap();
        let error = SampleError::Driver("injected device fault".into());
        let pending = Ready::Unit(Prepared {
            job: job(PREP_BOUND, 64),
            origin: Origin::Stream,
            data: Err(SampleError::TooLarge("invalid pending job".into())),
        });
        let (_salts_tx, salts) = mpsc::channel();
        reject_preparation(
            &mut preparation,
            Some(pending),
            &mut jobs,
            &salts,
            &out,
            &error,
            &CancelToken::default(),
        );
        for id in 0..PREP_BOUND + 3 {
            let result = results.try_recv().unwrap();
            assert_eq!(result.job_id, id.to_le_bytes());
            assert!(matches!(result.outcome, StreamOutcome::Completed(Err(_))));
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        drop(tx);
    }

    #[test]
    fn fault_cleanup_answers_non_exact_jobs_once_and_respects_cancellation() {
        let cancel = CancelToken::default();
        cancel.cancel_through(1);
        let non_exact = |id, watermark| {
            let mut job = job(id, 64);
            job.graph.j[0] = 0.5;
            job.watermark = Some(watermark);
            job
        };
        let mut preparation = Preparation::new().unwrap();
        preparation.submit(Source::Job(non_exact(0, 1)), CascadeSettings::default());
        preparation.submit(Source::Job(non_exact(1, 2)), CascadeSettings::default());
        let pending = Ready::Unit(Prepared {
            job: non_exact(2, 1),
            origin: Origin::Stream,
            data: Ok(None),
        });
        let (tx, mut jobs) = tokio::sync::mpsc::channel(2);
        tx.try_send(non_exact(3, 1)).unwrap();
        tx.try_send(non_exact(4, 2)).unwrap();
        let (out, mut results) = tokio::sync::mpsc::channel(5);
        let error = SampleError::Driver("injected fault".into());
        let (_salts_tx, salts) = mpsc::channel();
        reject_preparation(
            &mut preparation,
            Some(pending),
            &mut jobs,
            &salts,
            &out,
            &error,
            &cancel,
        );
        for id in 0usize..5 {
            let result = results.try_recv().unwrap();
            assert_eq!(result.job_id, id.to_le_bytes());
            if id == 1 || id == 4 {
                assert!(matches!(result.outcome, StreamOutcome::Completed(Err(_))));
            } else {
                assert!(matches!(result.outcome, StreamOutcome::Cancelled));
            }
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        assert!(tx.is_closed());
    }

    #[test]
    fn fault_rejects_pending_and_queued_jobs_without_waiting_for_eof() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let (out, mut results) = tokio::sync::mpsc::channel(3);
        tx.try_send(job(1, 8)).unwrap();
        tx.try_send(job(2, 8)).unwrap();
        let error = SampleError::Driver("injected fault".into());
        let (_salts_tx, salts) = mpsc::channel();
        reject_tail(
            Some(Ready::Unit(Prepared {
                job: job(0, 8),
                origin: Origin::Stream,
                data: Ok(None),
            })),
            &mut rx,
            &salts,
            &out,
            &error,
            &CancelToken::default(),
        );
        for id in 0usize..3 {
            let result = results.try_recv().expect("one rejection per submitted job");
            assert_eq!(result.job_id, id.to_le_bytes());
            match result.outcome {
                StreamOutcome::Completed(Err(actual)) => {
                    assert_eq!(actual, error.to_sample_error())
                }
                StreamOutcome::Completed(Ok(_)) | StreamOutcome::Cancelled => {
                    panic!("expected device fault")
                }
            }
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));
        // The sender stays open throughout fault cleanup.
        drop(tx);
    }

    #[test]
    fn admission_grows_when_governor_scale_rises() {
        use std::cell::{Cell, RefCell};

        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping governor growth test: no Metal device");
            }
            return;
        }
        let nominal = batch_size_for_reads(Kernel::Msa, 4);
        if nominal < 2 {
            tracing::warn!("skipping governor growth test: batch capacity is one");
            return;
        }
        struct Governor {
            raised: Cell<bool>,
            completed: RefCell<Vec<usize>>,
            out: Sender<StreamResult>,
        }
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                if self.raised.get() {
                    1.0
                } else {
                    0.25
                }
            }
            fn record_gpu_busy_us(&self, us: u64) {
                // Results stay buffered, so each delta counts a harvested batch.
                self.completed
                    .borrow_mut()
                    .push(self.out.max_capacity() - self.out.capacity());
                if us > 0 {
                    self.raised.set(true);
                }
            }
        }
        let count = nominal * 6;
        let (tx, rx) = tokio::sync::mpsc::channel(count);
        let (out, mut results) = tokio::sync::mpsc::channel(count);
        for id in 0..count {
            tx.try_send(job(id, 8)).unwrap();
        }
        drop(tx);
        let gov = Governor {
            raised: Cell::new(false),
            completed: RefCell::new(vec![0]),
            out: out.clone(),
        };
        let (_salts_tx, salts) = mpsc::channel();
        run(
            &MetalDevice::open(0).unwrap(),
            &Mutex::new(CascadeSettings::default()),
            &Mutex::new(None),
            rx,
            &salts,
            &out,
            &gov,
            &CancelToken::default(),
        );
        let mut completed = gov.completed.into_inner();
        completed.push(count - out.capacity());
        let batches: Vec<_> = completed
            .windows(2)
            .map(|p| p[1] - p[0])
            .filter(|&n| n > 0)
            .collect();
        assert_eq!(batches.first(), Some(&scale_budget(nominal, 0.25)));
        assert_eq!(
            batches.iter().max(),
            Some(&nominal),
            "admission stayed at initial scale: {batches:?}"
        );
        for id in 0..count {
            let result = results.try_recv().unwrap();
            match result.outcome {
                StreamOutcome::Completed(Ok(reads)) => assert_eq!(reads.len(), 4, "job {id}"),
                StreamOutcome::Completed(Err(error)) => panic!("{error}"),
                StreamOutcome::Cancelled => panic!("unexpected cancellation"),
            }
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn preparation_accepts_non_exact_jobs_with_ordinary_msa_limits() {
        let mut input = job(0, 64);
        input.graph.j[0] = 0.5;
        let mut preparer = Preparer::default();
        assert!(preparer
            .prepare(&mut input, CascadeSettings::default(), true)
            .unwrap()
            .is_none());
        input.params.num_sweeps = sampler::MAX_SWEEPS + 1;
        assert!(matches!(
            preparer.prepare(&mut input, CascadeSettings::default(), true),
            Err(SampleError::TooLarge(_))
        ));
        input.params.num_sweeps = 0;
        assert!(preparer
            .prepare(&mut input, CascadeSettings::default(), true)
            .unwrap()
            .is_none());
        input.params.num_sweeps = 64;
        input.graph.edges = (1..=sampler::MSA_MAX_DEG + 1).map(|v| (0, v)).collect();
        input.graph.j = vec![0.5; input.graph.edges.len()];
        assert!(matches!(
            preparer.prepare(&mut input, CascadeSettings::default(), true),
            Err(SampleError::TooLarge(_))
        ));
        input
            .graph
            .h
            .resize(sampler::kernel_max_nodes(Kernel::Msa) + 1, 0.0);
        assert!(matches!(
            preparer.prepare(&mut input, CascadeSettings::default(), true),
            Err(SampleError::TooLarge(_))
        ));
    }

    #[test]
    fn zero_sweeps_select_batch_in_both_coefficient_domains() {
        let mut preparer = Preparer::default();
        for coefficient in [1.0, 0.5] {
            let mut input = job(0, 0);
            input.graph.j[0] = coefficient;
            assert!(preparer
                .prepare(&mut input, CascadeSettings::default(), true)
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn public_stream_preserves_empty_and_zero_sweep_results() {
        let device = MetalDevice::open(0).unwrap();
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for coefficient in [1.0, 0.5] {
            let mut input = job(inputs.len(), 0);
            input.graph.j[0] = coefficient;
            expected.push(
                sampler::sample_ising(&device, &input.graph, &input.params, Kernel::Msa).unwrap(),
            );
            inputs.push(input);
        }
        for reads in [0, 3] {
            let mut input = job(inputs.len(), 0);
            input.graph = IsingGraph::new(vec![], vec![], vec![]);
            input.params.num_reads = reads;
            expected.push(
                (0..reads.max(1))
                    .map(|_| quip_solver_core::SamplerResult {
                        spins: vec![],
                        energy_milli: 0,
                    })
                    .collect(),
            );
            inputs.push(input);
        }
        let count = inputs.len();
        let sampler = MetalSampler::new(
            device,
            crate::iokit_gov::UtilGovernor::start(0, 100, false),
            Kernel::Msa,
        );
        let (tx, rx) = tokio::sync::mpsc::channel(count);
        let (out, mut results) = tokio::sync::mpsc::channel(count);
        for input in inputs {
            tx.try_send(input).unwrap();
        }
        drop(tx);
        sampler.sample_stream(rx, out, CancelToken::default());
        for (id, reads) in expected.into_iter().enumerate() {
            let result = results.blocking_recv().unwrap();
            assert_eq!(result.job_id, id.to_le_bytes());
            match result.outcome {
                StreamOutcome::Completed(Ok(actual)) => assert_eq!(actual, reads),
                StreamOutcome::Completed(Err(error)) => panic!("{error}"),
                StreamOutcome::Cancelled => panic!("unexpected cancellation"),
            }
            if id >= 2 {
                assert_eq!(result.device_access_time_us, 0);
            }
        }
        assert!(results.blocking_recv().is_none());
    }

    #[test]
    fn fallback_consults_throttling_governor_before_submission() {
        use std::cell::Cell;
        struct Governor {
            checks: Cell<usize>,
            reported: Cell<u64>,
            start: Instant,
        }
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                self.checks.set(self.checks.get() + 1);
                self.checks.get() <= 2
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, us: u64) {
                assert!(self.checks.get() >= 3);
                assert!(self.start.elapsed() >= Duration::from_millis(50));
                self.reported.set(self.reported.get() + us);
            }
        }
        let device = MetalDevice::open(0).unwrap();
        let governor = Governor {
            checks: Cell::new(0),
            reported: Cell::new(0),
            start: Instant::now(),
        };
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        let mut input = job(0, 32);
        input.graph.j[0] = 0.5;
        tx.try_send(input).unwrap();
        drop(tx);
        let (_salts_tx, salts) = mpsc::channel();
        run(
            &device,
            &Mutex::new(CascadeSettings::default()),
            &Mutex::new(None),
            rx,
            &salts,
            &out,
            &governor,
            &CancelToken::default(),
        );
        let result = results.try_recv().unwrap();
        assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(_))));
        assert!(result.device_access_time_us > 0);
        assert_eq!(governor.reported.get(), result.device_access_time_us);
    }

    #[test]
    fn invalid_admissions_fail_before_allocating_schedules() {
        let mut invalid = job(0, sampler::MAX_SWEEPS + 1);
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.params.num_sweeps = 0;
        validate(&invalid).unwrap();
        invalid.params.num_sweeps = 32;
        invalid.graph.j.pop();
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.graph.j.push(-0.5);
        validate(&invalid).unwrap();
        invalid.graph.j.pop();
        invalid.graph.j.push(-1.0);
        validate(&invalid).unwrap();
    }

    #[test]
    fn cancelling_a_kept_slot_returns_once_and_releases_it() {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping kept-slot cancellation test: no Metal device");
            }
            return;
        }
        struct Governor;
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = MetalDevice::open(0).unwrap();
        let settings = CascadeSettings {
            stages: stage_array(&[8]),
            keep: 1.0,
            keep_min: 1.0,
            keep_max: 1.0,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let mut live_job = job(1000, 256);
        live_job.watermark = Some(1);
        // Train the cutoff above this ferromagnetic job's negative energy.
        // Direct settings isolate slot cancellation from TOML's minimum of two.
        for _ in 0..200 {
            let (mut ticket, _, _) = controller.admit(&live_job);
            controller.checkpoint(&mut ticket, 0, None);
            controller.finish(&ticket, None, false);
        }
        let mut pool = Pool::new(&device, &live_job, 1, any_plan(&live_job)).unwrap();
        let (ticket, prepared) = controller.admit_schedule(&live_job);
        let slot = pool
            .slots
            .admit(SlotJob {
                graph: live_job.graph.clone(),
                schedule: Arc::clone(&prepared.betas),
                checkpoints: prepared.checkpoints.clone(),
                fresh_from: prepared.fresh_from,
                seed: live_job.params.seed,
                observe: false,
            })
            .unwrap();
        pool.live[slot] = Some(Live {
            job: live_job,
            origin: Origin::Stream,
            ticket,
            accounted_us: 0,
            admitted: Instant::now(),
            last_target_milli: None,
        });
        let (out, mut results) = tokio::sync::mpsc::channel(2);
        let cancel = CancelToken::default();
        pool.slots.commit_step(8, 8).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert_eq!(pool.live[slot].as_ref().unwrap().ticket.stage, 1);
        assert_eq!(pool.slots.live(), 1);
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        pool.slots.commit_step(8, 8).unwrap();
        cancel.cancel_through(1);
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert!(matches!(
            results.try_recv().unwrap().outcome,
            StreamOutcome::Cancelled
        ));
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert_eq!(pool.slots.live(), 0);
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn salts_of_a_stopped_lease_are_dropped_at_admission() {
        let device = MetalDevice::open(0).unwrap();
        let gov = crate::iokit_gov::UtilGovernor::start(0, 100, false);
        let (reply, outcomes) = mpsc::channel();
        let (salts_tx, salts) = mpsc::sync_channel(PREP_BOUND);
        let stop = Arc::new(AtomicBool::new(true));
        for index in 0..3 {
            let mut salt = salt(index, &reply);
            salt.stop = Arc::clone(&stop);
            salts_tx.send(salt).unwrap();
        }
        let (tx, jobs) = tokio::sync::mpsc::channel(1);
        tx.try_send(job(100, 16)).unwrap();
        drop(tx);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        run(
            &device,
            &Mutex::new(CascadeSettings::default()),
            &Mutex::new(None),
            jobs,
            &salts,
            &out,
            &gov,
            &CancelToken::default(),
        );
        let mut dropped: Vec<u64> = outcomes
            .try_iter()
            .map(|outcome| match outcome {
                SaltOutcome::Dropped { index } => index,
                _ => panic!("a stopped lease's salt must not run"),
            })
            .collect();
        dropped.sort_unstable();
        assert_eq!(dropped, [0, 1, 2]);
        assert!(matches!(
            results.try_recv().unwrap().outcome,
            StreamOutcome::Completed(Ok(_))
        ));
    }

    #[test]
    fn stopping_a_lease_releases_its_live_unit_once() {
        struct Governor;
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = MetalDevice::open(0).unwrap();
        // open_gates keeps the unit past the first checkpoint deterministically;
        // this test is about the lease `stop` flag, not the target hit.
        let mut controller = Controller::new(CascadeSettings {
            stages: stage_array(&[8]),
            open_gates: true,
            ..CascadeSettings::default()
        });
        let live_job = job(1000, 256);
        let mut pool = Pool::new(&device, &live_job, 1, any_plan(&live_job)).unwrap();
        let (ticket, prepared) = controller.admit_schedule(&live_job);
        let slot = pool
            .slots
            .admit(SlotJob {
                graph: live_job.graph.clone(),
                schedule: Arc::clone(&prepared.betas),
                checkpoints: prepared.checkpoints.clone(),
                fresh_from: prepared.fresh_from,
                seed: live_job.params.seed,
                observe: true,
            })
            .unwrap();
        let (reply, outcomes) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        pool.live[slot] = Some(Live {
            job: live_job,
            origin: Origin::Salt {
                index: 5,
                round: 0,
                target: LiveTarget::new(None),
                top10: LeaseTopK::new(),
                stop: Arc::clone(&stop),
                reply,
            },
            ticket,
            accounted_us: 0,
            admitted: Instant::now(),
            last_target_milli: None,
        });
        let (out, _results) = tokio::sync::mpsc::channel(1);
        let cancel = CancelToken::default();
        pool.slots.commit_step(8, 8).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert_eq!(pool.slots.live(), 1, "the unit keeps running past stage 0");
        assert!(outcomes.try_recv().is_err());
        pool.slots.commit_step(8, 8).unwrap();
        stop.store(true, Ordering::Release);
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert!(matches!(
            outcomes.try_recv().unwrap(),
            SaltOutcome::Dropped { index: 5 }
        ));
        assert_eq!(pool.slots.live(), 0);
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert!(outcomes.try_recv().is_err());
    }
}
