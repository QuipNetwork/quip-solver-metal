// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Otsuka on validated twisted-ice-dimer jobs, Metal SA on every other job.
//! Port of the CUDA `src/cluster/sampler.rs`.
//!
//! The cluster device and Metal SA each have their own command queue, and
//! each plans its command buffers against the 500 ms ceiling on the
//! assumption that it owns the GPU. So the sampler never runs a cluster job
//! while an SA job is in flight: a family job waits until every plain job it
//! forwarded to the SA stream has answered, and the dispatcher forwards no
//! plain job while the cluster job runs.

use super::device::{ClusterDevice, ClusterDeviceError, ClusterParams};
use super::{PreparedGraph, Weights};
use crate::capacity::DeviceLimits;
use crate::combined::CombinedSampler;
use crate::Kernel;
use quip_solver_core::families::{Declaration, FamilyClaim, FamilyContext};
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::{
    CancelToken, IsingGraph, OpenError, SampleError, SampleParams, Sampler, SamplerResult,
    StreamJob, StreamOutcome, StreamResult, WarmStart, WarmStreamJob,
};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{channel, Receiver, Sender};

/// Stable binary identity.
pub const OTSUKA_SOLVER_ID: &str = "quip-metal-otsuka/1";

/// The degree bound of the Otsuka identity. The session enforces it for
/// family and plain jobs.
pub const MAX_DEGREE: usize = 128;

/// Reads per job at most: one anneal each, one threadgroup per anneal.
pub const MAX_READS: usize = 256;

const TWISTED_DIMER: &[FamilyClaim] = &[FamilyClaim {
    revisions: &[TWISTED_DIMER_V1],
    only_params: &[],
}];

/// How often a family job that waits for the SA stream to drain checks
/// whether the session has closed.
const DRAIN_POLL: Duration = Duration::from_millis(10);

/// The cluster anneal boundary, so the session tests can run without a GPU.
#[doc(hidden)]
pub trait ClusterBackend: Send + Sync + 'static {
    /// Anneal `params.anneals` copies of `prepared`, as
    /// [`ClusterDevice::anneal`] does.
    ///
    /// # Errors
    /// Refusals, cancellation, and device failures.
    fn anneal(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
        stop: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<i8>>, ClusterDeviceError>;
}

impl ClusterBackend for ClusterDevice {
    fn anneal(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
        stop: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<i8>>, ClusterDeviceError> {
        Self::anneal(self, prepared, params, stop)
    }
}

/// The Metal SA engine of `quip-metal-otsuka`: the sampler of
/// `quip-metal-sa`. [`Sampler::should_throttle`] reports the IOKit governor,
/// so a cluster job yields between command buffers.
pub struct MetalSa(CombinedSampler);

impl Sampler for MetalSa {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.0.sample(graph, params)
    }

    fn sample_stream(
        &self,
        jobs: Receiver<StreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.0.sample_stream(jobs, out, cancel);
    }

    fn stream_width(&self) -> usize {
        self.0.stream_width()
    }

    fn declared_stream_width() -> u32 {
        u32::try_from(crate::combined::stream_width(Kernel::Sa)).unwrap_or(u32::MAX)
    }

    fn utilization(&self) -> f64 {
        self.0.utilization()
    }

    fn should_throttle(&self) -> bool {
        self.0.governor_yields()
    }

    fn max_reads(&self) -> u32 {
        self.0.max_reads()
    }

    fn apply_config(&self, backend_toml: &str) {
        self.0.apply_config(backend_toml);
    }
}

/// Plain jobs the dispatcher forwarded to the SA stream that have not
/// answered yet.
#[derive(Default)]
struct InFlight {
    count: Mutex<usize>,
    idle: Condvar,
}

impl InFlight {
    fn lock(&self) -> MutexGuard<'_, usize> {
        self.count
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn begin(&self) {
        *self.lock() += 1;
    }

    fn end(&self) {
        let mut count = self.lock();
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.idle.notify_all();
        }
    }

    /// Wait until no SA job is in flight. False when `abandon` returns true
    /// first.
    fn wait_idle(&self, abandon: &dyn Fn() -> bool) -> bool {
        let mut count = self.lock();
        while *count > 0 {
            if abandon() {
                return false;
            }
            count = self
                .idle
                .wait_timeout(count, DRAIN_POLL)
                .unwrap_or_else(|poison| poison.into_inner())
                .0;
        }
        true
    }
}

/// Serial family jobs beside one persistent Metal SA stream, never at the
/// same time as an SA job.
pub struct OtsukaSampler<D = ClusterDevice, S = MetalSa> {
    cluster: D,
    sa: S,
    limits: DeviceLimits,
    /// Held by each direct call and each cluster job, so two direct calls
    /// never share the GPU.
    exclusive: Mutex<()>,
}

impl OtsukaSampler {
    /// Open the cluster kernel on Metal device `device`, and bind the SA
    /// engine of `quip-metal-sa` to the same device. The SA engine opens on
    /// first use, unless `check` opens it now.
    ///
    /// # Errors
    /// The cluster device or kernel fails to open, or `check` fails.
    pub fn open(
        device: usize,
        utilization: u32,
        yielding: bool,
        check: bool,
    ) -> Result<Self, OpenError> {
        let cluster =
            ClusterDevice::open(device).map_err(|e| OpenError(format!("cluster device: {e}")))?;
        let sa = CombinedSampler::new(Kernel::Sa, device, utilization, yielding);
        if check {
            sa.check()?;
        }
        Ok(Self::with_backends(
            cluster,
            MetalSa(sa),
            *crate::sampler::host_device_limits(),
        ))
    }
}

impl<D: ClusterBackend, S: Sampler> OtsukaSampler<D, S> {
    /// Inject a cluster backend and an SA engine.
    #[doc(hidden)]
    #[must_use]
    pub fn with_backends(cluster: D, sa: S, limits: DeviceLimits) -> Self {
        Self {
            cluster,
            sa,
            limits,
            exclusive: Mutex::new(()),
        }
    }

    fn exclusive(&self) -> MutexGuard<'_, ()> {
        self.exclusive
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Anneal one read per anneal with the measured default schedule, then
    /// score every read on the original model.
    fn reads(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        prepared: &PreparedGraph,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        if params.num_reads == 0 || params.num_reads > MAX_READS {
            return Err(SampleError::Capacity);
        }
        let cluster = ClusterParams {
            anneals: params.num_reads,
            seed: params.seed,
            ..ClusterParams::default()
        };
        // A governor yield refuses this job as busy. Only the session token
        // makes `finish` report Cancelled and drop the result.
        let stopped = || cancelled() || self.sa.should_throttle();
        let _gpu = self.exclusive();
        checkpoint(&stopped).map_err(sample_error)?;
        let spins = self
            .cluster
            .anneal(prepared, &cluster, &stopped)
            .map_err(sample_error)?;
        if spins.len() != params.num_reads
            || spins
                .iter()
                .any(|s| s.len() != graph.h.len() || s.iter().any(|v| ![-1, 1].contains(v)))
        {
            return Err(SampleError::DeviceFault(
                "incomplete or invalid cluster output".into(),
            ));
        }
        let mut reads = Vec::with_capacity(spins.len());
        for spins in spins {
            checkpoint(cancelled).map_err(sample_error)?;
            reads.push(SamplerResult {
                energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
                spins,
            });
        }
        Ok(reads)
    }

    fn prepared(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        family: &FamilyContext,
    ) -> Result<PreparedGraph, SampleError> {
        if family.instance().h().len() != graph.h.len()
            || family.instance().edges().len() != graph.edges.len()
        {
            return Err(SampleError::Capacity);
        }
        PreparedGraph::from_family(family, Weights::FULL, &self.limits, params.num_reads).map_err(
            |e| {
                tracing::warn!(error = %e, "cluster admission refused");
                SampleError::Capacity
            },
        )
    }

    fn family_reads(
        &self,
        job: &StreamJob,
        family: &FamilyContext,
        cancel: &CancelToken,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let prepared = self.prepared(&job.graph, &job.params, family)?;
        self.reads(&job.graph, &job.params, &prepared, &|| {
            cancel.is_cancelled(job.watermark)
        })
    }

    fn finish(
        job: &StreamJob,
        cancel: &CancelToken,
        run: impl FnOnce() -> Result<Vec<SamplerResult>, SampleError>,
    ) -> StreamResult {
        let start = Instant::now();
        let result = if cancel.is_cancelled(job.watermark) {
            Ok(Vec::new())
        } else {
            run()
        };
        let fatal = result.as_ref().is_err_and(SampleError::is_fatal);
        let outcome = if cancel.is_cancelled(job.watermark) && !fatal {
            StreamOutcome::Cancelled
        } else {
            StreamOutcome::Completed(result)
        };
        StreamResult {
            job_id: job.job_id.clone(),
            outcome,
            device_access_time_us: u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
        }
    }

    /// Run the production result and cancellation path on a prepared fixture.
    #[doc(hidden)]
    #[must_use]
    pub fn sample_prepared_for_test(
        &self,
        job: &StreamJob,
        prepared: &PreparedGraph,
        cancel: &CancelToken,
    ) -> StreamResult {
        Self::finish(job, cancel, || {
            self.reads(&job.graph, &job.params, prepared, &|| {
                cancel.is_cancelled(job.watermark)
            })
        })
    }
}

fn checkpoint(stopped: &dyn Fn() -> bool) -> Result<(), ClusterDeviceError> {
    if stopped() {
        Err(ClusterDeviceError::Cancelled)
    } else {
        Ok(())
    }
}

fn supported(family: &FamilyContext) -> bool {
    family.family_id() == TWISTED_DIMER_V1.family_id
        && family.revision() == TWISTED_DIMER_V1.revision
}

/// A refusal that does not fit is `Capacity`, with one refund. A stop is
/// `DeviceBusy`; `finish` reports it as Cancelled when the session cancelled
/// the job. A device failure is fatal.
fn sample_error(error: ClusterDeviceError) -> SampleError {
    match error {
        ClusterDeviceError::Cancelled => SampleError::DeviceBusy,
        e @ (ClusterDeviceError::Capacity(_)
        | ClusterDeviceError::Prepare(_)
        | ClusterDeviceError::SweepTooLong { .. }) => {
            tracing::warn!(error = %e, "cluster job refused for capacity");
            SampleError::Capacity
        }
        e @ (ClusterDeviceError::Parameter(_)
        | ClusterDeviceError::Metal(_)
        | ClusterDeviceError::CommandBuffer(_)
        | ClusterDeviceError::Spin(_)) => SampleError::DeviceFault(e.to_string()),
    }
}

impl<D: ClusterBackend, S: Sampler> Sampler for OtsukaSampler<D, S> {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let _gpu = self.exclusive();
        self.sa.sample(graph, params)
    }

    fn accepts_family_metadata() -> bool {
        true
    }

    fn declaration() -> Declaration {
        Declaration {
            solver_id: OTSUKA_SOLVER_ID,
            generated: TWISTED_DIMER,
            accepted: TWISTED_DIMER,
        }
    }

    fn sample_family(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        _warm: Option<&WarmStart>,
        family: &FamilyContext,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        if !supported(family) {
            return self.sample(graph, params);
        }
        let prepared = self.prepared(graph, params, family)?;
        self.reads(graph, params, &prepared, &|| false)
    }

    fn sample_stream(
        &self,
        jobs: Receiver<StreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        self.sa.sample_stream(jobs, out, cancel);
    }

    fn sample_stream_warm(
        &self,
        mut jobs: Receiver<WarmStreamJob>,
        out: Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        let in_flight = InFlight::default();
        std::thread::scope(|scope| {
            // Keep the SA stream alive across plain jobs. Its results pass
            // through a relay that counts the jobs still in flight.
            let width = self.sa.stream_width().max(1);
            let (sa_tx, sa_rx) = channel(width.saturating_mul(2));
            let (sa_out, mut sa_results) = channel(width);
            let sa_cancel = cancel.clone();
            scope.spawn(|| self.sa.sample_stream(sa_rx, sa_out, sa_cancel));
            let relay_out = out.clone();
            let in_flight = &in_flight;
            scope.spawn(move || {
                while let Some(result) = sa_results.blocking_recv() {
                    in_flight.end();
                    if relay_out.blocking_send(result).is_err() {
                        break;
                    }
                }
            });
            while let Some(item) = jobs.blocking_recv() {
                if out.is_closed() {
                    break;
                }
                let job = item.job;
                if let Some(family) = item.family.as_ref().filter(|f| supported(f)) {
                    if !in_flight.wait_idle(&|| out.is_closed()) {
                        break;
                    }
                    let result =
                        Self::finish(&job, &cancel, || self.family_reads(&job, family, &cancel));
                    if out.blocking_send(result).is_err() {
                        break;
                    }
                } else {
                    in_flight.begin();
                    if sa_tx.blocking_send(job).is_err() {
                        in_flight.end();
                        break;
                    }
                }
            }
            drop(sa_tx);
        });
    }

    fn stream_width(&self) -> usize {
        self.sa.stream_width()
    }

    fn declared_stream_width() -> u32 {
        S::declared_stream_width()
    }

    fn max_reads(&self) -> u32 {
        u32::try_from(MAX_READS).unwrap_or(u32::MAX)
    }

    fn utilization(&self) -> f64 {
        self.sa.utilization()
    }

    fn should_throttle(&self) -> bool {
        self.sa.should_throttle()
    }

    fn apply_config(&self, backend_toml: &str) {
        self.sa.apply_config(backend_toml);
    }
}

#[cfg(test)]
mod tests {
    use super::InFlight;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn a_family_job_waits_until_every_sa_job_answers() {
        let in_flight = InFlight::default();
        in_flight.begin();
        in_flight.begin();
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                assert!(in_flight.wait_idle(&|| false));
                done.store(true, Ordering::SeqCst);
            });
            in_flight.end();
            std::thread::sleep(super::DRAIN_POLL * 3);
            assert!(
                !done.load(Ordering::SeqCst),
                "one SA job is still in flight"
            );
            in_flight.end();
        });
        assert!(done.load(Ordering::SeqCst));
    }

    #[test]
    fn a_closed_session_stops_the_wait() {
        let in_flight = InFlight::default();
        in_flight.begin();
        assert!(!in_flight.wait_idle(&|| true));
    }
}
