// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The Otsuka session: family routing, cancellation, refusals, and the
//! serialization of cluster and SA jobs, with fake engines, then the real
//! `quip-metal-otsuka` on the Metal device. Port of the CUDA
//! `tests/cluster_session.rs`.
//!
//! Metal GPU tests: need a real device (Apple Silicon). Run with
//! `--test-threads=1`.

use quip_solver_core::{
    CancelToken, IsingGraph, SampleError, SampleParams, Sampler, SamplerResult, StreamJob,
    StreamOutcome, StreamResult,
};
use quip_solver_metal::capacity::{CapacityError, DeviceLimits};
use quip_solver_metal::cluster::device::{ClusterDeviceError, ClusterParams};
use quip_solver_metal::cluster::sampler::{
    ClusterBackend, ClusterGovernor, OtsukaSampler, MAX_READS,
};
use quip_solver_metal::cluster::{Model, Motifs, PreparedGraph, Weights};
use quip_solver_metal::iokit_gov::UtilGovernor;
use quip_solver_metal::metal_otsuka_identity;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[expect(
    dead_code,
    reason = "the shared lease coordinator has helpers this file does not call"
)]
mod support;

/// Longest wait for one session phase. It detects a hang, not a slow run.
const PHASE_LIMIT: Duration = Duration::from_secs(60);

/// Command buffers of one fake anneal: sweep chunks, then descent passes.
const SWEEP_CHUNKS: usize = 7;
const DESCENT_PASSES: usize = 3;

/// GPU microseconds a fake command buffer reports.
const LAUNCH_US: u64 = 1_000;

/// How long a fake SA job or a fake cluster job holds the fake GPU in the
/// serialization test.
const HOLD: Duration = Duration::from_millis(50);

const HEADLESS_LIMITS: DeviceLimits = DeviceLimits {
    memory_bytes: 64 << 20,
    max_buffer_bytes: 64 << 20,
};

/// Which engine holds the fake GPU, and how often both did.
#[derive(Clone, Default)]
struct Probe {
    sa_active: Arc<AtomicUsize>,
    cluster_active: Arc<AtomicBool>,
    overlaps: Arc<AtomicUsize>,
}

#[derive(Clone, Default)]
struct FakeDevice {
    launches: Arc<Mutex<usize>>,
    calls: Arc<Mutex<Vec<ClusterParams>>>,
    cancel: Option<CancelToken>,
    fail: bool,
    short: bool,
    cancel_after: usize,
    capacity_for_reads: Option<usize>,
    yield_after: Option<Arc<AtomicBool>>,
    probe: Option<Probe>,
}

impl ClusterBackend for FakeDevice {
    fn anneal(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
        stop: &dyn Fn() -> bool,
        busy_us: &dyn Fn(u64),
    ) -> Result<Vec<Vec<i8>>, ClusterDeviceError> {
        self.calls.lock().unwrap().push(params.clone());
        if self.capacity_for_reads == Some(params.anneals) {
            return Err(CapacityError::Memory {
                needed: 1024,
                usable: 512,
            }
            .into());
        }
        if let Some(probe) = &self.probe {
            if probe.sa_active.load(Ordering::SeqCst) > 0 {
                probe.overlaps.fetch_add(1, Ordering::SeqCst);
            }
            probe.cluster_active.store(true, Ordering::SeqCst);
            std::thread::sleep(HOLD);
            probe.cluster_active.store(false, Ordering::SeqCst);
        }
        for _ in 0..SWEEP_CHUNKS + DESCENT_PASSES {
            if stop() {
                return Err(ClusterDeviceError::Cancelled);
            }
            let launches = {
                let mut launches = self.launches.lock().unwrap();
                *launches += 1;
                *launches
            };
            busy_us(LAUNCH_US);
            if let Some(token) = &self.cancel {
                if launches >= self.cancel_after {
                    token.cancel_through(1);
                }
            }
            if self.fail {
                return Err(ClusterDeviceError::Parameter("scripted device error"));
            }
        }
        if let Some(yielding) = &self.yield_after {
            yielding.store(true, Ordering::SeqCst);
        }
        let count = if self.short { 1 } else { params.anneals };
        let n = prepared.original_graph().num_nodes();
        Ok((0..count)
            .map(|read| {
                (0..n)
                    .map(|node| if (read + node) % 2 == 0 { 1 } else { -1 })
                    .collect()
            })
            .collect())
    }
}

#[derive(Clone, Default)]
struct FakeSa {
    calls: Arc<Mutex<usize>>,
    yielding: bool,
    late_yield: Arc<AtomicBool>,
    streams: Arc<Mutex<usize>>,
    probe: Option<Probe>,
    governor: Option<Arc<UtilGovernor>>,
    busy_us: Arc<AtomicU64>,
}

impl ClusterGovernor for FakeSa {
    fn yields_to_others(&self) -> bool {
        self.yielding
            || self.late_yield.load(Ordering::SeqCst)
            || self
                .governor
                .as_ref()
                .is_some_and(|g| g.should_yield_to_others())
    }

    fn record_gpu_busy_us(&self, us: u64) {
        self.busy_us.fetch_add(us, Ordering::SeqCst);
        if let Some(governor) = &self.governor {
            governor.record_gpu_busy_us(us);
        }
    }
}

impl Sampler for FakeSa {
    fn sample(
        &self,
        _graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        *self.calls.lock().unwrap() += 1;
        Ok((0..params.num_reads)
            .map(|_| SamplerResult {
                spins: vec![1, 1],
                energy_milli: -1000,
            })
            .collect())
    }

    fn sample_stream(
        &self,
        mut jobs: tokio::sync::mpsc::Receiver<StreamJob>,
        out: tokio::sync::mpsc::Sender<StreamResult>,
        _cancel: CancelToken,
    ) {
        *self.streams.lock().unwrap() += 1;
        while let Some(job) = jobs.blocking_recv() {
            if let Some(probe) = &self.probe {
                if probe.cluster_active.load(Ordering::SeqCst) {
                    probe.overlaps.fetch_add(1, Ordering::SeqCst);
                }
                probe.sa_active.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(HOLD);
            }
            let result = self.sample(&job.graph, &job.params);
            if let Some(probe) = &self.probe {
                probe.sa_active.fetch_sub(1, Ordering::SeqCst);
            }
            if out
                .blocking_send(StreamResult {
                    job_id: job.job_id,
                    outcome: StreamOutcome::Completed(result),
                    device_access_time_us: 1,
                })
                .is_err()
            {
                return;
            }
        }
    }

    fn stream_width(&self) -> usize {
        4
    }

    fn declared_stream_width() -> u32 {
        0
    }
}

fn graph() -> IsingGraph {
    IsingGraph::new(vec![0.0; 2], vec![-1.0], vec![(0, 1)])
}

fn prepared(limits: &DeviceLimits) -> PreparedGraph {
    PreparedGraph::new(
        &Model {
            h_milli: &[0, 0],
            edges: &[(0, 1)],
            j_milli: &[-1000],
        },
        &Motifs {
            tiles: &[],
            gauges: &[],
            pairs: &[(0, 1)],
        },
        Weights::FULL,
        limits,
        3,
    )
    .unwrap()
}

fn job() -> StreamJob {
    StreamJob {
        job_id: b"cluster".to_vec(),
        graph: graph(),
        params: SampleParams {
            num_reads: 3,
            num_sweeps: 7,
            seed: 0x5eed,
            ..Default::default()
        },
        watermark: Some(1),
    }
}

fn sampler(device: FakeDevice, sa: FakeSa) -> OtsukaSampler<FakeDevice, FakeSa> {
    OtsukaSampler::with_backends(device, sa, HEADLESS_LIMITS)
}

#[test]
fn every_read_is_returned_in_order_and_rescored() {
    let sampler = sampler(FakeDevice::default(), FakeSa::default());
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
        panic!("expected complete reads")
    };
    assert_eq!(reads.len(), 3);
    assert_eq!(
        reads.iter().map(|r| r.spins.clone()).collect::<Vec<_>>(),
        [vec![1, -1], vec![-1, 1], vec![1, -1]]
    );
    // The sampler scores the original coefficients: J = -1 on opposite spins.
    assert!(reads.iter().all(|r| r.energy_milli == 1000));
}

/// The device gets the measured default schedule, one anneal per read, and
/// the job seed. The job's sweep count and ladder do not change it.
#[test]
fn the_production_schedule_and_the_job_seed_reach_the_device() {
    let device = FakeDevice::default();
    let sampler = sampler(device.clone(), FakeSa::default());
    let mut request = job();
    request.params.num_sweeps = 3_000_000_000;
    request.params.beta_range = Some((0.1, 0.2));
    let result = sampler.sample_prepared_for_test(
        &request,
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(reads)) if reads.len() == 3));
    let calls = device.calls.lock().unwrap();
    assert_eq!(
        *calls,
        [ClusterParams {
            anneals: 3,
            seed: 0x5eed,
            ..ClusterParams::default()
        }]
    );
}

#[test]
fn cancellation_between_chunks_returns_cancelled_outcome() {
    let cancel = CancelToken::default();
    let device = FakeDevice {
        cancel: Some(cancel.clone()),
        ..Default::default()
    };
    let sampler = sampler(device.clone(), FakeSa::default());
    let result = sampler.sample_prepared_for_test(&job(), &prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert_eq!(*device.launches.lock().unwrap(), 1);
}

#[test]
fn cancellation_during_final_descent_stops_before_the_next_pass() {
    let cancel = CancelToken::default();
    let device = FakeDevice {
        cancel: Some(cancel.clone()),
        cancel_after: SWEEP_CHUNKS + 1,
        ..Default::default()
    };
    let sampler = sampler(device.clone(), FakeSa::default());
    let result = sampler.sample_prepared_for_test(&job(), &prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert_eq!(*device.launches.lock().unwrap(), SWEEP_CHUNKS + 1);
}

#[test]
fn a_device_error_is_a_job_error() {
    let device = FakeDevice {
        fail: true,
        ..Default::default()
    };
    let sampler = sampler(device, FakeSa::default());
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    let StreamOutcome::Completed(Err(error)) = result.outcome else {
        panic!("expected job error")
    };
    assert!(error.is_fatal());
    assert!(error.to_string().contains("scripted device error"));
}

#[test]
fn a_device_fault_remains_visible_when_cancellation_races_it() {
    let cancel = CancelToken::default();
    let device = FakeDevice {
        cancel: Some(cancel.clone()),
        fail: true,
        ..Default::default()
    };
    let sampler = sampler(device, FakeSa::default());
    let result = sampler.sample_prepared_for_test(&job(), &prepared(&HEADLESS_LIMITS), &cancel);
    let StreamOutcome::Completed(Err(error)) = result.outcome else {
        panic!("device fault was suppressed")
    };
    assert!(error.is_fatal());
}

#[test]
fn incomplete_device_output_is_rejected() {
    let device = FakeDevice {
        short: true,
        ..Default::default()
    };
    let sampler = sampler(device, FakeSa::default());
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(result.outcome, StreamOutcome::Completed(Err(error)) if error.is_fatal()));
}

/// A launch that does not fit is a non-fatal `Capacity` refusal.
#[test]
fn a_device_capacity_refusal_is_a_capacity_reject() {
    let device = FakeDevice {
        capacity_for_reads: Some(3),
        ..Default::default()
    };
    let sampler = sampler(device.clone(), FakeSa::default());
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(
        result.outcome,
        StreamOutcome::Completed(Err(SampleError::Capacity))
    ));
    assert_eq!(*device.launches.lock().unwrap(), 0);
}

#[test]
fn read_counts_outside_the_cap_are_refusals_and_never_launch() {
    let device = FakeDevice::default();
    let sampler = sampler(device.clone(), FakeSa::default());
    for reads in [0, MAX_READS + 1] {
        let mut request = job();
        request.params.num_reads = reads;
        let result = sampler.sample_prepared_for_test(
            &request,
            &prepared(&HEADLESS_LIMITS),
            &CancelToken::default(),
        );
        assert!(matches!(
            result.outcome,
            StreamOutcome::Completed(Err(SampleError::Capacity))
        ));
    }
    assert!(device.calls.lock().unwrap().is_empty());
}

#[test]
fn governor_yield_is_busy_and_never_reaches_the_cluster_device() {
    let device = FakeDevice::default();
    let sa = FakeSa {
        yielding: true,
        ..Default::default()
    };
    let sampler = sampler(device.clone(), sa);
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(
        result.outcome,
        StreamOutcome::Completed(Err(SampleError::DeviceBusy))
    ));
    assert!(device.calls.lock().unwrap().is_empty());
}

#[test]
fn governor_yield_after_device_completion_keeps_finished_reads() {
    let late_yield = Arc::new(AtomicBool::new(false));
    let device = FakeDevice {
        yield_after: Some(late_yield.clone()),
        ..Default::default()
    };
    let sa = FakeSa {
        late_yield,
        ..Default::default()
    };
    let sampler = sampler(device, sa);
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(reads)) if reads.len() == 3));
}

/// The miner's own cluster load does not stop a family job: the total load
/// is above the ceiling, but the load of other processes is below it.
#[test]
fn own_gpu_load_above_the_ceiling_does_not_stop_a_family_job() {
    let device = FakeDevice::default();
    let sa = FakeSa {
        governor: Some(Arc::new(UtilGovernor::with_sample_for_test(
            80, true, 100, 90,
        ))),
        ..Default::default()
    };
    let sampler = sampler(device.clone(), sa.clone());
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(reads)) if reads.len() == 3));
    let launches = SWEEP_CHUNKS + DESCENT_PASSES;
    assert_eq!(*device.launches.lock().unwrap(), launches);
    // Every command buffer reports its GPU time as the miner's own load.
    assert_eq!(
        sa.busy_us.load(Ordering::SeqCst),
        LAUNCH_US * launches as u64
    );
}

/// Load of other processes above the ceiling stops a family job as busy.
#[test]
fn foreign_gpu_load_above_the_ceiling_stops_a_family_job() {
    let device = FakeDevice::default();
    let sa = FakeSa {
        governor: Some(Arc::new(UtilGovernor::with_sample_for_test(
            80, true, 100, 10,
        ))),
        ..Default::default()
    };
    let sampler = sampler(device.clone(), sa);
    let result = sampler.sample_prepared_for_test(
        &job(),
        &prepared(&HEADLESS_LIMITS),
        &CancelToken::default(),
    );
    assert!(matches!(
        result.outcome,
        StreamOutcome::Completed(Err(SampleError::DeviceBusy))
    ));
    assert_eq!(*device.launches.lock().unwrap(), 0);
}

#[test]
fn an_abandoned_job_never_reaches_the_device() {
    let cancel = CancelToken::default();
    cancel.cancel_through(1);
    let device = FakeDevice::default();
    let sampler = sampler(device.clone(), FakeSa::default());
    let result = sampler.sample_prepared_for_test(&job(), &prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Cancelled));
    assert!(device.calls.lock().unwrap().is_empty());
}

#[test]
fn a_job_without_a_watermark_keeps_all_its_reads() {
    let cancel = CancelToken::default();
    cancel.cancel_through(u64::MAX);
    let mut request = job();
    request.watermark = None;
    let sampler = sampler(FakeDevice::default(), FakeSa::default());
    let result = sampler.sample_prepared_for_test(&request, &prepared(&HEADLESS_LIMITS), &cancel);
    assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(reads)) if reads.len() == 3));
}

#[test]
fn plain_jobs_use_the_sa_stream() {
    let device = FakeDevice::default();
    let sa = FakeSa::default();
    let sampler = sampler(device.clone(), sa.clone());
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (out, mut results) = tokio::sync::mpsc::channel(1);
    tx.blocking_send(job()).unwrap();
    drop(tx);
    sampler.sample_stream(rx, out, CancelToken::default());
    assert!(matches!(
        results.blocking_recv().unwrap().outcome,
        StreamOutcome::Completed(Ok(_))
    ));
    assert_eq!(*sa.calls.lock().unwrap(), 1);
    assert_eq!(*sa.streams.lock().unwrap(), 1);
    assert!(device.calls.lock().unwrap().is_empty());
    assert_eq!(sampler.stream_width(), 4);
}

#[test]
fn capabilities_are_headless_and_bounded() {
    assert!(!<OtsukaSampler as Sampler>::generates_locally());
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_quip-metal-otsuka"))
        .args(["--capabilities", "--device", "2147483647"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(caps["solverId"], "quip-metal-otsuka/1");
    assert_eq!(caps["backend"], "metal");
    assert_eq!(caps["algorithm"], "otsuka");
    assert_eq!(
        caps["streamWidth"],
        <OtsukaSampler as Sampler>::declared_stream_width()
    );
    let id = metal_otsuka_identity(&quip_solver_metal::capacity::host_limits(), None).unwrap();
    assert_eq!(caps["maxNodes"], id.max_nodes);
    assert_eq!(caps["maxEdges"], id.max_edges);
    assert_eq!(caps["maxDegree"], 128);
    assert_eq!(caps["maxDegree"], id.limits.max_degree.unwrap());
    for list in ["generatedFamilies", "acceptedFamilies"] {
        assert_eq!(caps[list][0]["familyId"], "quip/twisted-ice-dimer");
        assert_eq!(caps[list][0]["revisions"], serde_json::json!([1]));
        assert!(caps[list][0].get("paramsDigests").is_none());
    }
}

/// The modes of [`headless_miner_entry`], passed in `QUIP_OTSUKA_TEST_MODE`.
const ROUTING: &str = "routing";
const SERIAL: &str = "serial";
const TINY: &str = "tiny";

/// Reads of the family job that the routing fake device refuses.
const REFUSED_READS: u32 = 5;

// The SDK constructs WarmStreamJob and FamilyContext only inside its session.
// The test executable runs as a headless miner to exercise that exact path.
#[test]
fn headless_miner_entry() {
    let Ok(socket) = std::env::var("QUIP_OTSUKA_TEST_COORDINATOR") else {
        return;
    };
    let mode = std::env::var("QUIP_OTSUKA_TEST_MODE").unwrap();
    let probe = Probe::default();
    let device = FakeDevice {
        capacity_for_reads: (mode == ROUTING).then_some(REFUSED_READS as usize),
        probe: (mode == SERIAL).then(|| probe.clone()),
        ..Default::default()
    };
    let sa = FakeSa {
        probe: (mode == SERIAL).then(|| probe.clone()),
        ..Default::default()
    };
    // The tiny budget refuses every family job in preparation.
    let limits = if mode == TINY {
        DeviceLimits {
            memory_bytes: 64,
            max_buffer_bytes: 64,
        }
    } else {
        HEADLESS_LIMITS
    };
    let sampler = OtsukaSampler::with_backends(device.clone(), sa.clone(), limits);
    let args = quip_solver_core::CommonArgs {
        quip_coordinator: Some(socket),
        miner_id: Some("headless-cluster".into()),
        capabilities: false,
        solve: false,
        check: false,
        log_level: "error".into(),
        sweeps_per_beta: None,
    };
    let identity =
        metal_otsuka_identity(&quip_solver_metal::capacity::host_limits(), None).unwrap();
    let exit = quip_solver_core::run_code(identity, &args, || Ok(sampler));
    assert_eq!(exit as u8, 0);
    assert_eq!(
        *sa.streams.lock().unwrap(),
        1,
        "plain jobs must share one persistent stream"
    );
    let annealed = device.calls.lock().unwrap().len();
    match mode.as_str() {
        ROUTING => {
            assert_eq!(*sa.calls.lock().unwrap(), 2);
            assert_eq!(annealed, 2, "both family jobs reach the cluster backend");
        }
        SERIAL => {
            assert_eq!(*sa.calls.lock().unwrap(), 4);
            assert_eq!(annealed, 2);
            assert_eq!(
                probe.overlaps.load(Ordering::SeqCst),
                0,
                "a cluster job ran while an SA job held the GPU"
            );
        }
        TINY => assert_eq!(annealed, 0, "a refused family job never reaches the device"),
        other => panic!("unknown mode {other}"),
    }
}

/// Start the test executable as a headless miner in `mode`.
async fn headless_session(mode: &str) -> support::Session {
    let mode = mode.to_owned();
    support::Session::start_with(4, "", move |socket| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "headless_miner_entry", "--nocapture"])
            .args(["--test-threads", "1"])
            .env("QUIP_OTSUKA_TEST_COORDINATOR", socket)
            .env("QUIP_OTSUKA_TEST_MODE", mode);
        command
    })
    .await
}

fn plain_job(id: &[u8]) -> quip_solver_core::quip_proto::v1::Job {
    use quip_solver_core::quip_proto::v1::{
        ising_problem, CoefficientEncoding, EdgeList, IsingProblem, Job, JobKind,
    };
    use quip_solver_core::quip_protocol::wire::encode_i32_le;
    Job {
        job_id: id.to_vec(),
        kind: JobKind::IsingSample as i32,
        generation: 1,
        ising: Some(IsingProblem {
            graph: Some(ising_problem::Graph::Edges(EdgeList {
                u: vec![0],
                v: vec![1],
            })),
            encoding: CoefficientEncoding::I32 as i32,
            scale: 1000,
            h: encode_i32_le(&[0, 0]),
            j: encode_i32_le(&[-1000]),
            num_reads: 3,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn warm_routing_keeps_sa_width_and_refuses_only_the_bad_family_job() {
    use quip_solver_core::quip_proto::v1::{coord_msg, RejectReason};
    let mut session = headless_session(ROUTING).await;
    // The initial credits resolve the live width and permit both jobs before either completes.
    assert!(session.log.credits[0] >= 4);
    session
        .send(coord_msg::Msg::SetTarget(support::target(i64::MAX, 4)))
        .await;
    for id in [b"plain-1", b"plain-2"] {
        session.send(coord_msg::Msg::Job(plain_job(id))).await;
    }
    session
        .until("both plain jobs", PHASE_LIMIT, |log| {
            log.results.len() == 2 && log.refunds() == 2
        })
        .await;
    let (base, params) = canonical::fixture();
    let instance = canonical::instance(&base, &params, [7; 32]).unwrap();
    session
        .send(coord_msg::Msg::Job(canonical::job(
            &instance,
            &params,
            b"bad-family",
            REFUSED_READS,
        )))
        .await;
    session
        .until("bad job refusal and refund", PHASE_LIMIT, |log| {
            log.refunds() == 3 || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert_eq!(session.log.rejects.len(), 1);
    assert_eq!(session.log.rejects[0].job_id, b"bad-family");
    assert_eq!(session.log.rejects[0].reason, RejectReason::TooLarge as i32);
    assert_eq!(session.log.results.len(), 2);
    session
        .send(coord_msg::Msg::Job(canonical::job(
            &instance,
            &params,
            b"good-family",
            3,
        )))
        .await;
    session
        .until(
            "next family completes in same session",
            PHASE_LIMIT,
            |log| log.results.len() == 3 && log.refunds() == 4,
        )
        .await;
    assert_eq!(session.log.results[2].job_id, b"good-family");
    assert!(session.log.fatal.is_none());
    assert_eq!(session.log.rejects.len(), 1);
    assert_eq!(session.shutdown(2000).await, 0);
}

/// Plain and family jobs sent together: each family job starts only after
/// every plain job ahead of it answers, and no plain job starts while a
/// family job runs. The headless miner counts overlaps and exits nonzero on
/// any.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_jobs_never_run_beside_sa_jobs() {
    use quip_solver_core::quip_proto::v1::coord_msg;
    let mut session = headless_session(SERIAL).await;
    session
        .send(coord_msg::Msg::SetTarget(support::target(i64::MAX, 4)))
        .await;
    let (base, params) = canonical::fixture();
    let instance = canonical::instance(&base, &params, [7; 32]).unwrap();
    let family = |id: &[u8]| canonical::job(&instance, &params, id, 3);
    // Each round fits the initial credits.
    assert!(session.log.credits[0] >= 4);
    let rounds = [
        vec![
            plain_job(b"plain-1"),
            plain_job(b"plain-2"),
            family(b"family-1"),
            plain_job(b"plain-3"),
        ],
        vec![family(b"family-2"), plain_job(b"plain-4")],
    ];
    let mut sent = 0;
    for round in rounds {
        sent += round.len();
        for job in round {
            session.send(coord_msg::Msg::Job(job)).await;
        }
        session
            .until("every job of the round answers", PHASE_LIMIT, |log| {
                log.results.len() == sent || log.fatal.is_some()
            })
            .await;
    }
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert!(session.log.rejects.is_empty(), "{:?}", session.log.rejects);
    assert_eq!(session.shutdown(2000).await, 0);
}

/// A family job past the cluster memory budget is a `TooLarge` reject with
/// one refund, refused in preparation before any allocation.
#[tokio::test(flavor = "multi_thread")]
async fn a_family_job_past_the_cluster_budget_is_refused_before_allocation() {
    use quip_solver_core::quip_proto::v1::{coord_msg, RejectReason};
    let mut session = headless_session(TINY).await;
    session
        .send(coord_msg::Msg::SetTarget(support::target(i64::MAX, 4)))
        .await;
    let (base, params) = canonical::fixture();
    let instance = canonical::instance(&base, &params, [7; 32]).unwrap();
    session
        .send(coord_msg::Msg::Job(canonical::job(
            &instance,
            &params,
            b"too-large",
            3,
        )))
        .await;
    session
        .until("refusal and refund", PHASE_LIMIT, |log| {
            log.refunds() == 1 || log.fatal.is_some()
        })
        .await;
    assert!(session.log.fatal.is_none(), "{:?}", session.log.fatal);
    assert_eq!(session.log.rejects.len(), 1);
    assert_eq!(session.log.rejects[0].reason, RejectReason::TooLarge as i32);
    assert_eq!(session.shutdown(2000).await, 0);
}

mod canonical {
    use quip_solver_conformance::families::{export_instances, reference_revision};
    use quip_solver_core::ising_generators::{
        descriptor::params_digest, family::BaseTopology, ice::TWISTED_DIMER_V1, instance::Instance,
    };
    use quip_solver_core::quip_proto::v1::{
        ising_problem, CoefficientEncoding, EdgeList, FamilyBinding, FamilyDescriptor,
        IsingProblem, Job, JobKind,
    };
    use quip_solver_core::quip_protocol::wire::encode_i32_le;

    /// The conformance topology of twisted-ice-dimer revision 1, and its
    /// first parameter set that generates an instance at nonce `[7; 32]`.
    pub(super) fn fixture() -> (BaseTopology, Vec<u8>) {
        let reference = reference_revision(TWISTED_DIMER_V1.family_id, 1).unwrap();
        let stored = quip_solver_conformance::topology(reference.topology).unwrap();
        let base = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
        let params = reference
            .params
            .iter()
            .find(|p| {
                export_instances(reference.contract, p, &base, &[[7; 32]])
                    .is_ok_and(|instances| instances.into_iter().any(|i| i.is_ok()))
            })
            .unwrap()
            .to_vec();
        (base, params)
    }

    pub(super) fn job(instance: &Instance, params: &[u8], id: &[u8], reads: u32) -> Job {
        Job {
            job_id: id.to_vec(),
            kind: JobKind::IsingSample as i32,
            generation: 1,
            ising: Some(IsingProblem {
                graph: Some(ising_problem::Graph::Edges(EdgeList {
                    u: instance.edges().iter().map(|&(u, _)| u).collect(),
                    v: instance.edges().iter().map(|&(_, v)| v).collect(),
                })),
                encoding: CoefficientEncoding::I32 as i32,
                scale: 1000,
                h: encode_i32_le(instance.h()),
                j: encode_i32_le(instance.j()),
                num_reads: reads,
                num_sweeps: 4,
                family: Some(FamilyBinding {
                    family: Some(descriptor(params)),
                    nonce: vec![7; 32],
                    instance_digest: instance.digest().to_vec(),
                    metadata: instance.metadata().to_vec(),
                    topology_order_digest: instance.topology_order_digest().to_vec(),
                    node_ids: instance.node_ids().to_vec(),
                    offset_milli: instance.offset_milli(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    pub(super) fn descriptor(params: &[u8]) -> FamilyDescriptor {
        FamilyDescriptor {
            family_id: TWISTED_DIMER_V1.family_id.into(),
            revision: 1,
            params: params.to_vec(),
            params_digest: params_digest(params).to_vec(),
        }
    }

    pub(super) fn instance(
        base: &BaseTopology,
        params: &[u8],
        nonce: [u8; 32],
    ) -> Option<Instance> {
        export_instances(&TWISTED_DIMER_V1, params, base, &[nonce])
            .unwrap()
            .pop()
            .unwrap()
            .ok()
    }
}

/// The real binary and the production sampler on the Metal device.
mod gpu {
    use super::canonical::{descriptor, fixture, instance};
    use super::support::{lease, target, Session};
    use super::PHASE_LIMIT;
    use quip_solver_core::ising_generators::instance::Instance;
    use quip_solver_core::quip_proto::v1::{coord_msg, EdgeList, GeneratorAlgorithm, Topology};
    use quip_solver_core::quip_protocol::lease::LeaseSpec;
    use quip_solver_core::quip_protocol::scoring::energy_from_milli;
    use quip_solver_core::quip_protocol::wire::decode_spins_packed;
    use quip_solver_core::{
        CancelToken, IsingGraph, SampleParams, Sampler, StreamJob, StreamOutcome,
    };
    use quip_solver_metal::cluster::sampler::OtsukaSampler;
    use quip_solver_metal::metal_device::MetalDevice;
    use quip_solver_metal::{sample_ising, Kernel};

    const BIN: &str = env!("CARGO_BIN_EXE_quip-metal-otsuka");
    const HASH: [u8; 32] = [0x73; 32];

    fn assert_scores(result: &quip_solver_core::quip_proto::v1::Result, instance: &Instance) {
        let edges: Vec<_> = instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        assert!(!result.solutions.is_empty());
        for solution in &result.solutions {
            let spins = decode_spins_packed(&solution.spins, instance.node_ids().len()).unwrap();
            assert_eq!(
                solution.energy_milli,
                energy_from_milli(&spins, instance.h(), instance.j(), &edges)
            );
        }
    }

    /// A ring with alternating couplings, as in `tests/stream_contract.rs`.
    fn ring(n: usize) -> IsingGraph {
        let edges: Vec<(usize, usize)> = (0..n).map(|u| (u, (u + 1) % n)).collect();
        let j = (0..n)
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        IsingGraph::new(vec![0.0; n], j, edges)
    }

    /// A plain job on `quip-metal-otsuka` runs the sampler of
    /// `quip-metal-sa`: the SA stream that the session path forwards plain
    /// jobs to returns the reads of `sample_ising` on SA with the same seed,
    /// and the direct call does too. The session seeds each job from OS
    /// entropy, so this compares the engines in process.
    #[test]
    fn a_plain_job_returns_the_sa_reads_for_the_same_seed() {
        let sampler = OtsukaSampler::open(0, 100, false, false).unwrap();
        let graph = ring(64);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 64,
            seed: 41,
            ..Default::default()
        };
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        tx.blocking_send(StreamJob {
            job_id: b"plain".to_vec(),
            graph: graph.clone(),
            params: params.clone(),
            watermark: None,
        })
        .unwrap();
        drop(tx);
        sampler.sample_stream(rx, out, CancelToken::default());
        let StreamOutcome::Completed(Ok(streamed)) = results.blocking_recv().unwrap().outcome
        else {
            panic!("the plain job did not complete")
        };
        let device = MetalDevice::open(0).unwrap();
        let direct = sample_ising(&device, &graph, &params, Kernel::Sa).unwrap();
        assert_eq!(streamed, direct);
        assert_eq!(sampler.sample(&graph, &params).unwrap(), direct);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explicit_twisted_dimer_job_scores_on_original_model() {
        let (base, params) = fixture();
        let instance = instance(&base, &params, [7; 32]).unwrap();
        let mut session = Session::start(BIN, 1, "").await;
        session
            .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
            .await;
        session
            .send(coord_msg::Msg::Job(super::canonical::job(
                &instance,
                &params,
                b"explicit-cluster",
                64,
            )))
            .await;
        session
            .until("explicit result and credit", PHASE_LIMIT, |log| {
                !log.results.is_empty() && log.refunds() == 1
            })
            .await;
        assert!(session.log.rejects.is_empty());
        assert_eq!(session.log.results[0].solutions.len(), 64);
        assert_scores(&session.log.results[0], &instance);
        assert_eq!(session.shutdown(2000).await, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn canonical_host_family_lease_reaches_cluster_and_refunds_credit() {
        let (base, params) = fixture();
        let mut request = lease(b"host-cluster", 1, HASH, 0, 2);
        let generator = request.generator.as_mut().unwrap();
        generator.algorithm = GeneratorAlgorithm::FamilyChacha8V1 as i32;
        generator.family = Some(descriptor(&params));
        generator.topology_order_digest = base.order_digest().to_vec();
        let spec = LeaseSpec::from_proto(generator).unwrap();
        let mut session = Session::start(BIN, 1, "").await;
        assert!(!session
            .log
            .hello
            .as_ref()
            .unwrap()
            .capabilities
            .as_ref()
            .unwrap()
            .features
            .iter()
            .any(|f| f == "local-generation"));
        session
            .send(coord_msg::Msg::Topology(Topology {
                hash: HASH.to_vec(),
                nodes: base.node_ids().to_vec(),
                edges: Some(EdgeList {
                    u: base.edges().iter().map(|&(u, _)| u).collect(),
                    v: base.edges().iter().map(|&(_, v)| v).collect(),
                }),
                allowed_h_milli: vec![0],
                allowed_j_milli: vec![-1000, 1000],
            }))
            .await;
        session
            .send(coord_msg::Msg::SetTarget(target(i64::MAX, 4)))
            .await;
        session.send(coord_msg::Msg::Job(request)).await;
        session
            .until("family lease and credit", PHASE_LIMIT, |log| {
                log.done_count(b"host-cluster") == 1 && log.refunds() == 1
            })
            .await;
        assert!(session.log.rejects.is_empty());
        let mut expected_results = 0;
        for index in 0..2 {
            let nonce = spec.nonce(index).unwrap();
            if let Some(instance) = instance(&base, &params, nonce) {
                expected_results += 1;
                let result = session
                    .log
                    .results
                    .iter()
                    .find(|r| r.nonce == nonce)
                    .unwrap();
                assert_scores(result, &instance);
            }
        }
        assert!(expected_results > 0);
        assert_eq!(session.log.results.len(), expected_results);
        assert_eq!(session.shutdown(2000).await, 0);
    }
}
