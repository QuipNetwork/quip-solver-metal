// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! A protocol-2 test coordinator for salt leases. One miner process connects
//! over a Unix socket. The coordinator folds every miner frame into a `Log`.

use quip_solver_core::quip_proto::v1::{
    coord_msg, miner_msg,
    miner_service_server::{MinerService, MinerServiceServer},
    Configure, CoordMsg, EdgeList, Fatal, GeneratorAlgorithm, Hello, IsingProblemGenerator, Job,
    JobKind, LeaseDone, MinerMsg, Ping, Reject, SetTarget, Shutdown, Topology, Welcome,
};
use quip_solver_core::quip_protocol::lease::TopologyView;
use quip_solver_core::quip_protocol::target::Target;
use std::sync::Mutex;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status, Streaming};

type Outbound = mpsc::Sender<Result<CoordMsg, Status>>;
type Connection = (Streaming<MinerMsg>, Outbound);

struct Coordinator(Mutex<Option<oneshot::Sender<Connection>>>);

#[tonic::async_trait]
impl MinerService for Coordinator {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let (send, receive) = mpsc::channel(256);
        let connection = self
            .0
            .lock()
            .expect("coordinator lock")
            .take()
            .ok_or_else(|| Status::already_exists("one miner connection expected"))?;
        connection
            .send((request.into_inner(), send))
            .map_err(|_| Status::cancelled("test stopped"))?;
        Ok(Response::new(ReceiverStream::new(receive)))
    }
}

/// Every frame the miner sent, folded by kind.
#[derive(Debug, Default)]
pub(crate) struct Log {
    pub(crate) hello: Option<Hello>,
    pub(crate) ready: bool,
    /// Credits per `JobRequest`. The first entry is the initial grant.
    pub(crate) credits: Vec<u32>,
    pub(crate) results: Vec<quip_solver_core::quip_proto::v1::Result>,
    pub(crate) lease_done: Vec<LeaseDone>,
    pub(crate) rejects: Vec<Reject>,
    /// `jobs_done` from the latest `Status`.
    pub(crate) jobs_done: Option<u64>,
    pub(crate) fatal: Option<Fatal>,
    pub(crate) closed: bool,
    /// Arrival order: `result:<job>`, `done:<job>`, `credit`.
    pub(crate) order: Vec<String>,
}

impl Log {
    /// Credits returned after the initial grant.
    pub(crate) fn refunds(&self) -> u32 {
        self.credits.iter().skip(1).sum()
    }

    pub(crate) fn done(&self, job: &[u8]) -> Option<&LeaseDone> {
        self.lease_done.iter().find(|d| d.job_id == job)
    }

    pub(crate) fn done_count(&self, job: &[u8]) -> usize {
        self.lease_done.iter().filter(|d| d.job_id == job).count()
    }

    pub(crate) fn results_for(&self, job: &[u8]) -> Vec<&quip_solver_core::quip_proto::v1::Result> {
        self.results.iter().filter(|r| r.job_id == job).collect()
    }
}

pub(crate) struct Session {
    pub(crate) log: Log,
    inbound: Streaming<MinerMsg>,
    outbound: Outbound,
    child: Child,
    _server: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}

impl Session {
    /// Spawn `binary`, finish the handshake, and wait for the first credits.
    pub(crate) async fn start(binary: &str, queue_depth: u32, backend_toml: &str) -> Self {
        let directory = tempfile::tempdir().expect("socket directory");
        let path = directory.path().join("coord.sock");
        let listener = UnixListener::bind(&path).expect("bind test coordinator");
        let (send, receive) = oneshot::channel();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MinerServiceServer::new(Coordinator(Mutex::new(Some(send)))))
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await
                .expect("serve test coordinator");
        });
        let child = Command::new(binary)
            .args(["--quip-coordinator", &format!("unix://{}", path.display())])
            .args(["--miner-id", "lease-test"])
            .env("QUIP_SESSION_TOKEN", "test-token")
            .kill_on_drop(true)
            .spawn()
            .expect("start miner");
        let (inbound, outbound) = tokio::time::timeout(Duration::from_secs(30), receive)
            .await
            .expect("miner connects within 30 s")
            .expect("coordinator connection");
        let mut session = Self {
            log: Log::default(),
            inbound,
            outbound,
            child,
            _server: server,
            _directory: directory,
        };
        session
            .until("hello", Duration::from_secs(30), |log| log.hello.is_some())
            .await;
        session
            .send(coord_msg::Msg::Welcome(Welcome {
                protocol_version: 2,
            }))
            .await;
        session
            .send(coord_msg::Msg::Configure(Configure {
                queue_depth,
                idle_timeout_s: 3_600,
                heartbeat_s: 15,
                reconnect_window_s: 60,
                backend_toml: backend_toml.to_owned(),
            }))
            .await;
        session
            .until("ready", Duration::from_secs(60), |log| {
                log.ready && !log.credits.is_empty()
            })
            .await;
        session
    }

    pub(crate) async fn send(&mut self, msg: coord_msg::Msg) {
        self.outbound
            .send(Ok(CoordMsg { msg: Some(msg) }))
            .await
            .expect("coordinator send");
    }

    /// Fold frames until `done` holds. Panics on a timeout or an early close.
    pub(crate) async fn until(
        &mut self,
        phase: &str,
        limit: Duration,
        done: impl Fn(&Log) -> bool,
    ) {
        let deadline = Instant::now() + limit;
        while !done(&self.log) {
            assert!(
                !self.log.closed,
                "{phase}: stream closed early, fatal {:?}",
                self.log.fatal
            );
            match tokio::time::timeout_at(deadline, self.inbound.message()).await {
                Ok(Ok(Some(frame))) => self.observe(frame),
                Ok(Ok(None)) => self.log.closed = true,
                Ok(Err(error)) => panic!("{phase}: transport error: {error}"),
                Err(_) => panic!("{phase}: timed out after {limit:?}"),
            }
        }
    }

    /// Ping and return `Status.jobs_done`, which counts finished salts.
    pub(crate) async fn jobs_done(&mut self) -> u64 {
        self.log.jobs_done = None;
        self.send(coord_msg::Msg::Ping(Ping {})).await;
        self.until("status", Duration::from_secs(10), |log| {
            log.jobs_done.is_some()
        })
        .await;
        self.log.jobs_done.unwrap_or(0)
    }

    /// Ping until at least `salts` salts have finished.
    pub(crate) async fn wait_for_salts(&mut self, salts: u64, limit: Duration) {
        let deadline = Instant::now() + limit;
        loop {
            let done = self.jobs_done().await;
            if done >= salts {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "only {done} salts after {limit:?}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Send `Shutdown`, read to the end of the stream, and return the exit code.
    pub(crate) async fn shutdown(&mut self, grace_ms: u32) -> i32 {
        let started = Instant::now();
        self.send(coord_msg::Msg::Shutdown(Shutdown { grace_ms }))
            .await;
        let limit = Duration::from_millis(u64::from(grace_ms)) + Duration::from_secs(10);
        self.until("shutdown", limit, |log| log.closed).await;
        let status = tokio::time::timeout(Duration::from_secs(10), self.child.wait())
            .await
            .expect("miner exits after shutdown")
            .expect("wait for miner");
        // Allow two seconds for transport and scheduler delays beyond grace.
        let tolerance = Duration::from_millis(2_000);
        let elapsed = started.elapsed();
        assert!(
            elapsed <= Duration::from_millis(u64::from(grace_ms)) + tolerance,
            "shutdown took {elapsed:?}, exceeding grace {grace_ms} ms plus {tolerance:?}"
        );
        status.code().unwrap_or(-1)
    }

    fn observe(&mut self, frame: MinerMsg) {
        let Some(message) = frame.msg else {
            panic!("empty miner message");
        };
        match message {
            miner_msg::Msg::Hello(hello) => self.log.hello = Some(hello),
            miner_msg::Msg::Ready(_) => self.log.ready = true,
            miner_msg::Msg::JobRequest(request) => {
                self.log.credits.push(request.credits);
                self.log.order.push("credit".to_owned());
            }
            miner_msg::Msg::Result(result) => {
                self.log.order.push(format!(
                    "result:{}",
                    String::from_utf8_lossy(&result.job_id)
                ));
                self.log.results.push(result);
            }
            miner_msg::Msg::LeaseDone(done) => {
                self.log
                    .order
                    .push(format!("done:{}", String::from_utf8_lossy(&done.job_id)));
                self.log.lease_done.push(done);
            }
            miner_msg::Msg::Reject(reject) => self.log.rejects.push(reject),
            miner_msg::Msg::Status(status) => self.log.jobs_done = Some(status.jobs_done),
            miner_msg::Msg::Fatal(fatal) => self.log.fatal = Some(fatal),
            miner_msg::Msg::Capabilities(_) => {}
        }
    }
}

/// A ring of `n` nodes, fields from {-1, 0, 1} and couplings from {-1, 1}.
pub(crate) fn ring(n: u32, hash: [u8; 32]) -> (Topology, TopologyView) {
    let topology = Topology {
        hash: hash.to_vec(),
        nodes: (0..n).collect(),
        edges: Some(EdgeList {
            u: (0..n).collect(),
            v: (0..n).map(|i| (i + 1) % n).collect(),
        }),
        allowed_h_milli: vec![-1000, 0, 1000],
        allowed_j_milli: vec![-1000, 1000],
    };
    let view = TopologyView::from_proto(&topology).expect("valid ring topology");
    (topology, view)
}

pub(crate) const AGLAIS_HASH: [u8; 32] = [0xcb; 32];
/// The Aglais chain's `current_difficulty` after qblock 3278 (2026-09-18).
pub(crate) const AGLAIS_TARGET_MILLI: i64 = -14_625_068;

/// Aglais topology `cbec1eb4e9dc…`: the committed Advantage2 System 1
/// fixture without edge (880, 2695). Fields {0}, couplings {-1, 1}.
pub(crate) fn aglais() -> (Topology, TopologyView) {
    let (mut u, mut v) = (Vec::new(), Vec::new());
    for line in include_str!("../fixtures/advantage2-system1.edges").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut nodes = line.split_whitespace();
        let a: u32 = nodes.next().expect("edge start").parse().expect("node id");
        let b: u32 = nodes.next().expect("edge end").parse().expect("node id");
        if (a, b) != (880, 2695) {
            u.push(a);
            v.push(b);
        }
    }
    assert_eq!(u.len(), 41_514, "Aglais edge count");
    let topology = Topology {
        hash: AGLAIS_HASH.to_vec(),
        nodes: (0..4577).collect(),
        edges: Some(EdgeList { u, v }),
        allowed_h_milli: vec![0],
        allowed_j_milli: vec![-1000, 1000],
    };
    let view = TopologyView::from_proto(&topology).expect("valid Aglais topology");
    (topology, view)
}

/// An `ISING_GENERATE` job over `salt_start..salt_start + salt_count`.
pub(crate) fn lease(
    job_id: &[u8],
    generation: u64,
    hash: [u8; 32],
    salt_start: u64,
    salt_count: u64,
) -> Job {
    Job {
        job_id: job_id.to_vec(),
        kind: JobKind::IsingGenerate as i32,
        generation,
        deadline_ms: 0,
        ising: None,
        provenance: None,
        generator: Some(IsingProblemGenerator {
            algorithm: GeneratorAlgorithm::Blake3Chacha8V1 as i32,
            topology_hash: hash.to_vec(),
            last_proof_block_hash: vec![0x11; 32],
            miner_account: vec![0x22; 32],
            base_salt: vec![0x44; 32],
            salt_start,
            salt_count,
            family: None,
            topology_order_digest: Vec::new(),
        }),
    }
}

/// 64 reads, one proof of up to 32 solutions below `max_energy_milli`.
pub(crate) fn target(max_energy_milli: i64, num_sweeps: u32) -> SetTarget {
    SetTarget {
        max_energy_milli,
        min_solutions: 1,
        min_diversity_milli: 0,
        num_reads: 64,
        num_sweeps,
        anneal_time_us: 0,
        max_proof_solutions: 32,
    }
}

/// A target that only checks a result's energies, nonce, and salt against
/// its redrawn problem — never its solution count or its energy against the
/// live session target. `LeaseSink::push` forwards every non-empty read set
/// unfiltered (no target check, no truncation to a proof-set size), so a
/// lease `Result` carries the salt's full read set: it can exceed the size
/// or miss the energy bound a session `Target` derived from `SetTarget`
/// would enforce for an on-chain proof. The coordinator, not this test,
/// decides which verified rows beat the live target.
pub(crate) fn permissive_target() -> Target {
    Target {
        max_energy_milli: i64::MAX,
        min_solutions: 1,
        min_diversity_milli: 0,
        max_proof_solutions: u32::MAX,
    }
}

/// Path of a binary in this crate's target profile, built if needed.
pub(crate) fn miner_binary(name: &str) -> String {
    let status = std::process::Command::new(env!("CARGO"))
        .args([
            "build",
            "--release",
            "-p",
            "quip-solver-metal",
            "--bin",
            name,
        ])
        .status()
        .expect("cargo build");
    assert!(status.success(), "failed to build {name}");
    let mut path = std::env::current_exe().expect("test exe path");
    path.pop();
    path.pop();
    path.push(name);
    assert!(path.exists(), "missing binary {}", path.display());
    path.to_string_lossy().into_owned()
}
