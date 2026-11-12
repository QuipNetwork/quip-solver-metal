// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Salt leases through the real session and the MSA miner.
//!
//! Metal GPU tests: need a real device (Apple Silicon). Run with
//! `--test-threads=1`.

mod support;

use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Cancel, CoefficientEncoding, EdgeList, IsingProblem,
    IsingProblemGenerator, Job, JobKind, RejectReason,
};
use quip_solver_core::quip_protocol::lease::verify_lease_result;
use quip_solver_core::quip_protocol::wire::encode_i32_le;
use std::collections::HashSet;
use std::time::Duration;
use support::{
    aglais, lease, miner_binary, permissive_target, ring, target, Session, AGLAIS_HASH,
    AGLAIS_TARGET_MILLI,
};

/// How many of a lease's best-finishing screens the miner streams back with
/// reads instead of merely screening. Mirrors `resident::LEASE_TOP_N`
/// (private to the miner crate, so the integration tests hardcode it).
const LEASE_TOP_N: usize = 10;

const RING_A: [u8; 32] = [0x51; 32];
const RING_B: [u8; 32] = [0x52; 32];
const LONG: Duration = Duration::from_secs(120);

fn msa() -> String {
    miner_binary("quip-metal-msa")
}

fn generator(job: &Job) -> IsingProblemGenerator {
    job.generator.clone().expect("lease job")
}

#[tokio::test(flavor = "multi_thread")]
async fn every_salt_is_counted_once() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 64)))
        .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-measure",
        3,
        RING_A,
        0,
        200,
    )))
    .await;
    s.until("measured lease done", LONG, |log| {
        log.done(b"lease-measure").is_some()
    })
    .await;
    let mut bests: Vec<i64> = s
        .log
        .results_for(b"lease-measure")
        .iter()
        .map(|r| {
            r.solutions
                .iter()
                .map(|x| x.energy_milli)
                .min()
                .expect("solutions")
        })
        .collect();
    assert_eq!(bests.len(), 200, "an i64::MAX target makes every salt win");
    bests.sort_unstable();
    let median = bests[bests.len() / 2];

    let t = target(median, 64);
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let job = lease(b"lease-median", 5, RING_A, 0, 200);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("median lease done", LONG, |log| {
        log.done(b"lease-median").is_some()
    })
    .await;

    let done = s.log.done(b"lease-median").expect("LeaseDone");
    assert_eq!(done.salts_done, 200, "every salt is counted once");
    assert_eq!(s.log.done_count(b"lease-median"), 1, "one summary");
    let results = s.log.results_for(b"lease-median");
    assert!(
        results.len() < 200,
        "the median target screens at least one salt"
    );
    let mut salts = HashSet::new();
    for result in &results {
        // A permissive target: `push` forwards every non-empty read set
        // unfiltered, so a reported salt's reads need not beat `t`, only
        // decode to the energies it claims for its redrawn problem.
        let verified = verify_lease_result(&spec, &view, &permissive_target(), result)
            .expect("every reported salt verifies against the lease");
        assert!(salts.insert(verified.salt), "one result per salt");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_i64_max_target_reports_every_salt_as_a_winner() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    let t = target(i64::MAX, 32);
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let job = lease(b"lease-all", 3, RING_A, 0, 64);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("lease done", LONG, |log| log.done(b"lease-all").is_some())
        .await;

    let done = s.log.done(b"lease-all").expect("LeaseDone");
    assert_eq!(done.salts_done, 64);
    let results = s.log.results_for(b"lease-all");
    assert_eq!(results.len(), 64, "an i64::MAX target makes every salt win");
    let mut salts = HashSet::new();
    for result in &results {
        // A permissive target: `push` forwards every non-empty read set
        // unfiltered, so a winner's full read set need not fit a proof-set
        // size; this only checks it decodes to the energies it claims.
        let verified = verify_lease_result(&spec, &view, &permissive_target(), result)
            .expect("every winner verifies against the lease");
        assert!(salts.insert(verified.salt), "one result per salt");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn screened_salts_never_send_one_result_per_salt() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    // 64 nodes and 64 edges: no spin assignment on unit couplings can go
    // below -(64 + 64) * 1000, so this target screens every salt. 256
    // sweeps (above the cascade's first calibrated stage, 32) gives every
    // salt a real gate checkpoint before its final one, so most salts are
    // gate-screened rather than running straight to the final checkpoint.
    let bound = -(64i64 + 64) * 1000 - 1;
    s.send(coord_msg::Msg::SetTarget(target(bound, 256))).await;
    let salt_count = 200;
    let job = lease(b"lease-none", 3, RING_A, 0, salt_count);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("lease done", LONG, |log| log.done(b"lease-none").is_some())
        .await;

    let done = s.log.done(b"lease-none").expect("LeaseDone");
    assert_eq!(done.salts_done, salt_count, "every salt is counted once");
    assert!(done.best_energy_milli < i64::MAX, "reads finished");
    assert!(
        done.best_energy_milli > bound,
        "the unreachable target is never met"
    );
    // No salt beats the unreachable target, so nothing is a real winner and
    // there are no final-checkpoint survivors either: every reported salt
    // got here through the lease's running top-N, streamed back with reads
    // as it displaces a worse entry, instead of merely screened. The set
    // itself never holds more than LEASE_TOP_N entries, but streaming as
    // units finish (rather than holding the set to lease end) reports each
    // displacement, so the lease's total Result count is not bounded by
    // LEASE_TOP_N. A lease's total Results are its final-checkpoint/
    // target-hit survivors (always pushed; none here) plus the salts that
    // ever entered the running top-N as a screen — expected about
    // k + k*ln(n/k) for k = LEASE_TOP_N and n salts, far below one Result
    // per salt.
    let results = s.log.results_for(b"lease-none");
    assert!(
        results.len() >= LEASE_TOP_N,
        "the first {LEASE_TOP_N} screens always fill the still-open top-{LEASE_TOP_N}, got {}",
        results.len()
    );
    let ceiling = salt_count as usize / 2;
    assert!(
        results.len() < ceiling,
        "expected roughly {}+{}*ln({salt_count}/{}) ~= {:.0} results, got {} (>= the {ceiling} ceiling)",
        LEASE_TOP_N,
        LEASE_TOP_N,
        LEASE_TOP_N,
        LEASE_TOP_N as f64 + LEASE_TOP_N as f64 * (salt_count as f64 / LEASE_TOP_N as f64).ln(),
        results.len()
    );
    let mut salts = HashSet::new();
    for result in &results {
        let verified = verify_lease_result(&spec, &view, &permissive_target(), result)
            .expect("every reported salt verifies against the lease");
        assert!(salts.insert(verified.salt), "one result per salt");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lease_without_plain_jobs_starts_the_device() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"lease-first", 3, RING_A, 0, 50)))
        .await;
    s.until("lease done", Duration::from_secs(60), |log| {
        log.done(b"lease-first").is_some()
    })
    .await;
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_ends_a_live_lease_with_one_summary() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        16_384,
    )))
    .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-cancel",
        7,
        support::AGLAIS_HASH,
        0,
        1_000_000,
    )))
    .await;
    s.wait_for_salts(1, LONG).await;
    s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 7 }))
        .await;
    s.until("cancel summary", Duration::from_secs(10), |log| {
        log.done(b"lease-cancel").is_some() && log.refunds() >= 1
    })
    .await;

    let done = s.log.done(b"lease-cancel").expect("LeaseDone");
    assert!(
        done.salts_done < 1_000_000,
        "the cancel cut the lease short"
    );
    assert_eq!(s.log.done_count(b"lease-cancel"), 1, "one summary");
    assert_eq!(s.log.refunds(), 1, "the cancelled lease refunds its credit");
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn shutdown_during_a_lease_sends_its_summary_and_exits_cleanly() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        14_336,
    )))
    .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-shutdown",
        9,
        support::AGLAIS_HASH,
        0,
        1_000_000,
    )))
    .await;
    s.wait_for_salts(100, LONG).await;

    assert_eq!(s.shutdown(3_000).await, 0);
    assert_eq!(s.log.done_count(b"lease-shutdown"), 1);
    assert_eq!(s.log.refunds(), 1, "the lease refunds its credit");
    let done = s
        .log
        .done(b"lease-shutdown")
        .expect("LeaseDone before close");
    assert!(done.salts_done >= 100 && done.salts_done < 1_000_000);
    let summary = s
        .log
        .order
        .iter()
        .position(|e| e == "done:lease-shutdown")
        .expect("summary in order log");
    assert!(
        !s.log.order[summary..]
            .iter()
            .any(|e| e == "result:lease-shutdown"),
        "no result after the shut-down lease's summary"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plain_job_during_a_lease_returns_all_its_reads() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        14_336,
    )))
    .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-long",
        3,
        support::AGLAIS_HASH,
        0,
        1_000_000,
    )))
    .await;
    s.wait_for_salts(1, LONG).await;
    s.send(coord_msg::Msg::Job(Job {
        job_id: b"plain-1".to_vec(),
        kind: JobKind::IsingSample as i32,
        generation: 3,
        deadline_ms: 0,
        ising: Some(IsingProblem {
            encoding: CoefficientEncoding::I32 as i32,
            scale: 1000,
            h: encode_i32_le(&[1000, -1000, 0]),
            j: encode_i32_le(&[1000, -1000]),
            num_reads: 16,
            num_sweeps: 64,
            graph: Some(ising_problem::Graph::Edges(EdgeList {
                u: vec![0, 1],
                v: vec![1, 2],
            })),
            ..Default::default()
        }),
        provenance: None,
        generator: None,
    }))
    .await;
    s.until("plain job result", LONG, |log| {
        !log.results_for(b"plain-1").is_empty()
    })
    .await;

    let results = s.log.results_for(b"plain-1");
    assert_eq!(results.len(), 1, "a plain job sends one result");
    let result = results[0];
    assert_eq!(result.solutions.len(), 16, "all reads come back");
    assert_eq!(result.meta.as_ref().expect("meta").sweeps, 64);
    assert_eq!(s.shutdown(2_000).await, 0);
}

/// Degree 21 is past the unit kernel's 20 neighbours but within the weighted
/// lane's 62, so every salt runs on the batch path and the lease completes.
#[tokio::test(flavor = "multi_thread")]
async fn a_degree_21_lease_runs() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (mut topology, _) = ring(22, RING_A);
    topology.edges = Some(EdgeList {
        u: vec![0; 21],
        v: (1..22).collect(),
    });
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"degree-21", 3, RING_A, 0, 8)))
        .await;
    s.until("degree-21 lease done", LONG, |log| {
        log.done(b"degree-21").is_some()
    })
    .await;
    assert_eq!(s.log.done(b"degree-21").expect("LeaseDone").salts_done, 8);
    assert_eq!(
        s.log.results_for(b"degree-21").len(),
        8,
        "an i64::MAX target makes every salt win"
    );
    assert!(s.log.rejects.is_empty(), "{:?}", s.log.rejects);
    assert_eq!(s.shutdown(2_000).await, 0);
    assert!(s.log.fatal.is_none());
}

/// Close a lease whose salts the capacity model refuses, then serve a ring
/// lease on the same session: the refusal is a zero-salt summary with one
/// refund, never a fatal error.
async fn refused_lease_then_ring_lease(s: &mut Session, refused: &[u8]) {
    s.until("refused lease done", LONG, |log| {
        log.done(refused).is_some() && log.refunds() >= 1
    })
    .await;
    assert_eq!(s.log.done(refused).expect("LeaseDone").salts_done, 0);
    assert!(s.log.results_for(refused).is_empty());
    assert_eq!(s.log.refunds(), 1, "one refund for the refused lease");
    assert!(s.log.fatal.is_none(), "{:?}", s.log.fatal);

    let (topology, _) = ring(64, RING_B);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::Job(lease(b"next", 3, RING_B, 0, 8)))
        .await;
    s.until("next lease done", LONG, |log| {
        log.done(b"next").is_some() && log.refunds() >= 2
    })
    .await;
    assert_eq!(s.log.done(b"next").expect("LeaseDone").salts_done, 8);
    assert_eq!(s.log.results_for(b"next").len(), 8);
    assert_eq!(s.log.refunds(), 2, "the refused lease and the next lease");
    assert_eq!(s.shutdown(2_000).await, 0);
    assert_eq!(s.log.done_count(refused), 1);
    assert!(s.log.rejects.is_empty(), "{:?}", s.log.rejects);
    assert!(s.log.fatal.is_none(), "{:?}", s.log.fatal);
}

/// A port of the CUDA test of the same name. A hub of degree 63 is past the
/// weighted lanes' 62 neighbours, and a repeated coupler keeps the model off
/// dense-matrix storage, so the capacity model refuses every salt. The lease
/// closes with a refund and the session serves the next lease.
#[tokio::test(flavor = "multi_thread")]
async fn degree_sixty_three_lease_returns_capacity_without_faulting() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (mut topology, _) = ring(64, RING_A);
    let mut v: Vec<u32> = (1..63).collect();
    v.push(1);
    topology.edges = Some(EdgeList { u: vec![0; 63], v });
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"degree-63", 3, RING_A, 0, 8)))
        .await;
    refused_lease_then_ring_lease(&mut s, b"degree-63").await;
}

/// A model of 5,000 nodes passes the node limit, and its degree-63 hub runs
/// only on dense-matrix storage, where one sweep of 5,000 * 5,000 columns is
/// estimated past the one-sweep limit. The lease closes with a refund and
/// the session serves the next lease.
#[tokio::test(flavor = "multi_thread")]
async fn a_lease_past_the_one_sweep_limit_returns_capacity_without_faulting() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (mut topology, _) = ring(5_000, RING_A);
    topology.edges = Some(EdgeList {
        u: vec![0; 63],
        v: (1..64).collect(),
    });
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"one-sweep", 3, RING_A, 0, 8)))
        .await;
    refused_lease_then_ring_lease(&mut s, b"one-sweep").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_leases_are_rejected_with_refunds() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::Job(lease(b"no-target", 3, RING_A, 0, 10)))
        .await;
    s.until("target missing", Duration::from_secs(10), |log| {
        !log.rejects.is_empty()
    })
    .await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"wrong-hash", 3, RING_B, 0, 10)))
        .await;
    s.until("rejects", Duration::from_secs(10), |log| {
        log.rejects.len() >= 2 && log.refunds() >= 2
    })
    .await;

    let reason = |id: &[u8]| {
        s.log
            .rejects
            .iter()
            .find(|r| r.job_id == id)
            .map(|r| r.reason)
    };
    assert_eq!(
        reason(b"no-target"),
        Some(RejectReason::TargetMissing as i32)
    );
    assert_eq!(
        reason(b"wrong-hash"),
        Some(RejectReason::TopologyMismatch as i32)
    );
    assert_eq!(s.shutdown(2_000).await, 0);
    assert_eq!(s.log.rejects.len(), 2);
    assert_eq!(s.log.refunds(), 2);
    assert!(
        s.log.lease_done.is_empty(),
        "a rejected lease sends no summary"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_topology_change_between_leases_uses_the_new_graph() {
    let mut s = Session::start(&msa(), 4, "").await;
    let t = target(i64::MAX, 32);
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let (first, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(first)).await;
    s.send(coord_msg::Msg::Job(lease(b"lease-a", 3, RING_A, 0, 40)))
        .await;
    s.until("first lease", LONG, |log| log.done(b"lease-a").is_some())
        .await;

    let (second, view) = ring(96, RING_B);
    s.send(coord_msg::Msg::Topology(second)).await;
    let job = lease(b"lease-b", 3, RING_B, 0, 40);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("second lease", LONG, |log| log.done(b"lease-b").is_some())
        .await;

    assert_eq!(s.log.done(b"lease-b").expect("LeaseDone").salts_done, 40);
    let results = s.log.results_for(b"lease-b");
    assert_eq!(results.len(), 40);
    for result in results {
        // A permissive target: `push` forwards every non-empty read set
        // unfiltered, so this only checks the 96-node ring was actually
        // used to redraw and score the salt.
        verify_lease_result(&spec, &view, &permissive_target(), result)
            .expect("second-lease winners use the 96-node ring");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

/// Salts per second over `measure` salts, after `warmup` salts have finished.
async fn steady_rate(s: &mut Session, warmup: u64, measure: u64) -> f64 {
    s.wait_for_salts(warmup, LONG).await;
    let start = s.jobs_done().await;
    let started = std::time::Instant::now();
    s.wait_for_salts(start + measure, LONG).await;
    let end = s.jobs_done().await;
    (end - start) as f64 / started.elapsed().as_secs_f64()
}

/// Lease throughput on the Aglais topology under the chain gates.
///
/// Prints the steady rate of one lease, of four leases sent together, and the
/// rate across five back-to-back leases (first salt to last salt finished,
/// so a deep survivor's tail does not count) against the one-lease rate. Needs a
/// quiet GPU: run alone, `--release --ignored --nocapture --test-threads=1`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU throughput measurement; needs a quiet machine"]
#[expect(clippy::print_stderr, reason = "the measurement is the output")]
async fn lease_throughput() {
    let (topology, _) = aglais();
    let chain = || target(AGLAIS_TARGET_MILLI, 14_336);
    // Backend TOML for every session, for example `cascade_prefilter = false`.
    let backend = std::env::var("QUIP_LEASE_BACKEND").unwrap_or_default();
    let only_one = std::env::var("QUIP_LEASE_ONE").is_ok();
    let measure: u64 = std::env::var("QUIP_LEASE_MEASURE").map_or(200_000, |v| v.parse().unwrap());

    let mut s = Session::start(&msa(), 8, &backend).await;
    s.send(coord_msg::Msg::Topology(topology.clone())).await;
    s.send(coord_msg::Msg::SetTarget(chain())).await;
    // Room for the warmup, the salts already done when the measurement starts,
    // and the measurement itself.
    let one_size = 1_000_000.max(2 * measure);
    s.send(coord_msg::Msg::Job(lease(
        b"one",
        3,
        AGLAIS_HASH,
        0,
        one_size,
    )))
    .await;
    let one = steady_rate(&mut s, 30_000, measure).await;
    eprintln!("one lease: {one:.0} salts/s");
    drop(s);
    if only_one {
        return;
    }

    let mut s = Session::start(&msa(), 8, &backend).await;
    s.send(coord_msg::Msg::Topology(topology.clone())).await;
    s.send(coord_msg::Msg::SetTarget(chain())).await;
    for (i, id) in [b"four-0", b"four-1", b"four-2", b"four-3"]
        .iter()
        .enumerate()
    {
        let start = i as u64 * 1_000_000;
        s.send(coord_msg::Msg::Job(lease(
            *id,
            3,
            AGLAIS_HASH,
            start,
            1_000_000,
        )))
        .await;
    }
    let four = steady_rate(&mut s, 30_000, 200_000).await;
    eprintln!("four leases: {four:.0} salts/s");
    drop(s);

    const PER: u64 = 60_000;
    const WARMUP: u64 = 30_000;
    // Distinct sizes tell the leases apart in the miner's debug log.
    let sizes: Vec<u64> = (0..5).map(|i| PER + i).collect();
    let ids: Vec<Vec<u8>> = (0..5).map(|i| format!("seq-{i}").into_bytes()).collect();
    let mut s = Session::start(&msa(), 8, &backend).await;
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(chain())).await;
    s.send(coord_msg::Msg::Job(lease(
        b"warmup",
        3,
        AGLAIS_HASH,
        0,
        WARMUP,
    )))
    .await;
    let mut start = WARMUP;
    for (id, &size) in ids.iter().zip(&sizes) {
        s.send(coord_msg::Msg::Job(lease(id, 3, AGLAIS_HASH, start, size)))
            .await;
        start += size;
    }
    let total: u64 = sizes.iter().sum();
    s.wait_for_salts(WARMUP, LONG).await;
    let first = s.jobs_done().await;
    let started = std::time::Instant::now();
    s.wait_for_salts(WARMUP + total, LONG).await;
    let rate = (WARMUP + total - first) as f64 / started.elapsed().as_secs_f64();
    s.until("five leases done", Duration::from_secs(600), |log| {
        ids.iter().all(|id| log.done(id).is_some())
    })
    .await;
    let order: Vec<&String> = s
        .log
        .order
        .iter()
        .filter(|e| e.starts_with("done:"))
        .collect();
    eprintln!(
        "five leases of {PER}: {rate:.0} salts/s, {:.1}% of one-lease rate; done order {order:?}",
        100.0 * rate / one
    );
    drop(s);
}
