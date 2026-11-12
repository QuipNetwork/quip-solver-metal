// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The `Sampler::sample_stream` contract, checked for every kernel through the
//! public API only.
//!
//! Metal GPU tests: need a real device (Apple Silicon). Run with
//! `--test-threads=1`. Each test skips when no Metal device opens.

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::{
    CancelToken, IsingGraph, SampleError, SampleParams, Sampler, StreamJob, StreamOutcome,
    StreamResult,
};
use quip_solver_metal::iokit_gov::UtilGovernor;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::sample_ising;
use quip_solver_metal::{GibbsTag, Kernel, MetalSampler, MsaTag, SaTag, TaggedSampler};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;

const KERNELS: [Kernel; 3] = [Kernel::Sa, Kernel::Msa, Kernel::Gibbs];
const TIMEOUT: Duration = Duration::from_secs(120);

/// Run `f` once per kernel.
fn each_kernel(f: impl Fn(Kernel)) {
    for kernel in KERNELS {
        f(kernel);
    }
}

/// A fresh sampler on device 0, or `None` (with a note) when no device opens.
fn open(kernel: Kernel) -> Option<Arc<MetalSampler>> {
    match MetalDevice::open(0) {
        Ok(device) => {
            let gov = UtilGovernor::start(0, 100, false);
            Some(Arc::new(MetalSampler::new(device, gov, kernel)))
        }
        Err(e) => {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping: no Metal device ({e})");
            }
            None
        }
    }
}

/// An `n`-node ring with all-zero biases and ferromagnetic couplings.
fn ring(n: usize) -> IsingGraph {
    let edges: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    IsingGraph::new(vec![0.0; n], vec![-1.0; edges.len()], edges)
}

fn params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
    SampleParams {
        num_reads,
        num_sweeps,
        seed,
        ..SampleParams::default()
    }
}

fn job(id: &str, graph: IsingGraph, params: SampleParams, watermark: Option<u64>) -> StreamJob {
    StreamJob {
        job_id: id.as_bytes().to_vec(),
        graph,
        params,
        watermark,
    }
}

/// Feed `jobs` to `sample_stream` on its own thread, call `after_send`, close
/// the job channel, and collect every result until the sampler closes its
/// output. Panics if the stream is still open after [`TIMEOUT`].
fn stream(
    sampler: &Arc<MetalSampler>,
    jobs: Vec<StreamJob>,
    cancel: &CancelToken,
    after_send: impl FnOnce(),
) -> Vec<StreamResult> {
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(jobs.len().max(1));
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(jobs.len().max(1));
    let worker = {
        let sampler = Arc::clone(sampler);
        let cancel = cancel.clone();
        std::thread::spawn(move || sampler.sample_stream(job_rx, out_tx, cancel))
    };
    for j in jobs {
        job_tx
            .blocking_send(j)
            .expect("sampler dropped the job channel");
    }
    after_send();
    drop(job_tx);

    let deadline = Instant::now() + TIMEOUT;
    let mut results = Vec::new();
    loop {
        match out_rx.try_recv() {
            Ok(r) => results.push(r),
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {
                assert!(
                    Instant::now() < deadline,
                    "sample_stream still open after {TIMEOUT:?} with {} results",
                    results.len()
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
    worker.join().expect("sample_stream panicked");
    results
}

fn id_of(r: &StreamResult) -> String {
    String::from_utf8_lossy(&r.job_id).into_owned()
}

#[test]
fn every_job_gets_one_outcome_with_its_id() {
    each_kernel(|kernel| {
        let Some(sampler) = open(kernel) else { return };
        let mut ids: Vec<String> = (0..12).map(|i| format!("job-{i}")).collect();
        let jobs = ids
            .iter()
            .zip(0u64..)
            .map(|(id, seed)| job(id, ring(16), params(8, 64, seed), None))
            .collect();
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        let mut got: Vec<String> = results.iter().map(id_of).collect();
        got.sort();
        ids.sort();
        assert_eq!(got, ids, "{kernel:?}: one outcome per job id");
    });
}

#[test]
fn a_job_returns_exactly_its_reads_at_every_budget() {
    let Some(sampler) = open(Kernel::Msa) else {
        return;
    };
    // Stages a screen would use if a plain stream job were ever screened.
    sampler.apply_config("cascade_stages = [8, 64]");
    for (num_reads, num_sweeps) in [(1, 8), (8, 64), (64, 256), (64, 16_384)] {
        let jobs = vec![job(
            "budget",
            ring(16),
            params(num_reads, num_sweeps, 1),
            None,
        )];
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        assert_eq!(results.len(), 1);
        match &results[0].outcome {
            StreamOutcome::Completed(Ok(reads)) => assert_eq!(
                reads.len(),
                num_reads,
                "reads at num_reads={num_reads} num_sweeps={num_sweeps}"
            ),
            StreamOutcome::Completed(Err(e)) => {
                panic!("num_reads={num_reads} num_sweeps={num_sweeps}: {e:?}")
            }
            StreamOutcome::Cancelled => {
                panic!("num_reads={num_reads} num_sweeps={num_sweeps}: cancelled")
            }
        }
    }
}

/// A 32-node ring with `h` and `J` drawn from {-1, 0, 1} by a fixed seed.
fn mixed_ring() -> IsingGraph {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut draw = || {
        // xorshift64: deterministic, no dependency needed.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        f64::from(u8::try_from(state % 3).expect("< 3")) - 1.0
    };
    let n = 32;
    let h = (0..n).map(|_| draw()).collect();
    let edges: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let j = edges.iter().map(|_| draw()).collect();
    IsingGraph::new(h, j, edges)
}

#[test]
fn every_read_is_valid_and_scored_exactly() {
    each_kernel(|kernel| {
        let Some(sampler) = open(kernel) else { return };
        let graph = mixed_ring();
        let jobs = vec![job("scored", graph.clone(), params(8, 64, 3), None)];
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        assert_eq!(results.len(), 1);
        let StreamOutcome::Completed(Ok(reads)) = &results[0].outcome else {
            panic!("{kernel:?}: job did not complete with reads");
        };
        assert_eq!(reads.len(), 8, "{kernel:?}");
        for read in reads {
            assert_eq!(read.spins.len(), graph.num_nodes(), "{kernel:?}");
            assert!(
                read.spins.iter().all(|&s| s == -1 || s == 1),
                "{kernel:?}: spin outside {{-1, +1}}: {:?}",
                read.spins
            );
            let host = energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges);
            assert_eq!(
                read.energy_milli, host,
                "{kernel:?}: energy is not the host score"
            );
        }
    });
}

#[test]
fn alternating_int8_and_wide_jobs_keep_exact_scores_and_wide_samples() {
    let Some(dev) = (MetalDevice::device_count() > 0).then(|| MetalDevice::open(0).unwrap()) else {
        return;
    };
    for kernel in [Kernel::Sa, Kernel::Gibbs] {
        let Some(sampler) = open(kernel) else { return };
        let graphs: Vec<_> = (0..8)
            .map(|i| {
                let mut graph = ring(16);
                if i % 2 == 1 {
                    graph.h[0] = 0.125;
                    graph.j[0] = -0.5;
                }
                graph
            })
            .collect();
        let jobs = graphs
            .iter()
            .enumerate()
            .map(|(i, graph)| {
                job(
                    &format!("job-{i}"),
                    graph.clone(),
                    params(8, 64, i as u64 + 17),
                    None,
                )
            })
            .collect();
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        assert_eq!(results.len(), graphs.len(), "{kernel:?}");
        let mut seen = [false; 8];
        for result in results {
            let id = id_of(&result);
            let i: usize = id.strip_prefix("job-").unwrap().parse().unwrap();
            assert!(!seen[i], "{kernel:?}: duplicate {id}");
            seen[i] = true;
            let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
                panic!("{kernel:?}: {id} did not complete with reads");
            };
            assert_eq!(reads.len(), 8, "{kernel:?}: {id}");
            for read in &reads {
                assert_eq!(
                    read.energy_milli,
                    energy_milli(&read.spins, &graphs[i].h, &graphs[i].j, &graphs[i].edges),
                    "{kernel:?}: {id}"
                );
            }
            if i % 2 == 1 {
                let direct =
                    sample_ising(&dev, &graphs[i], &params(8, 64, i as u64 + 17), kernel).unwrap();
                assert_eq!(reads, direct, "{kernel:?}: {id}");
            }
        }
        assert!(seen.into_iter().all(|present| present), "{kernel:?}");
    }
}

/// A complete model of `n` nodes on the Int32 grid: couplings and fields
/// of three milli magnitudes with alternating signs.
fn complete_int32(n: usize) -> IsingGraph {
    const COUPLINGS: [f64; 3] = [1.3, 0.127, 2.0];
    const FIELDS: [f64; 3] = [0.5, 0.0, 0.75];
    let sign = |k: usize| if k.is_multiple_of(2) { 1.0 } else { -1.0 };
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            let k = u * 13 + v * 5;
            edges.push((u, v));
            j.push(sign(k) * COUPLINGS[(k / 2) % 3]);
        }
    }
    let h = (0..n).map(|i| sign(i) * FIELDS[(i / 2) % 3]).collect();
    IsingGraph::new(h, j, edges)
}

/// A complete Int32 model of 1,000 nodes is inside the capacity limits on
/// dense-matrix storage. The stream runs it as `sample_ising` does: two
/// jobs of the same topology each return the reads of `sample_ising` with
/// their seed, every one scored exactly.
#[test]
#[cfg_attr(
    not(feature = "long-tests"),
    ignore = "over 30 s in a debug build: run with --features long-tests"
)]
fn a_complete_model_runs_dense_in_the_stream() {
    let graph = complete_int32(1_000);
    each_kernel(|kernel| {
        let Some(sampler) = open(kernel) else { return };
        let jobs = (0..2)
            .map(|i| {
                job(
                    &format!("dense-{i}"),
                    graph.clone(),
                    params(64, 64, i + 41),
                    None,
                )
            })
            .collect();
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        assert_eq!(results.len(), 2, "{kernel:?}");
        let dev = MetalDevice::open(0).unwrap();
        for result in results {
            let id = id_of(&result);
            let seed: u64 = id.strip_prefix("dense-").unwrap().parse().unwrap();
            let reads = match result.outcome {
                StreamOutcome::Completed(Ok(reads)) => reads,
                StreamOutcome::Completed(Err(error)) => {
                    panic!("{kernel:?}: {id} failed: {error:?}")
                }
                StreamOutcome::Cancelled => panic!("{kernel:?}: {id} cancelled"),
            };
            assert_eq!(reads.len(), 64, "{kernel:?}: {id}");
            for read in &reads {
                assert_eq!(
                    read.energy_milli,
                    energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges),
                    "{kernel:?}: {id}"
                );
            }
            let direct = sample_ising(&dev, &graph, &params(64, 64, seed + 41), kernel).unwrap();
            assert_eq!(reads, direct, "{kernel:?}: {id}");
        }
    });
}

/// A ring one node past the SA thread-private frame (4,593) and one past
/// the Gibbs threadgroup array (4,800) runs on the spill build, which the
/// capacity footprint bounds. The stream returns the reads of `sample_ising`
/// with the same seed, every one scored exactly.
#[test]
fn a_model_past_the_kernel_frame_runs_in_the_stream() {
    for (kernel, n) in [(Kernel::Sa, 4_594), (Kernel::Gibbs, 4_801)] {
        let Some(sampler) = open(kernel) else { return };
        let graph = ring(n);
        let jobs = vec![job("past-frame", graph.clone(), params(64, 64, 7), None)];
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        assert_eq!(results.len(), 1, "{kernel:?}");
        let reads = match &results[0].outcome {
            StreamOutcome::Completed(Ok(reads)) => reads,
            StreamOutcome::Completed(Err(error)) => panic!("{kernel:?}: failed: {error:?}"),
            StreamOutcome::Cancelled => panic!("{kernel:?}: cancelled"),
        };
        assert_eq!(reads.len(), 64, "{kernel:?}");
        for read in reads {
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges),
                "{kernel:?}"
            );
        }
        let dev = MetalDevice::open(0).unwrap();
        let direct = sample_ising(&dev, &graph, &params(64, 64, 7), kernel).unwrap();
        assert_eq!(reads, &direct, "{kernel:?}");
    }
}

#[test]
fn a_cancelled_watermark_returns_cancelled() {
    each_kernel(|kernel| {
        let Some(sampler) = open(kernel) else { return };
        let ids: Vec<String> = (0..4).map(|i| format!("slow-{i}")).collect();
        let jobs = ids
            .iter()
            .zip(0u64..)
            .map(|(id, seed)| job(id, ring(16), params(8, 1_000_000, seed), Some(7)))
            .collect();
        let cancel = CancelToken::default();
        let results = stream(&sampler, jobs, &cancel, || cancel.cancel_through(7));
        assert_eq!(results.len(), ids.len(), "{kernel:?}: one outcome per job");
        let mut cancelled = 0;
        for r in &results {
            match &r.outcome {
                StreamOutcome::Cancelled => cancelled += 1,
                StreamOutcome::Completed(Ok(_)) => {}
                StreamOutcome::Completed(Err(e)) => {
                    panic!("{kernel:?}: {} failed: {e:?}", id_of(r))
                }
            }
        }
        assert!(cancelled >= 1, "{kernel:?}: no job reported Cancelled");
    });
}

/// A complete model past each kernel's dense one-sweep limit is refused:
/// 1,700 nodes for SA and Gibbs, and for MSA 4,899 nodes, whose 11,997,651
/// edges also pass the advertised `max_edges` of 11,992,753.
#[test]
fn an_oversized_job_fails_without_a_panic() {
    each_kernel(|kernel| {
        let Some(sampler) = open(kernel) else { return };
        let oversized = complete_int32(if kernel == Kernel::Msa { 4_899 } else { 1_700 });
        let jobs = vec![
            job("oversized", oversized, params(8, 64, 1), None),
            job("valid", ring(16), params(8, 64, 2), None),
        ];
        let results = stream(&sampler, jobs, &CancelToken::default(), || {});
        assert_eq!(results.len(), 2, "{kernel:?}");
        for r in &results {
            match (id_of(r).as_str(), &r.outcome) {
                ("oversized", StreamOutcome::Completed(Err(SampleError::Capacity))) => {}
                ("valid", StreamOutcome::Completed(Ok(reads))) => {
                    assert_eq!(reads.len(), 8, "{kernel:?}");
                }
                (id, StreamOutcome::Completed(Ok(_))) => {
                    panic!("{kernel:?}: {id} unexpectedly completed")
                }
                (id, StreamOutcome::Completed(Err(e))) => panic!("{kernel:?}: {id} failed: {e:?}"),
                (id, StreamOutcome::Cancelled) => panic!("{kernel:?}: {id} cancelled"),
            }
        }
    });
}

#[test]
fn stream_width_matches_the_declared_width() {
    each_kernel(|kernel| {
        let Some(sampler) = open(kernel) else { return };
        // The declared width counts the MSA binary's one ANE slot on top of
        // the Metal stream width; SA and Gibbs run on Metal alone.
        let expected = match kernel {
            Kernel::Sa => TaggedSampler::<SaTag>::declared_stream_width(),
            Kernel::Gibbs => TaggedSampler::<GibbsTag>::declared_stream_width(),
            Kernel::Msa => TaggedSampler::<MsaTag>::declared_stream_width() - 1,
        };
        let expected = usize::try_from(expected).expect("width fits usize");
        assert!(expected >= 1, "{kernel:?}: declared width below one");
        assert_eq!(sampler.stream_width(), expected, "{kernel:?}");
    });
}

/// Every log line any test of this binary emits, from every thread.
static LOG: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());

/// Appends to [`LOG`].
struct LogWriter;

impl std::io::Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        LOG.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Route the INFO log of every thread to [`LOG`]. The resident runner logs
/// on its preparation workers, so a thread-local subscriber misses it.
fn capture_log() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(|| LogWriter)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("one global subscriber");
    });
}

/// The Advantage2 System 1 working graph with couplings and fields of ±1
/// drawn by a fixed xorshift stream: a unit model for the slot kernel.
fn advantage2_unit() -> IsingGraph {
    let edges: Vec<(usize, usize)> = include_str!("fixtures/advantage2-system1.edges")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(|t| t.parse::<usize>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    let mut sign = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if x & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    };
    let h = (0..4577).map(|_| sign()).collect();
    let j = edges.iter().map(|_| sign()).collect();
    IsingGraph::new(h, j, edges)
}

/// A circulant graph of `n` nodes and degree 20 with milli-grid couplings
/// and fields: a weighted model.
fn weighted_circulant(n: usize) -> IsingGraph {
    let couplings = [-1.3, 0.127, 2.0];
    let fields = [-0.5, 0.0, 0.75];
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for i in 0..n {
        for d in 1..=10 {
            edges.push((i, (i + d) % n));
            j.push(couplings[(i * 7 + d) % couplings.len()]);
        }
    }
    let h = (0..n).map(|i| fields[i % fields.len()]).collect();
    IsingGraph::new(h, j, edges)
}

/// A weighted job of 20,000 nodes, past the 6,016 spin words the multi-spin
/// kernel keeps in threadgroup memory, runs on the batch path's spill build
/// in the same session as a unit job on the resident slots.
#[test]
fn a_spill_job_and_a_unit_job_share_a_session() {
    capture_log();
    let Some(sampler) = open(Kernel::Msa) else {
        return;
    };
    let unit = advantage2_unit();
    let spill = weighted_circulant(20_000);
    let graphs = [("unit", &unit), ("spill", &spill)];
    let jobs = graphs
        .iter()
        .zip(1u64..)
        .map(|((id, graph), seed)| job(id, (*graph).clone(), params(32, 64, seed), None))
        .collect();
    let results = stream(&sampler, jobs, &CancelToken::default(), || {});
    assert_eq!(results.len(), graphs.len());
    for result in &results {
        let id = id_of(result);
        let graph = graphs
            .iter()
            .find(|(name, _)| *name == id)
            .map(|(_, graph)| *graph)
            .expect("a known job id");
        let reads = match &result.outcome {
            StreamOutcome::Completed(Ok(reads)) => reads,
            StreamOutcome::Completed(Err(e)) => panic!("{id}: {e:?}"),
            StreamOutcome::Cancelled => panic!("{id}: cancelled"),
        };
        assert_eq!(reads.len(), 32, "{id}");
        for read in reads {
            assert_eq!(read.spins.len(), graph.num_nodes(), "{id}");
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges),
                "{id}: energy is not the host score"
            );
        }
    }
    let log = String::from_utf8_lossy(
        &LOG.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
    .into_owned();
    assert!(
        log.lines()
            .any(|line| line.contains("cascade topology prepared") && line.contains("nodes=4577")),
        "the unit job's cascade line is missing from the log:\n{log}"
    );
}
