// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The dense-matrix Gibbs kernel (`kernels/gibbs_dense.metal`): each spin
//! update reads one row of an `N * N` coupling matrix instead of a CSR row.
//! It keeps the seed, the colour classes, the RNG streams, and the
//! arithmetic of the sparse chromatic kernel, so a model that both paths
//! hold returns the same reads on each.
//!
//! ```sh
//! cargo test --release --test gibbs_dense
//! cargo test --release --test gibbs_dense -- --ignored --nocapture
//! ```

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::capacity::Storage;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::model_file::read_model;
use quip_solver_metal::sampler::{sample_ising_tuned, Kernel, SampleError, TunedRun, Tuning};
use quip_solver_metal::{IsingGraph, SamplerResult};

/// Coefficient precisions of the dense matrix. The Gibbs kernel runs a unit
/// model and a whole-unit model on the same `int8_t` matrix, so `Binary` and
/// `Whole` reach one build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    /// Couplings and fields in {-1, 0, 1}: an `int8_t` matrix.
    Binary,
    /// Whole units: an `int8_t` matrix.
    Whole,
    /// Milli coefficients that fit `int8_t` after division by their gcd: an
    /// `int8_t` matrix times one integer scale.
    Int8,
    /// Milli coefficients with gcd 1: an `int` matrix.
    Int32,
    /// Coefficients off the milli grid: a `float` matrix.
    Float32,
}

/// The precisions of the ground-state and rate tests.
const PRECISIONS: [Precision; 4] = [
    Precision::Binary,
    Precision::Int8,
    Precision::Int32,
    Precision::Float32,
];

/// Every matrix build.
const ALL_PRECISIONS: [Precision; 5] = [
    Precision::Binary,
    Precision::Whole,
    Precision::Int8,
    Precision::Int32,
    Precision::Float32,
];

/// Coupling magnitudes and field magnitudes of `precision`, picked by `k`.
fn magnitude(precision: Precision, k: usize, field: bool) -> f64 {
    let (couplings, fields): (&[f64], &[f64]) = match precision {
        Precision::Binary => (&[1.0], &[1.0, 0.0]),
        Precision::Whole => (&[2.0, 1.0, 3.0], &[1.0, 0.0, 2.0]),
        Precision::Int8 => (&[0.5, 1.0, 1.5], &[0.5, 0.0, 1.0]),
        Precision::Int32 => (&[1.3, 0.127, 2.0], &[0.5, 0.0, 0.75]),
        Precision::Float32 => (&[1.300_65, 0.127_063_5, 2.001], &[0.500_25, 0.0, 0.750_375]),
    };
    let values = if field { fields } else { couplings };
    values[k % values.len()]
}

/// Signed coefficients of `precision`, picked by `k`.
fn signed(precision: Precision, k: usize, field: bool) -> f64 {
    let value = magnitude(precision, k / 2, field);
    if k.is_multiple_of(2) {
        value
    } else {
        -value
    }
}

/// A complete model on `n` nodes whose ground state is `planted`: every
/// coupling is `-|w| s_u s_v` and every field `-|a| s_i`, so `planted`
/// satisfies every term, and a nonzero field breaks the global flip.
fn planted_complete(n: usize, precision: Precision) -> (IsingGraph, Vec<i8>) {
    let planted: Vec<i8> = (0..n)
        .map(|i| if (i * 7 + i / 3) % 5 < 2 { -1 } else { 1 })
        .collect();
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            edges.push((u, v));
            let w = magnitude(precision, u * 13 + v * 5, false);
            j.push(-w * f64::from(planted[u]) * f64::from(planted[v]));
        }
    }
    let h = (0..n)
        .map(|i| -magnitude(precision, i, true) * f64::from(planted[i]))
        .collect();
    (IsingGraph::new(h, j, edges), planted)
}

/// Circulant graph: node i links to i+1 ..= i+degree/2 (mod n), so every
/// node has `degree` neighbours when n > degree.
fn circulant(n: usize, degree: usize, precision: Precision) -> IsingGraph {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for i in 0..n {
        for d in 1..=degree / 2 {
            edges.push((i, (i + d) % n));
            j.push(signed(precision, i * 7 + d, false));
        }
    }
    let h = (0..n).map(|i| signed(precision, i, true)).collect();
    IsingGraph::new(h, j, edges)
}

/// A complete graph on `n` nodes with every coupling `j`.
fn complete(n: usize, j: f64) -> IsingGraph {
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
        .collect();
    IsingGraph::new(vec![0.0; n], vec![j; edges.len()], edges)
}

/// `graph` plus one more coupler.
fn with_edge(graph: &IsingGraph, edge: (usize, usize)) -> IsingGraph {
    let mut edges = graph.edges.to_vec();
    edges.push(edge);
    let mut j = graph.j.to_vec();
    j.push(1.0);
    IsingGraph::new(graph.h.to_vec(), j, edges)
}

fn params(reads: usize, sweeps: usize, sweeps_per_beta: usize, seed: u64) -> SampleParams {
    SampleParams {
        num_reads: reads,
        num_sweeps: sweeps,
        sweeps_per_beta,
        seed,
        ..SampleParams::default()
    }
}

fn run(dev: &MetalDevice, graph: &IsingGraph, p: &SampleParams, tuning: Tuning) -> TunedRun {
    sample_ising_tuned(dev, graph, p, Kernel::Gibbs, tuning).expect("gibbs samples")
}

fn dense(tuning: Tuning) -> Tuning {
    Tuning {
        force_dense: true,
        ..tuning
    }
}

fn spins(reads: &[SamplerResult]) -> Vec<Vec<i8>> {
    reads.iter().map(|r| r.spins.clone()).collect()
}

/// A planning rate so high that the anneal is one command buffer.
const ONE_CHUNK: f64 = 1e18;

/// The best read of `graph` reaches `planted`, whose energy is the ground
/// energy, and every read's energy is its host score.
fn assert_ground_state(out: &TunedRun, graph: &IsingGraph, planted: &[i8], label: &str) {
    let ground = energy_milli(planted, &graph.h, &graph.j, &graph.edges);
    for read in &out.reads {
        assert_eq!(
            read.energy_milli,
            energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges),
            "{label}"
        );
    }
    let best = out.reads.iter().min_by_key(|r| r.energy_milli).unwrap();
    assert_eq!(best.energy_milli, ground, "{label}");
    assert_eq!(best.spins, planted, "{label}");
}

#[test]
fn a_complete_model_reaches_its_ground_state_at_each_precision() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 1_024, 16, 21);
    for precision in PRECISIONS {
        let (graph, planted) = planted_complete(64, precision);
        // Degree 63 packs into a quarter of the CSR bytes, so the model
        // routes dense.
        let out = run(&dev, &graph, &p, Tuning::default());
        assert_eq!(out.storage, Storage::Dense, "{precision:?}");
        assert_eq!(out.reads.len(), 64, "{precision:?}");
        assert_ground_state(&out, &graph, &planted, &format!("{precision:?}"));
    }
}

/// The dense kernel keeps the sparse kernel's seed, colour classes, RNG
/// streams, and arithmetic, so a model both paths hold returns the same
/// reads, bit for bit, at every precision. Sweep windows change nothing
/// either, and neither does a sparse run that spills.
#[test]
fn the_dense_path_matches_the_sparse_path() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 64, 4, 7);
    for precision in ALL_PRECISIONS {
        let graph = circulant(200, 20, precision);
        let whole = Tuning {
            planning_rate: Some(ONE_CHUNK),
            ..Tuning::default()
        };
        let sparse = run(&dev, &graph, &p, whole);
        assert_eq!(sparse.storage, Storage::Threadgroup, "{precision:?}");
        let dense_run = run(&dev, &graph, &p, dense(whole));
        assert_eq!(dense_run.storage, Storage::Dense, "{precision:?}");
        assert_eq!(dense_run.plan.len(), 1, "{precision:?}");
        assert_eq!(
            spins(&dense_run.reads),
            spins(&sparse.reads),
            "{precision:?}"
        );
        assert_eq!(dense_run.reads, sparse.reads, "{precision:?} energies");
        // The anneal moved: the reads are not all one state.
        assert!(
            sparse
                .reads
                .iter()
                .any(|r| r.spins != sparse.reads[0].spins),
            "{precision:?}"
        );
        // The sparse spill build, and one-sweep windows that resume every
        // chunk from the carry-over, return the same reads.
        let spilled = run(
            &dev,
            &graph,
            &p,
            Tuning {
                gibbs_prefix_cap: Some(100),
                ..whole
            },
        );
        assert_eq!(spilled.storage, Storage::Spill, "{precision:?}");
        assert_eq!(spilled.reads, sparse.reads, "{precision:?}: sparse spill");
        let windows = run(
            &dev,
            &graph,
            &p,
            dense(Tuning {
                planning_rate: Some(1.0),
                ..Tuning::default()
            }),
        );
        assert!(windows.plan.len() > 16, "{precision:?}");
        assert_eq!(
            windows.reads,
            sparse.reads,
            "{precision:?}: {} chunks",
            windows.plan.len()
        );
    }
    // Rows of 30 neighbours, then 1,400-node models whose rows span many
    // matrix words, then fields that round differently in another order of
    // addition.
    let mut shapes = vec![
        (circulant(200, 30, Precision::Binary), p.clone()),
        (circulant(200, 30, Precision::Int8), p),
    ];
    for precision in ALL_PRECISIONS {
        shapes.push((circulant(1_400, 4, precision), params(64, 16, 4, 7)));
    }
    shapes.push((cancelling_float_model(10), cold(params(64, 64, 4, 7))));
    for (graph, p) in shapes {
        let label = format!("{} nodes, {} edges", graph.num_nodes(), graph.edges.len());
        let sparse = run(&dev, &graph, &p, Tuning::default());
        assert_ne!(sparse.storage, Storage::Dense, "{label}");
        let dense_run = run(&dev, &graph, &p, dense(Tuning::default()));
        assert_eq!(dense_run.storage, Storage::Dense, "{label}");
        assert_eq!(dense_run.reads, sparse.reads, "{label}");
        assert!(
            sparse
                .reads
                .iter()
                .any(|r| r.spins != sparse.reads[0].spins),
            "{label}"
        );
    }
}

/// `params` on a cold fixed schedule, so a field of 0 and a field of 1 give
/// clearly different heat-bath odds.
fn cold(p: SampleParams) -> SampleParams {
    SampleParams {
        beta_range: Some((1.0, 8.0)),
        ..p
    }
}

/// `gadgets` copies of a node `x` with three neighbours, in ascending
/// columns: couplings `2^24`, `1`, and `-2^24`. In `f32`, `2^24 + 1` rounds
/// to `2^24` but `-2^24 + 1` is exact, so the field of `x` is 0 in one order
/// of addition and 1 in another whenever the two large terms cancel. `x`
/// takes the first colour, so the first sweep updates it from the random
/// start, where the large terms cancel for half of the reads. Its
/// heat-bath odds then differ between the two orders, so one order of
/// addition changes the spin that a draw selects, and with it every later
/// read. One extra node with a field off the milli grid makes the model
/// Float32.
fn cancelling_float_model(gadgets: usize) -> IsingGraph {
    let big = f64::from(1_u32 << 24);
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for g in 0..gadgets {
        let x = 4 * g;
        for (offset, value) in [(1, big), (2, 1.0), (3, -big)] {
            edges.push((x, x + offset));
            j.push(value);
        }
    }
    let n = 4 * gadgets + 1;
    let mut h = vec![0.0; n];
    h[n - 1] = 0.0005;
    IsingGraph::new(h, j, edges)
}

/// A repeated coupler has no single matrix cell, so such a model keeps
/// sparse storage: the automatic route takes CSR for it, and a forced dense
/// run refuses it before any buffer. A self-loop is refused at every storage.
#[test]
fn a_repeated_coupler_or_a_self_loop_stays_sparse() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(32, 16, 4, 3);
    let clean = complete(80, -1.0);
    assert_eq!(
        run(&dev, &clean, &p, Tuning::default()).storage,
        Storage::Dense
    );
    let repeated = with_edge(&clean, (1, 0));
    let sparse = run(&dev, &repeated, &p, Tuning::default());
    assert_ne!(sparse.storage, Storage::Dense);
    assert_eq!(sparse.reads.len(), 32);
    let err = sample_ising_tuned(&dev, &repeated, &p, Kernel::Gibbs, dense(Tuning::default()))
        .unwrap_err();
    assert!(matches!(err, SampleError::TooLarge(_)), "{err:?}");
    let looped = with_edge(&clean, (5, 5));
    for tuning in [Tuning::default(), dense(Tuning::default())] {
        let err = sample_ising_tuned(&dev, &looped, &p, Kernel::Gibbs, tuning).unwrap_err();
        assert!(matches!(err, SampleError::TooLarge(_)), "{err:?}");
    }
}

/// The largest complete Int32 model whose one sweep the planner estimates
/// at 400 ms or less runs dense at 256 reads, with every chunk below the
/// 500 ms ceiling. One more node is refused before any buffer: see
/// `a_dense_gibbs_sweep_past_one_chunk_is_refused_before_allocation` in
/// `src/sampler.rs`.
#[test]
#[ignore = "host scoring of 256 reads over millions of couplers: run with --release --ignored"]
fn the_largest_admitted_dense_sweep_runs() {
    let dev = MetalDevice::open(0).unwrap();
    let (graph, _) = planted_complete(LIMIT_NODES, Precision::Int32);
    let p = params(MAX_READS, 2, 1, 5);
    let out = run(&dev, &graph, &p, Tuning::default());
    assert_eq!(out.storage, Storage::Dense);
    assert_eq!(out.reads.len(), MAX_READS);
    writeln!(
        io::stdout().lock(),
        "{LIMIT_NODES} nodes: {} chunks, longest {:.1} ms",
        out.plan.len(),
        out.max_chunk_us as f64 / 1e3
    )
    .unwrap();
    assert!(out.max_chunk_us < 500_000, "{} us", out.max_chunk_us);
    let (over, _) = planted_complete(LIMIT_NODES + 1, Precision::Int32);
    let err =
        sample_ising_tuned(&dev, &over, &p, Kernel::Gibbs, dense(Tuning::default())).unwrap_err();
    assert!(
        matches!(&err, SampleError::TooLarge(m) if m.contains("one dense-matrix sweep")),
        "{err:?}"
    );
}

/// A complete model one node past the one-sweep limit cannot plan dense
/// chunks. Auto routing and forced dense both refuse it before any buffer,
/// as Capacity.
#[test]
fn a_model_past_the_sweep_limit_is_refused_under_auto_and_forced_dense() {
    let dev = MetalDevice::open(0).unwrap();
    let (graph, _) = planted_complete(LIMIT_NODES + 1, Precision::Int32);
    let p = params(32, 2, 1, 5);
    for tuning in [Tuning::default(), dense(Tuning::default())] {
        let err = sample_ising_tuned(&dev, &graph, &p, Kernel::Gibbs, tuning).unwrap_err();
        assert!(
            matches!(&err, SampleError::TooLarge(m) if m.contains("one dense-matrix sweep")),
            "{err:?}"
        );
    }
}

fn portfolio_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/portfolio")
}

/// The proven ground energies, offset included, of every fixture in
/// file-name order.
fn proven_ground_states() -> Vec<(String, f64)> {
    let manifest = std::fs::read_to_string(portfolio_dir().join("ground_states.json")).unwrap();
    let records: Vec<serde_json::Value> = serde_json::from_str(&manifest).unwrap();
    let mut proven: Vec<(String, f64)> = records
        .iter()
        .filter(|r| r["status"] == "optimal" && r["gap"].as_f64() == Some(0.0))
        .map(|r| {
            (
                r["file"].as_str().unwrap().to_owned(),
                r["energy"].as_f64().unwrap(),
            )
        })
        .collect();
    proven.sort_by(|a, b| a.0.cmp(&b.0));
    proven
}

/// `E(s) = sum_i h_i s_i + sum_k J_k s_u s_v` in `f64`, without an offset.
fn real_energy(spins: &[i8], graph: &IsingGraph) -> f64 {
    let fields: f64 = graph
        .h
        .iter()
        .zip(spins)
        .map(|(&h, &s)| h * f64::from(s))
        .sum();
    let couplings: f64 = graph
        .j
        .iter()
        .zip(graph.edges.iter())
        .map(|(&j, &(u, v))| j * f64::from(spins[u]) * f64::from(spins[v]))
        .sum();
    fields + couplings
}

/// Port of CUDA `933088c`: each fixture in `range` of the proven list, forced
/// dense, reaches its proven ground energy with 128 reads of 8,192 sweeps in
/// 64 beta steps at seed 42.
fn portfolio_fixtures_run_dense(range: std::ops::Range<usize>) {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(128, 8_192, 128, 42);
    let proven = proven_ground_states();
    let end = range.end.min(proven.len());
    let mut failures = Vec::new();
    for (file, ground) in &proven[range.start..end] {
        let model = read_model(&portfolio_dir().join(file)).unwrap();
        let out = run(&dev, &model.graph, &p, dense(Tuning::default()));
        assert_eq!(out.storage, Storage::Dense, "{file}");
        assert_eq!(out.reads.len(), 128, "{file}");
        let best = out
            .reads
            .iter()
            .map(|r| real_energy(&r.spins, &model.graph) + model.offset)
            .fold(f64::INFINITY, f64::min);
        let tolerance = 1e-6 * ground.abs().max(1.0);
        if (best - ground).abs() > tolerance {
            failures.push(format!("{file}: best {best} against proven {ground}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_portfolio_fixture_runs_dense() {
    portfolio_fixtures_run_dense(0..6);
}

#[test]
#[ignore = "the long-tests feature does not exist until Task 7.1; the fixtures after the first six take minutes: run with --release --ignored"]
fn every_later_portfolio_fixture_runs_dense() {
    portfolio_fixtures_run_dense(6..usize::MAX);
}

/// Reads one dispatch runs at most: the batch path's read cap.
const MAX_READS: usize = 256;

/// The nodes of the largest complete model whose one sweep the admission
/// accepts at the planning rate in `docs/perf/2026-11-12-gibbs-dense.md`.
const LIMIT_NODES: usize = 1_673;

/// GPU time of the dense kernel on complete models, for the planning rate in
/// `docs/perf/2026-11-12-gibbs-dense.md`. Each shape runs 256 reads with the
/// production chunk plan and reports the measured column rate of one SIMD
/// group and the longest chunk. The large shapes run a few sweeps, one sweep
/// per chunk, up to the largest model the admission accepts.
#[test]
#[ignore = "minutes of device time: run with --release --ignored --nocapture"]
fn dense_rates() {
    let dev = MetalDevice::open(0).unwrap();
    let shapes = [
        (64, 1 << 20),
        (256, 1 << 20),
        (1_024, 1 << 20),
        (LIMIT_NODES, 1 << 17),
    ];
    for precision in PRECISIONS {
        for (nodes, budget) in shapes {
            let (graph, _) = planted_complete(nodes, precision);
            let sweeps = (budget / nodes).max(4);
            let p = params(MAX_READS, sweeps, (sweeps / 64).max(1), 9);
            let out = run(&dev, &graph, &p, Tuning::default());
            assert_eq!(out.storage, Storage::Dense);
            assert_eq!(out.reads.len(), MAX_READS, "{precision:?} {nodes}");
            // One column walk per node and sweep, at the worst case of every
            // spin flipping.
            let columns = (nodes * sweeps) as f64 * nodes as f64;
            let gpu = out.gpu_time_us.max(1) as f64;
            // The per-chunk worst case: the columns of the largest chunk over
            // the time of the longest one.
            let largest = out
                .plan
                .iter()
                .map(|c| i64::from(c.beta_count) * i64::from(c.sweep_count))
                .max()
                .unwrap_or(0);
            let worst =
                (largest * nodes as i64) as f64 * nodes as f64 / out.max_chunk_us.max(1) as f64;
            writeln!(
                io::stdout().lock(),
                "{{\"precision\":\"{precision:?}\",\"nodes\":{nodes},\"sweeps\":{sweeps},\"gpu_ms\":{:.1},\"mcolumns_per_s\":{:.1},\"max_chunk_ms\":{:.1},\"worst_chunk_mcolumns_per_s\":{worst:.1},\"chunks\":{}}}",
                gpu / 1e3,
                columns / gpu,
                out.max_chunk_us as f64 / 1e3,
                out.plan.len()
            )
            .unwrap();
            // A chunk of several sweeps is the planner's own choice, so the
            // planning margin holds it near 200 ms on the slowest shape.
            if out.plan.iter().all(|c| c.beta_count * c.sweep_count > 1) {
                assert!(
                    out.max_chunk_us <= 250_000,
                    "{precision:?} {nodes}: {} us",
                    out.max_chunk_us
                );
            }
        }
    }
}

/// GPU time of the sparse Gibbs build on the complete models of
/// `dense_rates`, for the comparison in `docs/perf/2026-11-12-gibbs-dense.md`.
/// The sparse build keeps the CSR of every model, so the rate counts the same
/// `n * n` columns per sweep as the dense rate.
#[test]
#[ignore = "minutes of device time: run with --release --ignored --nocapture"]
fn sparse_rates() {
    let dev = MetalDevice::open(0).unwrap();
    let sparse = Tuning {
        force_sparse: true,
        ..Tuning::default()
    };
    for precision in [Precision::Int8, Precision::Int32] {
        // Four sweeps only: the sparse planner has no degree term, so it
        // would put far more sweeps than these in one command buffer.
        for nodes in [1_024, LIMIT_NODES] {
            let (graph, _) = planted_complete(nodes, precision);
            let sweeps = 4;
            let p = params(MAX_READS, sweeps, 1, 9);
            let out = run(&dev, &graph, &p, sparse);
            assert_ne!(out.storage, Storage::Dense);
            assert_eq!(out.reads.len(), MAX_READS, "{precision:?} {nodes}");
            let columns = (nodes * sweeps) as f64 * nodes as f64;
            let gpu = out.gpu_time_us.max(1) as f64;
            writeln!(
                io::stdout().lock(),
                "{{\"storage\":\"{:?}\",\"precision\":\"{precision:?}\",\"nodes\":{nodes},\"sweeps\":{sweeps},\"gpu_ms\":{:.1},\"mcolumns_per_s\":{:.1},\"max_chunk_ms\":{:.1},\"chunks\":{}}}",
                out.storage,
                gpu / 1e3,
                columns / gpu,
                out.max_chunk_us as f64 / 1e3,
                out.plan.len()
            )
            .unwrap();
        }
    }
}

/// The sparse Gibbs planner has no degree term (bead quip-miner-metal-9sp
/// for SA): it plans in spin updates, and an update of a complete model walks
/// a row of `n - 1` columns. One planned run of a complete model of 1,024
/// nodes at 256 reads on CSR storage reports its plan and its longest chunk.
/// The run is short on purpose: the planner's own chunk would hold far more
/// sweeps than this.
#[test]
#[ignore = "seconds of device time per sweep: run with --release --ignored --nocapture"]
fn sparse_high_degree_chunk() {
    let dev = MetalDevice::open(0).unwrap();
    let sparse = Tuning {
        force_sparse: true,
        ..Tuning::default()
    };
    let nodes = 1_024;
    let (graph, _) = planted_complete(nodes, Precision::Binary);
    for sweeps in [8, 64, 128] {
        let p = params(MAX_READS, sweeps, 1, 9);
        let out = run(&dev, &graph, &p, sparse);
        assert_ne!(out.storage, Storage::Dense);
        assert_eq!(out.reads.len(), MAX_READS);
        writeln!(
            io::stdout().lock(),
            "{{\"storage\":\"{:?}\",\"nodes\":{nodes},\"sweeps\":{sweeps},\"chunks\":{},\"gpu_ms\":{:.1},\"max_chunk_ms\":{:.1}}}",
            out.storage,
            out.plan.len(),
            out.gpu_time_us as f64 / 1e3,
            out.max_chunk_us as f64 / 1e3
        )
        .unwrap();
    }
}
