// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The weighted multi-spin kernel with spin words spilled past threadgroup
//! memory, and with rungs split into sweep windows. Spill placement and
//! windows change neither the RNG stream nor the update order, so every run
//! must return the reads of the plain run of the same seed.
//!
//! ```sh
//! cargo test --release --test msa_spill
//! cargo test --release --test msa_spill -- --ignored --nocapture
//! ```

use std::collections::HashSet;
use std::io::{self, Write};

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising_tuned, Kernel, TunedRun, Tuning};
use quip_solver_metal::{IsingGraph, SamplerResult};

/// Coefficient paths of the weighted lane.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// Milli coefficients with gcd 1: `int` storage.
    Int,
    /// Milli coefficients that fit `int8_t` after division by their gcd.
    Int8,
    /// Coefficients off the milli grid: the Float32 lane.
    Float,
    /// Couplings and fields in {-1, 0, 1}: the unit models the weighted lane
    /// takes when they are too dense or too large for the unit kernel.
    Unit,
}

/// The coupling and field values of `path`, picked by `k`.
fn coefficient(path: Path, k: usize, field: bool) -> f64 {
    let (couplings, fields): (&[f64], &[f64]) = match path {
        Path::Int => (&[-1.3, 0.127, 2.0], &[-0.5, 0.0, 0.75]),
        Path::Int8 => (&[-1.5, 0.5, 1.0], &[-0.5, 0.0, 0.5]),
        Path::Float => (
            &[-1.300_65, 0.127_063_5, 2.001],
            &[-0.500_25, 0.0, 0.750_375],
        ),
        Path::Unit => (&[-1.0, 1.0, 1.0], &[-1.0, 0.0, 1.0]),
    };
    let values = if field { fields } else { couplings };
    values[k % values.len()]
}

/// Circulant graph: node i links to i+1 ..= i+degree/2 (mod n), so every
/// node has `degree` neighbours when n > degree.
fn circulant(n: usize, degree: usize, path: Path) -> IsingGraph {
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for i in 0..n {
        for d in 1..=degree / 2 {
            edges.push((i, (i + d) % n));
            j.push(coefficient(path, i * 7 + d, false));
        }
    }
    let h = (0..n).map(|i| coefficient(path, i, true)).collect();
    IsingGraph::new(h, j, edges)
}

/// Complete graph on `n` nodes with couplings drawn from `path`'s values by
/// an xorshift stream of `seed`.
fn complete(n: usize, seed: u64, path: Path) -> IsingGraph {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut draw = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        usize::try_from(x >> 40).unwrap()
    };
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            edges.push((u, v));
            j.push(coefficient(path, draw(), false));
        }
    }
    let h = (0..n).map(|_| coefficient(path, draw(), true)).collect();
    IsingGraph::new(h, j, edges)
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
    sample_ising_tuned(dev, graph, p, Kernel::Msa, tuning).expect("msa samples")
}

fn spins(reads: &[SamplerResult]) -> Vec<Vec<i8>> {
    reads.iter().map(|r| r.spins.clone()).collect()
}

/// A planning rate so high that the anneal is one command buffer.
const ONE_CHUNK: f64 = 1e18;

#[test]
fn a_spilled_run_matches_an_all_threadgroup_run() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 64, 4, 7);
    for (path, degree) in [
        (Path::Int, 20),
        (Path::Int8, 20),
        (Path::Float, 20),
        (Path::Int, 40),
    ] {
        let graph = circulant(5_000, degree, path);
        let whole = Tuning {
            planning_rate: Some(ONE_CHUNK),
            ..Tuning::default()
        };
        // K = 5,000: every word in threadgroup memory.
        let all = run(&dev, &graph, &p, whole);
        assert_eq!(all.plan.len(), 1, "{path:?} degree {degree}");
        assert_eq!(all.reads.len(), 64);
        for (cap, rate) in [
            (1_000, Some(ONE_CHUNK)),
            (0, Some(ONE_CHUNK)),
            (4_999, Some(ONE_CHUNK)),
            // Spilled and split into one-sweep windows: the threadgroup
            // prefix persists across every chunk.
            (1_000, Some(1.0)),
        ] {
            let tuning = Tuning {
                shared_spins_cap: Some(cap),
                planning_rate: rate,
                ..Tuning::default()
            };
            let spilled = run(&dev, &graph, &p, tuning);
            assert_eq!(
                spins(&spilled.reads),
                spins(&all.reads),
                "{path:?} degree {degree}: K = {cap}, {} chunks",
                spilled.plan.len()
            );
            assert_eq!(spilled.reads, all.reads, "{path:?} energies at K = {cap}");
        }
    }
}

#[test]
fn a_unit_model_past_threadgroup_memory_spills() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 32, 4, 11);
    let graph = circulant(7_000, 20, Path::Unit);
    let natural = run(&dev, &graph, &p, Tuning::default());
    let small = run(
        &dev,
        &graph,
        &p,
        Tuning {
            shared_spins_cap: Some(1_000),
            ..Tuning::default()
        },
    );
    assert_eq!(spins(&small.reads), spins(&natural.reads));
    for read in natural.reads.iter().take(4) {
        assert_eq!(
            read.energy_milli,
            energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
}

#[test]
fn a_long_rung_splits_into_sweep_windows() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 16, 4, 5);
    for path in [Path::Int, Path::Float] {
        let graph = circulant(1_000, 20, path);
        let unsplit = run(
            &dev,
            &graph,
            &p,
            Tuning {
                planning_rate: Some(ONE_CHUNK),
                ..Tuning::default()
            },
        );
        assert_eq!(unsplit.plan.len(), 1, "{path:?}");
        // One update per second: no rung fits a dispatch, so every sweep is
        // its own window.
        let split = run(
            &dev,
            &graph,
            &p,
            Tuning {
                planning_rate: Some(1.0),
                ..Tuning::default()
            },
        );
        assert_eq!(split.plan.len(), 16, "{path:?}: {:?}", split.plan);
        for (index, chunk) in split.plan.iter().enumerate() {
            let index = i32::try_from(index).unwrap();
            assert_eq!(chunk.beta_start, index / 4, "{chunk:?}");
            assert_eq!(chunk.beta_count, 1, "{chunk:?}");
            assert_eq!(chunk.sweep_start, index % 4, "{chunk:?}");
            assert_eq!(chunk.sweep_count, 1, "{chunk:?}");
        }
        assert_eq!(split.reads, unsplit.reads, "{path:?}");
    }
}

/// CUDA `94fd798` (QUI-1596): CPU MSA shares one acceptance draw across
/// lanes, and on a dense model every read then merges into one
/// configuration. The weighted lane here must keep its reads apart: more
/// than an eighth distinct after 16 sweeps, more than one after longer
/// anneals, where every read may reach the ground state or its flip.
#[test]
fn dense_reads_stay_distinct() {
    let dev = MetalDevice::open(0).unwrap();
    let reads = 128;
    for sweeps in [16, 64, 4_096] {
        let p = params(reads, sweeps, 1, 3);
        let mut counts = Vec::new();
        for path in [Path::Unit, Path::Int, Path::Float] {
            let graph = complete(28, 7, path);
            let out = run(&dev, &graph, &p, Tuning::default());
            assert_eq!(out.reads.len(), reads);
            let distinct = spins(&out.reads).into_iter().collect::<HashSet<_>>().len();
            counts.push((path, distinct));
        }
        writeln!(
            io::stdout().lock(),
            "{sweeps} sweeps, {reads} reads: {counts:?}"
        )
        .unwrap();
        for (path, distinct) in counts {
            let floor = if sweeps == 16 { reads / 8 } else { 1 };
            assert!(
                distinct > floor,
                "{path:?} at {sweeps} sweeps: {distinct} distinct rows of {reads}"
            );
        }
    }
}

/// CUDA's ceiling: 200,000 nodes at degree 40 is 4,000,000 edges. Every read
/// returns, the host-scored energy equals consensus scoring, and no command
/// buffer passes the 500 ms dispatch ceiling.
#[test]
#[ignore = "minutes of device time: run with --ignored --nocapture"]
fn a_200k_node_model_completes() {
    let dev = MetalDevice::open(0).unwrap();
    let graph = circulant(200_000, 40, Path::Int);
    assert_eq!(graph.edges.len(), 4_000_000);
    let p = params(128, 16, 4, 11);
    let out = run(&dev, &graph, &p, Tuning::default());
    writeln!(
        io::stdout().lock(),
        "200k nodes: {} chunks, max chunk {} us, GPU {} us",
        out.plan.len(),
        out.max_chunk_us,
        out.gpu_time_us
    )
    .unwrap();
    assert_eq!(out.reads.len(), 128);
    for read in out.reads.iter().take(2) {
        assert_eq!(
            read.energy_milli,
            energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    assert!(out.max_chunk_us < 500_000, "{} us", out.max_chunk_us);
}

/// Word-update rates of the two spill layouts on models above `K`: the
/// threadgroup prefix (the default `K`) and every word in device memory
/// (`K = 0`, the layout `kernels/msa64.metal` tried). For
/// `docs/perf/2026-11-12-msa-spill.md`.
#[test]
#[ignore = "minutes of device time: run with --release --ignored --nocapture"]
fn spill_layout_rates() {
    let dev = MetalDevice::open(0).unwrap();
    let reads = 128;
    let sweeps = 256;
    let p = params(reads, sweeps, 4, 9);
    for (nodes, degree) in [(20_000, 20), (20_000, 40), (60_000, 20)] {
        let graph = circulant(nodes, degree, Path::Int);
        let mut prefix_reads = None;
        for (layout, cap) in [("prefix", None), ("device", Some(0)), ("prefix", None)] {
            let tuning = Tuning {
                shared_spins_cap: cap,
                ..Tuning::default()
            };
            let mut times: Vec<u64> = (0..3)
                .map(|_| run(&dev, &graph, &p, tuning).gpu_time_us)
                .collect();
            times.sort_unstable();
            let out = run(&dev, &graph, &p, tuning);
            if let Some(want) = &prefix_reads {
                assert_eq!(&out.reads, want, "{layout} reads differ");
            } else {
                prefix_reads = Some(out.reads.clone());
            }
            let words = (reads / 32 * nodes * sweeps) as f64;
            let median = times[1].max(1) as f64;
            writeln!(
                io::stdout().lock(),
                "{{\"nodes\":{nodes},\"degree\":{degree},\"layout\":\"{layout}\",\"gpu_ms\":{:.1},\"gwords_per_s\":{:.4},\"max_chunk_ms\":{:.1},\"chunks\":{}}}",
                median / 1e3,
                words / median / 1e3,
                out.max_chunk_us as f64 / 1e3,
                out.plan.len()
            )
            .unwrap();
        }
    }
}

/// The worst admitted shape of the multi-spin spill build, one problem of
/// 256 reads: degree 62, the most the kernel walks, at its one-sweep limit
/// of 1,433,223 nodes, extrapolated past the grid. The anneal's longest
/// chunk stays below 400 ms, and one node more is refused as a capacity
/// error. The limit is that of `docs/perf/2026-11-metal-capacity.md`.
#[test]
#[ignore = "device timing: run with --release --ignored --nocapture"]
fn the_largest_admitted_spill_sweep_stays_under_400_ms() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(256, 16, 1, 9);
    let limit = 1_433_223;
    let out = sample_ising_tuned(
        &dev,
        &circulant(limit, 62, Path::Int),
        &p,
        Kernel::Msa,
        Tuning::default(),
    )
    .unwrap();
    let max_ms = out.max_chunk_us as f64 / 1e3;
    let chunks: Vec<f64> = out.chunk_us.iter().map(|&us| us as f64 / 1e3).collect();
    writeln!(
        io::stdout().lock(),
        "{{\"shape\":\"spill degree 62\",\"nodes\":{limit},\"storage\":\"{:?}\",\"chunks\":{},\"max_chunk_ms\":{max_ms:.1},\"chunk_ms\":{chunks:.1?}}}",
        out.storage,
        out.plan.len(),
    )
    .unwrap();
    assert!(max_ms < 400.0, "{max_ms} ms");
    let err = sample_ising_tuned(
        &dev,
        &circulant(limit + 1, 62, Path::Int),
        &p,
        Kernel::Msa,
        Tuning::default(),
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("past the 400 ms limit of one chunk"),
        "{err:?}"
    );
}
