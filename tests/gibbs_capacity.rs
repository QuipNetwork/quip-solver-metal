// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The Gibbs spill build, which keeps the first spins of each read in
//! threadgroup memory and the rest in a per-read device buffer, so Gibbs runs
//! models past its 4,800-spin threadgroup array. Placement and sweep windows
//! change neither the RNG stream nor the update order, so a spilled run
//! returns the reads of a threadgroup run of the same seed.
//!
//! ```sh
//! cargo test --test gibbs_capacity
//! cargo test --release --test gibbs_capacity -- --ignored --nocapture
//! ```

use std::io::{self, Write};

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising_tuned, Kernel, TunedRun, Tuning};
use quip_solver_metal::IsingGraph;

/// The threadgroup array of the local build: `shared_state[4800]` in
/// `kernels/gibbs.metal`.
const LOCAL_FRAME: usize = 4_800;

/// Coefficient paths of the Gibbs kernel.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// Couplings and fields in {-1, 0, 1}: the int8 build.
    Unit,
    /// Milli coefficients with gcd 1: the wide build on `int` storage.
    Int,
    /// Milli coefficients that fit `int8_t` after division by their gcd.
    Int8,
    /// Coefficients off the milli grid: the Float32 build.
    Float,
}

const PATHS: [Path; 4] = [Path::Unit, Path::Int, Path::Int8, Path::Float];

/// The coupling and field values of `path`, picked by `k`.
fn coefficient(path: Path, k: usize, field: bool) -> f64 {
    let (couplings, fields): (&[f64], &[f64]) = match path {
        Path::Unit => (&[-1.0, 1.0, 1.0], &[-1.0, 0.0, 1.0]),
        Path::Int => (&[-1.3, 0.127, 2.0], &[-0.5, 0.0, 0.75]),
        Path::Int8 => (&[-1.5, 0.5, 1.0], &[-0.5, 0.0, 0.5]),
        Path::Float => (
            &[-1.300_65, 0.127_063_5, 2.001],
            &[-0.500_25, 0.0, 0.750_375],
        ),
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

/// A planning rate so high that the anneal is one command buffer.
const ONE_CHUNK: f64 = 1e18;

#[test]
fn a_spilled_gibbs_run_matches_a_threadgroup_run() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 8, 2, 7);
    for path in PATHS {
        let graph = circulant(4_000, 20, path);
        let local = run(&dev, &graph, &p, Tuning::default());
        assert_eq!(local.reads.len(), 64, "{path:?}");
        // One update per second windows every rung into single sweeps on the
        // threadgroup build, which must compose to the whole run.
        let windowed = run(
            &dev,
            &graph,
            &p,
            Tuning {
                planning_rate: Some(1.0),
                ..Tuning::default()
            },
        );
        assert_eq!(windowed.plan.len(), 8, "{path:?}: {:?}", windowed.plan);
        assert_eq!(
            windowed.reads, local.reads,
            "{path:?}: windowed local build"
        );
        for rate in [Some(ONE_CHUNK), None, Some(1.0)] {
            let tuning = Tuning {
                gibbs_prefix_cap: Some(1_000),
                planning_rate: rate,
                ..Tuning::default()
            };
            let spilled = run(&dev, &graph, &p, tuning);
            assert_eq!(
                spilled.reads,
                local.reads,
                "{path:?}: spill build in {} chunks",
                spilled.plan.len()
            );
        }
    }
}

#[test]
fn a_20k_node_gibbs_model_completes_with_exact_energies() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(32, 4, 2, 11);
    for path in [Path::Unit, Path::Int] {
        let graph = circulant(20_000, 20, path);
        let whole = run(
            &dev,
            &graph,
            &p,
            Tuning {
                planning_rate: Some(ONE_CHUNK),
                ..Tuning::default()
            },
        );
        assert_eq!(whole.plan.len(), 1, "{path:?}");
        assert_eq!(whole.reads.len(), 32, "{path:?}");
        for read in &whole.reads {
            assert_eq!(read.spins.len(), 20_000);
            assert!(read.spins.iter().all(|&s| s == -1 || s == 1));
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        // One update per second: every sweep is its own window, so the
        // device state carries over four command buffers.
        let split = run(
            &dev,
            &graph,
            &p,
            Tuning {
                planning_rate: Some(1.0),
                ..Tuning::default()
            },
        );
        assert_eq!(split.plan.len(), 4, "{path:?}: {:?}", split.plan);
        for (index, chunk) in split.plan.iter().enumerate() {
            let index = i32::try_from(index).unwrap();
            assert_eq!(chunk.beta_start, index / 2, "{chunk:?}");
            assert_eq!(chunk.beta_count, 1, "{chunk:?}");
            assert_eq!(chunk.sweep_start, index % 2, "{chunk:?}");
            assert_eq!(chunk.sweep_count, 1, "{chunk:?}");
        }
        assert_eq!(split.reads, whole.reads, "{path:?}");
    }
}

/// A node cap below the frame is the limit itself and does not move a model
/// of at most the frame onto the spill build.
#[test]
fn an_explicit_cap_is_honored_and_keeps_the_threadgroup_build() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(32, 4, 1, 3);
    let capped = Tuning {
        node_cap: Some(1_000),
        ..Tuning::default()
    };
    let fits = circulant(1_000, 20, Path::Unit);
    assert_eq!(
        run(&dev, &fits, &p, capped).reads,
        run(&dev, &fits, &p, Tuning::default()).reads
    );
    let over = circulant(1_001, 20, Path::Unit);
    let err = sample_ising_tuned(&dev, &over, &p, Kernel::Gibbs, capped).unwrap_err();
    assert!(
        err.to_string()
            .contains("graph N=1001 exceeds Gibbs kernel limit 1000"),
        "{err}"
    );
}

/// Spin-update rates of the threadgroup and spill builds, and the largest
/// chunk of the spill build at its planned size. For
/// `docs/perf/2026-11-12-gibbs-spill.md`.
#[test]
#[ignore = "minutes of device time: run with --release --ignored --nocapture"]
fn spill_rates() {
    let dev = MetalDevice::open(0).unwrap();
    let sweeps = 64;
    // The kernel runs at most 256 reads (`MAX_READS`), so every shape asks
    // for no more and the update count uses the reads that ran.
    let reads = 256;
    for (nodes, degree) in [
        (4_000, 20),
        (20_000, 20),
        (20_000, 40),
        (60_000, 20),
        (200_000, 20),
    ] {
        for path in PATHS {
            let graph = circulant(nodes, degree, path);
            let p = params(reads, sweeps, 4, 9);
            let builds: &[(&str, Option<usize>)] = if nodes <= LOCAL_FRAME {
                &[("local", None), ("spill", Some(1_000))]
            } else {
                &[("spill", None)]
            };
            for &(build, prefix) in builds {
                let unplanned = Tuning {
                    gibbs_prefix_cap: prefix,
                    planning_rate: Some(ONE_CHUNK),
                    ..Tuning::default()
                };
                let planned = Tuning {
                    gibbs_prefix_cap: prefix,
                    ..Tuning::default()
                };
                let mut times: Vec<u64> = (0..3)
                    .map(|_| run(&dev, &graph, &p, unplanned).gpu_time_us)
                    .collect();
                times.sort_unstable();
                let out = run(&dev, &graph, &p, planned);
                assert_eq!(out.reads.len(), reads, "every requested read ran");
                let updates = (reads * nodes * sweeps) as f64;
                let median = times[1].max(1) as f64;
                writeln!(
                    io::stdout().lock(),
                    "{{\"nodes\":{nodes},\"degree\":{degree},\"reads\":{reads},\"path\":\"{path:?}\",\"build\":\"{build}\",\"gpu_ms\":{:.1},\"gupd_per_s\":{:.4},\"planned_chunks\":{},\"planned_max_chunk_ms\":{:.1}}}",
                    median / 1e3,
                    updates / median / 1e3,
                    out.plan.len(),
                    out.max_chunk_us as f64 / 1e3,
                )
                .unwrap();
            }
        }
    }
}
