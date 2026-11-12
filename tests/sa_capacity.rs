// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The SA global-state build, which keeps each read's packed spins and
//! `delta_energy` in device memory so SA runs models past its 4,593-node
//! thread-private frame. Placement and sweep windows change neither the RNG
//! stream nor the update order, so a global run returns the reads of a local
//! run of the same seed.
//!
//! ```sh
//! cargo test --test sa_capacity
//! cargo test --release --test sa_capacity -- --ignored --nocapture
//! ```

use std::io::{self, Write};

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::capacity::{advertised_nodes, host_limits};
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising_tuned, Kernel, SampleError, TunedRun, Tuning};
use quip_solver_metal::IsingGraph;

/// The thread-private frame of the local build: `delta_energy[4593]` in
/// `kernels/sa.metal`.
const LOCAL_FRAME: usize = 4_593;

/// Coefficient paths of the SA kernel.
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
    sample_ising_tuned(dev, graph, p, Kernel::Sa, tuning).expect("sa samples")
}

/// A planning rate so high that the anneal is one command buffer.
const ONE_CHUNK: f64 = 1e18;

#[test]
fn global_state_matches_local_state() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(64, 32, 4, 7);
    for path in PATHS {
        let graph = circulant(4_000, 20, path);
        let local = run(&dev, &graph, &p, Tuning::default());
        assert_eq!(local.reads.len(), 64, "{path:?}");
        for rate in [Some(ONE_CHUNK), None, Some(1.0)] {
            let tuning = Tuning {
                sa_global: true,
                planning_rate: rate,
                ..Tuning::default()
            };
            let global = run(&dev, &graph, &p, tuning);
            assert_eq!(
                global.reads,
                local.reads,
                "{path:?}: global build in {} chunks",
                global.plan.len()
            );
        }
    }
}

#[test]
fn a_20k_node_model_completes_with_exact_energies() {
    let dev = MetalDevice::open(0).unwrap();
    let p = params(32, 16, 4, 11);
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
        // device state carries over 16 command buffers.
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
        assert_eq!(split.reads, whole.reads, "{path:?}");
    }
}

/// CUDA `32801ee`: a node cap below the local frame is the limit itself. It
/// is not raised to the frame, and it does not change the build a model of
/// at most the frame runs on.
#[test]
fn an_explicit_cap_below_the_local_frame_is_honored() {
    let limits = host_limits();
    assert_eq!(
        advertised_nodes(Kernel::Sa, &limits, Some(1_000)),
        Ok(1_000)
    );
    let memory_limit = advertised_nodes(Kernel::Sa, &limits, None).unwrap();
    assert!(memory_limit > LOCAL_FRAME, "{memory_limit}");

    let dev = MetalDevice::open(0).unwrap();
    let p = params(32, 8, 1, 3);
    let capped = Tuning {
        node_cap: Some(1_000),
        ..Tuning::default()
    };
    let fits = circulant(1_000, 20, Path::Unit);
    let capped_run = run(&dev, &fits, &p, capped);
    assert_eq!(capped_run.reads.len(), 32);
    // The cap does not move a model onto the global build.
    assert_eq!(
        capped_run.reads,
        run(&dev, &fits, &p, Tuning::default()).reads
    );

    let over = circulant(1_001, 20, Path::Unit);
    let err = sample_ising_tuned(&dev, &over, &p, Kernel::Sa, capped).unwrap_err();
    let SampleError::TooLarge(msg) = err else {
        panic!("expected a capacity refusal, got {err:?}");
    };
    assert!(
        msg.contains("graph N=1001 exceeds Sa kernel limit 1000"),
        "{msg}"
    );
}

/// Spin-update rates of the local and global builds, and the largest chunk
/// of the global build at its planned size. For
/// `docs/perf/2026-11-12-sa-global.md`.
#[test]
#[ignore = "minutes of device time: run with --release --ignored --nocapture"]
fn global_state_rates() {
    let dev = MetalDevice::open(0).unwrap();
    let sweeps = 256;
    for (nodes, degree, reads) in [
        (4_000, 20, 256),
        (4_000, 20, 1_024),
        (20_000, 20, 256),
        (20_000, 40, 256),
        (60_000, 20, 1_024),
    ] {
        for path in PATHS {
            let graph = circulant(nodes, degree, path);
            let p = params(reads, sweeps, 4, 9);
            let builds: &[(&str, bool)] = if nodes <= LOCAL_FRAME {
                &[("local", false), ("global", true)]
            } else {
                &[("global", true)]
            };
            for &(build, sa_global) in builds {
                let unplanned = Tuning {
                    sa_global,
                    planning_rate: Some(ONE_CHUNK),
                    ..Tuning::default()
                };
                let planned = Tuning {
                    sa_global,
                    ..Tuning::default()
                };
                let mut times: Vec<u64> = (0..3)
                    .map(|_| run(&dev, &graph, &p, unplanned).gpu_time_us)
                    .collect();
                times.sort_unstable();
                let out = run(&dev, &graph, &p, planned);
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
