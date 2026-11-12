// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Real-valued models on the Float32 kernels. Each case is a helper that takes
//! the kernel, and each kernel module calls the helpers under the case names.
//!
//! Metal GPU tests: each case returns early when the host has no Metal device.

use std::path::{Path, PathBuf};

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::model_file::read_model;
use quip_solver_metal::sampler::{sample_ising, sample_ising_chunked, Kernel, SampleError};
use quip_solver_metal::IsingGraph;

fn device() -> Option<MetalDevice> {
    (MetalDevice::device_count() > 0).then(|| MetalDevice::open(0).unwrap())
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

/// A 12-node ring with chords. No coefficient is a whole number of milli
/// units, so only a Float32 kernel runs it.
fn twelve_node_model() -> IsingGraph {
    let n = 12;
    let mut edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    edges.extend((0..n / 2).map(|i| (i, i + n / 2)));
    let palette = [0.37, -1.25e-3, -0.8125, 0.0421, 1.3333, -0.5005];
    let j = (0..edges.len())
        .map(|k| palette[k % palette.len()])
        .collect();
    let h = (0..n)
        .map(|i| palette[(i * 5 + 1) % palette.len()] * 0.5)
        .collect();
    IsingGraph::new(h, j, edges)
}

fn a_real_model_returns_exact_energies(kernel: Kernel) {
    let Some(dev) = device() else {
        return;
    };
    let graph = twelve_node_model();
    let params = SampleParams {
        num_reads: 64,
        num_sweeps: 256,
        seed: 5,
        ..Default::default()
    };
    let reads = sample_ising(&dev, &graph, &params, kernel).unwrap();
    assert_eq!(reads.len(), 64);
    for read in &reads {
        assert_eq!(read.spins.len(), 12);
        assert!(read.spins.iter().all(|&s| s == 1 || s == -1));
        assert_eq!(
            read.energy_milli,
            energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
        );
        assert!(real_energy(&read.spins, &graph).is_finite());
    }
}

fn portfolio_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/portfolio")
}

/// The proven ground energies, offset included, of the first `count` fixtures
/// in file-name order.
fn proven_ground_states(count: usize) -> Vec<(String, f64)> {
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
    proven.truncate(count);
    assert_eq!(proven.len(), count);
    proven
}

/// 128 reads of 8,192 sweeps in 64 beta steps at seed 42, as the fixture
/// README states for the CUDA ground-state test.
fn portfolio_models_reach_their_ground_states(kernel: Kernel) {
    let Some(dev) = device() else {
        return;
    };
    let params = SampleParams {
        num_reads: 128,
        num_sweeps: 8192,
        sweeps_per_beta: 128,
        seed: 42,
        ..Default::default()
    };
    for (file, proven) in proven_ground_states(6) {
        let model = read_model(&portfolio_dir().join(&file)).unwrap();
        let reads = sample_ising(&dev, &model.graph, &params, kernel).unwrap();
        let best = reads
            .iter()
            .map(|r| real_energy(&r.spins, &model.graph) + model.offset)
            .fold(f64::INFINITY, f64::min);
        let tolerance = 1e-6 * proven.abs().max(1.0);
        assert!(
            (best - proven).abs() <= tolerance,
            "{kernel:?} {file}: best {best} against proven {proven}"
        );
    }
}

/// 64 fields of 1.2e-38: the default schedule's cold beta, about 3.6e38, is
/// past `f32::MAX` (CUDA commit `7d97e41`). A large coupling alone is
/// admitted, as in CUDA.
fn a_model_whose_schedule_overflows_f32_is_refused(kernel: Kernel) {
    let Some(dev) = device() else {
        return;
    };
    let graph = IsingGraph::new(vec![1.2e-38; 64], vec![], vec![]);
    let params = SampleParams {
        num_reads: 32,
        num_sweeps: 64,
        ..Default::default()
    };
    let err = sample_ising(&dev, &graph, &params, kernel).unwrap_err();
    assert!(
        matches!(&err, SampleError::TooLarge(m) if m.contains("beta") && m.contains("f32")),
        "{err:?}"
    );
    assert_eq!(
        err.to_sample_error(),
        quip_solver_core::SampleError::Capacity
    );
}

/// Zero fields with a beta past half of `f32::MAX`: fast math can double beta
/// first, and `inf * 0` would turn every heat-bath probability into NaN.
fn a_schedule_past_half_f32_max_is_refused(kernel: Kernel) {
    let Some(dev) = device() else {
        return;
    };
    // One off-grid field makes the model Float32; the other spins have none.
    let mut h = vec![0.0; 32];
    h[0] = 0.0005;
    let graph = IsingGraph::new(h, vec![], vec![]);
    let params = SampleParams {
        num_reads: 32,
        num_sweeps: 64,
        beta_range: Some((2e38, 3e38)),
        ..Default::default()
    };
    let err = sample_ising(&dev, &graph, &params, kernel).unwrap_err();
    assert!(
        matches!(&err, SampleError::TooLarge(m) if m.contains("half of f32::MAX")),
        "{err:?}"
    );
}

fn chunked_and_unchunked_real_anneals_match(kernel: Kernel) {
    let Some(dev) = device() else {
        return;
    };
    let graph = twelve_node_model();
    let params = SampleParams {
        num_reads: 64,
        num_sweeps: 96,
        sweeps_per_beta: 1,
        beta_range: Some((0.1, 3.0)),
        seed: 9,
        ..Default::default()
    };
    let whole = sample_ising(&dev, &graph, &params, kernel).unwrap();
    let split = sample_ising_chunked(&dev, &graph, &params, kernel, 1).unwrap();
    assert_eq!(whole, split);
    // The anneal moved: the reads are not all one state.
    assert!(whole.iter().any(|r| r.spins != whole[0].spins));
}

mod sa {
    use super::Kernel;

    #[test]
    fn a_real_model_returns_exact_energies() {
        super::a_real_model_returns_exact_energies(Kernel::Sa);
    }

    #[test]
    fn portfolio_models_reach_their_ground_states() {
        super::portfolio_models_reach_their_ground_states(Kernel::Sa);
    }

    #[test]
    fn a_model_whose_schedule_overflows_f32_is_refused() {
        super::a_model_whose_schedule_overflows_f32_is_refused(Kernel::Sa);
    }

    #[test]
    fn a_schedule_past_half_f32_max_is_refused() {
        super::a_schedule_past_half_f32_max_is_refused(Kernel::Sa);
    }

    #[test]
    fn chunked_and_unchunked_real_anneals_match() {
        super::chunked_and_unchunked_real_anneals_match(Kernel::Sa);
    }
}

mod gibbs {
    use super::Kernel;

    #[test]
    fn a_real_model_returns_exact_energies() {
        super::a_real_model_returns_exact_energies(Kernel::Gibbs);
    }

    #[test]
    fn portfolio_models_reach_their_ground_states() {
        super::portfolio_models_reach_their_ground_states(Kernel::Gibbs);
    }

    #[test]
    fn a_model_whose_schedule_overflows_f32_is_refused() {
        super::a_model_whose_schedule_overflows_f32_is_refused(Kernel::Gibbs);
    }

    #[test]
    fn a_schedule_past_half_f32_max_is_refused() {
        super::a_schedule_past_half_f32_max_is_refused(Kernel::Gibbs);
    }

    #[test]
    fn chunked_and_unchunked_real_anneals_match() {
        super::chunked_and_unchunked_real_anneals_match(Kernel::Gibbs);
    }
}
