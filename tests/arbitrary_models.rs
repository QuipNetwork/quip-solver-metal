// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! QUI-78 acceptance cases on Metal: exact or refused, never quantized. Each
//! kernel anneals a model off the milli grid in `f32` and the host scores it
//! exactly.

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{model_footprint, sample_ising, Kernel, SampleError};
use quip_solver_metal::IsingGraph;

fn device() -> Option<MetalDevice> {
    (MetalDevice::device_count() > 0).then(|| MetalDevice::open(0).unwrap())
}

fn params(sweeps: usize) -> SampleParams {
    SampleParams {
        num_reads: 64,
        num_sweeps: sweeps,
        seed: 11,
        ..Default::default()
    }
}

fn brute_force_min(graph: &IsingGraph) -> i64 {
    let n = graph.h.len();
    (0..1_u32 << n)
        .map(|bits| {
            let s: Vec<i8> = (0..n)
                .map(|i| if bits >> i & 1 == 1 { -1 } else { 1 })
                .collect();
            energy_milli(&s, &graph.h, &graph.j, &graph.edges)
        })
        .min()
        .unwrap()
}

fn best(dev: &MetalDevice, graph: &IsingGraph, kernel: Kernel, sweeps: usize) -> i64 {
    let reads = sample_ising(dev, graph, &params(sweeps), kernel).unwrap();
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    reads.iter().map(|r| r.energy_milli).min().unwrap()
}

/// A 10-variable QUBO with quarter-unit coefficients, as Ising.
fn quarter_qubo() -> IsingGraph {
    let n = 10;
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if (u * 7 + v * 3) % 4 != 0 {
                edges.push((u, v));
                j.push(f64::from(((u + 2 * v) % 7) as i32 - 3) * 0.25);
            }
        }
    }
    let h = (0..n)
        .map(|i| f64::from((i % 5) as i32 - 2) * 0.25)
        .collect();
    IsingGraph::new(h, j, edges)
}

#[test]
fn a_quarter_unit_qubo_reaches_the_brute_force_minimum() {
    let Some(dev) = device() else { return };
    let graph = quarter_qubo();
    let want = brute_force_min(&graph);
    for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
        assert_eq!(best(&dev, &graph, kernel, 2000), want, "{kernel:?}");
    }
}

#[test]
fn large_and_wide_range_coefficients_are_solved_or_refused() {
    let Some(dev) = device() else { return };
    let solved = IsingGraph::new(
        vec![0.001, -1000.0, 0.0],
        vec![5000.0, -0.002],
        vec![(0, 1), (1, 2)],
    );
    let too_wide = IsingGraph::new(vec![0.0, 0.0], vec![300_000.0], vec![(0, 1)]);
    for kernel in [Kernel::Sa, Kernel::Msa] {
        assert_eq!(
            best(&dev, &solved, kernel, 500),
            brute_force_min(&solved),
            "{kernel:?}"
        );
    }
    // Past the wide bound, every kernel runs its Float32 build.
    for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
        assert_eq!(
            best(&dev, &too_wide, kernel, 500),
            brute_force_min(&too_wide),
            "{kernel:?}"
        );
    }
}

#[test]
fn complete_graphs_at_16_64_65_and_200_nodes_report_exact_energies() {
    let Some(dev) = device() else { return };
    for n in [16_usize, 64, 65, 200] {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .collect();
        let j = edges
            .iter()
            .map(|&(u, v)| if (u + v) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let graph = IsingGraph::new(vec![0.0; n], j, edges);
        // best() asserts reported == recomputed for every read.
        best(&dev, &graph, Kernel::Sa, 64);
        best(&dev, &graph, Kernel::Gibbs, 64);
        // MSA walks CSR rows of at most 62 neighbours: K_16 runs on CSR, and
        // the denser graphs run on dense-matrix storage.
        best(&dev, &graph, Kernel::Msa, 64);
    }
}

/// K_63 with unit antiferromagnetic couplings, degree 62: past the unit
/// kernel's 20 neighbours, so MSA runs it on the weighted lane. With `M` the
/// spin sum, `E = sum_{i<j} s_i s_j = (M^2 - n) / 2`, and odd `n` makes `M`
/// odd, so the ground energy is `(1 - n) / 2`, at `|M| = 1`.
fn a_dense_unit_model_with_degree_over_20_runs_on_msa(n: usize, sweeps: usize) {
    let Some(dev) = device() else { return };
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
        .collect();
    let graph = IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges);
    let ground = (1 - n as i64) * 500;
    let reads = sample_ising(&dev, &graph, &params(sweeps), Kernel::Msa).unwrap();
    assert_eq!(reads.len(), 64);
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
        assert_eq!(r.energy_milli, ground, "every read reaches the ground");
    }
}

/// The smallest odd complete graph above the unit kernel degree limit still
/// exercises MSA's weighted lane, with the same exact-ground assertion.
#[test]
fn a_small_dense_model_with_degree_over_20_runs_on_msa() {
    a_dense_unit_model_with_degree_over_20_runs_on_msa(23, 1000);
}

#[test]
#[cfg_attr(
    not(feature = "long-tests"),
    ignore = "degree 62 model takes over 5 s: run with --features long-tests"
)]
fn a_degree_62_model_runs_on_msa() {
    a_dense_unit_model_with_degree_over_20_runs_on_msa(63, 1000);
}

/// A weighted star whose hub has 30 leaves, so the hub's row takes two row
/// chunks of the weighted lane. Each leaf `i` settles against the hub spin
/// `s0` at `-|J_i s0 + h_i|`, so the ground energy is
/// `min over s0 of (h_0 s0 - sum_i |J_i s0 + h_i|)`.
#[test]
fn a_weighted_star_past_one_row_chunk_reaches_its_ground_state() {
    let Some(dev) = device() else { return };
    let leaves = 30;
    let j: Vec<f64> = (0..leaves)
        .map(|i| f64::from((i % 7) as i32 - 3) * 0.75)
        .collect();
    let mut h: Vec<f64> = (0..leaves)
        .map(|i| f64::from((i % 5) as i32 - 2) * 0.5)
        .collect();
    h.insert(0, 1.25);
    let edges = (1..=leaves).map(|v| (0, v)).collect();
    let graph = IsingGraph::new(h.clone(), j.clone(), edges);
    let want = [1.0, -1.0]
        .iter()
        .map(|&s0| {
            h[0] * s0
                - j.iter()
                    .zip(&h[1..])
                    .map(|(&jj, &hh)| (jj * s0 + hh).abs())
                    .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    let want = (want * 1000.0).round() as i64;
    let reads = sample_ising(&dev, &graph, &params(500), Kernel::Msa).unwrap();
    for r in &reads {
        assert_eq!(
            r.energy_milli,
            energy_milli(&r.spins, &graph.h, &graph.j, &graph.edges)
        );
    }
    // The 32 lanes of a word share each acceptance threshold, so on a star a
    // whole word can freeze with the hub against the ground orientation. Over
    // seeds 1 to 24 about three words in ten do, with or without the node
    // each sweep leaves out. No read may pass the ground energy, and the best
    // read must reach it.
    assert!(reads.iter().all(|r| r.energy_milli >= want), "want {want}");
    assert_eq!(reads.iter().map(|r| r.energy_milli).min(), Some(want));
}

#[test]
fn the_same_topology_at_1x_and_1000x_reaches_scaled_ground_states() {
    let Some(dev) = device() else { return };
    let n = 64;
    let edges: Vec<(usize, usize)> = (0..n).map(|v| (v, (v + 1) % n)).collect();
    let ring = |scale: f64| IsingGraph::new(vec![-scale; n], vec![-scale; n], edges.clone());
    // At 1x MSA takes the unit kernel; at 1000x, the weighted one.
    for kernel in [Kernel::Sa, Kernel::Msa] {
        let one = best(&dev, &ring(1.0), kernel, 2000);
        let thousand = best(&dev, &ring(1000.0), kernel, 2000);
        assert_eq!(one, -128_000, "{kernel:?}");
        assert_eq!(thousand, -128_000_000, "{kernel:?}");
    }
}

#[test]
fn a_non_representable_model_is_refused_as_capacity() {
    let Some(dev) = device() else { return };
    let off_step = IsingGraph::new(vec![0.0005, 0.0], vec![1.0], vec![(0, 1)]);
    let non_finite = IsingGraph::new(vec![f64::NAN, 0.0], vec![1.0], vec![(0, 1)]);
    for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
        let err = sample_ising(&dev, &non_finite, &params(16), kernel).unwrap_err();
        assert!(
            matches!(err, SampleError::TooLarge(_)),
            "{kernel:?}: {err:?}"
        );
        // Every kernel runs the off-step model on its Float32 build.
        best(&dev, &off_step, kernel, 64);
    }
}

#[test]
fn every_kernel_refuses_self_loops_for_unit_and_milli_models() {
    let Some(dev) = device() else { return };
    for j in [1.0, 0.5] {
        let graph = IsingGraph::new(vec![0.0], vec![j], vec![(0, 0)]);
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let err = sample_ising(&dev, &graph, &params(16), kernel).unwrap_err();
            assert!(
                matches!(&err, SampleError::TooLarge(message) if message.contains("self-loop")),
                "{kernel:?}, J={j}: {err:?}"
            );
        }
    }
}

#[test]
fn the_wide_path_repeats_exactly_for_a_fixed_seed() {
    let Some(dev) = device() else { return };
    let graph = quarter_qubo();
    for kernel in [Kernel::Sa, Kernel::Msa] {
        let a = sample_ising(&dev, &graph, &params(300), kernel).unwrap();
        let b = sample_ising(&dev, &graph, &params(300), kernel).unwrap();
        assert_eq!(a, b, "{kernel:?}");
    }
}

/// A whole-unit model whose sign model has a different ground state: the
/// weight-3 bond must win over two unit bonds. The unit kernel, reading signs
/// only, would settle the frustrated triangle anywhere.
#[test]
fn msa_anneals_weights_not_signs() {
    let Some(dev) = device() else { return };
    let triangle = IsingGraph::new(
        vec![0.0; 3],
        vec![3.0, -1.0, -1.0],
        vec![(0, 1), (1, 2), (0, 2)],
    );
    let want = brute_force_min(&triangle);
    let reads = sample_ising(&dev, &triangle, &params(500), Kernel::Msa).unwrap();
    assert!(reads.iter().all(|r| r.energy_milli == want), "{reads:?}");
}

/// A ring whose coefficients are multiples of 250 milli with one quotient
/// above 63, so the model is wide and takes `int8_t` storage at scale 250.
/// With `fine` set, one coefficient is 1 milli, so the gcd is 1 and a quotient
/// of 100 000 does not fit: the same shape on plain `int` storage.
fn scaled_ring(fine: bool) -> IsingGraph {
    let n = 100;
    let edges: Vec<(usize, usize)> = (0..n).map(|v| (v, (v + 1) % n)).collect();
    let mut j: Vec<f64> = (0..n)
        .map(|k| f64::from((k % 7) as i32 - 3) * 0.25)
        .collect();
    j[0] = 25.0;
    let mut h = vec![0.25; n];
    if fine {
        h[1] = 0.001;
    }
    IsingGraph::new(h, j, edges)
}

#[test]
fn a_scaled_i8_model_uses_one_byte_per_coupler() {
    let scaled = scaled_ring(false);
    let plain = scaled_ring(true);
    let edges = scaled.edges.len();
    let nodes = scaled.h.len();
    for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
        let narrow = model_footprint(&scaled, &params(64), kernel).unwrap();
        let wide = model_footprint(&plain, &params(64), kernel).unwrap();
        assert_eq!(narrow.storage, wide.storage, "{kernel:?}");
        // The J values (two per edge) and the h values drop from four bytes
        // to one. Deltas and fields stay `int`.
        let saved = 3 * (2 * edges + nodes);
        assert_eq!(
            wide.per_instance_bytes - narrow.per_instance_bytes,
            saved,
            "{kernel:?}"
        );
        assert_eq!(wide.shared_bytes, narrow.shared_bytes, "{kernel:?}");
    }
}
