// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! QUI-78 acceptance cases on Metal: exact or refused, never quantized.

use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::SampleParams;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising, Kernel, SampleError};
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
    for kernel in [Kernel::Sa, Kernel::Gibbs] {
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
    assert_eq!(
        best(&dev, &solved, Kernel::Sa, 500),
        brute_force_min(&solved)
    );
    let too_wide = IsingGraph::new(vec![0.0, 0.0], vec![300_000.0], vec![(0, 1)]);
    let err = sample_ising(&dev, &too_wide, &params(16), Kernel::Sa).unwrap_err();
    assert!(matches!(err, SampleError::TooLarge(_)), "{err:?}");
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
    }
}

#[test]
fn the_same_topology_at_1x_and_1000x_reaches_scaled_ground_states() {
    let Some(dev) = device() else { return };
    let n = 64;
    let edges: Vec<(usize, usize)> = (0..n).map(|v| (v, (v + 1) % n)).collect();
    let ring = |scale: f64| IsingGraph::new(vec![-scale; n], vec![-scale; n], edges.clone());
    let one = best(&dev, &ring(1.0), Kernel::Sa, 2000);
    let thousand = best(&dev, &ring(1000.0), Kernel::Sa, 2000);
    assert_eq!(one, -128_000);
    assert_eq!(thousand, -128_000_000);
    assert_eq!(thousand, 1000 * one);
}

#[test]
fn a_non_representable_model_is_refused_as_capacity() {
    let Some(dev) = device() else { return };
    for graph in [
        IsingGraph::new(vec![0.0005, 0.0], vec![1.0], vec![(0, 1)]),
        IsingGraph::new(vec![f64::NAN, 0.0], vec![1.0], vec![(0, 1)]),
    ] {
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            let err = sample_ising(&dev, &graph, &params(16), kernel).unwrap_err();
            assert!(
                matches!(err, SampleError::TooLarge(_)),
                "{kernel:?}: {err:?}"
            );
        }
    }
}

#[test]
fn the_wide_path_repeats_exactly_for_a_fixed_seed() {
    let Some(dev) = device() else { return };
    let graph = quarter_qubo();
    let a = sample_ising(&dev, &graph, &params(300), Kernel::Sa).unwrap();
    let b = sample_ising(&dev, &graph, &params(300), Kernel::Sa).unwrap();
    assert_eq!(a, b);
}
