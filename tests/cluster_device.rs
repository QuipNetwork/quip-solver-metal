// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Cluster anneal on the Metal device: the host reference, chunking, the final
//! descent, and an Advantage2-size instance. Ports of the CUDA device tests.

use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_metal::capacity::DeviceLimits;
use quip_solver_metal::cluster::device::{ClusterDevice, ClusterDeviceError, ClusterParams};
use quip_solver_metal::cluster::{Model, Motifs, PreparedGraph, Weights};

#[path = "support/cluster_reference.rs"]
mod reference;

const WEIGHTS: [Weights; 3] = [
    Weights::FULL,
    Weights {
        tile_milli: 250,
        pair_milli: 250,
    },
    Weights {
        tile_milli: 1,
        pair_milli: 0,
    },
];

fn limits() -> DeviceLimits {
    DeviceLimits {
        memory_bytes: 1 << 30,
        max_buffer_bytes: 1 << 30,
    }
}

fn device() -> ClusterDevice {
    ClusterDevice::open(0).unwrap()
}

// The corner-sharing fixture, with nontrivial gauges and residual edges.
fn corner_sharing(weights: Weights) -> PreparedGraph {
    let tiles = [[0, 1, 2, 3], [3, 4, 5, 6]];
    let gauges = [[1, -1, 1, 1], [-1, 1, 1, -1]];
    let mut edges = vec![(0, 4), (5, 7), (1, 5), (2, 6), (1, 4), (6, 7)];
    let mut j = vec![-1000, -1000, 300, -200, 700, 500];
    for (tile, gauge) in tiles.iter().zip(gauges) {
        for a in 0..4 {
            for b in a + 1..4 {
                edges.push((tile[a], tile[b]));
                j.push(1000 * i32::from(gauge[a] * gauge[b]));
            }
        }
    }
    PreparedGraph::new(
        &Model {
            h_milli: &[0; 8],
            edges: &edges,
            j_milli: &j,
        },
        &Motifs {
            tiles: &tiles,
            gauges: &gauges,
            pairs: &[(0, 4), (5, 7)],
        },
        weights,
        &limits(),
        32,
    )
    .unwrap()
}

/// A canonical twisted-ice-dimer instance on the 4,577-node Advantage2 working
/// graph, in adjacency mode with `weak_milli` 62 and 256 pairs.
fn advantage2_instance() -> Instance {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for line in include_str!("fixtures/advantage2-system1.edges").lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut ends = line.split_whitespace().map(|x| x.parse::<u32>().unwrap());
        edges.push((ends.next().unwrap(), ends.next().unwrap()));
    }
    nodes.extend(0..4577_u32);
    let topology = BaseTopology::new(nodes, &edges).unwrap();
    let mut params = quip_solver_conformance::families::twisted::P_TWISTED_ADJ_W45_P6;
    params[22..24].copy_from_slice(&62_u16.to_le_bytes());
    params[30..34].copy_from_slice(&256_u32.to_le_bytes());
    quip_solver_conformance::families::export_instances(
        &TWISTED_DIMER_V1,
        &params,
        &topology,
        &[[9; 32]],
    )
    .unwrap()
    .remove(0)
    .unwrap()
}

fn advantage2(anneals: usize) -> PreparedGraph {
    PreparedGraph::from_instance(
        &advantage2_instance(),
        &TWISTED_DIMER_V1,
        Weights {
            tile_milli: 250,
            pair_milli: 250,
        },
        &limits(),
        anneals,
    )
    .unwrap()
}

/// Every single flip of every state leaves the energy no lower.
fn assert_local_minima(prepared: &PreparedGraph, states: &[Vec<i8>]) {
    for spins in states {
        let energy = prepared.energy(spins).unwrap();
        let mut moved = spins.clone();
        for node in 0..spins.len() {
            moved[node] *= -1;
            assert!(
                prepared.energy(&moved).unwrap() >= energy,
                "flipping node {node} lowers the energy"
            );
            moved[node] *= -1;
        }
    }
}

// Missing annealing, wrong residual signs, lost RNG, and omitted tail sweeps
// must disagree before final descent can hide their effects.
#[test]
fn sweeps_match_the_host_reference_before_final_descent() {
    let device = device();
    for weights in WEIGHTS {
        let p = corner_sharing(weights);
        for (sweeps, single_every, knee) in [(1, 0, false), (2, 1, false), (7, 2, true)] {
            for seed in [1, u64::MAX] {
                let q = ClusterParams {
                    sweeps,
                    single_every,
                    rounds: 3,
                    beta_min: 0.3,
                    beta_max: 4.0,
                    beta_knee: if knee { 1.2 } else { 0.0 },
                    knee_frac: if knee { 0.4 } else { 0.0 },
                    anneals: 4,
                    seed,
                    ladder: None,
                };
                let (expected, rng) = reference::run(&p, &q);
                let actual = device
                    .trace_for_test(&p, &q, false, None, &|| false)
                    .unwrap();
                assert_eq!(
                    actual.spins, expected,
                    "spins: {q:?}, {weights:?}, seed {seed}"
                );
                assert_eq!(actual.rng, rng, "RNG: {q:?}, {weights:?}, seed {seed}");
            }
        }
    }
}

// The same comparison where union-find is contended: the Advantage2 instance
// has thousands of tiles that share nodes, spread over all 512 threads. Each
// `(sweeps, single_every, rounds, seed)` keeps every reference decision more
// than 1e-5 from its threshold; most other seeds trip the reference's
// near-tie guard, which is not a mismatch.
#[test]
fn sweeps_match_the_host_reference_at_4577_nodes() {
    let device = device();
    let p = advantage2(4);
    for (sweeps, single_every, rounds, seed) in [
        (1, 0, 2, 2),
        (1, 0, 2, 4),
        (3, 1, 2, 4),
        (3, 1, 2, 5),
        (2, 1, 3, 1),
        (2, 2, 3, 4),
        (4, 2, 3, 33),
    ] {
        let q = ClusterParams {
            sweeps,
            single_every,
            rounds,
            beta_min: 0.3,
            beta_max: 4.0,
            anneals: 4,
            seed,
            ..ClusterParams::default()
        };
        let (expected, rng) = reference::run(&p, &q);
        let actual = device
            .trace_for_test(&p, &q, false, None, &|| false)
            .unwrap();
        assert_eq!(actual.spins, expected, "spins: {q:?}");
        assert_eq!(actual.rng, rng, "RNG: {q:?}");
    }
}

#[test]
fn chunked_and_unchunked_anneals_match() {
    let device = device();
    for weights in WEIGHTS {
        let p = corner_sharing(weights);
        let one_sweep = ClusterParams {
            sweeps: 32,
            rounds: 3,
            beta_knee: 3.0,
            knee_frac: 0.4,
            anneals: 32,
            seed: 7,
            ..ClusterParams::default()
        };
        // One sweep per command buffer, and one command buffer.
        let trace = |rate, descent| {
            device
                .trace_for_test(&p, &one_sweep, descent, Some(rate), &|| false)
                .unwrap()
        };
        let chunked = trace(1.0, false);
        let unchunked = trace(1e18, false);
        assert_eq!(chunked.chunks.len(), 32);
        assert!(chunked.chunks.iter().all(|c| c.sweeps == 1));
        assert_eq!(unchunked.chunks.len(), 1);
        assert_eq!(unchunked.chunks[0].sweeps, 32);
        assert_eq!(chunked.spins, unchunked.spins);
        assert_eq!(chunked.rng, unchunked.rng);
        // With the final descent, and through the production entry point.
        let a = trace(1.0, true).spins;
        assert_eq!(a, trace(1e18, true).spins);
        assert_eq!(a, device.anneal(&p, &one_sweep, &|| false).unwrap());
        assert_eq!(a, device.anneal(&p, &one_sweep, &|| false).unwrap());
    }
}

/// A ladder run keeps each command buffer inside one rung at that rung's
/// beta. One sweep per command buffer and one command buffer per rung give
/// the same state. A one-rung ladder equals the kernel schedule at that
/// constant beta.
#[test]
fn ladder_runs_match_across_chunking_and_a_constant_schedule() {
    use quip_solver_metal::cluster::device::BetaLadder;
    let device = device();
    for weights in WEIGHTS {
        let p = corner_sharing(weights);
        let ladder = ClusterParams {
            sweeps: 12,
            rounds: 3,
            anneals: 16,
            seed: 11,
            ladder: Some(BetaLadder {
                betas: vec![0.4, 1.1, 2.5],
                sweeps_per_beta: 4,
            }),
            ..ClusterParams::default()
        };
        let trace = |q: &ClusterParams, rate| {
            device
                .trace_for_test(&p, q, false, Some(rate), &|| false)
                .unwrap()
        };
        let chunked = trace(&ladder, 1.0);
        let per_rung = trace(&ladder, 1e18);
        assert_eq!(chunked.chunks.len(), 12);
        assert_eq!(
            per_rung.chunks.iter().map(|c| c.sweeps).collect::<Vec<_>>(),
            [4, 4, 4]
        );
        assert_eq!(chunked.spins, per_rung.spins);
        assert_eq!(chunked.rng, per_rung.rng);
        let one_rung = ClusterParams {
            sweeps: 4,
            ladder: Some(BetaLadder {
                betas: vec![1.1],
                sweeps_per_beta: 4,
            }),
            ..ladder.clone()
        };
        let constant = ClusterParams {
            beta_min: 1.1,
            beta_max: 1.1,
            ladder: None,
            ..one_rung.clone()
        };
        let a = trace(&one_rung, 1e18);
        let b = trace(&constant, 1e18);
        assert_eq!(a.spins, b.spins);
        assert_eq!(a.rng, b.rng);
        // A different rung order changes the anneal.
        let reversed = ClusterParams {
            ladder: Some(BetaLadder {
                betas: vec![2.5, 1.1, 0.4],
                sweeps_per_beta: 4,
            }),
            ..ladder.clone()
        };
        assert_ne!(trace(&reversed, 1e18).rng, per_rung.rng);
    }
}

#[test]
fn the_final_descent_reaches_a_local_minimum() {
    // Tiny anneals reach the exhaustive ground state.
    let device = device();
    for weights in WEIGHTS {
        let p = corner_sharing(weights);
        let ground = (0..256_u32)
            .map(|bits| {
                let s: Vec<i8> = (0..8)
                    .map(|i| if bits >> i & 1 == 0 { 1 } else { -1 })
                    .collect();
                p.energy(&s).unwrap()
            })
            .min()
            .unwrap();
        let q = ClusterParams {
            sweeps: 32,
            anneals: 32,
            seed: 1,
            ..ClusterParams::default()
        };
        let states = device.anneal(&p, &q, &|| false).unwrap();
        assert_eq!(states.len(), 32);
        assert!(states.iter().all(|s| s.len() == 8));
        assert_local_minima(&p, &states);
        assert_eq!(
            states.iter().map(|s| p.energy(s).unwrap()).min(),
            Some(ground)
        );
    }
    // More than ten strides of the 512-thread node loops, an odd spin count,
    // a knee, and no periodic single-spin sweeps: only the descent relaxes.
    let mut edges = vec![(4, 5500), (3, 5500)];
    let mut couplings = vec![-1000, 300];
    for first in 0..4 {
        for second in first + 1..4 {
            edges.push((first, second));
            couplings.push(1000);
        }
    }
    let mut weak = 17_u32;
    for u in 5..5500_u32 {
        edges.push((u, u + 1));
        weak = weak.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        couplings.push(if weak >> 16 & 1 == 0 { 300 } else { -300 });
    }
    let p = PreparedGraph::new(
        &Model {
            h_milli: &vec![0; 5501],
            edges: &edges,
            j_milli: &couplings,
        },
        &Motifs {
            tiles: &[[0, 1, 2, 3]],
            gauges: &[[1; 4]],
            pairs: &[(4, 5500)],
        },
        Weights {
            tile_milli: 250,
            pair_milli: 250,
        },
        &limits(),
        4,
    )
    .unwrap();
    let q = ClusterParams {
        sweeps: 5,
        single_every: 0,
        rounds: 3,
        beta_knee: 3.0,
        knee_frac: 0.4,
        anneals: 4,
        seed: u64::MAX,
        ..ClusterParams::default()
    };
    let before = device
        .trace_for_test(&p, &q, false, Some(1.0), &|| false)
        .unwrap();
    let after = device
        .trace_for_test(&p, &q, true, Some(1.0), &|| false)
        .unwrap();
    assert_eq!(after.spins.len(), 4);
    assert_ne!(before.spins, after.spins, "the descent changed nothing");
    assert_local_minima(&p, &after.spins);
    // Descent passes follow the five sweeps, one command buffer each.
    assert!(after.chunks.len() > 5);
    assert!(after.chunks[5..].iter().all(|c| c.sweeps == 0));
    assert_eq!(after.spins, device.anneal(&p, &q, &|| false).unwrap());
}

#[test]
#[cfg_attr(
    not(feature = "long-tests"),
    ignore = "over 10 s in a debug build: run with --features long-tests"
)]
fn a_4577_node_instance_runs() {
    let device = device();
    let p = advantage2(8);
    assert_eq!(p.original_graph().num_nodes(), 4577);
    let q = ClusterParams {
        sweeps: 64,
        anneals: 8,
        seed: 3,
        ..ClusterParams::default()
    };
    let states = device.anneal(&p, &q, &|| false).unwrap();
    assert_eq!(states.len(), 8);
    assert!(states
        .iter()
        .all(|s| s.len() == 4577 && s.iter().all(|&v| v == 1 || v == -1)));
    assert_local_minima(&p, &states);
    // Cancellation before the first command buffer returns no reads.
    assert!(matches!(
        device.anneal(&p, &q, &|| true),
        Err(ClusterDeviceError::Cancelled)
    ));
}

#[test]
fn stop_between_command_buffers_cancels() {
    let device = device();
    let p = advantage2(8);
    let q = ClusterParams {
        sweeps: 1024,
        anneals: 8,
        seed: 3,
        ..ClusterParams::default()
    };
    // `stop` runs once before allocation and once before each command
    // buffer: `sweep_chunks` sweep windows, then the descent passes.
    let calls = std::cell::Cell::new(0_usize);
    let counted = || {
        calls.set(calls.get() + 1);
        false
    };
    let full = device.trace_for_test(&p, &q, true, None, &counted).unwrap();
    let sweep_chunks = full.chunks.iter().filter(|c| c.sweeps > 0).count();
    let passes = full.chunks.len() - sweep_chunks;
    assert!(sweep_chunks >= 2, "{sweep_chunks} sweep chunks");
    assert!(passes >= 2, "{passes} descent passes");
    assert_eq!(calls.get(), 1 + full.chunks.len());
    // True before the second sweep chunk, then before the second pass.
    for stop_at in [3, 1 + sweep_chunks + 2] {
        let calls = std::cell::Cell::new(0_usize);
        let stop = || {
            calls.set(calls.get() + 1);
            calls.get() == stop_at
        };
        assert!(
            matches!(
                device.anneal(&p, &q, &stop),
                Err(ClusterDeviceError::Cancelled)
            ),
            "stop at call {stop_at}"
        );
        assert_eq!(calls.get(), stop_at);
    }
}

#[test]
fn the_two_word_generator_has_full_period() {
    // The step is linear over GF(2) on 64 state bits. It has period 2^64 - 1
    // when its matrix T satisfies T^(2^64-1) = I and T^((2^64-1)/p) != I for
    // each prime p of 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417.
    type Matrix = [u64; 64];
    fn apply(m: &Matrix, v: u64) -> u64 {
        (0..64).fold(
            0,
            |out, bit| {
                if v >> bit & 1 == 1 {
                    out ^ m[bit]
                } else {
                    out
                }
            },
        )
    }
    fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
        // Column `bit` of a * b is a applied to column `bit` of b.
        std::array::from_fn(|bit| apply(a, b[bit]))
    }
    fn power(m: &Matrix, mut e: u64) -> Matrix {
        let mut result: Matrix = std::array::from_fn(|bit| 1 << bit);
        let mut base = *m;
        while e > 0 {
            if e & 1 == 1 {
                result = multiply(&result, &base);
            }
            base = multiply(&base, &base);
            e >>= 1;
        }
        result
    }
    let step: Matrix = std::array::from_fn(|bit| {
        let v = 1_u64 << bit;
        let mut s = [v as u32, (v >> 32) as u32];
        reference::next(&mut s);
        u64::from(s[0]) | u64::from(s[1]) << 32
    });
    let identity: Matrix = std::array::from_fn(|bit| 1 << bit);
    assert_eq!(power(&step, u64::MAX), identity);
    for p in [3, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(
            power(&step, u64::MAX / p),
            identity,
            "order divides 2^64-1 / {p}"
        );
    }
}

/// Measure GPU time per command buffer at the Advantage2 size, with the
/// production chunk plan. Run:
/// `cargo test --release --test cluster_device -- --ignored --nocapture cluster_chunks`
#[test]
#[ignore = "measurement: several seconds of GPU time per anneal count"]
#[expect(clippy::print_stderr, reason = "report measured chunk times")]
fn cluster_chunks_stay_under_the_ceiling() {
    let device = device();
    let mut worst = 0.0_f64;
    for anneals in [1, 8, 32, 64, 128, 256] {
        let p = advantage2(anneals);
        let q = ClusterParams {
            sweeps: 4096,
            anneals,
            seed: 5,
            ..ClusterParams::default()
        };
        let trace = device
            .trace_for_test(&p, &q, true, None, &|| false)
            .unwrap();
        let sweeps: Vec<_> = trace.chunks.iter().filter(|c| c.sweeps > 0).collect();
        let max = sweeps.iter().map(|c| c.gpu_ms).fold(0.0, f64::max);
        let ratio = sweeps
            .iter()
            .map(|c| c.gpu_ms / c.planned_ms)
            .fold(0.0, f64::max);
        let descent = trace
            .chunks
            .iter()
            .filter(|c| c.sweeps == 0)
            .map(|c| c.gpu_ms)
            .fold(0.0, f64::max);
        eprintln!(
            "anneals {anneals}: {} sweep chunks of {} sweeps, max {max:.1} ms, \
             measured/planned {ratio:.3}, {} descent passes, max {descent:.2} ms",
            sweeps.len(),
            sweeps[0].sweeps,
            trace.chunks.len() - sweeps.len(),
        );
        worst = worst.max(max).max(descent);
    }
    assert!(worst < 400.0, "a chunk measured {worst:.1} ms");
}

/// Wall time of one production family job: 256 reads at 2,048 sweeps, one
/// sweep per beta (the session default), on the Advantage2 instance with
/// full weights, as `quip-metal-otsuka` runs it.
#[test]
#[ignore = "measurement: several seconds of GPU time"]
#[expect(clippy::print_stderr, reason = "report the measured job time")]
fn a_production_family_job_time() {
    let device = device();
    let instance = advantage2_instance();
    let graph = quip_solver_core::IsingGraph::new(
        instance
            .h()
            .iter()
            .map(|&h| f64::from(h) / 1000.0)
            .collect(),
        instance
            .j()
            .iter()
            .map(|&j| f64::from(j) / 1000.0)
            .collect(),
        instance
            .edges()
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
    );
    let params = quip_solver_core::SampleParams {
        num_reads: 256,
        num_sweeps: 2048,
        seed: 3,
        ..Default::default()
    };
    let q = quip_solver_metal::cluster::sampler::cluster_params(&graph, &params).unwrap();
    let p =
        PreparedGraph::from_instance(&instance, &TWISTED_DIMER_V1, Weights::FULL, &limits(), 256)
            .unwrap();
    let started = std::time::Instant::now();
    let trace = device
        .trace_for_test(&p, &q, true, None, &|| false)
        .unwrap();
    let wall = started.elapsed().as_secs_f64();
    let gpu: f64 = trace.chunks.iter().map(|c| c.gpu_ms).sum::<f64>() / 1000.0;
    let sweeps = trace.chunks.iter().filter(|c| c.sweeps > 0).count();
    eprintln!(
        "256 reads, 2048 sweeps: {} command buffers ({sweeps} sweep, {} descent), \
         wall {wall:.2} s, GPU {gpu:.2} s, {:.3} ms wall per buffer",
        trace.chunks.len(),
        trace.chunks.len() - sweeps,
        wall * 1000.0 / trace.chunks.len() as f64,
    );
}
