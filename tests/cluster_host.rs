// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Host admission and device layout checks. No Metal device is opened.

use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_metal::capacity::{cluster_footprint, ClusterShape, DeviceLimits};
use quip_solver_metal::cluster::{
    GraphError, Model, Motifs, PrepareError, PreparedGraph, SpinError, Weights,
};

#[derive(Clone)]
struct Fixture {
    h: Vec<i32>,
    edges: Vec<(u32, u32)>,
    j: Vec<i32>,
    tiles: Vec<[u32; 4]>,
    gauges: Vec<[i8; 4]>,
    pairs: Vec<(u32, u32)>,
}

impl Fixture {
    fn prepare(
        &self,
        weights: Weights,
        limits: &DeviceLimits,
    ) -> Result<PreparedGraph, PrepareError> {
        PreparedGraph::new(
            &Model {
                h_milli: &self.h,
                edges: &self.edges,
                j_milli: &self.j,
            },
            &Motifs {
                tiles: &self.tiles,
                gauges: &self.gauges,
                pairs: &self.pairs,
            },
            weights,
            limits,
            3,
        )
    }

    fn direct(&self, spins: &[i8]) -> i64 {
        self.edges
            .iter()
            .zip(&self.j)
            .map(|(&(u, v), &j)| {
                i64::from(j)
                    * i64::from(spins[usize::try_from(u).unwrap()])
                    * i64::from(spins[usize::try_from(v).unwrap()])
            })
            .sum()
    }
}

fn limits() -> DeviceLimits {
    DeviceLimits {
        memory_bytes: 1 << 30,
        max_buffer_bytes: 1 << 30,
    }
}

// Source: the reviewed CPU tests/otsuka_kernel.rs corner_sharing fixture.
fn corner_sharing() -> Fixture {
    let mut f = Fixture {
        h: vec![0; 8],
        edges: vec![(0, 4), (5, 7), (1, 5), (2, 6), (1, 4), (6, 7)],
        j: vec![-1000, -1000, 300, -200, 700, 500],
        tiles: vec![[0, 1, 2, 3], [3, 4, 5, 6]],
        gauges: vec![[1, -1, 1, 1], [-1, 1, 1, -1]],
        pairs: vec![(0, 4), (5, 7)],
    };
    for (tile, gauge) in f.tiles.iter().zip(&f.gauges) {
        for a in 0..4 {
            for b in a + 1..4 {
                f.edges.push((tile[a], tile[b]));
                f.j.push(1000 * i32::from(gauge[a] * gauge[b]));
            }
        }
    }
    f
}

fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
    (0..1_u32 << n).map(move |bits| {
        (0..n)
            .map(|i| if bits >> i & 1 == 0 { 1 } else { -1 })
            .collect()
    })
}

const WEIGHTS: [Weights; 5] = [
    Weights::FULL,
    Weights {
        pair_milli: 600,
        tile_milli: 1000,
    },
    Weights {
        pair_milli: 0,
        tile_milli: 1000,
    },
    Weights {
        pair_milli: 1000,
        tile_milli: 400,
    },
    Weights {
        pair_milli: 250,
        tile_milli: 1,
    },
];

// Decode the ABI independently: slot-major words, high u16 node, low i16 J.
fn ell_energy(deg: &[i32], ell: &[i32], spins: &[i8]) -> i64 {
    let mut twice = 0;
    for u in 0..spins.len() {
        for slot in 0..usize::try_from(deg[u]).unwrap() {
            let bytes = ell[slot * spins.len() + u].to_le_bytes();
            let v = usize::from(u16::from_le_bytes([bytes[2], bytes[3]]));
            let j = i16::from_le_bytes([bytes[0], bytes[1]]);
            twice += i64::from(j) * i64::from(spins[u]) * i64::from(spins[v]);
        }
    }
    assert_eq!(twice % 2, 0);
    twice / 2
}

#[test]
fn every_state_preserves_energy_in_both_orders_and_all_layers() {
    let f = corner_sharing();
    for weights in WEIGHTS {
        let p = f.prepare(weights, &limits()).unwrap();
        let d = p.layout();
        for s in states(8) {
            let device = p.permute(&s).unwrap();
            assert_eq!(p.unpermute(&device).unwrap(), s);
            assert_eq!(p.permute(&p.unpermute(&s).unwrap()).unwrap(), s);
            let total = f.direct(&s);
            assert_eq!(p.energy(&s).unwrap(), total);
            let (tile, link, residual) = p.split_energies(&s).unwrap();
            assert_eq!(tile + link, p.strong_energy(&s).unwrap());
            assert_eq!(residual, p.weak_energy(&s).unwrap());
            assert_eq!(tile + link + residual, total);
            assert_eq!(ell_energy(&d.all_deg, &d.all_ell, &device), total);
            assert_eq!(ell_energy(&d.weak_deg, &d.weak_ell, &device), residual);
            let mut strong = 0;
            for (t, (nodes, gauge)) in d
                .tiles
                .as_chunks::<4>()
                .0
                .iter()
                .zip(d.gauges.as_chunks::<4>().0.iter())
                .enumerate()
            {
                for a in 0..4 {
                    for b in a + 1..4 {
                        strong += i64::from(d.tile_w[t])
                            * i64::from(gauge[a])
                            * i64::from(gauge[b])
                            * i64::from(device[usize::try_from(nodes[a]).unwrap()])
                            * i64::from(device[usize::try_from(nodes[b]).unwrap()]);
                    }
                }
            }
            for (p, pair) in d.pairs.as_chunks::<2>().0.iter().enumerate() {
                strong -= i64::from(d.link_w[p])
                    * i64::from(device[usize::try_from(pair[0]).unwrap()])
                    * i64::from(device[usize::try_from(pair[1]).unwrap()]);
            }
            assert_eq!(strong, tile + link);
        }
    }
}

#[test]
fn strong_and_weak_match_reviewed_cpu_port_for_every_state() {
    // Frozen from the unmodified CPU graph.rs at ea67aaaa75a0cdaca04e15c9770a9b72bd92448b.
    // Each row holds CPU strong and weak energies for one weight and state.
    let expected: Vec<Vec<[i64; 2]>> =
        serde_json::from_str(include_str!("fixtures/cluster_cpu_energy.json")).unwrap();
    assert_eq!(expected.len(), WEIGHTS.len());
    for (weights, rows) in WEIGHTS.into_iter().zip(expected) {
        let p = corner_sharing().prepare(weights, &limits()).unwrap();
        assert_eq!(rows.len(), 256);
        for (s, [strong, weak]) in states(8).zip(rows) {
            assert_eq!(p.strong_energy(&s).unwrap(), strong);
            assert_eq!(p.weak_energy(&s).unwrap(), weak);
        }
    }
}

#[test]
fn malformed_family_metadata_is_an_explicit_error() {
    use quip_solver_core::ising_generators::instance::{Instance, InstanceLimits, InstanceParts};
    let instance = Instance::new(
        InstanceParts {
            descriptor_digest: [0; 32],
            topology_order_digest: [0; 32],
            nonce: None,
            node_ids: vec![0],
            edges: vec![],
            h: vec![0],
            j: vec![],
            offset_milli: 0,
            metadata: vec![0, 0],
        },
        &InstanceLimits::FAM_37,
        &[1, 2, 3],
        None,
    )
    .unwrap();
    assert_eq!(
        PreparedGraph::from_instance(&instance, &TWISTED_DIMER_V1, Weights::FULL, &limits(), 3)
            .unwrap_err(),
        PrepareError::Metadata
    );
}

#[test]
fn permutation_is_explicit_and_colors_are_proper() {
    let f = corner_sharing();
    let p = f.prepare(Weights::FULL, &limits()).unwrap();
    assert_ne!(p.device_to_original(), &(0..8).collect::<Vec<_>>());
    for (new, &old) in p.device_to_original().iter().enumerate() {
        assert_eq!(p.original_to_device()[old], new);
    }
    for (old, &new) in p.original_to_device().iter().enumerate() {
        assert_eq!(p.device_to_original()[new], old);
    }
    let mut colors = [0; 8];
    for (c, off) in p.layout().color_off.windows(2).enumerate() {
        for new in off[0]..off[1] {
            colors[p.device_to_original()[usize::try_from(new).unwrap()]] = c;
        }
    }
    for &(u, v) in &f.edges {
        assert_ne!(
            colors[usize::try_from(u).unwrap()],
            colors[usize::try_from(v).unwrap()]
        );
    }
}

#[test]
fn weak_edges_above_research_threshold_stay_weak() {
    let p = corner_sharing().prepare(Weights::FULL, &limits()).unwrap();
    assert_eq!(
        p.layout().pairs.len(),
        4,
        "the 700 and 500 milli edges are not metadata pairs"
    );
    assert_eq!(p.layout().link_w, [1000, 1000]);
    assert_eq!(p.layout().tile_w, [1000, 1000]);
    assert!(p.layout().h.iter().all(|&h| h == 0));
}

#[test]
fn edgeless_and_zero_coupling_models_keep_every_node() {
    let f = Fixture {
        h: vec![0; 3],
        edges: vec![(0, 1)],
        j: vec![0],
        tiles: vec![],
        gauges: vec![],
        pairs: vec![],
    };
    let p = f.prepare(Weights::FULL, &limits()).unwrap();
    assert!(p.layout().all_ell.is_empty());
    assert!(p.layout().weak_ell.is_empty());
    assert_eq!(p.layout().all_deg, [0, 0, 0]);
    assert_eq!(p.layout().header, [3, 0, 0, 0, 0, 2]);
    assert_eq!(p.layout().color_off.last(), Some(&3));
    for s in states(3) {
        assert_eq!(p.unpermute(&p.permute(&s).unwrap()).unwrap(), s);
        assert_eq!(p.energy(&s).unwrap(), 0);
    }
    let f = Fixture {
        edges: vec![],
        j: vec![],
        ..f
    };
    let p = f.prepare(Weights::FULL, &limits()).unwrap();
    assert_eq!(p.layout().color_off, [0, 3]);
    assert_eq!(p.original_graph().num_nodes(), 3);
}

#[test]
fn cpu_validation_errors_are_preserved() {
    let base = corner_sharing();
    let mut cases = Vec::new();
    let mut add = |error, edit: fn(&mut Fixture)| {
        let mut f = base.clone();
        edit(&mut f);
        cases.push((f, error));
    };
    add(GraphError::Empty, |f| f.h.clear());
    add(GraphError::Fields, |f| f.h[0] = 1);
    add(GraphError::Lengths, |f| {
        f.j.pop();
    });
    add(GraphError::Lengths, |f| {
        f.gauges.pop();
    });
    add(GraphError::Edge { index: 0 }, |f| f.edges[0].0 = 8);
    add(GraphError::Edge { index: 0 }, |f| {
        f.edges[0].0 = f.edges[0].1;
    });
    add(GraphError::Edge { index: 0 }, |f| f.j[0] = i32::MIN);
    add(GraphError::DuplicateEdge { index: 1 }, |f| {
        f.edges[1] = (4, 0);
    });
    add(GraphError::Tile { tile: 0 }, |f| f.tiles[0][0] = 8);
    add(GraphError::Tile { tile: 0 }, |f| f.tiles[0][0] = 1);
    add(GraphError::Gauge { tile: 0 }, |f| f.gauges[0][0] = 0);
    add(GraphError::TileCoupling { tile: 0 }, |f| f.j[6] *= -1);
    add(GraphError::TileCoupling { tile: 0 }, |f| {
        f.j.remove(6);
        f.edges.remove(6);
    });
    add(GraphError::TileOverlap { tile: 1 }, |f| {
        f.tiles[1] = f.tiles[0];
        f.gauges[1] = f.gauges[0];
    });
    add(GraphError::Pair { pair: 0 }, |f| f.pairs[0] = (8, 0));
    add(GraphError::Pair { pair: 0 }, |f| f.pairs[0] = (0, 0));
    add(GraphError::Pair { pair: 1 }, |f| f.pairs[1] = f.pairs[0]);
    add(GraphError::Pair { pair: 0 }, |f| f.j[0] = -999);
    add(GraphError::Pair { pair: 0 }, |f| f.pairs[0] = (0, 1));
    add(GraphError::StrongEdge { index: 4 }, |f| f.j[4] = 1000);
    for (f, want) in cases {
        assert_eq!(
            f.prepare(Weights::FULL, &limits()).unwrap_err(),
            PrepareError::Graph(want)
        );
    }
    for weights in [
        Weights {
            pair_milli: -1,
            tile_milli: 1,
        },
        Weights {
            pair_milli: 1001,
            tile_milli: 1,
        },
        Weights {
            pair_milli: 1,
            tile_milli: 0,
        },
        Weights {
            pair_milli: 1,
            tile_milli: 1001,
        },
    ] {
        assert_eq!(
            base.prepare(weights, &limits()).unwrap_err(),
            PrepareError::Graph(GraphError::Weights)
        );
    }
}

#[test]
fn capacity_checks_exact_device_boundary() {
    let f = corner_sharing();
    let p = f.prepare(Weights::FULL, &limits()).unwrap();
    let mut l = limits();
    // The fixture launches three anneals in one batch.
    let total = p.footprint().total(3, 1);
    l.memory_bytes = total.div_ceil(4) * 5;
    f.prepare(Weights::FULL, &l).unwrap();
    l.memory_bytes -= 5;
    assert_eq!(
        f.prepare(Weights::FULL, &l).unwrap_err(),
        PrepareError::DeviceMemory {
            requested: total,
            budget: l.usable_bytes(),
        }
    );
}

#[test]
fn packed_node_bound_is_checked_before_building_tables() {
    let mut f = Fixture {
        h: vec![0; 65_537],
        edges: vec![],
        j: vec![],
        tiles: vec![],
        gauges: vec![],
        pairs: vec![],
    };
    assert!(matches!(
        f.prepare(Weights::FULL, &limits()),
        Err(PrepareError::IndexOverflow { .. })
    ));
    f.h.pop();
    f.edges.push((0, 65_535));
    f.j.push(-999);
    let p = f.prepare(Weights::FULL, &limits()).unwrap();
    assert_eq!(p.layout().header[0], 65_536);
    let words = &p.layout().all_ell;
    assert!(words
        .iter()
        .any(|&w| (u32::from_ne_bytes(w.to_ne_bytes()) >> 16) == 65_535));
}

#[test]
fn footprint_counts_padded_tables_weights_state_and_rng() {
    let shape = ClusterShape {
        nodes: 8,
        tiles: 2,
        links: 2,
        weak_width: 3,
        all_width: 6,
        colors: 4,
    };
    let f = cluster_footprint(shape);
    // 32 tile + 8 gauge + 16 pair + 64 degree + 288 ELL + 20 colors
    // + 32 field + 8 tile weight + 8 link weight.
    assert_eq!(f.shared_bytes, 476);
    // parent and acc (8 bytes) plus one spin byte per node, then the
    // per-thread random state.
    assert_eq!(f.per_instance_bytes, 8 * 9 + 512 * 8);
    assert_eq!(f.total(3, 1), 476 + 3 * (8 * 9 + 512 * 8));
    let huge = ClusterShape {
        nodes: usize::MAX,
        ..shape
    };
    assert_eq!(cluster_footprint(huge).total(usize::MAX, 1), usize::MAX);
}

#[test]
fn bad_spin_states_and_empty_launches_return_errors() {
    let f = corner_sharing();
    let p = f.prepare(Weights::FULL, &limits()).unwrap();
    for state in [&[1_i8; 7][..], &[0_i8; 8][..]] {
        let error = if state.len() == 7 {
            SpinError::Length
        } else {
            SpinError::Value { node: 0 }
        };
        assert_eq!(p.permute(state).unwrap_err(), error);
        assert_eq!(p.unpermute(state).unwrap_err(), error);
        assert_eq!(p.energy(state).unwrap_err(), error);
        assert_eq!(p.strong_energy(state).unwrap_err(), error);
        assert_eq!(p.weak_energy(state).unwrap_err(), error);
        assert_eq!(p.split_energies(state).unwrap_err(), error);
    }
    assert_eq!(
        PreparedGraph::new(
            &Model {
                h_milli: &[0],
                edges: &[],
                j_milli: &[]
            },
            &Motifs {
                tiles: &[],
                gauges: &[],
                pairs: &[]
            },
            Weights::FULL,
            &limits(),
            0
        )
        .unwrap_err(),
        PrepareError::EmptyLaunch
    );
}

#[test]
fn canonical_generator_instance_prepares_and_rescores_every_state() {
    // Source: core 0.0.8 canonical twisted-dimer generator, adjacency mode.
    // Stored conformance vectors start at 80 nodes, too large to enumerate.
    let edges: Vec<_> = (0..4)
        .flat_map(|u| (u + 1..4).map(move |v| (u, v)))
        .collect();
    let topology = BaseTopology::new(
        vec![17, 2, 99, 8],
        &edges
            .iter()
            .map(|&(u, v)| ([17, 2, 99, 8][u], [17, 2, 99, 8][v]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut params = quip_solver_conformance::families::twisted::P_TWISTED_ADJ_W45_P6;
    params[30..34].copy_from_slice(&0_u32.to_le_bytes());
    let instance = quip_solver_conformance::families::export_instances(
        &TWISTED_DIMER_V1,
        &params,
        &topology,
        &[[4; 32]],
    )
    .unwrap()
    .remove(0)
    .unwrap();
    let dense_edges: Vec<_> = instance
        .edges()
        .iter()
        .map(|&(u, v)| (usize::try_from(u).unwrap(), usize::try_from(v).unwrap()))
        .collect();
    for weights in WEIGHTS {
        let p = PreparedGraph::from_instance(&instance, &TWISTED_DIMER_V1, weights, &limits(), 3)
            .unwrap();
        assert_eq!(p.layout().tiles.len(), 4);
        for s in states(4) {
            let direct = quip_solver_core::quip_protocol::scoring::energy_from_milli(
                &s,
                instance.h(),
                instance.j(),
                &dense_edges,
            );
            assert_eq!(p.energy(&s).unwrap(), direct);
            let (t, l, r) = p.split_energies(&s).unwrap();
            assert_eq!(t + l + r, direct);
        }
    }
    let unsupported = quip_solver_core::ising_generators::descriptor::Contract {
        revision: 2,
        ..TWISTED_DIMER_V1
    };
    assert_eq!(
        PreparedGraph::from_instance(&instance, &unsupported, Weights::FULL, &limits(), 3)
            .unwrap_err(),
        PrepareError::UnsupportedFamily
    );
}
