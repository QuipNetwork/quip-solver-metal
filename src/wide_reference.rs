// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Integer host model of the wide SA and node-parallel Gibbs kernels.

use quip_solver_core::{IsingGraph, SampleParams};

use crate::metal_device::MetalDevice;
use crate::sampler::{build_beta_schedule, sample_ising, Kernel};
use crate::topology::{fill_h_j_wide, SelfFeedingTopology};
use crate::wide_tables::{heat_bath_rows, metropolis_rows, WIDE_ROW};

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn splitmix32(z: &mut u32) -> u32 {
    *z = z.wrapping_add(0x9e37_79b9);
    let mut r = *z;
    r = (r ^ (r >> 16)).wrapping_mul(0x85eb_ca6b);
    r = (r ^ (r >> 13)).wrapping_mul(0xc2b2_ae35);
    r ^ (r >> 16)
}

struct Xoshiro([u32; 4]);

impl Xoshiro {
    fn seeded(seed: u32) -> Self {
        let mut z = seed;
        Self([
            splitmix32(&mut z),
            splitmix32(&mut z),
            splitmix32(&mut z),
            splitmix32(&mut z),
        ])
    }

    fn next(&mut self) -> u32 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(11);
        result
    }
}

struct Csr {
    row: Vec<i32>,
    col: Vec<i32>,
    j: Vec<i32>,
    h: Vec<i32>,
}

fn csr(graph: &IsingGraph) -> (SelfFeedingTopology, Csr) {
    let topo = SelfFeedingTopology::build(graph);
    let (j, h) = fill_h_j_wide(&topo, graph);
    let csr = Csr {
        row: topo.row_ptr.clone(),
        col: topo.col_ind.clone(),
        j,
        h,
    };
    (topo, csr)
}

fn field(c: &Csr, var: usize, spins: &[i32]) -> i32 {
    let (a, b) = (c.row[var] as usize, c.row[var + 1] as usize);
    (a..b).fold(c.h[var], |acc, p| acc + c.j[p] * spins[c.col[p] as usize])
}

pub(crate) fn reference_sa(graph: &IsingGraph, params: &SampleParams) -> Vec<Vec<i8>> {
    let (_, c) = csr(graph);
    let n = graph.h.len();
    let (betas, sweeps_per) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let rows = metropolis_rows(&betas, params.seed);
    let base = (params.seed as u32).wrapping_add(1);
    (0..params.num_reads as u32)
        .map(|thread_id| {
            let mut rng = (if base == 0 { 1 } else { base }) ^ thread_id.wrapping_mul(12345);
            if rng == 0 {
                rng = 1;
            }
            let mut s: Vec<i32> = (0..n)
                .map(|_| if xorshift32(&mut rng) & 1 == 1 { -1 } else { 1 })
                .collect();
            let mut de: Vec<i32> = (0..n).map(|v| -2 * s[v] * field(&c, v, &s)).collect();
            for row in rows.as_chunks::<WIDE_ROW>().0 {
                for _ in 0..sweeps_per {
                    for var in 0..n {
                        let d = de[var];
                        let flip =
                            d <= 0 || d <= row[(xorshift32(&mut rng) as usize) & (WIDE_ROW - 1)];
                        if flip {
                            let m = 4 * s[var];
                            for p in c.row[var] as usize..c.row[var + 1] as usize {
                                let nb = c.col[p] as usize;
                                de[nb] += m * c.j[p] * s[nb];
                            }
                            s[var] = -s[var];
                            de[var] = -d;
                        }
                    }
                }
            }
            s.into_iter().map(|v| v as i8).collect()
        })
        .collect()
}

pub(crate) fn reference_gibbs_parallel(
    graph: &IsingGraph,
    params: &SampleParams,
    group: usize,
) -> Vec<Vec<i8>> {
    let (topo, c) = csr(graph);
    let colors = &topo.colors;
    let n = graph.h.len();
    let (betas, sweeps_per) = build_beta_schedule(
        graph,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );
    let rows = heat_bath_rows(&betas, params.seed);
    let base = (params.seed as u32).wrapping_add(1);
    let base = if base == 0 { 1 } else { base };
    (0..params.num_reads as u32)
        .map(|sample_id| {
            let mut rngs: Vec<Xoshiro> = (0..group as u32)
                .map(|t| {
                    Xoshiro::seeded(
                        base ^ sample_id.wrapping_mul(2_654_435_761)
                            ^ t.wrapping_mul(2_246_822_519),
                    )
                })
                .collect();
            let mut s = vec![0_i32; n];
            for (t, rng) in rngs.iter_mut().enumerate() {
                for var in (t..n).step_by(group) {
                    s[var] = if rng.next() & 1 == 1 { -1 } else { 1 };
                }
            }
            for row in rows.as_chunks::<WIDE_ROW>().0 {
                for _ in 0..sweeps_per {
                    for color in 0..colors.num_colors as usize {
                        let start = colors.starts[color] as usize;
                        let count = colors.counts[color] as usize;
                        for (t, rng) in rngs.iter_mut().enumerate() {
                            for i in (t..count).step_by(group) {
                                let var = colors.nodes[start + i] as usize;
                                let f = field(&c, var, &s);
                                let r = rng.next() as usize & (WIDE_ROW - 1);
                                s[var] = if f <= row[r] { 1 } else { -1 };
                            }
                        }
                    }
                }
            }
            s.into_iter().map(|v| v as i8).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quarter_model() -> IsingGraph {
        let n = 12;
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for v in 0..n {
            edges.push((v, (v + 1) % n));
            j.push(if v % 3 == 0 { -0.75 } else { 0.25 });
            if v % 4 == 0 {
                edges.push((v, (v + 5) % n));
                j.push(0.5);
            }
        }
        let h = (0..n)
            .map(|v| if v % 2 == 0 { 0.125 } else { -0.375 })
            .collect();
        IsingGraph::new(h, j, edges)
    }

    fn params() -> SampleParams {
        SampleParams {
            num_reads: 64,
            num_sweeps: 48,
            sweeps_per_beta: 2,
            seed: 0x5EED,
            beta_range: Some((0.1, 4.0)),
            ..Default::default()
        }
    }

    #[test]
    fn wide_sa_matches_the_host_reference_bit_for_bit() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        let graph = quarter_model();
        let got: Vec<Vec<i8>> = sample_ising(&dev, &graph, &params(), Kernel::Sa)
            .unwrap()
            .into_iter()
            .map(|r| r.spins)
            .collect();
        assert_eq!(got, reference_sa(&graph, &params()));
    }

    #[test]
    fn wide_gibbs_matches_the_host_reference_bit_for_bit() {
        if MetalDevice::device_count() == 0 || !crate::sampler::gibbs_node_parallel() {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        let graph = quarter_model();
        let got: Vec<Vec<i8>> = sample_ising(&dev, &graph, &params(), Kernel::Gibbs)
            .unwrap()
            .into_iter()
            .map(|r| r.spins)
            .collect();
        let want = reference_gibbs_parallel(&graph, &params(), crate::sampler::WIDE_GIBBS_GROUP);
        assert_eq!(got, want);
    }

    #[test]
    fn wide_gibbs_sequential_dispatches_when_selected() {
        if MetalDevice::device_count() == 0 || crate::sampler::gibbs_node_parallel() {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        let graph = quarter_model();
        let got = sample_ising(&dev, &graph, &params(), Kernel::Gibbs).unwrap();
        assert_eq!(got.len(), params().num_reads);
        assert!(got.iter().all(|read| read.spins.len() == graph.h.len()));
    }
}
