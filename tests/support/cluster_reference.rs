// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Independent serial port of the Metal `cluster_anneal` sweeps, from the CUDA
//! reference `tests/support/cluster_reference.rs` (gpucluster.cu:76-106,
//! 113-136,160-256,309-319). The generator is the two-word 32-bit xorshift of
//! `kernels/cluster.metal`, not the 64-bit xorshift of the CUDA kernel.
//! Thread ownership controls draw order. Components use graph traversal rather
//! than the device union-find, so the oracle does not share its implementation.

// Preserve the research source's unsigned truncation, signed packed decoding,
// and int-to-float rounding. Preparation bounds all indices and counts to i32.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the oracle mirrors the kernel's 32-bit casts"
)]

use quip_solver_metal::cluster::device::ClusterParams;
use quip_solver_metal::cluster::PreparedGraph;

const THREADS: usize = 512;

/// One step of the two-word xorshift: Marsaglia 2003, triple (10, 13, 10).
pub(crate) fn next(s: &mut [u32; 2]) -> u32 {
    let t = s[0] ^ (s[0] << 10);
    s[0] = s[1];
    s[1] = (s[1] ^ (s[1] >> 10)) ^ (t ^ (t >> 13));
    s[1]
}

fn uniform(s: &mut [u32; 2]) -> f32 {
    (next(s) >> 8) as f32 / 16_777_216.0
}

fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn xs64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

// gpucluster.cu:311-319, independently expanded. Each splitmix64 value splits
// into its low word and its high word.
fn initialize(n: usize, anneals: usize, seed: u64) -> (Vec<i8>, Vec<[u32; 2]>) {
    let mut x = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(0x0123_4567);
    let rng = (0..anneals * THREADS)
        .map(|_| {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let z = (z ^ (z >> 31)) | 1;
            [z as u32, (z >> 32) as u32]
        })
        .collect();
    let mut y = seed ^ 0xd1b5_4a32_d192_ed03;
    let spins = (0..n * anneals)
        .map(|_| if (xs64(&mut y) >> 33) & 1 == 1 { 1 } else { -1 })
        .collect();
    (spins, rng)
}

fn components(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut adjacent = vec![Vec::new(); n];
    for &(a, b) in edges {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut roots = vec![usize::MAX; n];
    for root in 0..n {
        if roots[root] != usize::MAX {
            continue;
        }
        roots[root] = root;
        let mut todo = vec![root];
        while let Some(u) = todo.pop() {
            for &v in &adjacent[u] {
                if roots[v] == usize::MAX {
                    roots[v] = root;
                    todo.push(v);
                }
            }
        }
    }
    roots
}

fn beta_at(q: &ClusterParams, sweep: usize) -> f32 {
    let f = if q.sweeps > 1 {
        sweep as f32 / (q.sweeps - 1) as f32
    } else {
        1.0
    };
    if q.beta_knee > 0.0 && q.knee_frac > 0.0 && q.knee_frac < 1.0 {
        if f < q.knee_frac {
            return q.beta_min * (q.beta_knee / q.beta_min).powf(f / q.knee_frac);
        }
        return q.beta_knee
            * (q.beta_max / q.beta_knee).powf((f - q.knee_frac) / (1.0 - q.knee_frac));
    }
    q.beta_min * (q.beta_max / q.beta_min).powf(f)
}

fn decode(word: i32) -> (usize, i32) {
    ((word as u32 >> 16) as usize, i32::from(word as i16))
}

// CPU exp and pow and the Metal library can differ in their last bits. Each
// tested random decision must stay clear of its threshold, without relaxing
// spins.
fn below(draw: f32, threshold: f32) -> bool {
    assert!(
        (draw - threshold).abs() > 1e-5,
        "reference decision near floating-point boundary"
    );
    draw < threshold
}

/// Every anneal's spins in original order and every thread's two RNG words,
/// anneal-major, after the cluster sweeps and before the final descent.
// Keep the sweep phases together for comparison with gpucluster.cu:160-256.
#[expect(clippy::too_many_lines, reason = "one sweep reads as one unit")]
pub(crate) fn run(prepared: &PreparedGraph, schedule: &ClusterParams) -> (Vec<Vec<i8>>, Vec<u32>) {
    let layout = prepared.layout();
    let node_count = layout.h.len();
    let (mut spins, mut rng) = initialize(node_count, schedule.anneals, schedule.seed);
    for (block, (s, rng)) in spins
        .chunks_exact_mut(node_count)
        .zip(rng.as_chunks_mut::<THREADS>().0)
        .enumerate()
    {
        for sweep in 0..schedule.sweeps {
            let beta = beta_at(schedule, sweep);
            let mut bonds = Vec::new();
            // gpucluster.cu:171-209. Each tile and pair consumes its owner's RNG.
            for (t, (nodes, gauge)) in layout
                .tiles
                .as_chunks::<4>()
                .0
                .iter()
                .zip(layout.gauges.as_chunks::<4>().0)
                .enumerate()
            {
                let mut pos = Vec::new();
                let mut neg = Vec::new();
                for (&u, &g) in nodes.iter().zip(gauge) {
                    let u = u as usize;
                    if s[u] * g > 0 {
                        pos.push(u);
                    } else {
                        neg.push(u);
                    }
                }
                if pos.is_empty() || neg.is_empty() {
                    continue;
                }
                let r = uniform(&mut rng[t % THREADS]);
                let bt = beta * layout.tile_w[t] as f32 * 0.001;
                let z = (-2.0 * bt).exp();
                let zero = z * z * z * z;
                let single = zero + 4.0 * (z - zero) / 3.0;
                if pos.len() == 2 {
                    if below(r, zero) {
                        continue;
                    }
                    if below(r, single) {
                        let b = next(&mut rng[t % THREADS]);
                        bonds.push((pos[(b & 1) as usize], neg[((b >> 1) & 1) as usize]));
                    } else {
                        let ch = (next(&mut rng[t % THREADS]) & 1) as usize;
                        bonds.push((pos[0], neg[ch]));
                        bonds.push((pos[1], neg[1 - ch]));
                    }
                } else if !below(r, (-6.0 * bt).exp()) {
                    let b = next(&mut rng[t % THREADS]);
                    bonds.push((
                        pos[(b & 0xffff) as usize % pos.len()],
                        neg[(b >> 16) as usize % neg.len()],
                    ));
                }
            }
            for (pair, nodes) in layout.pairs.as_chunks::<2>().0.iter().enumerate() {
                let (a, b) = (nodes[0] as usize, nodes[1] as usize);
                let w = layout.link_w[pair];
                let satisfied = if w > 0 { s[a] == s[b] } else { s[a] != s[b] };
                let bond = 1.0 - (-2.0 * beta * (w as f32).abs() * 0.001).exp();
                if satisfied && below(uniform(&mut rng[pair % THREADS]), bond) {
                    bonds.push((a, b));
                }
            }
            let roots = components(node_count, &bonds);
            let salt = mix((sweep as u32)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add((block as u32).wrapping_mul(0x85eb_ca6b)));
            // gpucluster.cu:214-249. Sum integer residual boundaries before
            // any root decisions or simultaneous flips in this round.
            for round in 0..schedule.rounds {
                let selected = |r: usize| mix(r as u32 ^ salt) as usize % schedule.rounds == round;
                let mut acc = vec![0_i32; node_count];
                for u in 0..node_count {
                    let root = roots[u];
                    if !selected(root) {
                        continue;
                    }
                    let mut a = layout.h[u];
                    for e in 0..layout.weak_deg[u] as usize {
                        let (v, w) = decode(layout.weak_ell[e * node_count + u]);
                        if roots[v] != root {
                            a += w * i32::from(s[v]);
                        }
                    }
                    acc[root] += i32::from(s[u]) * a;
                }
                // Roots visited in ascending order preserve each thread's
                // strided order, even for models larger than one threadgroup.
                for r in 0..node_count {
                    if roots[r] != r || !selected(r) {
                        continue;
                    }
                    let delta = -2.0 * acc[r] as f32;
                    acc[r] = i32::from(below(
                        uniform(&mut rng[r % THREADS]),
                        1.0 / (1.0 + (beta * delta * 0.001).exp()),
                    ));
                }
                for u in 0..node_count {
                    if selected(roots[u]) && acc[roots[u]] == 1 {
                        s[u] *= -1;
                    }
                }
            }
            // gpucluster.cu:113-136,251-253. Nonadjacent nodes of a color
            // commute. Thread IDs restart at zero at each color's first node.
            if schedule.single_every > 0 && (sweep + 1) % schedule.single_every == 0 {
                for color in layout.color_off.windows(2) {
                    let start = color[0] as usize;
                    for u in start..color[1] as usize {
                        let mut field = layout.h[u];
                        for e in 0..layout.all_deg[u] as usize {
                            let (v, w) = decode(layout.all_ell[e * node_count + u]);
                            field += w * i32::from(s[v]);
                        }
                        let delta = -2 * i32::from(s[u]) * field;
                        if delta <= 0
                            || below(
                                uniform(&mut rng[(u - start) % THREADS]),
                                (-beta * delta as f32 * 0.001).exp(),
                            )
                        {
                            s[u] *= -1;
                        }
                    }
                }
            }
        }
    }
    let states = spins
        .chunks_exact(node_count)
        .map(|s| prepared.unpermute(s).unwrap())
        .collect();
    (states, rng.into_iter().flatten().collect())
}
