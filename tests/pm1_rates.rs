// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! ±1 throughput and quality on the Advantage2 topology, for a before and
//! after comparison of the int8 path. Run on an idle host:
//!
//! ```sh
//! cargo test --release --test pm1_rates -- --ignored --nocapture
//! ```
//!
//! `QUIP_PM1_SCALE=wide` multiplies every coefficient by 1.5 so the same jobs
//! run on the wide path, for the cost factor in `sampler.rs`.

use std::io::{self, Write};
use std::time::Instant;

use quip_solver_core::SampleParams;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising, Kernel};
use quip_solver_metal::IsingGraph;

fn edges() -> Vec<(usize, usize)> {
    include_str!("fixtures/advantage2-system1.edges")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(|t| t.parse::<usize>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect()
}

fn pm1_job(edges: &[(usize, usize)], seed: u64, scale: f64) -> IsingGraph {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut sign = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if x & 1 == 1 {
            scale
        } else {
            -scale
        }
    };
    let h = (0..4577).map(|_| sign()).collect();
    let j = edges.iter().map(|_| sign()).collect();
    IsingGraph::new(h, j, edges.to_vec())
}

#[test]
#[ignore = "minutes of device time; see the module comment"]
fn pm1_rates() {
    let dev = MetalDevice::open(0).unwrap();
    let scale = if std::env::var("QUIP_PM1_SCALE").as_deref() == Ok("wide") {
        1.5
    } else {
        1.0
    };
    let edges = edges();
    for kernel in [Kernel::Sa, Kernel::Gibbs] {
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 4096,
            seed: 3,
            ..Default::default()
        };
        let mut seconds = Vec::new();
        let mut bests = Vec::new();
        for job in 0..8 {
            let graph = pm1_job(&edges, job, scale);
            let start = Instant::now();
            let reads = sample_ising(&dev, &graph, &params, kernel).unwrap();
            seconds.push(start.elapsed().as_secs_f64());
            bests.push(reads.iter().map(|r| r.energy_milli).min().unwrap());
        }
        seconds.sort_by(f64::total_cmp);
        let mean_best = bests.iter().sum::<i64>() as f64 / bests.len() as f64;
        writeln!(
            io::stdout().lock(),
            "{{\"kernel\":\"{kernel:?}\",\"scale\":{scale},\"median_s\":{:.4},\"mean_best_milli\":{mean_best:.1}}}",
            seconds[seconds.len() / 2]
        )
        .unwrap();
    }
}
