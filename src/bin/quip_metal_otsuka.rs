// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Metal Otsuka miner (`quip-metal-otsuka`). macOS-only at runtime. Runs the
//! Otsuka cluster anneal on validated `quip/twisted-ice-dimer` revision 1
//! jobs, and Metal simulated annealing on every other job.

// Binary is a separate crate; lib.rs crate-root panic discipline does not apply here.
#![deny(clippy::unwrap_used)]
#![deny(clippy::panic)]
#![warn(clippy::expect_used)]

use clap::Parser;
use quip_solver_core::{run, CommonArgs, OpenError};
use quip_solver_metal::cluster::sampler::OtsukaSampler;
use quip_solver_metal::host_otsuka_identity;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 2"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Metal device index. Default 0 → miner id `metal-0`.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Target GPU utilization ceiling percent (1–100). Used by IOKit governor.
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    /// Yield to other GPU users under thermal/display pressure.
    #[arg(long, default_value_t = false)]
    yielding: bool,
    /// Advertise at most this many nodes. Default: the largest count the SA
    /// capacity model and the one-sweep limit admit on this host. A value
    /// past that limit is a configuration error.
    #[arg(long)]
    max_nodes: Option<usize>,
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("metal-{}", cli.device));
    }
    let identity = match host_otsuka_identity(cli.max_nodes) {
        Ok(identity) => identity,
        Err(code) => return code,
    };
    if cli.common.capabilities {
        return run(identity, &cli.common, || {
            Err::<OtsukaSampler, _>(OpenError("capabilities open no device".into()))
        });
    }
    run(identity, &cli.common, || {
        OtsukaSampler::open(cli.device, cli.utilization, cli.yielding, cli.common.check)
    })
}
