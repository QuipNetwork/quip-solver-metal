//! Metal heat-bath Gibbs miner (`quip-metal-gibbs`). macOS-only at runtime.

// Binary is a separate crate; lib.rs crate-root panic discipline does not apply here.
#![deny(clippy::unwrap_used)]
#![deny(clippy::panic)]
#![warn(clippy::expect_used)]

use clap::Parser;
use quip_solver_core::CommonArgs;
use quip_solver_metal::{host_identity, run_metal, GibbsTag, Kernel};
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
    /// Advertise at most this many nodes. Default: the largest count the
    /// capacity model and the one-sweep limit admit on this host. A value
    /// past that limit is a configuration error.
    #[arg(long)]
    max_nodes: Option<usize>,
    /// Print the largest model each connectivity and precision admits, then exit. Opens no device.
    #[arg(long, conflicts_with_all = ["quip_coordinator", "capabilities", "check", "solve"])]
    probe_capacity: bool,
    /// Print the probe as JSON. Only with `--probe-capacity`.
    #[arg(long, requires = "probe_capacity")]
    json: bool,
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if cli.probe_capacity {
        quip_solver_metal::probe::print_probe(
            quip_solver_metal::sampler::Kernel::Gibbs,
            &quip_solver_metal::capacity::host_limits(),
            cli.json,
        );
        return ExitCode::SUCCESS;
    }
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("metal-{}", cli.device));
    }
    let identity = match host_identity(Kernel::Gibbs, cli.max_nodes) {
        Ok(identity) => identity,
        Err(code) => return code,
    };
    run_metal::<GibbsTag>(
        identity,
        &cli.common,
        cli.device,
        cli.utilization,
        cli.yielding,
    )
}
