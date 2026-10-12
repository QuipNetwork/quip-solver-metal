//! Metal Ising samplers.
//!
//! Three binaries share this library:
//! - `quip-metal-sa` — Metropolis simulated annealing on one Apple GPU
//! - `quip-metal-msa` — multi-spin annealing on selected Metal and ANE engines
//! - `quip-metal-gibbs` — single-site heat-bath Gibbs on one Apple GPU
//!
//! Kernels take **explicit per-job** CSR buffers from the host (no kernel-side
//! nonce economy / rotating slots). Solution energies are always scored with
//! [`quip_solver_core::quip_protocol::scoring::energy_milli`] for consensus
//! (host f64 — Metal has no fp64). The coordinator session loop lives in
//! `quip-solver-core`.
//!
//! # Platform
//!
//! macOS only. Metal and IOKit have no implementation on any other OS, so this
//! crate does not build elsewhere and deliberately offers no stub path — a
//! binary that cannot mine is not worth the dead code needed to produce it.
//! In practice a non-macOS build fails while compiling the Apple-only
//! dependencies (`core-graphics-types`: "link kind `framework` is only
//! supported on Apple targets") before reaching this crate at all; the
//! `compile_error!` below states the contract for the case where it does.

// Panic discipline for the *library* only. These cannot live in `Cargo.toml`'s
// `[lints]` table: that applies to every target in the package, and the
// integration tests in `tests/` use `unwrap`/`expect`/`panic!` freely and
// legitimately (a panicking assertion is how a test reports failure). A
// crate-root attribute scopes them to this crate, which is where a panic
// would take down a running miner. Everything else lives in `[lints]`.
#![deny(clippy::unwrap_used)]
#![deny(clippy::panic)]
#![warn(clippy::expect_used)]
// `#[cfg(test)] mod tests` blocks inside `src/` compile as part of *this*
// crate, so the three denials above would land on the co-located unit tests
// too. Relax them under the `test` cfg only: a panicking assertion is how a
// unit test reports failure, exactly as in `tests/`. Non-test builds keep the
// full discipline.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::panic, clippy::expect_used))]

// States the platform contract in code. The Apple-only dependencies usually
// fail first, so this is a backstop rather than the error you will actually
// see; it still fires for any build that reaches this crate.
#[cfg(not(target_os = "macos"))]
compile_error!(
    "quip-solver-metal is macOS-only: it depends on Metal and IOKit, which exist \
     on no other platform. Build it on macOS (Apple Silicon)."
);

pub mod sampler;

mod cascade;
mod combined;
mod cutoff;
mod draw;
mod lease_turn;
mod model_checks;
mod resident;
mod slots;

pub mod iokit_gov;
pub mod metal_device;
pub mod streaming;
pub mod topology;

pub use quip_solver_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};

pub use sampler::{sample_ising, Kernel};

use quip_solver_core::{run, BackendIdentity, CommonArgs};
use std::process::ExitCode;

const DEFAULT_MAX_EDGES: u32 = 1_000_000;

// Compile-time guard that the sampler's `num_sweeps` rejection threshold
// still admits every job this backend advertises it will accept.
//
// `quip-solver-core` doubles the resolved sweeps for Gibbs
// (`GIBBS_SWEEP_MULTIPLIER`), so an adapt-driven job can legitimately arrive
// at `2 * max_sweeps`. Raising `METAL_ADAPT`'s `max_sweeps` past half of
// `sampler::MAX_SWEEPS` would make the miner reject work it just told the
// coordinator it could do; this turns that drift into a build failure rather
// than a runtime reject. Anonymous `const _`: a named const is only evaluated
// where it is used, so it would assert nothing.
const _: () = assert!(
    sampler::MAX_SWEEPS >= 2 * METAL_ADAPT.max_sweeps as usize,
    "sampler::MAX_SWEEPS must admit the Gibbs-doubled METAL_ADAPT.max_sweeps"
);

/// Metal adapt envelope (from `GPU/metal_miner.py`).
const METAL_ADAPT: quip_solver_core::adapt::AdaptBounds = quip_solver_core::adapt::AdaptBounds {
    min_sweeps: 256,
    max_sweeps: 2048,
    min_reads: 64,
    max_reads: 256,
    reads_solution_min_factor: 0,
    reads_solution_max_factor: 0,
    reads_solution_floor_factor: 0,
};

/// Multi-spin adapt envelope.
///
/// Reads are pinned to 64: two 32-lane words per problem. Sweeps run from
/// 4096 to 14336 with the difficulty. At the testnet's current target the
/// difficulty saturates at 1.0, so a job is 64 reads at 14336 sweeps. Every
/// accepted job is still bounded by `sampler::MAX_SWEEPS`.
///
/// Measured 2026-09-18 on Apple M4 Max (40 GPU cores) against 60 Aglais
/// testnet qblocks regenerated from their nonces, five seeds each, with the
/// jobs/s of each shape from the batched bench
/// (`docs/perf/2026-09-18-testnet-sweeps-study.md` and
/// `docs/perf/2026-09-18-testnet-sweeps-intermediates.md`). Against the
/// previous envelope of 128 reads at 16384 sweeps:
///
/// ```text
/// reads  sweeps   jobs/s   P(valid)   valid/s   winner-beating/s   valid/s at min_solutions 5
/// 128    16384     6.24      0.54      3.37       2.00               1.21
/// 64     16384    12.19      0.40      4.92       2.48               1.38
/// 64     14336    13.99      0.40      5.64       3.03               1.40
/// 64      8192    22.49      0.23      5.17       2.17               0.67
/// 64      4096    41.50      0.06      2.49       1.25               0.00
/// ```
///
/// 64 reads at 14336 sweeps gives 1.67x the valid proofs per second of the
/// old envelope (90% bootstrap interval 1.46 to 1.88), 1.51x the
/// winner-beating proofs (1.23 to 1.82), and 1.40 against 1.21 proofs per
/// second under the chain's default `min_solutions` of 5. P(valid) is flat
/// from 14336 to 16384 sweeps. The floor stays at 4096: the chance per job
/// collapses between 4096 and 2048 sweeps at that target.
const METAL_MSA_ADAPT: quip_solver_core::adapt::AdaptBounds =
    quip_solver_core::adapt::AdaptBounds {
        min_sweeps: 4096,
        max_sweeps: 14336,
        min_reads: 64,
        max_reads: 64,
        reads_solution_min_factor: 0,
        reads_solution_max_factor: 0,
        reads_solution_floor_factor: 0,
    };

const _: () = assert!(
    sampler::MAX_SWEEPS >= METAL_MSA_ADAPT.max_sweeps as usize,
    "sampler::MAX_SWEEPS must admit METAL_MSA_ADAPT.max_sweeps"
);

/// Features every kernel advertises: a real `sample_stream` override, the
/// IOKit governor, and a host-built ladder that honours
/// `SampleParams::schedule`.
const FEATURES: &[&str] = &[
    "streaming",
    "governor",
    quip_solver_core::LINEAR_SCHEDULE_FEATURE,
];

/// One unit, in milli. Every kernel truncates each coefficient to an `int8` of
/// whole units (`topology::quantize_i8`), so only whole units are exact.
const UNIT_MILLI: u32 = 1000;

/// SA's model limits. `kernels/sa.metal` keeps each flip's energy change in an
/// `int8_t`, `-2 * s * (h + Σ J s)`, so the local field must stay within 63
/// units or the change wraps.
pub const METAL_SA_LIMITS: quip_solver_core::ModelLimits = quip_solver_core::ModelLimits {
    max_degree: Some(0),
    max_local_field_milli: Some(63 * UNIT_MILLI),
    exact_resolution_milli: Some(UNIT_MILLI),
};

/// Gibbs's model limits. `kernels/gibbs.metal` sums the field in `float`, so
/// only the `int8` coefficients bound it: a field of at most 127 units keeps
/// every coefficient inside one.
pub const METAL_GIBBS_LIMITS: quip_solver_core::ModelLimits = quip_solver_core::ModelLimits {
    max_degree: Some(0),
    max_local_field_milli: Some(127 * UNIT_MILLI),
    exact_resolution_milli: Some(UNIT_MILLI),
};

/// MSA's model limits, on the GPU and the ANE alike. Both unroll
/// `sampler::MSA_MAX_DEG` neighbours, which caps the degree, and the field
/// limit is one unit per neighbour plus one for the bias. Like every
/// `ModelLimits`, these are checked on the model as received. Both MSA engines
/// anneal on signs only, so a model inside these limits with a coefficient
/// above one unit in magnitude runs as its sign model, with only its energies
/// rescored. `ModelLimits` cannot state a ±1-only contract yet (QUI-1832);
/// refusing such a model is QUI-1642.
pub const METAL_MSA_LIMITS: quip_solver_core::ModelLimits = quip_solver_core::ModelLimits {
    max_degree: Some(crate::sampler::MSA_MAX_DEG as u32),
    max_local_field_milli: Some((crate::sampler::MSA_MAX_DEG as u32 + 1) * UNIT_MILLI),
    exact_resolution_milli: Some(UNIT_MILLI),
};

/// Backend identity for `quip-metal-sa`.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::METAL_SA_IDENTITY;
/// use quip_solver_core::quip_proto::v1::{Algorithm, Backend};
///
/// assert_eq!(METAL_SA_IDENTITY.backend, Backend::Metal);
/// assert_eq!(METAL_SA_IDENTITY.algorithm, Algorithm::Sa);
/// assert!(METAL_SA_IDENTITY.max_nodes > 0);
/// ```
pub const METAL_SA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_solver_core::quip_proto::v1::Backend::Metal,
    algorithm: quip_solver_core::quip_proto::v1::Algorithm::Sa,
    // Single source of truth with the sampler's runtime guard: the advertised
    // cap and the guard cannot drift because this is the same constant. A job
    // over it would overrun the SA kernel's `thread int8_t delta_energy[4593]`
    // (`kernels/sa.metal`), so it must reject `TooLarge` rather than clamp.
    // `const` context, so the narrowing cast is checked at compile time.
    max_nodes: crate::sampler::SA_MAX_NODES as u32,
    max_edges: DEFAULT_MAX_EDGES,
    features: FEATURES,
    adapt: METAL_ADAPT,
    limits: METAL_SA_LIMITS,
};

/// Backend identity for `quip-metal-gibbs`.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::METAL_GIBBS_IDENTITY;
/// use quip_solver_core::quip_proto::v1::{Algorithm, Backend};
///
/// assert_eq!(METAL_GIBBS_IDENTITY.backend, Backend::Metal);
/// assert_eq!(METAL_GIBBS_IDENTITY.algorithm, Algorithm::Gibbs);
/// assert!(METAL_GIBBS_IDENTITY.max_nodes > 0);
/// ```
pub const METAL_GIBBS_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_solver_core::quip_proto::v1::Backend::Metal,
    algorithm: quip_solver_core::quip_proto::v1::Algorithm::Gibbs,
    // Same single-source-of-truth rule as SA above; the Gibbs cap comes from
    // `thread int8_t packed_state[600]` (600*8 bits) in `kernels/gibbs.metal`.
    max_nodes: crate::sampler::GIBBS_MAX_NODES as u32,
    max_edges: DEFAULT_MAX_EDGES,
    features: FEATURES,
    adapt: METAL_ADAPT,
    limits: METAL_GIBBS_LIMITS,
};

/// Backend identity for `quip-metal-msa`.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::METAL_MSA_IDENTITY;
/// use quip_solver_core::quip_proto::v1::{Algorithm, Backend};
///
/// assert_eq!(METAL_MSA_IDENTITY.backend, Backend::Metal);
/// assert_eq!(METAL_MSA_IDENTITY.algorithm, Algorithm::Msa);
/// assert_eq!(METAL_MSA_IDENTITY.adapt.min_reads, 64);
/// ```
pub const METAL_MSA_IDENTITY: BackendIdentity = BackendIdentity {
    backend: quip_solver_core::quip_proto::v1::Backend::Metal,
    algorithm: quip_solver_core::quip_proto::v1::Algorithm::Msa,
    // The GPU kernel's capacity, the same constant as its dispatch guard. The
    // ANE holds up to 16,384 nodes, but only with unit coefficients, at most
    // 128 reads, and while it is enabled, so advertising its cap would admit
    // models that the binary then refuses. `const` context, so the narrowing
    // cast is checked at compile time.
    max_nodes: crate::sampler::MSA_MAX_NODES as u32,
    max_edges: DEFAULT_MAX_EDGES,
    features: FEATURES,
    adapt: METAL_MSA_ADAPT,
    limits: METAL_MSA_LIMITS,
};

/// Metal sampler backend: one Apple GPU device plus an IOKit utilization
/// governor. macOS-only.
///
/// # Examples
///
/// ```no_run
/// use quip_solver_metal::{
///     Kernel, MetalSampler,
///     iokit_gov::UtilGovernor,
///     metal_device::MetalDevice,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = MetalDevice::open(0)?;
/// let gov = UtilGovernor::start(0, 100, false);
/// let sampler = MetalSampler::new(device, gov, Kernel::Sa);
/// let _ = sampler;
/// # Ok(())
/// # }
/// ```
pub struct MetalSampler {
    device: crate::metal_device::MetalDevice,
    gov: crate::iokit_gov::UtilGovernor,
    kernel: Kernel,
    cascade: std::sync::Mutex<cascade::CascadeSettings>,
    controller: std::sync::Mutex<Option<cascade::Controller>>,
    salts_tx: std::sync::mpsc::SyncSender<resident::Salt>,
    salts_rx: std::sync::Mutex<std::sync::mpsc::Receiver<resident::Salt>>,
    /// Cleared when the resident runner exits, so a lease stops waiting on it.
    runner_live: std::sync::atomic::AtomicBool,
    lease_turns: lease_turn::LeaseTurns,
    /// Counts lease cancellations, as the miner's view of the coordinator's
    /// rounds: a reseed cancels every queued lease. A lease bumps it while it
    /// still holds its place in `lease_turns`, so every lease that feeds after
    /// a reseed reads the new value.
    lease_round: std::sync::atomic::AtomicU64,
}

/// Stores a value into a flag when dropped, on every exit path.
struct StoreOnDrop<'a>(&'a std::sync::atomic::AtomicBool, bool);

impl Drop for StoreOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(self.1, std::sync::atomic::Ordering::Release);
    }
}

impl std::fmt::Debug for MetalSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSampler")
            .field("device", &self.device)
            .field("gov", &self.gov)
            .field("kernel", &self.kernel)
            .field("cascade", &self.cascade)
            .finish_non_exhaustive()
    }
}

/// Metal backend config, parsed from the verbatim `config.toml` subsection in
/// `Configure.backend_toml`. Unrecognized keys land in `unknown`;
/// `warn_unknown_fields` filters session-level keys (e.g. `num_sweeps`).
#[derive(serde::Deserialize, Default)]
struct MetalConfig {
    /// GPU utilization ceiling 1–100 (governor throttle threshold when yielding).
    utilization: Option<u32>,
    /// Yield the GPU to siblings when util exceeds the ceiling.
    yielding: Option<bool>,
    enable_ane: Option<bool>,
    enable_metal: Option<bool>,
    #[serde(flatten)]
    cascade: cascade::CascadeToml,
    #[serde(flatten)]
    unknown: std::collections::BTreeMap<String, toml::Value>,
}

impl MetalSampler {
    /// Bind an opened [`crate::metal_device::MetalDevice`] and
    /// [`crate::iokit_gov::UtilGovernor`] to a kernel.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use quip_solver_metal::{
    ///     Kernel, MetalSampler,
    ///     iokit_gov::UtilGovernor,
    ///     metal_device::MetalDevice,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = MetalDevice::open(0)?;
    /// let gov = UtilGovernor::start(0, 100, false);
    /// let _sampler = MetalSampler::new(device, gov, Kernel::Gibbs);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        device: crate::metal_device::MetalDevice,
        gov: crate::iokit_gov::UtilGovernor,
        kernel: Kernel,
    ) -> Self {
        let (salts_tx, salts_rx) = std::sync::mpsc::sync_channel(resident::PREP_BOUND);
        Self {
            device,
            gov,
            kernel,
            cascade: std::sync::Mutex::new(cascade::CascadeSettings::default()),
            controller: std::sync::Mutex::new(None),
            salts_tx,
            salts_rx: std::sync::Mutex::new(salts_rx),
            runner_live: std::sync::atomic::AtomicBool::new(true),
            lease_turns: lease_turn::LeaseTurns::default(),
            lease_round: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Report a lease's outcomes until every queued salt has one, the
    /// feeder has stopped, or the lease stops.
    fn report_lease(
        &self,
        sink: &quip_solver_core::LeaseSink,
        outcomes: &std::sync::mpsc::Receiver<resident::SaltOutcome>,
        live_target: &resident::LiveTarget,
        fed: &std::sync::atomic::AtomicU64,
        feeding: &std::sync::atomic::AtomicBool,
    ) -> Result<(), quip_solver_core::SampleError> {
        use std::sync::atomic::Ordering;
        let mut reported = 0u64;
        while !sink.is_stopped() {
            self.require_runner()?;
            live_target.set(sink.target_energy_milli());
            // `feeding` is cleared after the last increment of `fed`.
            if !feeding.load(Ordering::Acquire) && reported == fed.load(Ordering::Acquire) {
                break;
            }
            let Ok(first) = outcomes.recv_timeout(std::time::Duration::from_millis(1)) else {
                continue;
            };
            for outcome in std::iter::once(first).chain(outcomes.try_iter()) {
                reported += 1;
                let sent = match outcome {
                    resident::SaltOutcome::Screened {
                        index,
                        energy_milli,
                    } => sink.screen(index, energy_milli),
                    resident::SaltOutcome::Survived { index, reads } => sink.push(index, reads),
                    resident::SaltOutcome::Dropped { index } => {
                        tracing::debug!(index, "lease salt dropped before reporting");
                        Ok(())
                    }
                    resident::SaltOutcome::Failed { index, error } => {
                        return Err(quip_solver_core::SampleError::DeviceFault(format!(
                            "lease salt {index}: {error}"
                        )));
                    }
                };
                if sent.is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn require_runner(&self) -> Result<(), quip_solver_core::SampleError> {
        if self.runner_live.load(std::sync::atomic::Ordering::Acquire) {
            Ok(())
        } else {
            Err(quip_solver_core::SampleError::DeviceFault(
                "resident runner stopped during a lease".into(),
            ))
        }
    }

    pub(crate) fn set_cascade(&self, settings: cascade::CascadeSettings) {
        *self
            .cascade
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = settings;
    }
}

impl quip_solver_core::Sampler for MetalSampler {
    fn sample_lease(
        &self,
        lease: &quip_solver_core::Lease,
        topology: &quip_solver_core::quip_protocol::lease::TopologyView,
        params: &SampleParams,
        sink: &quip_solver_core::LeaseSink,
    ) -> Result<(), quip_solver_core::SampleError> {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        if self.kernel != Kernel::Msa {
            return Err(quip_solver_core::SampleError::DeviceFault(
                "local lease generation needs the MSA kernel".into(),
            ));
        }
        let topology = std::sync::Arc::new(topology.clone());
        let (reply, outcomes) = std::sync::mpsc::channel();
        // Queued, preparing, and live units of this lease stop once it returns.
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let _stop = StoreOnDrop(&stop, true);
        // One lease feeds at a time; see `lease_turn`.
        let mut turn = self.lease_turns.join();
        let runner_live = || self.runner_live.load(Ordering::Acquire);
        if !turn.wait(|| sink.is_stopped() || !runner_live()) {
            if sink.is_stopped() {
                self.lease_round.fetch_add(1, Ordering::AcqRel);
            }
            return self.require_runner();
        }
        let round = self.lease_round.load(Ordering::Acquire);
        tracing::debug!(
            turn = turn.id(),
            salts = lease.salt_count(),
            "lease feeding"
        );
        // Shared with every unit of this lease. quip-solver-core keeps the
        // session target in a live `watch` channel (LeaseSink::target_energy_milli,
        // a cheap borrow); the reporting loop below refreshes this cell on
        // every pass instead of units each capturing a target snapshot at
        // enqueue time. The resident runner only reads it, at each checkpoint.
        let live_target = resident::LiveTarget::new(sink.target_energy_milli());
        // Shared with every unit of this lease. A new lease starts empty;
        // see `resident::LeaseTopK`.
        let top10 = resident::LeaseTopK::new();
        // Salts queued so far, and whether the feeder may queue more.
        let (fed, feeding) = (AtomicU64::new(0), AtomicBool::new(true));
        std::thread::scope(|scope| {
            // The feeder only queues salts, blocking while the intake channel
            // is full, so the runner never waits on this lease's reporting.
            // It hands the turn to the next lease once its last salt is queued.
            let (fed, feeding, stop) = (&fed, &feeding, &stop);
            let (target, top10, unit_reply) = (live_target.clone(), top10.clone(), reply.clone());
            let topology = std::sync::Arc::clone(&topology);
            scope.spawn(move || {
                let _done = StoreOnDrop(feeding, false);
                for index in 0..lease.salt_count() {
                    if sink.is_stopped() || stop.load(Ordering::Acquire) {
                        if sink.is_stopped() {
                            // Still holding the turn; see `lease_round`.
                            self.lease_round.fetch_add(1, Ordering::AcqRel);
                        }
                        return;
                    }
                    let salt = resident::Salt {
                        topology: std::sync::Arc::clone(&topology),
                        nonce: lease.nonce(index),
                        index,
                        round,
                        params: params.clone(),
                        stop: std::sync::Arc::clone(stop),
                        target: target.clone(),
                        top10: top10.clone(),
                        reply: unit_reply.clone(),
                    };
                    if self.salts_tx.send(salt).is_err() {
                        return;
                    }
                    fed.fetch_add(1, Ordering::Release);
                }
                turn.release();
                tracing::debug!(turn = turn.id(), "lease fed");
            });
            drop(reply);
            let reported = self.report_lease(sink, &outcomes, &live_target, fed, feeding);
            // Stop the feeder before the scope joins it: queued salts of a
            // stopped lease drain without running.
            stop.store(true, Ordering::Release);
            reported
        })
    }

    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, quip_solver_core::SampleError> {
        sample_ising(&self.device, graph, params, self.kernel).map_err(|e| {
            // `kind` carries the local `sampler::SampleError` variant (Debug),
            // so a kernel compile failure and a device reset stay
            // distinguishable in the log even though both map to
            // `DeviceFault` below — see `sampler::SampleError::to_sample_error`.
            // A capacity refusal is the one case that maps elsewhere
            // (`Capacity`).
            tracing::error!(error = %e, kind = ?e, "metal sample failed");
            e.to_sample_error()
        })
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<quip_solver_core::StreamJob>,
        out: tokio::sync::mpsc::Sender<quip_solver_core::StreamResult>,
        cancel: quip_solver_core::CancelToken,
    ) {
        // Both runners report device time to the same governor that sizes work.
        if self.kernel == Kernel::Msa {
            self.runner_live
                .store(true, std::sync::atomic::Ordering::Release);
            let _exited = StoreOnDrop(&self.runner_live, false);
            resident::run(
                &self.device,
                &self.cascade,
                &self.controller,
                jobs,
                &self
                    .salts_rx
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()),
                &out,
                &self.gov,
                &cancel,
            );
        } else {
            streaming::run_stream(&self.device, self.kernel, jobs, &out, &self.gov, &cancel);
        }
    }

    fn stream_width(&self) -> usize {
        streaming::stream_width(&self.device, self.kernel)
    }

    fn utilization(&self) -> f64 {
        self.gov.utilization() as f64
    }

    fn should_throttle(&self) -> bool {
        self.gov.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        streaming::max_reads(self.kernel)
    }

    fn apply_config(&self, backend_toml: &str) {
        // Pure parse/merge lives in `resolve_governor_config` so unit tests
        // exercise the untrusted TOML path without opening a Metal device.
        let (ceiling, yielding) = resolve_governor_config(
            backend_toml,
            self.gov.utilization_ceiling(),
            self.gov.yielding(),
        );
        self.gov.reconfigure(ceiling, yielding);
        if let Ok(cfg) = toml::from_str::<MetalConfig>(backend_toml) {
            self.cascade
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .merge(&cfg.cascade);
        }
    }
}

/// Parse coordinator `backend_toml` and resolve the utilization ceiling and
/// yielding flag against the values the governor already holds (CLI defaults
/// until the first `Configure`).
///
/// Malformed TOML falls back to an empty config (`unwrap_or_default`), so the
/// current values are kept. Present keys override via
/// `quip_solver_core::config::config_override`; unknown keys are warned via
/// `warn_unknown_fields`. Out-of-range utilization is returned as-is —
/// `UtilGovernor::reconfigure` clamps to `1..=100`.
fn resolve_governor_config(
    backend_toml: &str,
    current_ceiling: u32,
    current_yielding: bool,
) -> (u32, bool) {
    use quip_solver_core::config::{config_override, warn_unknown_fields};
    let cfg: MetalConfig = toml::from_str(backend_toml).unwrap_or_default();
    warn_unknown_fields("metal", cfg.unknown.keys());
    // config over CLI (the governor holds the CLI-set values until now).
    let ceiling = config_override("utilization", current_ceiling, cfg.utilization);
    let yielding = config_override("yielding", current_yielding, cfg.yielding);
    (ceiling, yielding)
}

/// Kernel selection at the type level, one tag per Metal binary.
///
/// [`quip_solver_core::Sampler::declared_stream_width`] is associated —
/// `--capabilities` answers it with no device — so a width that differs per
/// kernel needs a `Sampler` type per binary. [`run_metal`] takes the tag
/// and builds the matching [`TaggedSampler`].
///
/// `Send + Sync + 'static` because `Sampler` requires them of the whole
/// sampler type; a zero-sized tag satisfies all three trivially.
pub trait KernelTag: Send + Sync + 'static {
    /// The kernel this tag selects.
    const KERNEL: Kernel;
}

/// Tag for `quip-metal-sa`.
pub struct SaTag;

impl KernelTag for SaTag {
    const KERNEL: Kernel = Kernel::Sa;
}

/// Tag for `quip-metal-gibbs`.
pub struct GibbsTag;

impl KernelTag for GibbsTag {
    const KERNEL: Kernel = Kernel::Gibbs;
}

/// Tag for `quip-metal-msa`.
pub struct MsaTag;

impl KernelTag for MsaTag {
    const KERNEL: Kernel = Kernel::Msa;
}

/// Engine router bound to its binary's kernel. MSA has one ANE slot in
/// addition to the Metal stream width. SA and Gibbs execute on Metal only.
/// [`run_metal`] keeps the tag and runtime kernel equal by construction.
pub struct TaggedSampler<A: KernelTag> {
    inner: combined::CombinedSampler,
    _kernel: std::marker::PhantomData<A>,
}

impl<A: KernelTag> quip_solver_core::Sampler for TaggedSampler<A> {
    fn generates_locally() -> bool {
        A::KERNEL == Kernel::Msa
    }

    fn sample_lease(
        &self,
        lease: &quip_solver_core::Lease,
        topology: &quip_solver_core::quip_protocol::lease::TopologyView,
        params: &SampleParams,
        sink: &quip_solver_core::LeaseSink,
    ) -> Result<(), quip_solver_core::SampleError> {
        self.inner.sample_lease(lease, topology, params, sink)
    }

    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, quip_solver_core::SampleError> {
        self.inner.sample(graph, params)
    }

    fn sample_stream(
        &self,
        jobs: tokio::sync::mpsc::Receiver<quip_solver_core::StreamJob>,
        out: tokio::sync::mpsc::Sender<quip_solver_core::StreamResult>,
        cancel: quip_solver_core::CancelToken,
    ) {
        self.inner.sample_stream(jobs, out, cancel);
    }

    fn stream_width(&self) -> usize {
        self.inner.stream_width()
    }

    /// Combined admission bound, independent of which engines configuration
    /// enables. Disabled slots remain unused rather than changing capabilities.
    fn declared_stream_width() -> u32 {
        u32::try_from(combined::stream_width(A::KERNEL)).unwrap_or(u32::MAX)
    }

    fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    fn should_throttle(&self) -> bool {
        self.inner.should_throttle()
    }

    fn max_reads(&self) -> u32 {
        self.inner.max_reads()
    }

    fn apply_config(&self, backend_toml: &str) {
        self.inner.apply_config(backend_toml);
    }
}

/// Run one miner identity with the selected engines. Session engines open on
/// first use, after backend configuration arrives. `--check` opens the default
/// engines: Metal for every algorithm, plus ANE for MSA.
///
/// # Examples
///
/// ```no_run
/// use quip_solver_core::CommonArgs;
/// use quip_solver_metal::{run_metal, SaTag, METAL_SA_IDENTITY};
///
/// let common = CommonArgs {
///     quip_coordinator: None,
///     miner_id: None,
///     capabilities: true,
///     solve: false,
///     check: false,
///     log_level: "info".into(),
///     sweeps_per_beta: None,
/// };
/// let _code = run_metal::<SaTag>(METAL_SA_IDENTITY, &common, 0, 100, false);
/// ```
pub fn run_metal<A: KernelTag>(
    id: BackendIdentity,
    common: &CommonArgs,
    device: usize,
    utilization: u32,
    yielding: bool,
) -> ExitCode {
    run(id, common, || {
        let inner = combined::CombinedSampler::new(A::KERNEL, device, utilization, yielding);
        if common.check {
            inner.check()?;
        }
        Ok(TaggedSampler::<A> {
            inner,
            _kernel: std::marker::PhantomData,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_governor_config;

    /// A swapped tag constant would silently advertise the other kernel's
    /// width; pin the tag → kernel binding. Device-free on purpose: the
    /// declared width must be answerable without a GPU.
    #[test]
    fn tagged_declared_widths_follow_their_kernels() {
        use super::{GibbsTag, Kernel, MsaTag, SaTag, TaggedSampler};
        use quip_solver_core::Sampler;
        assert_eq!(
            TaggedSampler::<SaTag>::declared_stream_width(),
            u32::try_from(crate::streaming::declared_stream_width(Kernel::Sa)).unwrap_or(u32::MAX)
        );
        assert_eq!(
            TaggedSampler::<GibbsTag>::declared_stream_width(),
            u32::try_from(crate::streaming::declared_stream_width(Kernel::Gibbs))
                .unwrap_or(u32::MAX)
        );
        assert_eq!(
            TaggedSampler::<MsaTag>::declared_stream_width(),
            u32::try_from(crate::combined::stream_width(Kernel::Msa)).unwrap_or(u32::MAX)
        );
    }

    #[test]
    fn msa_identity_advertises_the_multi_spin_kernel() {
        use super::{METAL_MSA_IDENTITY, METAL_SA_IDENTITY};
        assert_eq!(
            METAL_MSA_IDENTITY.backend,
            quip_solver_core::quip_proto::v1::Backend::Metal
        );
        assert_eq!(
            METAL_MSA_IDENTITY.algorithm,
            quip_solver_core::quip_proto::v1::Algorithm::Msa
        );
        assert_eq!(METAL_MSA_IDENTITY.max_nodes, 6016);
        assert_eq!(METAL_MSA_IDENTITY.features, METAL_SA_IDENTITY.features);
        assert!(METAL_MSA_IDENTITY
            .features
            .contains(&quip_solver_core::LINEAR_SCHEDULE_FEATURE));
        assert_eq!(METAL_MSA_IDENTITY.limits.max_degree, Some(20));
        // Reads are pinned to whole words.
        assert_eq!(METAL_MSA_IDENTITY.adapt.min_reads % 32, 0);
        assert_eq!(
            METAL_MSA_IDENTITY.adapt.min_reads,
            METAL_MSA_IDENTITY.adapt.max_reads
        );
        const {
            assert!(METAL_MSA_IDENTITY.adapt.min_sweeps <= METAL_MSA_IDENTITY.adapt.max_sweeps);
        }
    }

    #[test]
    fn msa_envelope_resolves_to_the_measured_shape_at_the_testnet_target() {
        use super::METAL_MSA_IDENTITY;
        use quip_solver_core::adapt::adapt_params;
        // Aglais qblock 3278: zero-field Advantage2 graph, target below the
        // adapt model's hardest energy, so the difficulty saturates at 1.0 and
        // the job is the envelope's maximum: the shape the sweeps study picked.
        let job = adapt_params(
            -14_624_068,
            1,
            4_577,
            41_514,
            &[0],
            &METAL_MSA_IDENTITY.adapt,
        );
        assert_eq!(job.num_reads, 64);
        assert_eq!(job.num_sweeps, 14_336);
        // An easy target lands on the floor, still a whole number of words.
        let easy = adapt_params(0, 1, 4_577, 41_514, &[0], &METAL_MSA_IDENTITY.adapt);
        assert_eq!(easy.num_reads, 64);
        assert_eq!(easy.num_sweeps, 4_096);
    }

    /// CLI defaults the pure resolver starts from in every case below.
    const CLI_CEILING: u32 = 80;
    const CLI_YIELDING: bool = false;

    #[test]
    fn empty_toml_keeps_current_values() {
        let (ceiling, yielding) = resolve_governor_config("", CLI_CEILING, CLI_YIELDING);
        assert_eq!(ceiling, CLI_CEILING);
        assert_eq!(yielding, CLI_YIELDING);
    }

    #[test]
    fn valid_utilization_overrides_ceiling() {
        let (ceiling, yielding) =
            resolve_governor_config("utilization = 60", CLI_CEILING, CLI_YIELDING);
        assert_eq!(ceiling, 60);
        assert_eq!(yielding, CLI_YIELDING);
    }

    #[test]
    fn valid_yielding_overrides_flag() {
        let (ceiling, yielding) =
            resolve_governor_config("yielding = true", CLI_CEILING, CLI_YIELDING);
        assert_eq!(ceiling, CLI_CEILING);
        assert!(yielding);
    }

    #[test]
    fn both_keys_override_together() {
        let toml = "utilization = 50\nyielding = true\n";
        let (ceiling, yielding) = resolve_governor_config(toml, CLI_CEILING, CLI_YIELDING);
        assert_eq!(ceiling, 50);
        assert!(yielding);
    }

    #[test]
    fn malformed_toml_falls_back_to_current_values() {
        // `unwrap_or_default` must swallow parse failure — no panic, no change.
        let (ceiling, yielding) =
            resolve_governor_config("not = [valid", CLI_CEILING, CLI_YIELDING);
        assert_eq!(ceiling, CLI_CEILING);
        assert_eq!(yielding, CLI_YIELDING);
    }

    #[test]
    fn unknown_keys_do_not_block_valid_overrides() {
        // Unknowns are warned (observable via tracing) but ignored for merge.
        let toml = "utilization = 40\nnot_a_real_key = 1\nyielding = true\n";
        let (ceiling, yielding) = resolve_governor_config(toml, CLI_CEILING, CLI_YIELDING);
        assert_eq!(ceiling, 40);
        assert!(yielding);
    }

    #[test]
    fn out_of_range_utilization_passes_through_unclamped() {
        // `config_override` returns the config value as-is; clamping to
        // `1..=100` is `UtilGovernor::reconfigure`'s job, not the parser's.
        let (zero, _) = resolve_governor_config("utilization = 0", CLI_CEILING, CLI_YIELDING);
        assert_eq!(zero, 0);
        let (high, _) = resolve_governor_config("utilization = 250", CLI_CEILING, CLI_YIELDING);
        assert_eq!(high, 250);
    }

    #[test]
    fn same_as_cli_is_a_silent_no_op() {
        // config_override keeps the CLI value when the config restates it.
        let (ceiling, yielding) = resolve_governor_config(
            "utilization = 80\nyielding = false\n",
            CLI_CEILING,
            CLI_YIELDING,
        );
        assert_eq!(ceiling, CLI_CEILING);
        assert_eq!(yielding, CLI_YIELDING);
    }
}
