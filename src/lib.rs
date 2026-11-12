//! Metal Ising samplers.
//!
//! Four binaries share this library:
//! - `quip-metal-sa` — Metropolis simulated annealing on one Apple GPU
//! - `quip-metal-msa` — multi-spin annealing on selected Metal and ANE engines
//! - `quip-metal-gibbs` — single-site heat-bath Gibbs on one Apple GPU
//! - `quip-metal-otsuka` — the Otsuka cluster anneal on twisted-ice-dimer
//!   family jobs, and Metal SA on every other job
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

pub mod capacity;
pub mod cluster;
pub mod probe;
pub mod sampler;

mod cascade;
mod combined;
mod cutoff;
mod draw;
mod frustration;
mod lease_turn;
mod model_checks;
mod representation;
mod resident;
mod screen;
mod slots;
#[cfg(test)]
mod wide_reference;
mod wide_tables;

pub mod iokit_gov;
pub mod metal_device;
pub mod model_file;
pub mod streaming;
pub mod topology;

pub use quip_solver_core::{Algorithm, IsingGraph, SampleParams, SamplerResult};

pub use sampler::{sample_ising, Kernel};

use crate::capacity::{CapacityError, DeviceLimits};
use quip_solver_core::{run, BackendIdentity, CommonArgs};
use std::process::ExitCode;

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
/// IOKit governor, a host-built ladder that honours `SampleParams::schedule`,
/// and the exact coefficient refusal. Each SA or Gibbs model runs on the int8
/// or the wide integer kernel, whichever represents it exactly
/// (`representation::classify`); each MSA model runs on the unit or the
/// weighted kernel (`representation::classify_msa`). Any other model is
/// refused.
const FEATURES: &[&str] = &[
    "streaming",
    "governor",
    quip_solver_core::LINEAR_SCHEDULE_FEATURE,
    quip_solver_core::EXACT_COEFFICIENTS_FEATURE,
];

/// SA's model limits. A model the int8 kernel cannot hold runs on the wide
/// kernel, which keeps milli units in `int`, so the field bound is the wide
/// path's. The degree bound is [`ADVERTISED_MAX_DEGREE`].
pub const METAL_SA_LIMITS: quip_solver_core::ModelLimits = quip_solver_core::ModelLimits {
    max_degree: Some(ADVERTISED_MAX_DEGREE),
    max_local_field_milli: Some(crate::representation::WIDE_MAX_FIELD_MILLI),
    exact_resolution_milli: Some(1),
};

/// Gibbs's model limits, the same wide bound as [`METAL_SA_LIMITS`].
pub const METAL_GIBBS_LIMITS: quip_solver_core::ModelLimits = METAL_SA_LIMITS;

/// MSA's model limits, the same as [`METAL_SA_LIMITS`]. A unit model runs on
/// the unit kernel (or the ANE) up to `sampler::MSA_UNIT_MAX_DEG`
/// neighbours, and any other model on the weighted kernel up to
/// `sampler::MSA_SPARSE_MAX_DEG`.
pub const METAL_MSA_LIMITS: quip_solver_core::ModelLimits = METAL_SA_LIMITS;

/// The degree bound every identity advertises: the weighted multi-spin
/// kernel's sparse limit, `sampler::MSA_SPARSE_MAX_DEG`. Under it the
/// session admits every shape inside `max_nodes` and `max_edges`. Past it, a
/// model runs only on dense-matrix storage, which holds a few thousand nodes
/// (`capacity::advertised_couplers`), and the CSR one-sweep limit refuses
/// SA rows far shorter than the dense size. A denser model, a complete one
/// included, still runs through the library API (`sample_ising`) when the
/// session admission accepts it.
const ADVERTISED_MAX_DEGREE: u32 = {
    assert!(crate::sampler::MSA_SPARSE_MAX_DEG <= u32::MAX as usize);
    crate::sampler::MSA_SPARSE_MAX_DEG as u32
};

/// The identity of `kernel` on a device with `limits`: `max_nodes` is
/// [`capacity::advertised_nodes`], lowered to `cap`, and `max_edges` is
/// [`capacity::advertised_couplers`]. Both come from the admission the
/// session applies (`capacity::session_admits`), so a job at the stated
/// corners is admitted.
fn identity(
    kernel: Kernel,
    adapt: quip_solver_core::adapt::AdaptBounds,
    model_limits: quip_solver_core::ModelLimits,
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<BackendIdentity, CapacityError> {
    let nodes = capacity::advertised_nodes(kernel, limits, cap)?;
    let edges = capacity::advertised_couplers(kernel, limits);
    identity_with(kernel, adapt, model_limits, nodes, edges)
}

/// The identity of `kernel` with `nodes` and `edges` as its limits.
fn identity_with(
    kernel: Kernel,
    adapt: quip_solver_core::adapt::AdaptBounds,
    model_limits: quip_solver_core::ModelLimits,
    nodes: usize,
    edges: usize,
) -> Result<BackendIdentity, CapacityError> {
    use quip_solver_core::quip_proto::v1::Algorithm as Proto;
    let to_u32 = |count: usize| {
        u32::try_from(count).map_err(|_| CapacityError::Nodes {
            max: u32::MAX as usize,
        })
    };
    Ok(BackendIdentity {
        backend: quip_solver_core::quip_proto::v1::Backend::Metal,
        algorithm: match kernel {
            Kernel::Sa => Proto::Sa,
            Kernel::Gibbs => Proto::Gibbs,
            Kernel::Msa => Proto::Msa,
        },
        max_nodes: to_u32(nodes)?,
        max_edges: to_u32(edges)?,
        features: FEATURES,
        adapt,
        limits: model_limits,
    })
}

/// Backend identity for `quip-metal-sa` on a device with `limits`, with
/// `max_nodes` lowered to `cap` (`--max-nodes`).
///
/// # Errors
///
/// [`CapacityError::Nodes`] naming the limit when `cap` is 0 or exceeds it,
/// or when no model fits `limits`.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::{capacity, metal_sa_identity};
/// use quip_solver_core::quip_proto::v1::{Algorithm, Backend};
///
/// let identity = metal_sa_identity(&capacity::host_limits(), None).unwrap();
/// assert_eq!(identity.backend, Backend::Metal);
/// assert_eq!(identity.algorithm, Algorithm::Sa);
/// assert!(identity.max_nodes > 0);
/// ```
pub fn metal_sa_identity(
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<BackendIdentity, CapacityError> {
    identity(Kernel::Sa, METAL_ADAPT, METAL_SA_LIMITS, limits, cap)
}

/// Backend identity for `quip-metal-gibbs`; see [`metal_sa_identity`].
///
/// # Errors
///
/// The errors of [`metal_sa_identity`].
///
/// # Examples
///
/// ```
/// use quip_solver_metal::{capacity, metal_gibbs_identity};
/// use quip_solver_core::quip_proto::v1::{Algorithm, Backend};
///
/// let identity = metal_gibbs_identity(&capacity::host_limits(), None).unwrap();
/// assert_eq!(identity.backend, Backend::Metal);
/// assert_eq!(identity.algorithm, Algorithm::Gibbs);
/// assert!(identity.max_nodes > 0);
/// ```
pub fn metal_gibbs_identity(
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<BackendIdentity, CapacityError> {
    identity(Kernel::Gibbs, METAL_ADAPT, METAL_GIBBS_LIMITS, limits, cap)
}

/// Backend identity for `quip-metal-msa`; see [`metal_sa_identity`]. The
/// limits are the GPU's. The ANE holds up to 16,384 nodes, but only with
/// unit coefficients, at most 128 reads, and while it is enabled, so its
/// bounds do not widen the identity.
///
/// # Errors
///
/// The errors of [`metal_sa_identity`].
///
/// # Examples
///
/// ```
/// use quip_solver_metal::{capacity, metal_msa_identity};
/// use quip_solver_core::quip_proto::v1::{Algorithm, Backend};
///
/// let identity = metal_msa_identity(&capacity::host_limits(), None).unwrap();
/// assert_eq!(identity.backend, Backend::Metal);
/// assert_eq!(identity.algorithm, Algorithm::Msa);
/// assert_eq!(identity.adapt.min_reads, 64);
/// ```
pub fn metal_msa_identity(
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<BackendIdentity, CapacityError> {
    identity(Kernel::Msa, METAL_MSA_ADAPT, METAL_MSA_LIMITS, limits, cap)
}

/// Model limits of `quip-metal-otsuka`: those of [`METAL_SA_LIMITS`] with
/// the degree bound [`cluster::sampler::MAX_DEGREE`]. Plain jobs run on the
/// SA kernel, whose sparse storage has no row-length limit.
pub const METAL_OTSUKA_LIMITS: quip_solver_core::ModelLimits = quip_solver_core::ModelLimits {
    max_degree: Some(OTSUKA_MAX_DEGREE),
    ..METAL_SA_LIMITS
};

const OTSUKA_MAX_DEGREE: u32 = {
    assert!(cluster::sampler::MAX_DEGREE <= u32::MAX as usize);
    cluster::sampler::MAX_DEGREE as u32
};

/// Backend identity for `quip-metal-otsuka`. Plain jobs run on Metal SA, so
/// the limits come from the SA capacity model, as for [`metal_sa_identity`],
/// with the degree bound [`cluster::sampler::MAX_DEGREE`]. The SA one-sweep
/// limit refuses the SA couplers at that degree, so `max_edges` is
/// [`capacity::advertised_couplers_within_degree`] and `max_nodes` is
/// [`capacity::advertised_nodes_for`] those couplers, lowered to `cap`. A
/// family job that does not fit the cluster layout is refused as `Capacity`
/// before any allocation.
///
/// # Errors
///
/// The errors of [`metal_sa_identity`].
///
/// # Examples
///
/// ```
/// use quip_solver_metal::{capacity, metal_otsuka_identity, metal_sa_identity};
/// use quip_solver_core::quip_proto::v1::Algorithm;
///
/// let limits = capacity::host_limits();
/// let otsuka = metal_otsuka_identity(&limits, None).unwrap();
/// let sa = metal_sa_identity(&limits, None).unwrap();
/// assert_eq!(otsuka.algorithm, Algorithm::Otsuka);
/// assert_eq!(otsuka.limits.max_degree, Some(128));
/// assert!(otsuka.max_edges <= sa.max_edges);
/// ```
pub fn metal_otsuka_identity(
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<BackendIdentity, CapacityError> {
    let edges = capacity::advertised_couplers_within_degree(
        Kernel::Sa,
        limits,
        cluster::sampler::MAX_DEGREE,
    );
    let nodes = capacity::advertised_nodes_for(Kernel::Sa, limits, edges, cap)?;
    let mut id = identity_with(Kernel::Sa, METAL_ADAPT, METAL_OTSUKA_LIMITS, nodes, edges)?;
    id.algorithm = quip_solver_core::quip_proto::v1::Algorithm::Otsuka;
    Ok(id)
}

/// The identity of `kernel` on this host with `max_nodes` lowered to `cap`,
/// or, when `cap` is 0 or passes the limit, the exit code of an invalid
/// configuration after a message on standard error. The binaries call it
/// before they open anything, so `--capabilities` prints the same limits.
///
/// # Errors
///
/// [`quip_solver_core::quip_protocol::session::ExitCode::ConfigInvalid`]
/// (64) when the identity is an error.
pub fn host_identity(kernel: Kernel, cap: Option<usize>) -> Result<BackendIdentity, ExitCode> {
    let limits = capacity::host_limits();
    let identity = match kernel {
        Kernel::Sa => metal_sa_identity(&limits, cap),
        Kernel::Gibbs => metal_gibbs_identity(&limits, cap),
        Kernel::Msa => metal_msa_identity(&limits, cap),
    };
    let binary = match kernel {
        Kernel::Sa => "quip-metal-sa",
        Kernel::Gibbs => "quip-metal-gibbs",
        Kernel::Msa => "quip-metal-msa",
    };
    config_exit(identity, binary, cap)
}

/// [`host_identity`] for `quip-metal-otsuka`: [`metal_otsuka_identity`] on
/// this host.
///
/// # Errors
///
/// The errors of [`host_identity`].
pub fn host_otsuka_identity(cap: Option<usize>) -> Result<BackendIdentity, ExitCode> {
    config_exit(
        metal_otsuka_identity(&capacity::host_limits(), cap),
        "quip-metal-otsuka",
        cap,
    )
}

/// The identity, or the configuration exit code after
/// [`config_message`] on standard error.
fn config_exit(
    identity: Result<BackendIdentity, CapacityError>,
    binary: &str,
    cap: Option<usize>,
) -> Result<BackendIdentity, ExitCode> {
    identity.map_err(|e| {
        #[expect(
            clippy::print_stderr,
            reason = "the binary exits before the session installs its subscriber"
        )]
        {
            eprintln!("{}", config_message(binary, cap, e));
        }
        let code = quip_solver_core::quip_protocol::session::ExitCode::ConfigInvalid.as_i32();
        ExitCode::from(u8::try_from(code).unwrap_or(u8::MAX))
    })
}

/// The configuration error of `binary` for an identity error `e`.
fn config_message(binary: &str, cap: Option<usize>, e: CapacityError) -> String {
    match (cap, e) {
        (Some(cap), CapacityError::Nodes { max }) => format!(
            "--max-nodes {cap} is outside 1 to {max}, the nodes this host admits for {binary}"
        ),
        (_, e) => format!("no model fits this host for {binary}: {e}"),
    }
}

/// The couplers the identity of `kernel` advertises on this host, computed
/// once per kernel. The session gates that bound a job by edge count read
/// it, so they refuse exactly past `max_edges`.
pub(crate) fn advertised_edges(kernel: Kernel) -> usize {
    static EDGES: [std::sync::OnceLock<usize>; 3] = [const { std::sync::OnceLock::new() }; 3];
    let slot = match kernel {
        Kernel::Sa => 0,
        Kernel::Gibbs => 1,
        Kernel::Msa => 2,
    };
    *EDGES[slot]
        .get_or_init(|| capacity::advertised_couplers(kernel, crate::sampler::host_device_limits()))
}

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
    /// Stage F for chain-topology leases, built by the first such lease.
    prefilter: std::sync::Mutex<Option<frustration::Prefilter>>,
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
        // Deep enough that preparation keeps the screen fed while the runner
        // waits on the GPU.
        let (salts_tx, salts_rx) = std::sync::mpsc::sync_channel(resident::SCREEN_PREP_BOUND);
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
            prefilter: std::sync::Mutex::new(None),
        }
    }

    /// Report a lease's outcomes until every queued salt has one, the
    /// feeder has stopped, or the lease stops.
    ///
    /// A salt that finishes within [`RERUN_MARGIN_MILLI`] above the live
    /// target is held, and `rerun` queues [`RERUN_COPIES`] more deep runs of
    /// it. The sink takes one push per salt, so the salt is pushed once: on
    /// the first copy below the target, or with the best read set once every
    /// copy has answered.
    fn report_lease(
        &self,
        sink: &quip_solver_core::LeaseSink,
        outcomes: &std::sync::mpsc::Receiver<resident::SaltOutcome>,
        live_target: &resident::LiveTarget,
        fed: &std::sync::atomic::AtomicU64,
        feeding: &std::sync::atomic::AtomicBool,
        rerun: &dyn Fn(u64, u8) -> bool,
    ) -> Result<(), quip_solver_core::SampleError> {
        use std::sync::atomic::Ordering;
        let mut reported = 0u64;
        // Rerun copies queued; each answers once, like a fed salt.
        let mut copies = 0u64;
        let mut held = std::collections::HashMap::<u64, Held>::new();
        while !sink.is_stopped() {
            self.require_runner()?;
            let target = sink.target_energy_milli();
            live_target.set(target);
            // `feeding` is cleared after the last increment of `fed`.
            if !feeding.load(Ordering::Acquire) && reported == fed.load(Ordering::Acquire) + copies
            {
                break;
            }
            let Ok(first) = outcomes.recv_timeout(std::time::Duration::from_millis(1)) else {
                continue;
            };
            for outcome in std::iter::once(first).chain(outcomes.try_iter()) {
                reported += 1;
                let sent = match outcome {
                    // Every salt of a lease shares its topology and
                    // parameters, so a capacity refusal of one salt refuses
                    // the lease. `Capacity` is not fatal: the session closes
                    // the lease with a refund and serves the next one.
                    resident::SaltOutcome::Failed {
                        index,
                        error: quip_solver_core::SampleError::Capacity,
                    } => {
                        tracing::warn!(index, "lease refused for capacity");
                        return Err(quip_solver_core::SampleError::Capacity);
                    }
                    resident::SaltOutcome::Failed { index, error } => {
                        return Err(quip_solver_core::SampleError::DeviceFault(format!(
                            "lease salt {index}: {error}"
                        )));
                    }
                    outcome if held.contains_key(&outcome.index()) => {
                        let index = outcome.index();
                        let Some(entry) = held.get_mut(&index) else {
                            continue;
                        };
                        entry.waiting -= 1;
                        if let resident::SaltOutcome::Survived { reads, .. } = outcome {
                            let best = best_energy(&reads);
                            if best < entry.best {
                                entry.best = best;
                                entry.reads = reads;
                            }
                        }
                        let hit = target.is_some_and(|target| entry.best < target);
                        let sent = if !entry.pushed && (hit || entry.waiting == 0) {
                            entry.pushed = true;
                            tracing::info!(
                                index,
                                first = entry.first,
                                best = entry.best,
                                ?target,
                                hit,
                                "near-miss rerun finished"
                            );
                            sink.push(index, std::mem::take(&mut entry.reads))
                        } else {
                            Ok(())
                        };
                        if entry.waiting == 0 {
                            held.remove(&index);
                        }
                        sent
                    }
                    resident::SaltOutcome::Survived { index, reads } => {
                        let best = best_energy(&reads);
                        let near = target.is_some_and(|target| {
                            best >= target && best - target <= RERUN_MARGIN_MILLI
                        });
                        let queued = if near {
                            (1..=RERUN_COPIES)
                                .take_while(|&copy| rerun(index, copy))
                                .count()
                        } else {
                            0
                        };
                        if queued == 0 {
                            sink.push(index, reads)
                        } else {
                            tracing::info!(index, best, ?target, queued, "near-miss rerun");
                            copies += queued as u64;
                            held.insert(
                                index,
                                Held {
                                    reads,
                                    best,
                                    first: best,
                                    waiting: queued,
                                    pushed: false,
                                },
                            );
                            Ok(())
                        }
                    }
                    resident::SaltOutcome::Screened {
                        index,
                        energy_milli,
                    } => sink.screen(index, energy_milli),
                    resident::SaltOutcome::Dropped { index } => {
                        tracing::debug!(index, "lease salt dropped before reporting");
                        Ok(())
                    }
                };
                if sent.is_err() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// The stage F prefilter when this lease's salts take it, built on first
    /// use. The guard holds it for the lease's feed; lease turns feed one
    /// lease at a time.
    fn prefilter_for(
        &self,
        topology: &quip_solver_core::quip_protocol::lease::TopologyView,
        params: &SampleParams,
    ) -> Option<std::sync::MutexGuard<'_, Option<frustration::Prefilter>>> {
        let settings = *self
            .cascade
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let chain = cascade::topology_fingerprint(topology.num_nodes, &topology.edges)
            == cascade::CHAIN_GATES.fingerprint;
        if !(settings.screen && settings.prefilter && settings.chain_gated(chain, params)) {
            return None;
        }
        let mut guard = self
            .prefilter
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.is_none() {
            match frustration::Prefilter::for_view(&self.device, topology)? {
                Ok(prefilter) => *guard = Some(prefilter),
                Err(error) => {
                    tracing::error!(%error, "prefilter unavailable");
                    return None;
                }
            }
        }
        Some(guard)
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

/// Rerun copies of a near-miss salt, each a 64-read deep run to
/// [`cascade::ChainGates::full_sweeps`] with its own seed.
const RERUN_COPIES: u8 = 3;

/// A salt whose short deep run finishes at most this far above the live
/// target takes the long rerun copies.
const RERUN_MARGIN_MILLI: i64 = 40_000;

/// A near-miss salt waiting on its rerun copies.
struct Held {
    /// The best read set so far.
    reads: Vec<SamplerResult>,
    best: i64,
    /// The salt's own best, before any copy.
    first: i64,
    /// Copies that have not answered.
    waiting: usize,
    pushed: bool,
}

fn best_energy(reads: &[SamplerResult]) -> i64 {
    reads
        .iter()
        .map(|read| read.energy_milli)
        .min()
        .unwrap_or(i64::MAX)
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
            let (rerun_topology, rerun_top10) = (std::sync::Arc::clone(&topology), top10.clone());
            let (fed, feeding, stop) = (&fed, &feeding, &stop);
            let (target, top10, unit_reply) = (live_target.clone(), top10.clone(), reply.clone());
            let topology = std::sync::Arc::clone(&topology);
            scope.spawn(move || {
                let _done = StoreOnDrop(feeding, false);
                let stopped = || {
                    if sink.is_stopped() {
                        // Still holding the turn; see `lease_round`.
                        self.lease_round.fetch_add(1, Ordering::AcqRel);
                        return true;
                    }
                    stop.load(Ordering::Acquire)
                };
                let feed = |index: u64, nonce: [u8; 32]| {
                    let salt = resident::Salt {
                        topology: std::sync::Arc::clone(&topology),
                        nonce,
                        index,
                        round,
                        rerun: 0,
                        params: params.clone(),
                        stop: std::sync::Arc::clone(stop),
                        target: target.clone(),
                        top10: top10.clone(),
                        reply: unit_reply.clone(),
                    };
                    if self.salts_tx.send(salt).is_err() {
                        return false;
                    }
                    fed.fetch_add(1, Ordering::Release);
                    true
                };
                let mut prefilter = self.prefilter_for(&topology, params);
                if let Some(prefilter) = prefilter.as_mut().and_then(|guard| guard.as_mut()) {
                    // A rejected salt answers straight to the sink: it never
                    // reaches the runner, so it is not counted as fed.
                    let mut fed_all = true;
                    let ran = prefilter.run(
                        0..lease.salt_count(),
                        |i| lease.nonce(i),
                        |index, kept| {
                            fed_all = !stopped()
                                && if kept {
                                    feed(index, lease.nonce(index))
                                } else {
                                    sink.screen(index, i64::MAX).is_ok()
                                };
                            fed_all
                        },
                    );
                    if let Err(error) = ran {
                        tracing::error!(%error, "prefilter failed; feeding the rest unfiltered");
                    } else if !fed_all {
                        return;
                    } else {
                        turn.release();
                        tracing::debug!(turn = turn.id(), "lease fed");
                        return;
                    }
                }
                drop(prefilter);
                for index in 0..lease.salt_count() {
                    if stopped() || !feed(index, lease.nonce(index)) {
                        return;
                    }
                }
                turn.release();
                tracing::debug!(turn = turn.id(), "lease fed");
            });
            // A near-miss rerun copy goes straight to the runner: the
            // prefilter and screen already passed this salt.
            let rerun = |index: u64, copy: u8| {
                let salt = resident::Salt {
                    topology: std::sync::Arc::clone(&rerun_topology),
                    nonce: lease.nonce(index),
                    index,
                    round,
                    rerun: copy,
                    params: params.clone(),
                    stop: std::sync::Arc::clone(stop),
                    target: live_target.clone(),
                    top10: rerun_top10.clone(),
                    reply: reply.clone(),
                };
                !stop.load(Ordering::Acquire) && self.salts_tx.send(salt).is_ok()
            };
            let reported = self.report_lease(sink, &outcomes, &live_target, fed, feeding, &rerun);
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
/// use quip_solver_metal::{host_identity, run_metal, Kernel, SaTag};
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
/// let identity = host_identity(Kernel::Sa, None).unwrap();
/// let _code = run_metal::<SaTag>(identity, &common, 0, 100, false);
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

    fn host_identities() -> [(super::Kernel, quip_solver_core::BackendIdentity); 3] {
        use super::Kernel;
        let limits = crate::sampler::host_device_limits();
        [
            (Kernel::Sa, super::metal_sa_identity(limits, None).unwrap()),
            (
                Kernel::Gibbs,
                super::metal_gibbs_identity(limits, None).unwrap(),
            ),
            (
                Kernel::Msa,
                super::metal_msa_identity(limits, None).unwrap(),
            ),
        ]
    }

    #[test]
    fn msa_identity_advertises_the_multi_spin_kernel() {
        let [_, _, (_, msa)] = host_identities();
        assert_eq!(
            msa.backend,
            quip_solver_core::quip_proto::v1::Backend::Metal
        );
        assert_eq!(
            msa.algorithm,
            quip_solver_core::quip_proto::v1::Algorithm::Msa
        );
        assert_eq!(msa.features, super::FEATURES);
        // Reads are pinned to whole words.
        assert_eq!(msa.adapt.min_reads % 32, 0);
        assert_eq!(msa.adapt.min_reads, msa.adapt.max_reads);
        assert!(msa.adapt.min_sweeps <= msa.adapt.max_sweeps);
    }

    /// Every identity lists `linear-schedule` and its model limits, and
    /// claims exact coefficients. Device-free, like `--capabilities`.
    #[test]
    fn identities_advertise_linear_schedule_and_their_limits() {
        use quip_solver_core::ModelLimits;
        let limits = ModelLimits {
            max_degree: Some(crate::sampler::MSA_SPARSE_MAX_DEG as u32),
            max_local_field_milli: Some(268_435_000),
            exact_resolution_milli: Some(1),
        };
        for (kernel, identity) in host_identities() {
            for feature in [
                quip_solver_core::LINEAR_SCHEDULE_FEATURE,
                quip_solver_core::EXACT_COEFFICIENTS_FEATURE,
            ] {
                assert!(identity.features.contains(&feature), "{kernel:?}");
            }
            assert_eq!(identity.limits, limits, "{kernel:?}");
        }
    }

    /// The identity states the limits the session admits: `max_nodes` and
    /// `max_edges` are the capacity model's advertised values on this host,
    /// and the edge gates of the session read the same `max_edges`.
    #[test]
    fn the_identity_matches_the_capacity_model() {
        use crate::capacity::{advertised_couplers, advertised_nodes, session_admits};
        let limits = crate::sampler::host_device_limits();
        for (kernel, identity) in host_identities() {
            let nodes = advertised_nodes(kernel, limits, None).unwrap();
            let edges = advertised_couplers(kernel, limits);
            assert_eq!(identity.max_nodes as usize, nodes, "{kernel:?}");
            assert_eq!(identity.max_edges as usize, edges, "{kernel:?}");
            assert_eq!(super::advertised_edges(kernel), edges, "{kernel:?}");
            assert!(session_admits(kernel, nodes, edges, limits), "{kernel:?}");
            // The kernel arrays no longer bound the identity.
            assert!(
                nodes > crate::sampler::kernel_max_nodes(kernel),
                "{kernel:?}: {nodes}"
            );
            // `--max-nodes` lowers the node limit only.
            let capped =
                super::identity(kernel, identity.adapt, identity.limits, limits, Some(1_000))
                    .unwrap();
            assert_eq!(capped.max_nodes, 1_000, "{kernel:?}");
            assert_eq!(capped.max_edges, identity.max_edges, "{kernel:?}");
        }
    }

    /// The configuration error names the binary.
    #[test]
    fn the_configuration_error_names_the_binary() {
        use crate::capacity::CapacityError;
        assert_eq!(
            super::config_message("quip-metal-otsuka", Some(70_000), CapacityError::Nodes { max: 60_743 }),
            "--max-nodes 70000 is outside 1 to 60743, the nodes this host admits for quip-metal-otsuka"
        );
        assert_eq!(
            super::config_message("quip-metal-sa", None, CapacityError::Nodes { max: 0 }),
            "no model fits this host for quip-metal-sa: 0 nodes is the most a kernel can index"
        );
    }

    /// A `--max-nodes` past the limit fails before any device opens, with
    /// the limit in the error.
    #[test]
    fn max_nodes_above_the_limit_fails_at_open() {
        let limits = crate::sampler::host_device_limits();
        for (kernel, identity) in host_identities() {
            let limit = identity.max_nodes as usize;
            let over = super::identity(
                kernel,
                identity.adapt,
                identity.limits,
                limits,
                Some(limit + 1),
            );
            assert_eq!(
                over.map(|_| ()),
                Err(crate::capacity::CapacityError::Nodes { max: limit }),
                "{kernel:?}"
            );
            assert!(
                super::host_identity(kernel, Some(limit + 1)).is_err(),
                "{kernel:?}"
            );
            // The harness reads `max_nodes == 0` as no bound, so a zero cap
            // would lift the limit it was meant to lower.
            let zero = super::identity(kernel, identity.adapt, identity.limits, limits, Some(0));
            assert_eq!(
                zero.map(|_| ()),
                Err(crate::capacity::CapacityError::Nodes { max: limit }),
                "{kernel:?}"
            );
            assert!(super::host_identity(kernel, Some(0)).is_err(), "{kernel:?}");
            assert_eq!(
                super::host_identity(kernel, Some(limit)).map(|id| id.max_nodes),
                Ok(identity.max_nodes),
                "{kernel:?}"
            );
        }
    }

    /// `count` values from 1 to `max`, spaced evenly on a log scale, with
    /// `max` itself.
    fn log_grid(max: usize, count: u32) -> Vec<usize> {
        let mut values: Vec<usize> = (0..count)
            .map(|k| {
                (max as f64)
                    .powf(f64::from(k) / f64::from(count - 1))
                    .round() as usize
            })
            .chain([max])
            .collect();
        values.dedup();
        values
    }

    /// Every shape inside the advertised box is admitted: each node count
    /// up to `max_nodes` with each coupler count up to `max_edges` whose mean
    /// degree the advertised degree bound allows (a graph's mean degree is at
    /// most its largest). A coarse log grid covers the box, including its
    /// edges. With no degree bound the grid reaches the dense and high-degree
    /// shapes that the session refuses. The Otsuka identity runs plain jobs
    /// on SA, so its box, with degree bound 128, is checked against SA.
    #[test]
    fn every_shape_in_the_advertised_box_is_admitted() {
        use crate::capacity::session_admits;
        let limits = crate::sampler::host_device_limits();
        let otsuka = (
            super::Kernel::Sa,
            super::metal_otsuka_identity(limits, None).unwrap(),
        );
        for (kernel, identity) in host_identities().into_iter().chain([otsuka]) {
            let bound = identity
                .limits
                .max_degree
                .filter(|&d| d != 0)
                .map(|d| d as usize);
            let mut refused = Vec::new();
            for nodes in log_grid(identity.max_nodes as usize, 48) {
                let complete = nodes * (nodes - 1) / 2;
                let most = bound.map_or(complete, |d| complete.min(nodes * d / 2));
                let most = most.min(identity.max_edges as usize);
                for edges in log_grid(most.max(1), 24) {
                    if edges <= most && !session_admits(kernel, nodes, edges, limits) {
                        refused.push((nodes, edges));
                    }
                }
            }
            assert!(
                refused.is_empty(),
                "{kernel:?}: {} refused shapes, first {:?}",
                refused.len(),
                &refused[..refused.len().min(4)]
            );
        }
    }

    /// The harness refuses a node past the advertised degree bound, and the
    /// session admits a multi-spin model whose hub reaches it: a ring of
    /// 20,000 nodes with one hub of degree 62 is admitted, and the same ring
    /// with a hub of degree 63 is refused by the identity. The session
    /// refuses it too: no sparse multi-spin tier walks the row, and the
    /// model is too large for dense-matrix storage.
    #[test]
    fn a_multi_spin_hub_at_the_degree_bound_is_admitted() {
        use quip_solver_core::SampleParams;
        let [_, _, (_, msa)] = host_identities();
        let bound = msa.limits.max_degree.unwrap() as usize;
        assert_eq!(bound, crate::sampler::MSA_SPARSE_MAX_DEG);
        let hub = |degree: usize| {
            let n = 20_000;
            let mut edges: Vec<(usize, usize)> = (0..n).map(|u| (u, (u + 1) % n)).collect();
            // Node 0 already has two ring neighbours, 1 and n - 1.
            edges.extend((0..degree - 2).map(|k| (0, 2 + 2 * k)));
            let j = (0..edges.len())
                .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            super::IsingGraph::new(vec![0.0; n], j, edges)
        };
        let max_degree = |graph: &super::IsingGraph| {
            let mut degree = vec![0usize; graph.num_nodes()];
            for &(u, v) in graph.edges.iter() {
                degree[u] += 1;
                degree[v] += 1;
            }
            degree.into_iter().max().unwrap()
        };
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 64,
            ..Default::default()
        };
        let at = hub(bound);
        assert_eq!(max_degree(&at), bound);
        crate::sampler::admit_one(&at, &params, super::Kernel::Msa).unwrap();
        let past = hub(bound + 1);
        assert!(max_degree(&past) > bound, "the identity refuses it");
        assert_eq!(
            crate::sampler::admit_one(&past, &params, super::Kernel::Msa)
                .map(|_| ())
                .map_err(|e| e.to_sample_error()),
            Err(quip_solver_core::SampleError::Capacity)
        );
    }

    /// Plain jobs on `quip-metal-otsuka` run on SA up to degree 128. The
    /// corners of the Otsuka box pass the SA session admission, and so does
    /// a ring at the Otsuka node limit with one hub of degree 128.
    #[test]
    fn an_sa_hub_at_the_otsuka_degree_bound_is_admitted() {
        use quip_solver_core::SampleParams;
        let limits = crate::sampler::host_device_limits();
        let otsuka = super::metal_otsuka_identity(limits, None).unwrap();
        let bound = otsuka.limits.max_degree.unwrap() as usize;
        assert_eq!(bound, 128);
        // Both corners of the box: every coupler at degree 128, and at the
        // node limit.
        let edges = otsuka.max_edges as usize;
        for nodes in [edges.div_ceil(bound / 2), otsuka.max_nodes as usize] {
            assert!(
                crate::capacity::session_admits(super::Kernel::Sa, nodes, edges, limits),
                "{nodes} nodes, {edges} couplers"
            );
        }
        let n = otsuka.max_nodes as usize;
        let mut edges: Vec<(usize, usize)> = (0..n).map(|u| (u, (u + 1) % n)).collect();
        edges.extend((0..bound - 2).map(|k| (0, 2 + 2 * k)));
        let j = (0..edges.len())
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let graph = super::IsingGraph::new(vec![0.0; n], j, edges);
        for reads in [32, 256] {
            let params = SampleParams {
                num_reads: reads,
                num_sweeps: 64,
                ..Default::default()
            };
            crate::sampler::admit_one(&graph, &params, super::Kernel::Sa).unwrap();
        }
    }

    /// `nodes` nodes with `edges` couplers: every pair when `edges` is the
    /// complete count, else circulant offsets 1, 2, ... in turn. `coupling`
    /// gives coupler `k` its value.
    fn corner_graph(nodes: usize, edges: usize, coupling: fn(usize) -> f64) -> super::IsingGraph {
        let mut pairs = Vec::with_capacity(edges);
        if edges == nodes * (nodes - 1) / 2 {
            pairs.extend((0..nodes).flat_map(|u| ((u + 1)..nodes).map(move |v| (u, v))));
        } else {
            'offsets: for offset in 1..=nodes / 2 {
                for u in 0..nodes {
                    // The offset of half an even ring meets each pair twice.
                    if 2 * offset == nodes && u >= offset {
                        continue;
                    }
                    if pairs.len() == edges {
                        break 'offsets;
                    }
                    pairs.push((u, (u + offset) % nodes));
                }
            }
        }
        assert_eq!(pairs.len(), edges);
        let j = (0..edges).map(coupling).collect();
        super::IsingGraph::new(vec![0.0; nodes], j, pairs)
    }

    /// The advertised corners run through the session admission
    /// (`sampler::admit_one`): `max_nodes` with `max_edges` couplers, at the
    /// reference degree `2 * max_edges / max_nodes`, and the complete model
    /// of `max_edges` couplers. The harness refuses the complete model, whose
    /// degree passes the advertised bound, so it runs only through the
    /// library API. Both are admitted on unit and on wide
    /// coefficients, at the smallest and the largest read count, and one
    /// node more is refused as a capacity error.
    #[test]
    #[cfg_attr(
        not(feature = "long-tests"),
        ignore = "over 30 s in a debug build: run with --features long-tests"
    )]
    fn the_advertised_corners_pass_the_session_admission() {
        use quip_solver_core::SampleParams;
        let unit: fn(usize) -> f64 = |k| if k % 2 == 0 { 1.0 } else { -1.0 };
        let wide: fn(usize) -> f64 = |k| if k % 2 == 0 { 1.3 } else { -0.127 };
        for (kernel, identity) in host_identities() {
            let nodes = identity.max_nodes as usize;
            let edges = identity.max_edges as usize;
            let complete = (2..).find(|n| n * (n - 1) / 2 >= edges).unwrap();
            assert_eq!(complete * (complete - 1) / 2, edges, "{kernel:?}");
            let graphs: Vec<super::IsingGraph> = [unit, wide]
                .into_iter()
                .flat_map(|coupling| {
                    [
                        corner_graph(nodes, edges, coupling),
                        corner_graph(complete, edges, coupling),
                        corner_graph(nodes + 1, edges, coupling),
                    ]
                })
                .collect();
            for reads in [32, 256] {
                let params = SampleParams {
                    num_reads: reads,
                    num_sweeps: 64,
                    ..Default::default()
                };
                for corners in graphs.chunks(3) {
                    for graph in &corners[..2] {
                        if let Err(e) = crate::sampler::admit_one(graph, &params, kernel) {
                            panic!(
                                "{kernel:?}: {} nodes, {reads} reads: {e:?}",
                                graph.num_nodes()
                            );
                        }
                    }
                    assert_eq!(
                        crate::sampler::admit_one(&corners[2], &params, kernel)
                            .map(|_| ())
                            .map_err(|e| e.to_sample_error()),
                        Err(quip_solver_core::SampleError::Capacity),
                        "{kernel:?}: {reads} reads"
                    );
                }
            }
        }
    }

    #[test]
    fn msa_envelope_resolves_to_the_measured_shape_at_the_testnet_target() {
        use quip_solver_core::adapt::adapt_params;
        let [_, _, (_, msa)] = host_identities();
        // Aglais qblock 3278: zero-field Advantage2 graph, target below the
        // adapt model's hardest energy, so the difficulty saturates at 1.0 and
        // the job is the envelope's maximum: the shape the sweeps study picked.
        let job = adapt_params(-14_624_068, 1, 4_577, 41_514, &[0], &msa.adapt);
        assert_eq!(job.num_reads, 64);
        assert_eq!(job.num_sweeps, 14_336);
        // An easy target lands on the floor, still a whole number of words.
        let easy = adapt_params(0, 1, 4_577, 41_514, &[0], &msa.adapt);
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
