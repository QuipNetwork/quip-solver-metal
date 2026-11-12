//! Metal device + runtime-compiled SA / Gibbs / multi-spin pipelines for one Apple GPU.
//!
//! One process owns one device (`[metal.N]` → device N / miner id `metal-N`).
//! Kernels are JIT-compiled from `.metal` source via
//! `new_library_with_source` (analogous to NVRTC in the CUDA crate).
//!
//! # Send / Sync
//!
//! The `metal` crate's owned wrapper types (`Device`, `CommandQueue`,
//! `ComputePipelineState`, `Buffer`, ...) are declared `Send + Sync` by the
//! crate itself (`foreign_type! { pub unsafe type ...: Sync + Send { ... } }`
//! in `metal::lib`), so `Arc<MetalDevice>` would in fact compile. This crate
//! still owns `MetalDevice` directly (no `Arc`) and drives `sample_ising`
//! synchronously inside `run_session`'s single `block_on` future — matching
//! the CUDA crate's one-thread-per-job model and keeping every GPU call on
//! the same OS thread for the life of the process, which is the simpler and
//! more conservative choice given Apple's own guidance that `MTLDevice` /
//! `MTLCommandQueue` usage from multiple threads needs care.

use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions};
use thiserror::Error;

const SA_SRC: &str = include_str!("../kernels/sa.metal");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.metal");

/// Prepended to a kernel source to build its wide variant.
const WIDE_DEFINE: &str = "#define QUIP_WIDE 1\n";

/// Prepended to a kernel source to build its Float32 variant.
const FLOAT_DEFINE: &str = "#define QUIP_FLOAT 1\n";

/// Prepended to a wide variant for `int8_t` coefficient storage.
const WIDE_I8_DEFINE: &str = "#define QUIP_WIDE_I8 1\n";

/// Prepended to a weighted multi-spin variant whose rows may pass one chunk.
const MSA_DENSE_DEFINE: &str = "#define QUIP_MSA_DENSE 1\n";

/// Prepended to a weighted multi-spin variant whose spin words spill past
/// threadgroup memory.
const MSA_SPILL_DEFINE: &str = "#define QUIP_MSA_SPILL 1\n";

/// Prepended to an SA variant that keeps each read's spins and
/// `delta_energy` in device memory instead of the thread-private frame.
const SA_GLOBAL_DEFINE: &str = "#define QUIP_SA_GLOBAL 1\n";

fn global_state(source: &str) -> String {
    format!("{SA_GLOBAL_DEFINE}{source}")
}

/// Prepended to a chromatic Gibbs variant that keeps the spins past its
/// threadgroup prefix in each read's slice of the persistent buffer.
const GIBBS_SPILL_DEFINE: &str = "#define QUIP_GIBBS_SPILL 1\n";

fn gibbs_spill(source: &str) -> String {
    format!("{GIBBS_SPILL_DEFINE}{source}")
}

fn msa_dense(source: &str) -> String {
    format!("{MSA_DENSE_DEFINE}{source}")
}

/// The milli spill build walks rows of any length, as the dense build does.
fn msa_spill(source: &str) -> String {
    format!("{MSA_SPILL_DEFINE}{MSA_DENSE_DEFINE}{source}")
}

fn float_spill(source: &str) -> String {
    format!("{MSA_SPILL_DEFINE}{FLOAT_DEFINE}{source}")
}

fn wide_i8(source: &str) -> String {
    format!("{WIDE_DEFINE}{WIDE_I8_DEFINE}{source}")
}

fn wide(source: &str) -> String {
    format!("{WIDE_DEFINE}{source}")
}

fn float(source: &str) -> String {
    format!("{FLOAT_DEFINE}{source}")
}

/// The dense-matrix SA kernel, appended to [`SA_SRC`] for its typedefs, RNG,
/// and spin helpers; see `kernels/sa_dense.metal`.
const SA_DENSE_SRC: &str = include_str!("../kernels/sa_dense.metal");

/// The dense-matrix SA kernel source after the `defines` of its variant. The
/// kernel keeps each read's state in the persistent buffers, so every
/// variant is a global build.
fn sa_matrix(defines: &str) -> String {
    format!("{SA_GLOBAL_DEFINE}{defines}{SA_SRC}\n{SA_DENSE_SRC}")
}

/// The dense-matrix Gibbs kernel, appended to [`GIBBS_SRC`] for its typedefs,
/// RNG, and heat-bath helper; see `kernels/gibbs_dense.metal`.
const GIBBS_DENSE_SRC: &str = include_str!("../kernels/gibbs_dense.metal");

/// The dense-matrix Gibbs kernel source after the `defines` of its variant.
fn gibbs_matrix(defines: &str) -> String {
    format!("{defines}{GIBBS_SRC}\n{GIBBS_DENSE_SRC}")
}

/// Multi-spin coded SA, see `kernels/msa.metal`.
const MSA_SRC: &str = include_str!("../kernels/msa.metal");

/// The dense-matrix multi-spin kernel, appended to [`MSA_SRC`] for its RNG,
/// spin stores, and coefficient types; see `kernels/msa_dense.metal`.
const MSA_DENSE_SRC: &str = include_str!("../kernels/msa_dense.metal");

/// Prepended to a dense-matrix multi-spin variant whose matrix holds two bit
/// planes per 32 columns (a unit model).
const MSA_MATRIX_BITS_DEFINE: &str = "#define QUIP_MSA_MATRIX_BITS 1\n";

/// Prepended to the bit-matrix variant that accepts with the unit kernel's
/// geometric draw instead of the weighted lane's threshold rows.
const MSA_MATRIX_UNIT_DEFINE: &str = "#define QUIP_MSA_MATRIX_UNIT 1\n";

/// The dense-matrix kernel source after the `defines` of its variant.
fn msa_matrix(defines: &str) -> String {
    format!("{defines}{MSA_SRC}\n{MSA_DENSE_SRC}")
}

/// Failure opening a Metal device or compiling its SA, Gibbs, or multi-spin pipelines.
#[derive(Debug, Error)]
pub enum MetalError {
    /// Driver refused pipeline-state creation for a compiled entry point.
    #[error("Metal driver: {0}")]
    Driver(String),
    /// Kernel source or entry point failed to compile.
    #[error("Metal compile: {0}")]
    Compile(String),
    /// No Metal device at the requested `Device::all()` index.
    #[error("no Metal device at index {0}")]
    NoDevice(usize),
}

/// Loaded pipelines + queue bound to a single device.
///
/// Kept on one OS thread by convention, not by type constraint — see the
/// module-level Send / Sync note above.
///
/// Fields are `pub(crate)`, not `pub`: the pipelines must belong to the
/// device opened at `device_index`, which only [`MetalDevice::open`] can
/// guarantee. Every reader is in-crate (`crate::sampler`, `crate::streaming`).
#[derive(Debug)]
pub struct MetalDevice {
    /// Which `Device::all()` slot this came from (the `N` in miner id
    /// `metal-N`). Nothing reads it yet — narrowing it from `pub` is what
    /// made that visible. Kept because it is the device's identity and the
    /// natural field for diagnostics to report; `expect` (not `allow`) so
    /// this marker fires the moment a reader appears and can be deleted.
    #[expect(dead_code, reason = "device identity retained for diagnostics")]
    pub(crate) device_index: usize,
    pub(crate) device: Device,
    pub(crate) queue: metal::CommandQueue,
    pub(crate) sa: ComputePipelineState,
    pub(crate) sa_wide: ComputePipelineState,
    /// `sa` for Float32 models: `float` coefficients and delta.
    pub(crate) sa_float: ComputePipelineState,
    /// `sa_wide` with `int8_t` coefficients times one integer scale.
    pub(crate) sa_wide_i8: ComputePipelineState,
    /// `sa` with each read's spins and `delta_energy` in device memory (the
    /// `QUIP_SA_GLOBAL` build), for a model past the thread-private frame of
    /// `sampler::SA_MAX_NODES` nodes.
    pub(crate) sa_global: ComputePipelineState,
    /// `sa_wide` on the global build.
    pub(crate) sa_wide_global: ComputePipelineState,
    /// `sa_float` on the global build.
    pub(crate) sa_float_global: ComputePipelineState,
    /// `sa_wide_i8` on the global build.
    pub(crate) sa_wide_i8_global: ComputePipelineState,
    pub(crate) gibbs: ComputePipelineState,
    pub(crate) gibbs_wide: ComputePipelineState,
    /// `gibbs` for Float32 models: `float` coefficients and heat-bath.
    pub(crate) gibbs_float: ComputePipelineState,
    /// `gibbs_wide` with `int8_t` coefficients times one integer scale.
    pub(crate) gibbs_wide_i8: ComputePipelineState,
    /// Chromatic (node-parallel) Gibbs: one threadgroup per sample, threads
    /// split the nodes of each color, `threadgroup`-shared state. Same buffer
    /// layout as `gibbs`, different dispatch geometry.
    pub(crate) gibbs_parallel: ComputePipelineState,
    pub(crate) gibbs_parallel_wide: ComputePipelineState,
    /// `gibbs_parallel` for Float32 models.
    pub(crate) gibbs_parallel_float: ComputePipelineState,
    /// `gibbs_parallel_wide` with `int8_t` coefficients times one integer scale.
    pub(crate) gibbs_parallel_wide_i8: ComputePipelineState,
    /// `gibbs_parallel` on the spill build (`QUIP_GIBBS_SPILL`), for a model
    /// past the threadgroup array of `sampler::GIBBS_MAX_NODES` spins.
    pub(crate) gibbs_parallel_spill: ComputePipelineState,
    /// `gibbs_parallel_wide` on the spill build.
    pub(crate) gibbs_parallel_wide_spill: ComputePipelineState,
    /// `gibbs_parallel_float` on the spill build.
    pub(crate) gibbs_parallel_float_spill: ComputePipelineState,
    /// `gibbs_parallel_wide_i8` on the spill build.
    pub(crate) gibbs_parallel_wide_i8_spill: ComputePipelineState,
    /// Multi-spin coded SA: one threadgroup per (problem, 32-replica word),
    /// threads split each colour class, spin words in `threadgroup` memory.
    /// Same buffer layout as `gibbs_parallel` with `words` at slot 19.
    pub(crate) msa: ComputePipelineState,
    /// `msa` for weighted models: milli `int` coefficients and host-built
    /// Metropolis threshold rows at slot 25.
    pub(crate) msa_weighted: ComputePipelineState,
    /// `msa_weighted` for Float32 models: `float` coefficients and a
    /// per-lane `float` Metropolis test, no threshold rows.
    pub(crate) msa_float: ComputePipelineState,
    /// `msa_weighted` with `int8_t` coefficients times one integer scale.
    pub(crate) msa_weighted_i8: ComputePipelineState,
    /// `msa_weighted` for a model with a row past one chunk of
    /// `sampler::MSA_UNIT_MAX_DEG` neighbours (the `QUIP_MSA_DENSE` build).
    /// It holds `int` coefficients only: the host never stores a dense
    /// batch as `int8_t`, which was the slower storage there.
    pub(crate) msa_weighted_dense: ComputePipelineState,
    /// `msa_weighted_dense` for a model of more nodes than threadgroup memory
    /// holds (the `QUIP_MSA_SPILL` build): words past `shared_spins` live in
    /// device memory. It holds `int` coefficients only.
    pub(crate) msa_weighted_spill: ComputePipelineState,
    /// `msa_float` with spin words past `shared_spins` in device memory.
    pub(crate) msa_float_spill: ComputePipelineState,
    /// Dense-matrix multi-spin (`kernels/msa_dense.metal`) for a unit model
    /// the unit kernel would run: a bit matrix and the unit kernel's
    /// geometric draw.
    pub(crate) msa_matrix_bits_unit: ComputePipelineState,
    /// Dense-matrix multi-spin for a unit model the weighted lane would run:
    /// a bit matrix and the milli threshold rows.
    pub(crate) msa_matrix_bits: ComputePipelineState,
    /// Dense-matrix multi-spin with an `int8_t` matrix times one scale.
    pub(crate) msa_matrix_i8: ComputePipelineState,
    /// Dense-matrix multi-spin with a milli `int` matrix.
    pub(crate) msa_matrix_int: ComputePipelineState,
    /// Dense-matrix multi-spin with a `float` matrix and the float lane's
    /// acceptance.
    pub(crate) msa_matrix_float: ComputePipelineState,
    /// Dense-matrix SA (`kernels/sa_dense.metal`) with an `int8_t` matrix and
    /// `int8_t` deltas. A unit model runs here too.
    pub(crate) sa_matrix_i8: ComputePipelineState,
    /// Dense-matrix SA with an `int8_t` matrix times one integer scale and
    /// milli `int` deltas.
    pub(crate) sa_matrix_wide_i8: ComputePipelineState,
    /// Dense-matrix SA with a milli `int` matrix.
    pub(crate) sa_matrix_int: ComputePipelineState,
    /// Dense-matrix SA with a `float` matrix.
    pub(crate) sa_matrix_float: ComputePipelineState,
    /// Dense-matrix Gibbs (`kernels/gibbs_dense.metal`) with an `int8_t`
    /// matrix of whole units. A unit model runs here too.
    pub(crate) gibbs_matrix_i8: ComputePipelineState,
    /// Dense-matrix Gibbs with an `int8_t` matrix times one integer scale and
    /// the wide build's integer field.
    pub(crate) gibbs_matrix_wide_i8: ComputePipelineState,
    /// Dense-matrix Gibbs with a milli `int` matrix.
    pub(crate) gibbs_matrix_int: ComputePipelineState,
    /// Dense-matrix Gibbs with a `float` matrix.
    pub(crate) gibbs_matrix_float: ComputePipelineState,
    /// Multi-spin slot-table dispatch with per-slot state and schedules.
    pub(crate) msa_slots: ComputePipelineState,
    /// Nonce-parallel screen: eight models of four reads per word.
    pub(crate) msa_screen: ComputePipelineState,
    /// Lease draw of 32 models per threadgroup into bit-sliced edge signs.
    pub(crate) msa_frustration_draw: ComputePipelineState,
    /// Frustrated-cycle counts of each drawn 32-model group.
    pub(crate) msa_frustration_count: ComputePipelineState,
    /// One CSR topology, replaced when `n`, edges, or the colouring flag change.
    pub(crate) topology_cache: crate::sampler::TopologyCache,
    /// Multi-spin and SA batch buffers returned after their command buffers retire.
    pub(crate) buffer_pool: std::sync::Arc<crate::sampler::BufferPool>,
}

impl MetalDevice {
    /// Open device `device_index` and compile the SA, Gibbs and multi-spin kernels.
    ///
    /// Indexing: `Device::all()` order. Index 0 is typically the system
    /// default (Apple Silicon integrated GPU). Higher indices map into
    /// `all()` when multiple Metal devices are present.
    ///
    /// # Errors
    ///
    /// - [`MetalError::NoDevice`] — no Metal devices are visible to this
    ///   process, or `device_index` is past the end of `Device::all()`.
    /// - [`MetalError::Compile`] — a kernel source in `kernels/` failed to
    ///   compile, or the compiled library has no such entry point.
    /// - [`MetalError::Driver`] — the driver refused to build a compute
    ///   pipeline state for an entry point that did compile.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use quip_solver_metal::metal_device::MetalDevice;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = MetalDevice::open(0)?;
    /// let _ = device;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(device_index: usize) -> Result<Self, MetalError> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err(MetalError::NoDevice(device_index));
        }
        let device = devices
            .into_iter()
            .nth(device_index)
            .ok_or(MetalError::NoDevice(device_index))?;

        let sa = compile_pipeline(&device, SA_SRC, "pure_simulated_annealing")?;
        let sa_wide = compile_pipeline(&device, &wide(SA_SRC), "pure_simulated_annealing")?;
        let sa_float = compile_pipeline(&device, &float(SA_SRC), "pure_simulated_annealing")?;
        let sa_wide_i8 = compile_pipeline(&device, &wide_i8(SA_SRC), "pure_simulated_annealing")?;
        let sa_global =
            compile_pipeline(&device, &global_state(SA_SRC), "pure_simulated_annealing")?;
        let sa_wide_global = compile_pipeline(
            &device,
            &global_state(&wide(SA_SRC)),
            "pure_simulated_annealing",
        )?;
        let sa_float_global = compile_pipeline(
            &device,
            &global_state(&float(SA_SRC)),
            "pure_simulated_annealing",
        )?;
        let sa_wide_i8_global = compile_pipeline(
            &device,
            &global_state(&wide_i8(SA_SRC)),
            "pure_simulated_annealing",
        )?;
        let gibbs = compile_pipeline(&device, GIBBS_SRC, "block_gibbs_sampler")?;
        let gibbs_wide = compile_pipeline(&device, &wide(GIBBS_SRC), "block_gibbs_sampler")?;
        let gibbs_float = compile_pipeline(&device, &float(GIBBS_SRC), "block_gibbs_sampler")?;
        let gibbs_wide_i8 = compile_pipeline(&device, &wide_i8(GIBBS_SRC), "block_gibbs_sampler")?;
        let gibbs_parallel = compile_pipeline(&device, GIBBS_SRC, "block_gibbs_parallel")?;
        let gibbs_parallel_wide =
            compile_pipeline(&device, &wide(GIBBS_SRC), "block_gibbs_parallel")?;
        let gibbs_parallel_float =
            compile_pipeline(&device, &float(GIBBS_SRC), "block_gibbs_parallel")?;
        let gibbs_parallel_wide_i8 =
            compile_pipeline(&device, &wide_i8(GIBBS_SRC), "block_gibbs_parallel")?;
        let gibbs_parallel_spill =
            compile_pipeline(&device, &gibbs_spill(GIBBS_SRC), "block_gibbs_parallel")?;
        let gibbs_parallel_wide_spill = compile_pipeline(
            &device,
            &gibbs_spill(&wide(GIBBS_SRC)),
            "block_gibbs_parallel",
        )?;
        let gibbs_parallel_float_spill = compile_pipeline(
            &device,
            &gibbs_spill(&float(GIBBS_SRC)),
            "block_gibbs_parallel",
        )?;
        let gibbs_parallel_wide_i8_spill = compile_pipeline(
            &device,
            &gibbs_spill(&wide_i8(GIBBS_SRC)),
            "block_gibbs_parallel",
        )?;
        let msa = compile_pipeline(&device, MSA_SRC, "msa_anneal")?;
        let msa_weighted = compile_pipeline(&device, MSA_SRC, "msa_anneal_weighted")?;
        let msa_float = compile_pipeline(&device, &float(MSA_SRC), "msa_anneal_weighted")?;
        let msa_weighted_i8 = compile_pipeline(&device, &wide_i8(MSA_SRC), "msa_anneal_weighted")?;
        let msa_weighted_dense =
            compile_pipeline(&device, &msa_dense(MSA_SRC), "msa_anneal_weighted")?;
        let msa_weighted_spill =
            compile_pipeline(&device, &msa_spill(MSA_SRC), "msa_anneal_weighted")?;
        let msa_float_spill =
            compile_pipeline(&device, &float_spill(MSA_SRC), "msa_anneal_weighted")?;
        let matrix =
            |defines: &str| compile_pipeline(&device, &msa_matrix(defines), "msa_matrix_anneal");
        let msa_matrix_bits_unit =
            matrix(&format!("{MSA_MATRIX_BITS_DEFINE}{MSA_MATRIX_UNIT_DEFINE}"))?;
        let msa_matrix_bits = matrix(MSA_MATRIX_BITS_DEFINE)?;
        let msa_matrix_i8 = matrix(&format!("{WIDE_DEFINE}{WIDE_I8_DEFINE}"))?;
        let msa_matrix_int = matrix("")?;
        let msa_matrix_float = matrix(FLOAT_DEFINE)?;
        let sa_matrix_pipeline =
            |defines: &str| compile_pipeline(&device, &sa_matrix(defines), "sa_matrix_anneal");
        let sa_matrix_i8 = sa_matrix_pipeline("")?;
        let sa_matrix_wide_i8 = sa_matrix_pipeline(&format!("{WIDE_DEFINE}{WIDE_I8_DEFINE}"))?;
        let sa_matrix_int = sa_matrix_pipeline(WIDE_DEFINE)?;
        let sa_matrix_float = sa_matrix_pipeline(FLOAT_DEFINE)?;
        let gibbs_matrix_pipeline =
            |defines: &str| compile_pipeline(&device, &gibbs_matrix(defines), "block_gibbs_matrix");
        let gibbs_matrix_i8 = gibbs_matrix_pipeline("")?;
        let gibbs_matrix_wide_i8 =
            gibbs_matrix_pipeline(&format!("{WIDE_DEFINE}{WIDE_I8_DEFINE}"))?;
        let gibbs_matrix_int = gibbs_matrix_pipeline(WIDE_DEFINE)?;
        let gibbs_matrix_float = gibbs_matrix_pipeline(FLOAT_DEFINE)?;
        let msa_slots = compile_pipeline(&device, MSA_SRC, "msa_anneal_slots")?;
        let msa_screen = compile_pipeline(&device, MSA_SRC, "msa_screen")?;
        let msa_frustration_draw = compile_pipeline(&device, MSA_SRC, "msa_frustration_draw")?;
        let msa_frustration_count = compile_pipeline(&device, MSA_SRC, "msa_frustration_count")?;
        // Fixed dispatch widths must fit before any job reaches the device.
        // Other pipelines cap their dispatch width to the compiled limit.
        for (name, pipeline, width) in [
            ("sa", &sa, crate::sampler::MAX_READS),
            ("sa_global", &sa_global, crate::sampler::MAX_READS),
            ("sa_matrix_i8", &sa_matrix_i8, crate::sampler::MAX_READS),
            ("gibbs", &gibbs, crate::sampler::MAX_READS),
            ("gibbs_parallel_wide", &gibbs_parallel_wide, 128),
            ("gibbs_parallel_wide_i8", &gibbs_parallel_wide_i8, 128),
            ("gibbs_parallel_wide_spill", &gibbs_parallel_wide_spill, 128),
            (
                "gibbs_parallel_wide_i8_spill",
                &gibbs_parallel_wide_i8_spill,
                128,
            ),
            ("gibbs_matrix_wide_i8", &gibbs_matrix_wide_i8, 128),
            ("gibbs_matrix_int", &gibbs_matrix_int, 128),
            (
                "msa_matrix_bits_unit",
                &msa_matrix_bits_unit,
                crate::sampler::MSA_LANES,
            ),
            (
                "msa_matrix_bits",
                &msa_matrix_bits,
                crate::sampler::MSA_LANES,
            ),
            ("msa_matrix_i8", &msa_matrix_i8, crate::sampler::MSA_LANES),
            ("msa_matrix_int", &msa_matrix_int, crate::sampler::MSA_LANES),
            (
                "msa_matrix_float",
                &msa_matrix_float,
                crate::sampler::MSA_LANES,
            ),
            (
                "gibbs_matrix_i8",
                &gibbs_matrix_i8,
                (gibbs_parallel.max_total_threads_per_threadgroup() as usize).clamp(1, 256),
            ),
            (
                "gibbs_matrix_float",
                &gibbs_matrix_float,
                (gibbs_parallel_float.max_total_threads_per_threadgroup() as usize).clamp(1, 256),
            ),
            ("msa_frustration_draw", &msa_frustration_draw, 64),
            ("msa_frustration_count", &msa_frustration_count, 256),
        ] {
            check_threadgroup_width(
                name,
                pipeline.max_total_threads_per_threadgroup() as usize,
                width,
            )?;
        }
        let queue = device.new_command_queue();

        Ok(Self {
            device_index,
            device,
            queue,
            sa,
            sa_wide,
            sa_float,
            sa_wide_i8,
            sa_global,
            sa_wide_global,
            sa_float_global,
            sa_wide_i8_global,
            gibbs,
            gibbs_wide,
            gibbs_float,
            gibbs_wide_i8,
            gibbs_parallel,
            gibbs_parallel_wide,
            gibbs_parallel_float,
            gibbs_parallel_wide_i8,
            gibbs_parallel_spill,
            gibbs_parallel_wide_spill,
            gibbs_parallel_float_spill,
            gibbs_parallel_wide_i8_spill,
            msa,
            msa_weighted,
            msa_float,
            msa_weighted_i8,
            msa_weighted_dense,
            msa_weighted_spill,
            msa_float_spill,
            msa_matrix_bits_unit,
            msa_matrix_bits,
            msa_matrix_i8,
            msa_matrix_int,
            msa_matrix_float,
            sa_matrix_i8,
            sa_matrix_wide_i8,
            sa_matrix_int,
            sa_matrix_float,
            gibbs_matrix_i8,
            gibbs_matrix_wide_i8,
            gibbs_matrix_int,
            gibbs_matrix_float,
            msa_slots,
            msa_screen,
            msa_frustration_draw,
            msa_frustration_count,
            topology_cache: crate::sampler::TopologyCache::default(),
            buffer_pool: std::sync::Arc::new(crate::sampler::BufferPool::default()),
        })
    }

    /// Number of Metal devices visible to this process.
    ///
    /// Enumeration cannot fail — `Device::all()` yields an empty list rather
    /// than an error when no device is present — so this returns a plain
    /// count. Callers that need a *usable* device want [`Self::check`], which
    /// also compiles the kernels.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_solver_metal::metal_device::MetalDevice;
    ///
    /// let a = MetalDevice::device_count();
    /// let b = MetalDevice::device_count();
    /// assert_eq!(a, b);
    /// ```
    #[must_use]
    pub fn device_count() -> usize {
        Device::all().len()
    }

    /// Probe that a device can open and compile kernels (`--check`).
    ///
    /// # Errors
    ///
    /// Propagates [`Self::open`] verbatim: [`MetalError::NoDevice`] when
    /// `device_index` names no device, [`MetalError::Compile`] on a kernel
    /// source or entry-point failure, and [`MetalError::Driver`] when
    /// pipeline-state creation is refused.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use quip_solver_metal::metal_device::MetalDevice;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// MetalDevice::check(0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn check(device_index: usize) -> Result<(), MetalError> {
        let _ = Self::open(device_index)?;
        Ok(())
    }

    /// Shared-storage buffer of `len` zeroed bytes.
    pub fn new_zeroed_buffer(&self, len: u64) -> metal::Buffer {
        self.device
            .new_buffer(len, MTLResourceOptions::StorageModeShared)
    }

    /// Shared-storage buffer filled from host slice bytes.
    pub fn new_buffer_from_slice<T: Copy>(&self, data: &[T]) -> metal::Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        if byte_len == 0 {
            // Metal rejects zero-length buffers; allocate a tiny stub.
            return self.new_zeroed_buffer(4);
        }
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

fn check_threadgroup_width(entry: &str, maximum: usize, dispatch: usize) -> Result<(), MetalError> {
    if dispatch > maximum {
        return Err(MetalError::Driver(format!(
            "{entry} dispatch needs {dispatch} threads per threadgroup, but the pipeline admits {maximum}"
        )));
    }
    Ok(())
}

fn compile_pipeline(
    device: &Device,
    source: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalError> {
    compile_with(device, source, entry, &CompileOptions::new())
}

/// [`compile_pipeline`] with fast math off, for a kernel whose `exp` and
/// `pow` must round as a host reference does (`kernels/cluster.metal`).
pub(crate) fn compile_precise_pipeline(
    device: &Device,
    source: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalError> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    compile_with(device, source, entry, &options)
}

fn compile_with(
    device: &Device,
    source: &str,
    entry: &str,
    options: &CompileOptions,
) -> Result<ComputePipelineState, MetalError> {
    let library = device
        .new_library_with_source(source, options)
        .map_err(|e| MetalError::Compile(format!("{entry}: {e}")))?;
    let function = library
        .get_function(entry, None)
        .map_err(|e| MetalError::Compile(format!("{entry} function: {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalError::Driver(format!("{entry} pipeline: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_rejects_a_dispatch_past_its_thread_limit() {
        check_threadgroup_width("test", 256, 256).unwrap();
        let error = check_threadgroup_width("test", 255, 256).unwrap_err();
        assert!(
            matches!(error, MetalError::Driver(message) if message.contains("256") && message.contains("255"))
        );
    }

    #[test]
    fn wide_row_matches_metal_sources() {
        assert_eq!(crate::wide_tables::WIDE_ROW, 2048);
        assert!(SA_SRC.contains("#define WIDE_ROW 2048"));
        assert!(GIBBS_SRC.contains("#define WIDE_ROW 2048"));
        assert!(MSA_SRC.contains("#define WIDE_ROW       2048"));
    }

    #[test]
    #[expect(clippy::print_stderr, reason = "report GPU pipeline thread maxima")]
    fn wide_and_float_pipelines_compile_and_admit_a_simd_group() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        for (name, pipeline) in [
            ("sa_wide", &dev.sa_wide),
            ("sa_float", &dev.sa_float),
            ("gibbs_wide", &dev.gibbs_wide),
            ("gibbs_float", &dev.gibbs_float),
            ("gibbs_parallel_wide", &dev.gibbs_parallel_wide),
            ("gibbs_parallel_float", &dev.gibbs_parallel_float),
            ("msa_weighted", &dev.msa_weighted),
            ("msa_float", &dev.msa_float),
            ("sa_wide_i8", &dev.sa_wide_i8),
            ("gibbs_wide_i8", &dev.gibbs_wide_i8),
            ("gibbs_parallel_wide_i8", &dev.gibbs_parallel_wide_i8),
            ("gibbs_parallel_wide_spill", &dev.gibbs_parallel_wide_spill),
            (
                "gibbs_parallel_float_spill",
                &dev.gibbs_parallel_float_spill,
            ),
            (
                "gibbs_parallel_wide_i8_spill",
                &dev.gibbs_parallel_wide_i8_spill,
            ),
            ("msa_weighted_i8", &dev.msa_weighted_i8),
            ("msa_weighted_dense", &dev.msa_weighted_dense),
            ("msa_weighted_spill", &dev.msa_weighted_spill),
            ("msa_float_spill", &dev.msa_float_spill),
            ("msa_matrix_bits_unit", &dev.msa_matrix_bits_unit),
            ("msa_matrix_bits", &dev.msa_matrix_bits),
            ("msa_matrix_i8", &dev.msa_matrix_i8),
            ("msa_matrix_int", &dev.msa_matrix_int),
            ("msa_matrix_float", &dev.msa_matrix_float),
            ("sa_matrix_i8", &dev.sa_matrix_i8),
            ("sa_matrix_wide_i8", &dev.sa_matrix_wide_i8),
            ("sa_matrix_int", &dev.sa_matrix_int),
            ("sa_matrix_float", &dev.sa_matrix_float),
            ("gibbs_matrix_i8", &dev.gibbs_matrix_i8),
            ("gibbs_matrix_wide_i8", &dev.gibbs_matrix_wide_i8),
            ("gibbs_matrix_int", &dev.gibbs_matrix_int),
            ("gibbs_matrix_float", &dev.gibbs_matrix_float),
        ] {
            let max = pipeline.max_total_threads_per_threadgroup();
            eprintln!("{name}: max threads per threadgroup {max}");
            assert!(max >= 32, "{name} admits only {max} threads");
        }
        assert!(dev.gibbs_parallel_wide.max_total_threads_per_threadgroup() >= 128);
        assert!(
            dev.gibbs_parallel_wide_i8
                .max_total_threads_per_threadgroup()
                >= 128
        );
    }

    #[test]
    fn device_count_is_consistent() {
        // Enumeration cannot fail; two back-to-back reads must agree.
        let a = MetalDevice::device_count();
        let b = MetalDevice::device_count();
        assert_eq!(a, b, "device_count drifted between calls");
    }

    #[test]
    fn open_past_the_end_is_no_device() {
        let n = MetalDevice::device_count();
        let err = MetalDevice::open(n).unwrap_err();
        match err {
            MetalError::NoDevice(i) => assert_eq!(i, n),
            other => panic!("expected NoDevice({n}), got {other:?}"),
        }
    }

    #[test]
    fn check_succeeds_on_a_valid_index() {
        // Apple Silicon always exposes at least one Metal device; if this
        // host somehow has none, open(0) must still report NoDevice cleanly.
        let n = MetalDevice::device_count();
        if n == 0 {
            let err = MetalDevice::check(0).unwrap_err();
            assert!(
                matches!(err, MetalError::NoDevice(0)),
                "expected NoDevice(0), got {err:?}"
            );
            return;
        }
        MetalDevice::check(0).unwrap();
    }

    #[test]
    fn msa_static_threadgroup_memory_is_the_lane_totals_only() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        for pipeline in [
            &dev.msa,
            &dev.msa_weighted,
            &dev.msa_weighted_i8,
            &dev.msa_weighted_dense,
            &dev.msa_weighted_spill,
            &dev.msa_float_spill,
            &dev.msa_float,
            &dev.msa_slots,
            &dev.msa_screen,
        ] {
            assert!(pipeline.static_threadgroup_memory_length() as usize <= 32 * 4 + 16);
        }
    }

    #[test]
    #[expect(clippy::print_stderr, reason = "report multi-spin pipeline limits")]
    fn msa_pipeline_compiles_and_admits_at_least_256_threads() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        // The host requests 1,024 threads per multi-spin threadgroup and clamps
        // to the pipeline maximum. Require support for at least 256 threads.
        for pipeline in [
            &dev.msa,
            &dev.msa_weighted,
            &dev.msa_weighted_i8,
            &dev.msa_weighted_dense,
            &dev.msa_weighted_spill,
            &dev.msa_float_spill,
            &dev.msa_float,
            &dev.msa_slots,
            &dev.msa_screen,
        ] {
            eprintln!(
                "msa pipeline: max threads {}, static tg bytes {}",
                pipeline.max_total_threads_per_threadgroup(),
                pipeline.static_threadgroup_memory_length()
            );
            assert!(pipeline.max_total_threads_per_threadgroup() >= 256);
            // Static threadgroup lane totals must leave room for the
            // largest advertised N at 4 bytes per spin under the 32 KB cap.
            let static_bytes = pipeline.static_threadgroup_memory_length() as usize;
            assert!(static_bytes <= 144, "static tg bytes {static_bytes}");
            assert!(
                static_bytes + 6016 * 4 <= dev.device.max_threadgroup_memory_length() as usize,
                "6016 spins do not fit beside {static_bytes} static bytes"
            );
        }
    }
}
