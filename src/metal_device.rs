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

fn wide(source: &str) -> String {
    format!("{WIDE_DEFINE}{source}")
}

fn float(source: &str) -> String {
    format!("{FLOAT_DEFINE}{source}")
}

/// Multi-spin coded SA, see `kernels/msa.metal`.
const MSA_SRC: &str = include_str!("../kernels/msa.metal");

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
    pub(crate) gibbs: ComputePipelineState,
    pub(crate) gibbs_wide: ComputePipelineState,
    /// `gibbs` for Float32 models: `float` coefficients and heat-bath.
    pub(crate) gibbs_float: ComputePipelineState,
    /// Chromatic (node-parallel) Gibbs: one threadgroup per sample, threads
    /// split the nodes of each color, `threadgroup`-shared state. Same buffer
    /// layout as `gibbs`, different dispatch geometry.
    pub(crate) gibbs_parallel: ComputePipelineState,
    pub(crate) gibbs_parallel_wide: ComputePipelineState,
    /// `gibbs_parallel` for Float32 models.
    pub(crate) gibbs_parallel_float: ComputePipelineState,
    /// Multi-spin coded SA: one threadgroup per (problem, 32-replica word),
    /// threads split each colour class, spin words in `threadgroup` memory.
    /// Same buffer layout as `gibbs_parallel` with `words` at slot 19.
    pub(crate) msa: ComputePipelineState,
    /// `msa` for weighted models: milli `int` coefficients and host-built
    /// Metropolis threshold rows at slot 25.
    pub(crate) msa_weighted: ComputePipelineState,
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
    /// Multi-spin batch buffers returned after their command buffers retire.
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
        let gibbs = compile_pipeline(&device, GIBBS_SRC, "block_gibbs_sampler")?;
        let gibbs_wide = compile_pipeline(&device, &wide(GIBBS_SRC), "block_gibbs_sampler")?;
        let gibbs_float = compile_pipeline(&device, &float(GIBBS_SRC), "block_gibbs_sampler")?;
        let gibbs_parallel = compile_pipeline(&device, GIBBS_SRC, "block_gibbs_parallel")?;
        let gibbs_parallel_wide =
            compile_pipeline(&device, &wide(GIBBS_SRC), "block_gibbs_parallel")?;
        let gibbs_parallel_float =
            compile_pipeline(&device, &float(GIBBS_SRC), "block_gibbs_parallel")?;
        let msa = compile_pipeline(&device, MSA_SRC, "msa_anneal")?;
        let msa_weighted = compile_pipeline(&device, MSA_SRC, "msa_anneal_weighted")?;
        let msa_slots = compile_pipeline(&device, MSA_SRC, "msa_anneal_slots")?;
        let msa_screen = compile_pipeline(&device, MSA_SRC, "msa_screen")?;
        let msa_frustration_draw = compile_pipeline(&device, MSA_SRC, "msa_frustration_draw")?;
        let msa_frustration_count = compile_pipeline(&device, MSA_SRC, "msa_frustration_count")?;
        let queue = device.new_command_queue();

        Ok(Self {
            device_index,
            device,
            queue,
            sa,
            sa_wide,
            sa_float,
            gibbs,
            gibbs_wide,
            gibbs_float,
            gibbs_parallel,
            gibbs_parallel_wide,
            gibbs_parallel_float,
            msa,
            msa_weighted,
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

fn compile_pipeline(
    device: &Device,
    source: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalError> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
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
        ] {
            let max = pipeline.max_total_threads_per_threadgroup();
            eprintln!("{name}: max threads per threadgroup {max}");
            assert!(max >= 32, "{name} admits only {max} threads");
        }
        assert!(dev.gibbs_parallel_wide.max_total_threads_per_threadgroup() >= 128);
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
        for pipeline in [&dev.msa, &dev.msa_slots, &dev.msa_screen] {
            assert!(pipeline.static_threadgroup_memory_length() as usize <= 32 * 4 + 16);
        }
    }

    #[test]
    fn msa_pipeline_compiles_and_admits_at_least_256_threads() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let dev = MetalDevice::open(0).unwrap();
        // The host requests 1,024 threads per multi-spin threadgroup and clamps
        // to the pipeline maximum. Require support for at least 256 threads.
        for pipeline in [&dev.msa, &dev.msa_slots, &dev.msa_screen] {
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
