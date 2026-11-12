// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Byte footprints per kernel and storage tier.
//!
//! [`footprint`] answers two questions for a model: which storage tier holds
//! it, and how many bytes that tier needs. Callers refuse a model that does
//! not fit before they allocate a buffer.
//!
//! # Tiers
//!
//! - [`Storage::Threadgroup`]: spin state lives in threadgroup memory. Only
//!   models up to the kernel node cap fit, and multi-spin also needs degree
//!   at most [`MSA_UNIT_MAX_DEG`] on the unit lane, or [`MSA_SPARSE_MAX_DEG`]
//!   on the weighted lanes. The sizes come from the buffers that
//!   `sampler::encode_batch` allocates.
//! - [`Storage::Spill`]: spin state past the threadgroup cap lives in device
//!   memory. The weighted multi-spin kernel keeps it in the carry-over
//!   buffer the threadgroup tier already counts, so the tier allocates the
//!   same buffers. The SA global build keeps each read's packed spins and
//!   `delta_energy` in its carry-over buffers in the same way. The Gibbs
//!   spill build keeps the spins past its threadgroup prefix in each read's
//!   slice of its carry-over spin buffer, which holds one byte per spin.
//! - [`Storage::Dense`]: an `N * N` coupling matrix instead of CSR. It
//!   admits any degree, so a multi-spin model past the sparse degree routes
//!   here. [`routed_footprint`] keeps a graph with a repeated coupler or a
//!   self-loop off this tier. Every kernel runs it (see
//!   [`runnable_storages`]).
//!
//! Every formula sits in one small function per kernel, so a later task can
//! correct one without touching the routing.

use crate::sampler::{
    simd_rounded_reads, Kernel, GIBBS_MAX_NODES, MAX_READS, MSA_LANES, MSA_MAX_NODES,
    MSA_SPARSE_MAX_DEG, MSA_THREADS, MSA_UNIT_MAX_DEG, SA_MAX_NODES, SIMD_WIDTH,
};
use crate::streaming::{batch_size_for_reads, STREAM_IN_FLIGHT};
use quip_solver_core::IsingGraph;

/// Largest node count the 32-bit kernel indexing can address.
pub const MAX_NODES: usize = i32::MAX as usize;

/// Largest CSR entry count (`2 * edges`) the 32-bit kernel indexing can
/// address.
const MAX_NNZ: usize = u32::MAX as usize;
/// Bytes of one `col_ind` entry.
const COL_IND_ENTRY_BYTES: usize = 4;
/// Beta entries the advertised limits plan against: a 64-rung schedule.
const ADVERTISED_BETAS: usize = 64;

/// Memory headroom numerator, as CUDA's `MEMORY_HEADROOM_NUM`.
const HEADROOM_NUM: usize = 4;
/// Memory headroom denominator, as CUDA's `MEMORY_HEADROOM_DEN`.
const HEADROOM_DEN: usize = 5;
/// Threads per chromatic-Gibbs threadgroup (`kernels/gibbs.metal`).
const GIBBS_GROUP_THREADS: usize = 256;
/// Bytes of xoshiro128** state per random stream (four `uint`).
const RNG_BYTES_PER_STREAM: usize = 16;

/// Memory limits of the device a miner runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Memory the miner may plan against, in bytes.
    pub memory_bytes: usize,
    /// Largest single buffer, in bytes.
    pub max_buffer_bytes: usize,
}

impl DeviceLimits {
    /// Bytes available to buffers after the memory headroom: 4/5 of
    /// [`Self::memory_bytes`].
    #[must_use]
    pub fn usable_bytes(&self) -> usize {
        self.memory_bytes / HEADROOM_DEN * HEADROOM_NUM
            + self.memory_bytes % HEADROOM_DEN * HEADROOM_NUM / HEADROOM_DEN
    }
}

unsafe extern "C" {
    /// `sysctlbyname(3)`: read a kernel value by name.
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> std::ffi::c_int;
}

/// Total physical memory from `hw.memsize`, or `None` when the call fails.
fn physical_memory_bytes() -> Option<usize> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the name is a NUL-terminated literal. `value` is a live u64
    // and `len` holds its size, so the kernel writes at most eight bytes
    // into it. A null `newp` with zero `newlen` makes the call read-only.
    let status = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            std::ptr::from_mut(&mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == 0 && len == std::mem::size_of::<u64>() {
        usize::try_from(value).ok()
    } else {
        None
    }
}

/// Limits of this host: half of `hw.memsize` as planning memory, and a
/// buffer limit equal to the planning memory.
///
/// When `hw.memsize` cannot be read, the memory is 0, so every model is
/// refused rather than planned against a guessed size.
#[must_use]
pub fn host_limits() -> DeviceLimits {
    let memory_bytes = physical_memory_bytes().map_or_else(
        || {
            tracing::warn!("hw.memsize unreadable; the capacity model refuses every model");
            0
        },
        |bytes| bytes / 2,
    );
    DeviceLimits {
        memory_bytes,
        max_buffer_bytes: memory_bytes,
    }
}

/// Numeric type of the couplings and fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Unit coefficients: two bit planes in a dense matrix (2 bits per
    /// cell); one byte in CSR.
    Binary,
    /// Signed 8-bit coefficients.
    Int8,
    /// Signed 32-bit integer coefficients (the wide milli representation).
    Int32,
    /// The wide milli representation stored as `int8_t` coefficients times one
    /// integer scale: one byte per coefficient, while deltas and fields stay
    /// 32-bit.
    Int8Scaled,
    /// 32-bit float coefficients.
    Float32,
}

impl Precision {
    /// Bytes per coefficient in the CSR value and field buffers.
    const fn coef_bytes(self) -> usize {
        match self {
            Self::Binary | Self::Int8 | Self::Int8Scaled => 1,
            Self::Int32 | Self::Float32 => 4,
        }
    }

    /// Bytes per `delta_energy` entry (`new_sa_persistent`): one for the
    /// narrow representation, four for every other.
    const fn delta_bytes(self) -> usize {
        match self {
            Self::Binary | Self::Int8 => 1,
            Self::Int8Scaled | Self::Int32 | Self::Float32 => 4,
        }
    }

    /// Bytes of the dense coupling matrix of an `nodes`-node model: `nodes`
    /// rows of [`dense_row_elems`] elements. A unit model holds two `uint`
    /// bit planes per 32 columns (the negative and the nonzero couplings),
    /// every other precision one coefficient per cell
    /// (`sampler::upload_dense_inputs`).
    const fn dense_bytes(self, nodes: usize) -> usize {
        let elem_bytes = match self {
            Self::Binary | Self::Int32 | Self::Float32 => 4,
            Self::Int8 | Self::Int8Scaled => 1,
        };
        nodes
            .saturating_mul(dense_row_elems(self, nodes))
            .saturating_mul(elem_bytes)
    }
}

/// Where a model's spin state and couplings live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// Spin state in threadgroup memory.
    Threadgroup,
    /// Spin state past the threadgroup cap in a device buffer.
    Spill,
    /// Dense `N * N` coupling matrix.
    Dense,
}

impl Storage {
    /// The tier name in CUDA's JSON: `local`, `global`, `dense` for SA, and
    /// `shared`, `spill`, `dense` for Gibbs and MSA.
    #[must_use]
    pub const fn name(self, kernel: Kernel) -> &'static str {
        match (kernel, self) {
            (Kernel::Sa, Self::Threadgroup) => "local",
            (Kernel::Sa, Self::Spill) => "global",
            (Kernel::Gibbs | Kernel::Msa, Self::Threadgroup) => "shared",
            (Kernel::Gibbs | Kernel::Msa, Self::Spill) => "spill",
            (_, Self::Dense) => "dense",
        }
    }
}

/// The size of one model, as the footprint formulas see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelShape {
    /// Spin count.
    pub nodes: usize,
    /// Coupling count (each edge appears twice in CSR).
    pub edges: usize,
    /// Largest number of neighbours of any node.
    pub max_degree: usize,
    /// Coefficient type on CSR storage.
    pub precision: Precision,
    /// Coefficient type of the dense coupling matrix. It differs from
    /// [`Self::precision`] where the two layouts differ: a unit multi-spin
    /// model past the unit lane runs `int` CSR rows but a bit matrix, and a
    /// wide model whose CSR rows must stay `int` can hold an `int8_t`
    /// matrix.
    pub dense_precision: Precision,
    /// Entries in the beta schedule.
    pub betas: usize,
    /// Reads per model, before rounding up to a whole SIMD group.
    pub reads: usize,
}

/// Bytes a model needs in one storage tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// The chosen tier.
    pub storage: Storage,
    /// Bytes that exist once per model.
    pub per_instance_bytes: usize,
    /// Bytes that exist once per kernel, whatever the model count.
    pub shared_bytes: usize,
}

impl Footprint {
    /// Bytes for `instances` models at once, held in `batches` batches that
    /// each allocate the shared buffers.
    #[must_use]
    pub fn total(&self, instances: usize, batches: usize) -> usize {
        self.per_instance_bytes
            .saturating_mul(instances)
            .saturating_add(self.shared_bytes.saturating_mul(batches))
    }
}

/// Why a model does not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CapacityError {
    /// More nodes than the 32-bit kernel indexing can address.
    #[error("{max} nodes is the most a kernel can index")]
    Nodes {
        /// Largest admitted node count.
        max: usize,
    },
    /// The models in flight need more than the usable memory.
    #[error("{needed} bytes needed; {usable} bytes usable")]
    Memory {
        /// Bytes the models in flight need.
        needed: usize,
        /// Usable bytes after the headroom.
        usable: usize,
    },
    /// One buffer exceeds the device's single-buffer limit.
    #[error("a {needed} byte buffer exceeds the {max} byte buffer limit")]
    Buffer {
        /// Size of the largest buffer.
        needed: usize,
        /// The device limit.
        max: usize,
    },
    /// An index the kernel computes in 32 bits would pass its range.
    #[error("{entries} entries exceed the {max} a 32-bit kernel index addresses")]
    Index {
        /// Entries the largest index addresses.
        entries: usize,
        /// The most the index addresses.
        max: usize,
    },
    /// The degree exceeds what every admitting tier supports.
    #[error("degree above {max} is not supported")]
    Degree {
        /// Largest admitted degree.
        max: usize,
    },
}

/// Models whose buffers exist at once.
///
/// SA and Gibbs return [`STREAM_IN_FLIGHT`] times [`batch_size_for_reads`]:
/// the stream keeps that many batches in flight, and its cap on the problems
/// of one batch never passes [`batch_size_for_reads`]
/// (`streaming::form_and_commit` and `sampler::one_sweep_problem_cap`). The
/// footprint of one model times this count therefore bounds the stream's
/// batches in flight. MSA returns the resident runner's bound on
/// queued, preparing, and ready jobs (`PREP_BOUND`, "At most 40 jobs" in
/// `src/resident.rs`).
#[must_use]
pub fn instances(kernel: Kernel, reads: usize) -> usize {
    match kernel {
        Kernel::Sa | Kernel::Gibbs => STREAM_IN_FLIGHT * batch_size_for_reads(kernel, reads),
        Kernel::Msa => crate::resident::PREP_BOUND,
    }
}

/// Batches whose shared buffers exist at once: [`STREAM_IN_FLIGHT`] for the
/// SA and Gibbs streams, each with its own beta schedule and colour list,
/// and 1 for the multi-spin batch path, which runs one batch at a time
/// beside the resident slot pools (`resident::run_fallback`).
#[must_use]
pub fn batches_in_flight(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Sa | Kernel::Gibbs => STREAM_IN_FLIGHT,
        Kernel::Msa => 1,
    }
}

/// One tier's buffers: the size of each per-model buffer and of each shared
/// buffer. Each buffer the kernel allocates appears exactly once.
struct Sizes {
    /// Each per-model buffer, in bytes.
    per_instance: Vec<usize>,
    /// Each buffer shared by all models, in bytes.
    shared: Vec<usize>,
}

/// Threadgroup node cap of a kernel.
const fn node_cap(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Sa => SA_MAX_NODES,
        Kernel::Gibbs => GIBBS_MAX_NODES,
        Kernel::Msa => MSA_MAX_NODES,
    }
}

/// Whether `storage` admits `shape` on `kernel`.
fn admits(kernel: Kernel, storage: Storage, shape: &ModelShape) -> bool {
    match storage {
        Storage::Threadgroup => {
            // `sampler::msa_max_degree`: the weighted lanes (milli, its
            // int8 storage, and Float32) walk rows up to the sparse limit;
            // the unit lane, sized as Binary or Int8, unrolls one chunk.
            let msa_degree = match shape.precision {
                Precision::Binary | Precision::Int8 => MSA_UNIT_MAX_DEG,
                Precision::Int32 | Precision::Int8Scaled | Precision::Float32 => MSA_SPARSE_MAX_DEG,
            };
            shape.nodes <= node_cap(kernel)
                && (kernel != Kernel::Msa || shape.max_degree <= msa_degree)
        }
        Storage::Spill => kernel != Kernel::Msa || shape.max_degree <= MSA_SPARSE_MAX_DEG,
        // A dense matrix has no row-length limit, so it admits any degree.
        Storage::Dense => true,
    }
}

/// Largest entry index a kernel computes in `uint`.
const MAX_UINT_INDEX: usize = u32::MAX as usize;
/// Largest value of an `int` CSR offset or `row_ptr` entry.
const MAX_INT_OFFSET: usize = i32::MAX as usize;
/// `uint` words of one RNG stream.
const RNG_WORDS_PER_STREAM: usize = RNG_BYTES_PER_STREAM / 4;

/// Refuse `problems` problems of `shape` on `storage` when an index that
/// `kernel` computes in 32 bits would pass its range.
///
/// The indices, from the kernels and from `sampler::upload_inputs`:
///
/// - SA: entry `i` of thread `t` is `i * num_threads + t` in the global and
///   dense builds, and `thread * N` in the local build, so every tier bounds
///   `samples * N`.
/// - Gibbs: the chromatic builds index the RNG streams as
///   `(sample * 256 + thread) * 4`. The shared and dense builds also index
///   the spins as `sample * N`; the spill build addresses them in 64 bits.
/// - Multi-spin: each threadgroup's words are `threadgroup * N`, with one
///   threadgroup per problem and 32-read word, and the packed output is
///   `(problem * reads + read) * packed_size + byte`.
/// - Dense: a row of one problem's matrix is `row * row_elems`.
/// - CSR: `row_ptr` and the per-problem offsets are `int`. Every kernel tiles
///   the couplings per problem. SA and Gibbs also tile the structure; the
///   multi-spin kernel reads one copy of it.
///
/// # Errors
///
/// [`CapacityError::Index`] naming the largest index that does not fit.
pub(crate) fn check_indices(
    kernel: Kernel,
    storage: Storage,
    shape: &ModelShape,
    problems: usize,
) -> Result<(), CapacityError> {
    let n = shape.nodes.max(1);
    let reads = simd_rounded_reads(shape.reads);
    let samples = problems.saturating_mul(reads);
    let mut uint = vec![match kernel {
        Kernel::Sa => samples.saturating_mul(n),
        Kernel::Gibbs => samples
            .saturating_mul(GIBBS_GROUP_THREADS)
            .saturating_mul(RNG_WORDS_PER_STREAM),
        Kernel::Msa => problems.saturating_mul(reads / MSA_LANES).saturating_mul(n),
    }];
    if kernel == Kernel::Gibbs && storage != Storage::Spill {
        uint.push(samples.saturating_mul(n));
    }
    if kernel == Kernel::Msa {
        uint.push(samples.saturating_mul(n.div_ceil(8)));
    }
    let mut int = Vec::new();
    if storage == Storage::Dense {
        uint.push(n.saturating_mul(dense_row_elems(shape.dense_precision, n)));
    } else {
        // The couplings are tiled per problem on every kernel; the multi-spin
        // kernel shares one `row_ptr`.
        int.push(problems.saturating_mul(shape.edges.saturating_mul(2).max(1)));
        let structures = if kernel == Kernel::Msa { 1 } else { problems };
        int.push(structures.saturating_mul(n.saturating_add(1)));
    }
    let over = |entries: &[usize], max: usize| {
        entries
            .iter()
            .copied()
            .filter(|&e| e > max)
            .max()
            .map(|entries| CapacityError::Index { entries, max })
    };
    match over(&uint, MAX_UINT_INDEX).or_else(|| over(&int, MAX_INT_OFFSET)) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Elements of one row of an `n`-node dense matrix: the two `uint` bit
/// planes per 32 columns of a unit model, else one coefficient per column
/// (`sampler::dense_row_elems`).
const fn dense_row_elems(precision: Precision, n: usize) -> usize {
    match precision {
        Precision::Binary => 2 * n.div_ceil(32),
        _ => n,
    }
}

/// Output buffers every kernel writes per model: packed samples and one
/// `i32` energy per read (`encode_batch`).
fn output_buffers(shape: &ModelShape) -> [usize; 2] {
    let reads = simd_rounded_reads(shape.reads);
    [
        reads.saturating_mul(shape.nodes.div_ceil(8).max(1)),
        reads.saturating_mul(4),
    ]
}

/// CSR buffers of one model: `row_ptr` and `col_ind` (4 bytes each entry),
/// then couplings and fields at the precision's width (`upload_inputs`).
fn csr_buffers(shape: &ModelShape) -> [usize; 4] {
    let nnz = shape.edges.saturating_mul(2).max(1);
    let width = shape.precision.coef_bytes();
    [
        shape.nodes.saturating_add(1).saturating_mul(4),
        nnz.saturating_mul(4),
        nnz.saturating_mul(width),
        shape.nodes.saturating_mul(width),
    ]
}

/// Buffers every kernel shares: the beta schedule and the colour-class node
/// list (`encode_batch`, `build_cached_topology`).
fn common_shared(shape: &ModelShape) -> Vec<usize> {
    vec![shape.betas.saturating_mul(4), shape.nodes.saturating_mul(4)]
}

/// Carry-over buffers of one model, whatever the tier. They hold the spin
/// state of every node, so they already cover the nodes past the threadgroup
/// cap.
fn persistent_buffers(kernel: Kernel, shape: &ModelShape) -> Vec<usize> {
    match kernel {
        Kernel::Sa => sa_persistent(shape).to_vec(),
        Kernel::Gibbs => gibbs_persistent(shape).to_vec(),
        Kernel::Msa => msa_persistent(shape).to_vec(),
    }
}

/// SA carry-over buffers of one model (`new_sa_persistent`): packed spins,
/// `delta_energy` per spin, the RNG word, and the running energy, per read.
fn sa_persistent(shape: &ModelShape) -> [usize; 4] {
    let reads = simd_rounded_reads(shape.reads);
    [
        reads.saturating_mul(shape.nodes.div_ceil(8).max(1)),
        reads
            .saturating_mul(shape.nodes.max(1))
            .saturating_mul(shape.precision.delta_bytes()),
        reads.saturating_mul(4),
        reads.saturating_mul(4),
    ]
}

/// Gibbs carry-over buffers of one model (`new_gibbs_persistent`, chromatic
/// layout): one unpacked byte per spin per sample, and one RNG stream per
/// thread of each sample's threadgroup.
fn gibbs_persistent(shape: &ModelShape) -> [usize; 2] {
    let reads = simd_rounded_reads(shape.reads);
    [
        reads.saturating_mul(shape.nodes.max(1)),
        reads
            .saturating_mul(GIBBS_GROUP_THREADS)
            .saturating_mul(RNG_BYTES_PER_STREAM),
    ]
}

/// MSA carry-over buffers of one model (`new_msa_persistent`): one `uint`
/// per word and spin, and one RNG stream per word and thread.
fn msa_persistent(shape: &ModelShape) -> [usize; 2] {
    let words = simd_rounded_reads(shape.reads) / MSA_LANES;
    [
        words.saturating_mul(shape.nodes.max(1)).saturating_mul(4),
        words
            .saturating_mul(MSA_THREADS)
            .saturating_mul(RNG_BYTES_PER_STREAM),
    ]
}

/// Threadgroup-tier sizes. SA and Gibbs tile the CSR structure per model;
/// MSA keeps one untiled copy, which counts as shared.
fn threadgroup_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    let csr = csr_buffers(shape);
    let mut per_instance = Vec::new();
    per_instance.extend(output_buffers(shape));
    let mut shared = common_shared(shape);
    if kernel == Kernel::Msa {
        per_instance.extend(&csr[2..]);
        shared.extend(&csr[..2]);
    } else {
        per_instance.extend(csr);
    }
    per_instance.extend(persistent_buffers(kernel, shape));
    Sizes {
        per_instance,
        shared,
    }
}

/// Spill-tier sizes. The weighted multi-spin kernel keeps the words of the
/// `nodes - K` nodes past its threadgroup cap `K` in their slots of the
/// carry-over buffer (`persistent_state`, one `uint` per word and spin),
/// which the threadgroup tier already counts, so the spill tier allocates
/// the same buffers and adds none (`sampler::encode_batch_storage`). The SA
/// global build likewise works in place in its carry-over buffers
/// ([`sa_persistent`]): packed spins and `delta_energy` for every node of
/// every read. The Gibbs spill build keeps the spins past its prefix in the
/// carry-over spin buffer ([`gibbs_persistent`]), one byte per spin per
/// read, so it too adds no buffer.
fn spill_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    threadgroup_sizes(kernel, shape)
}

/// Dense-tier sizes: the coupling matrix at [`ModelShape::dense_precision`]
/// ([`Precision::dense_bytes`]) replaces the CSR structure and couplings.
/// The fields, the outputs, and the carry-over buffers remain, at the same
/// precision. The multi-spin and Gibbs kernels also share one colour start
/// and one count per node.
fn dense_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    let dense = ModelShape {
        precision: shape.dense_precision,
        ..*shape
    };
    let mut per_instance = vec![
        dense.precision.dense_bytes(dense.nodes),
        dense.nodes.saturating_mul(dense.precision.coef_bytes()),
    ];
    per_instance.extend(output_buffers(&dense));
    per_instance.extend(persistent_buffers(kernel, &dense));
    let mut shared = common_shared(&dense);
    if kernel != Kernel::Sa {
        // The colour start and count arrays: a complete model has one
        // colour per node (`sampler::encode_batch_storage`).
        shared.extend([dense.nodes.saturating_mul(4); 2]);
    }
    Sizes {
        per_instance,
        shared,
    }
}

/// Largest single buffer: each per-model buffer counted for the `tiling`
/// models one batch buffer holds, and each shared buffer on its own.
fn largest_buffer(sizes: &Sizes, tiling: usize) -> usize {
    let per_instance = sizes.per_instance.iter().map(|&b| b.saturating_mul(tiling));
    per_instance
        .chain(sizes.shared.iter().copied())
        .max()
        .unwrap_or(0)
}

/// Models that share one `MTLBuffer` of per-model data.
///
/// All three kernels size a buffer for one batch of
/// [`batch_size_for_reads`] models: SA and Gibbs through `num_problems` in
/// `encode_batch_inner` (`src/sampler.rs`), and MSA through the slot pool
/// capacity (`SlotPool::new`, `src/slots.rs`, whose per-slot regions are
/// `capacity * size`), which `src/resident.rs` sets to `batch_size_for_reads`
/// for each of its two pools. [`instances`] counts both batches in flight.
fn models_per_buffer(kernel: Kernel, reads: usize) -> usize {
    batch_size_for_reads(kernel, reads)
}

/// Choose the cheapest tier that admits `shape` and fits `limits`.
///
/// Tiers are tried from cheapest to dearest by per-model bytes; ties go to
/// the earlier tier in the order threadgroup, spill, dense. The first tier
/// that passes both checks is returned. When none passes, the error is the
/// cheapest tier's.
///
/// # Errors
///
/// - [`CapacityError::Nodes`] when `shape.nodes` exceeds [`MAX_NODES`].
/// - [`CapacityError::Buffer`] when `2 * shape.edges` exceeds `u32::MAX`, the
///   most `col_ind` entries 32-bit indexing addresses. It reports the
///   `col_ind` bytes needed and the bytes `u32::MAX` entries occupy.
/// - [`CapacityError::Degree`] when no tier admits the degree.
/// - [`CapacityError::Index`] when one model passes a 32-bit index of the
///   tier ([`check_indices`]). Of the per-tier checks this runs first; the
///   node and `col_ind` bounds above run before any tier.
/// - [`CapacityError::Memory`] when `total(instances, batches_in_flight)`
///   exceeds
///   `limits.usable_bytes()`.
/// - [`CapacityError::Buffer`] when the largest buffer exceeds
///   `limits.max_buffer_bytes`. A per-model buffer counts for the
///   [`models_per_buffer`] models it holds; a shared buffer counts once.
pub fn footprint(
    kernel: Kernel,
    shape: &ModelShape,
    limits: &DeviceLimits,
) -> Result<Footprint, CapacityError> {
    footprint_in(
        kernel,
        shape,
        limits,
        &[Storage::Threadgroup, Storage::Spill, Storage::Dense],
    )
}

/// [`footprint`] for a graph: the tier choice of [`footprint`], except that a
/// graph that is not a dense candidate (a repeated coupler or a self-loop,
/// see `topology::dense_candidate`) never takes [`Storage::Dense`].
///
/// The pair scan runs only when [`footprint`] picks dense storage, so a
/// sparse model pays nothing for it. `shape` must describe `graph`.
///
/// # Errors
///
/// The errors of [`footprint`]. A non-candidate that no sparse tier admits,
/// such as a multi-spin model past the sparse degree, is a
/// [`CapacityError::Degree`].
pub fn routed_footprint(
    kernel: Kernel,
    graph: &IsingGraph,
    shape: &ModelShape,
    limits: &DeviceLimits,
) -> Result<Footprint, CapacityError> {
    routed_on(
        kernel,
        graph,
        shape,
        limits,
        &[Storage::Threadgroup, Storage::Spill, Storage::Dense],
    )
}

/// [`routed_footprint`] over only the tiers in [`runnable_storages`]: the
/// tier and bytes the sampler's batch path allocates, which its admission
/// checks before any buffer exists.
///
/// # Errors
///
/// The errors of [`routed_footprint`].
pub fn routed_runnable_footprint(
    kernel: Kernel,
    graph: &IsingGraph,
    shape: &ModelShape,
    limits: &DeviceLimits,
) -> Result<Footprint, CapacityError> {
    routed_on(kernel, graph, shape, limits, runnable_storages(kernel))
}

/// The routing of [`routed_footprint`] over `tiers`: the cheapest tier that
/// fits, except that a graph that is not a dense candidate takes the
/// cheapest of the other tiers.
fn routed_on(
    kernel: Kernel,
    graph: &IsingGraph,
    shape: &ModelShape,
    limits: &DeviceLimits,
    tiers: &[Storage],
) -> Result<Footprint, CapacityError> {
    route(kernel, shape, limits, tiers, || {
        crate::topology::dense_candidate(graph)
    })
}

/// The cheapest tier of `tiers` that fits, except that a model for which
/// `dense_candidate` is false takes the cheapest of the other tiers. The
/// check runs only when the cheapest tier is [`Storage::Dense`].
///
/// # Errors
///
/// The errors of [`footprint`].
pub(crate) fn route(
    kernel: Kernel,
    shape: &ModelShape,
    limits: &DeviceLimits,
    tiers: &[Storage],
    dense_candidate: impl FnOnce() -> bool,
) -> Result<Footprint, CapacityError> {
    let chosen = footprint_in(kernel, shape, limits, tiers)?;
    if chosen.storage == Storage::Dense && !dense_candidate() {
        let sparse: Vec<Storage> = tiers
            .iter()
            .copied()
            .filter(|&tier| tier != Storage::Dense)
            .collect();
        return footprint_in(kernel, shape, limits, &sparse);
    }
    Ok(chosen)
}

/// The tiers a kernel of this crate runs. A kernel with no dense build would
/// upload CSR whatever tier [`footprint`] prices; a dense kernel adds its tier
/// here. Every kernel runs dense storage on the batch path
/// (`kernels/msa_dense.metal`, `kernels/sa_dense.metal`,
/// `kernels/gibbs_dense.metal`).
#[must_use]
pub const fn runnable_storages(kernel: Kernel) -> &'static [Storage] {
    match kernel {
        Kernel::Sa | Kernel::Gibbs | Kernel::Msa => {
            &[Storage::Threadgroup, Storage::Spill, Storage::Dense]
        }
    }
}

/// [`footprint`] over only the tiers in [`runnable_storages`]: the bytes the
/// sampler allocates, which its admission checks before any buffer exists.
///
/// # Errors
///
/// The errors of [`footprint`].
pub fn runnable_footprint(
    kernel: Kernel,
    shape: &ModelShape,
    limits: &DeviceLimits,
) -> Result<Footprint, CapacityError> {
    footprint_in(kernel, shape, limits, runnable_storages(kernel))
}

/// [`footprint`] over the tiers in `tiers`, in their cost order: the
/// sampler's admission when a path allocates only some tiers, such as the
/// multi-spin gates that run CSR storage alone.
///
/// # Errors
///
/// The errors of [`footprint`].
pub(crate) fn footprint_in(
    kernel: Kernel,
    shape: &ModelShape,
    limits: &DeviceLimits,
    tiers: &[Storage],
) -> Result<Footprint, CapacityError> {
    if shape.nodes > MAX_NODES {
        return Err(CapacityError::Nodes { max: MAX_NODES });
    }
    if shape.edges.saturating_mul(2) > MAX_NNZ {
        // `col_ind` indexes its entries with 32 bits, whatever the memory.
        return Err(CapacityError::Buffer {
            needed: shape.edges.saturating_mul(2 * COL_IND_ENTRY_BYTES),
            max: MAX_NNZ * COL_IND_ENTRY_BYTES,
        });
    }
    let mut admitted: Vec<(Footprint, usize)> = [
        (Storage::Threadgroup, threadgroup_sizes(kernel, shape)),
        (Storage::Spill, spill_sizes(kernel, shape)),
        (Storage::Dense, dense_sizes(kernel, shape)),
    ]
    .into_iter()
    .filter(|(storage, _)| tiers.contains(storage) && admits(kernel, *storage, shape))
    .map(|(storage, sizes)| {
        let tiling = models_per_buffer(kernel, shape.reads);
        let footprint = Footprint {
            storage,
            per_instance_bytes: sum(&sizes.per_instance),
            shared_bytes: sum(&sizes.shared),
        };
        (footprint, largest_buffer(&sizes, tiling))
    })
    .collect();
    // Stable: equal costs keep the tier order.
    admitted.sort_by_key(|(footprint, _)| footprint.per_instance_bytes);
    let in_flight = instances(kernel, shape.reads);
    let mut first_error = None;
    for (footprint, largest) in admitted {
        let needed = footprint.total(in_flight, batches_in_flight(kernel));
        let error = if let Err(error) = check_indices(kernel, footprint.storage, shape, 1) {
            error
        } else if needed > limits.usable_bytes() {
            CapacityError::Memory {
                needed,
                usable: limits.usable_bytes(),
            }
        } else if largest > limits.max_buffer_bytes {
            CapacityError::Buffer {
                needed: largest,
                max: limits.max_buffer_bytes,
            }
        } else {
            return Ok(footprint);
        };
        first_error.get_or_insert(error);
    }
    Err(first_error.unwrap_or(CapacityError::Degree {
        max: MSA_SPARSE_MAX_DEG,
    }))
}

/// How many couplings a model has for a given node count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connectivity {
    /// Every node has this many neighbours: `nodes * degree / 2` edges,
    /// rounded down, and at most the complete count.
    Degree(usize),
    /// Every pair of nodes is coupled.
    Complete,
}

impl Connectivity {
    /// `(edges, max_degree)` of a model with `nodes` nodes.
    fn size(self, nodes: usize) -> (usize, usize) {
        let complete_degree = nodes.saturating_sub(1);
        let complete_edges = nodes.saturating_mul(complete_degree) / 2;
        match self {
            Self::Complete => (complete_edges, complete_degree),
            Self::Degree(degree) => (
                (nodes.saturating_mul(degree) / 2).min(complete_edges),
                degree.min(complete_degree),
            ),
        }
    }
}

/// The largest model of `connectivity` that `footprint` admits, with its
/// footprint, or `None` when not even one node fits.
///
/// Binary search on the node count. Fit is monotone in it: a model with
/// fewer nodes has no more edges, degree, or bytes in any tier, and every
/// tier that admits the larger model admits the smaller.
#[must_use]
pub fn max_model(
    kernel: Kernel,
    connectivity: Connectivity,
    precision: Precision,
    limits: &DeviceLimits,
    reads: usize,
) -> Option<(ModelShape, Footprint)> {
    let probe = |nodes: usize| {
        let (edges, max_degree) = connectivity.size(nodes);
        let shape = ModelShape {
            nodes,
            edges,
            max_degree,
            precision,
            dense_precision: precision,
            betas: ADVERTISED_BETAS,
            reads,
        };
        footprint(kernel, &shape, limits)
            .ok()
            .map(|footprint| (shape, footprint))
    };
    let mut best = None;
    // Invariant: `low` fits or is 0; `high + 1` does not fit or exceeds the
    // node bound.
    let (mut low, mut high) = (0, MAX_NODES);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if let Some(found) = probe(mid) {
            best = Some(found);
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    best
}

/// Read counts the miner accepts, after the dispatch rounds each up to a
/// whole SIMD group: 32, 64, ..., [`MAX_READS`].
fn accepted_reads() -> impl Iterator<Item = usize> {
    (1..=MAX_READS / SIMD_WIDTH).map(|groups| groups * SIMD_WIDTH)
}

/// The `(CSR, dense)` coefficient layouts the batch path runs a model of
/// `kernel` on (`sampler::model_shape`). SA and Gibbs hold a unit model as
/// Int8. A multi-spin unit model past the unit lane walks `int` rows but
/// keeps a bit matrix, and a wide one whose rows pass one chunk walks `int`
/// rows but may keep an `int8_t` matrix.
const fn layouts(kernel: Kernel) -> &'static [(Precision, Precision)] {
    match kernel {
        Kernel::Sa | Kernel::Gibbs => &[
            (Precision::Int8, Precision::Int8),
            (Precision::Int8Scaled, Precision::Int8Scaled),
            (Precision::Int32, Precision::Int32),
            (Precision::Float32, Precision::Float32),
        ],
        Kernel::Msa => &[
            (Precision::Binary, Precision::Binary),
            (Precision::Int32, Precision::Binary),
            (Precision::Int8Scaled, Precision::Int8Scaled),
            (Precision::Int32, Precision::Int8Scaled),
            (Precision::Int32, Precision::Int32),
            (Precision::Float32, Precision::Float32),
        ],
    }
}

/// Whether the session admits one model of `nodes` nodes and `edges`
/// couplers, spread as evenly as a simple graph allows, with
/// [`ADVERTISED_BETAS`] betas: at every read count the miner accepts, on
/// every coefficient layout, under the capacity model, the 32-bit index
/// guards, and the one-sweep limit (`sampler::session_admits_shape`).
#[must_use]
pub fn session_admits(kernel: Kernel, nodes: usize, edges: usize, limits: &DeviceLimits) -> bool {
    let complete = nodes.saturating_mul(nodes.saturating_sub(1)) / 2;
    if nodes == 0 || edges > complete {
        return false;
    }
    let max_degree = edges.saturating_mul(2).div_ceil(nodes).min(nodes - 1);
    accepted_reads().all(|reads| {
        layouts(kernel).iter().all(|&(precision, dense_precision)| {
            let shape = ModelShape {
                nodes,
                edges,
                max_degree,
                precision,
                dense_precision,
                betas: ADVERTISED_BETAS,
                reads,
            };
            crate::sampler::session_admits_shape(kernel, &shape, limits)
        })
    })
}

/// The largest `n` in `0..=high` for which `fits(n)` holds, or 0. A binary
/// search: it returns an `n` that fits and whose successor does not, which
/// is the largest when `fits` is monotone.
fn largest_fitting(high: usize, fits: impl Fn(usize) -> bool) -> usize {
    // Invariant: `low` fits or is 0; `high + 1` does not fit.
    let (mut low, mut high) = (0, high);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// The node count past which no dense-matrix sweep fits one chunk on any
/// kernel: the dense row index alone stops below it ([`check_indices`]).
const DENSE_SEARCH_NODES: usize = 1 << 16;

/// The most couplers a miner advertises: the couplers of the largest
/// complete model the session admits ([`session_admits`]), or 0 when none
/// fits. Admission of a complete model falls as its nodes grow, so the
/// search finds the largest.
#[must_use]
pub fn advertised_couplers(kernel: Kernel, limits: &DeviceLimits) -> usize {
    let complete = |n: usize| n * n.saturating_sub(1) / 2;
    complete(largest_fitting(DENSE_SEARCH_NODES, |n| {
        session_admits(kernel, n, complete(n), limits)
    }))
}

/// The most nodes a miner advertises: the largest node count the session
/// admits with the advertised couplers ([`advertised_couplers`]), or every
/// coupler of a complete model when it has fewer. The reference degree of
/// that model is `2 * couplers / nodes`. `cap` lowers the limit.
///
/// Admission is not monotone in the node count at a fixed coupler count: a
/// model just past the largest complete one takes dense-matrix storage and
/// fails the one-sweep limit, and a larger one takes CSR storage and fits.
/// The binary search returns a node count that the session admits and whose
/// successor it refuses.
///
/// # Errors
///
/// [`CapacityError::Nodes`] naming the limit when `cap` is 0 or exceeds it,
/// or when the limit is 0: the harness reads a zero `max_nodes` as no bound
/// at all.
pub fn advertised_nodes(
    kernel: Kernel,
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<usize, CapacityError> {
    let couplers = advertised_couplers(kernel, limits);
    let limit = largest_fitting(MAX_NODES, |n| {
        let complete = n * n.saturating_sub(1) / 2;
        session_admits(kernel, n, couplers.min(complete), limits)
    });
    match cap {
        _ if limit == 0 => Err(CapacityError::Nodes { max: 0 }),
        Some(cap) if cap == 0 || cap > limit => Err(CapacityError::Nodes { max: limit }),
        Some(cap) => Ok(cap),
        None => Ok(limit),
    }
}

/// Saturating sum of buffer sizes.
fn sum(sizes: &[usize]) -> usize {
    sizes.iter().fold(0, |total, &b| total.saturating_add(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNELS: [Kernel; 3] = [Kernel::Sa, Kernel::Gibbs, Kernel::Msa];

    const ROOMY: DeviceLimits = DeviceLimits {
        memory_bytes: 1 << 40,
        max_buffer_bytes: 1 << 40,
    };

    fn unit_shape() -> ModelShape {
        ModelShape {
            nodes: 4_577,
            edges: 41_515,
            max_degree: 20,
            precision: Precision::Binary,
            dense_precision: Precision::Binary,
            betas: 64,
            reads: 100,
        }
    }

    fn sparse_shape() -> ModelShape {
        ModelShape {
            nodes: 200_000,
            edges: 4_000_000,
            max_degree: 40,
            precision: Precision::Int32,
            dense_precision: Precision::Int32,
            betas: 64,
            reads: 100,
        }
    }

    fn complete_shape() -> ModelShape {
        ModelShape {
            nodes: 2_000,
            edges: 1_999_000,
            max_degree: 1_999,
            precision: Precision::Int8,
            dense_precision: Precision::Int8,
            betas: 64,
            reads: 100,
        }
    }

    #[test]
    fn usable_bytes_keeps_one_fifth_headroom() {
        let limits = DeviceLimits {
            memory_bytes: 1000,
            max_buffer_bytes: 1000,
        };
        assert_eq!(limits.usable_bytes(), 800);
    }

    #[test]
    fn usable_bytes_does_not_overflow_on_huge_memory() {
        let limits = DeviceLimits {
            memory_bytes: usize::MAX,
            max_buffer_bytes: usize::MAX,
        };
        assert!(limits.usable_bytes() > usize::MAX / 5 * 4 - 1);
    }

    #[test]
    fn host_limits_read_half_of_physical_memory() {
        let limits = host_limits();
        assert!(limits.memory_bytes > 0);
        assert_eq!(limits.max_buffer_bytes, limits.memory_bytes);
    }

    #[test]
    fn storage_names_match_the_cuda_json() {
        assert_eq!(Storage::Threadgroup.name(Kernel::Sa), "local");
        assert_eq!(Storage::Spill.name(Kernel::Sa), "global");
        assert_eq!(Storage::Dense.name(Kernel::Sa), "dense");
        for kernel in [Kernel::Gibbs, Kernel::Msa] {
            assert_eq!(Storage::Threadgroup.name(kernel), "shared");
            assert_eq!(Storage::Spill.name(kernel), "spill");
            assert_eq!(Storage::Dense.name(kernel), "dense");
        }
    }

    #[test]
    fn advantage2_unit_models_stay_on_threadgroup_storage() {
        for kernel in KERNELS {
            let fp = footprint(kernel, &unit_shape(), &ROOMY).unwrap();
            assert_eq!(fp.storage, Storage::Threadgroup, "{kernel:?}");
        }
    }

    #[test]
    fn a_sparse_model_past_the_threadgroup_routes_to_spill() {
        for kernel in KERNELS {
            let fp = footprint(kernel, &sparse_shape(), &ROOMY).unwrap();
            assert_eq!(fp.storage, Storage::Spill, "{kernel:?}");
        }
    }

    /// The spill build keeps one byte per spin of every read in its
    /// carry-over buffer, so the footprint past the threadgroup array
    /// counts it.
    #[test]
    fn gibbs_past_the_threadgroup_array_counts_the_per_read_spill() {
        let shape = ModelShape {
            nodes: 20_000,
            edges: 200_000,
            max_degree: 20,
            precision: Precision::Int8,
            dense_precision: Precision::Int8,
            betas: 64,
            reads: 100,
        };
        let fp = footprint(Kernel::Gibbs, &shape, &ROOMY).unwrap();
        assert_eq!(fp.storage, Storage::Spill);
        assert_eq!(fp.storage.name(Kernel::Gibbs), "spill");
        let spill = simd_rounded_reads(100) * 20_000;
        assert!(fp.per_instance_bytes > spill);
        let smaller = ModelShape {
            nodes: GIBBS_MAX_NODES,
            ..shape
        };
        let local = footprint(Kernel::Gibbs, &smaller, &ROOMY).unwrap();
        assert_eq!(local.storage, Storage::Threadgroup);
    }

    /// The global build keeps packed spins and `delta_energy` for every
    /// node of every read in its carry-over buffers.
    #[test]
    fn sa_past_the_local_frame_counts_the_per_read_state() {
        for (precision, delta_bytes) in [(Precision::Binary, 1), (Precision::Int32, 4)] {
            let shape = ModelShape {
                nodes: 20_000,
                edges: 200_000,
                max_degree: 20,
                precision,
                dense_precision: precision,
                betas: 64,
                reads: 100,
            };
            let fp = footprint(Kernel::Sa, &shape, &ROOMY).unwrap();
            assert_eq!(fp.storage, Storage::Spill, "{precision:?}");
            assert_eq!(fp.storage.name(Kernel::Sa), "global");
            let reads = simd_rounded_reads(100);
            let state = reads * 20_000usize.div_ceil(8) + reads * 20_000 * delta_bytes;
            assert!(fp.per_instance_bytes > state, "{precision:?}");
            let smaller = ModelShape {
                nodes: SA_MAX_NODES,
                ..shape
            };
            let local = footprint(Kernel::Sa, &smaller, &ROOMY).unwrap();
            assert_eq!(local.storage, Storage::Threadgroup, "{precision:?}");
        }
    }

    /// A complete model has one colour class per node, so the dense tier of
    /// the colour-walking kernels prices a colour start and a count per node
    /// beside the colour node list; SA walks no colours and prices none. The
    /// sampler's admission takes the dense tier for Gibbs.
    #[test]
    fn the_dense_tier_prices_the_colour_arrays_of_the_colour_kernels() {
        let shape = ModelShape {
            nodes: 1_000,
            edges: 1_000 * 999 / 2,
            max_degree: 999,
            precision: Precision::Int8,
            dense_precision: Precision::Int8,
            betas: 64,
            reads: 256,
        };
        let list = common_shared(&shape);
        let arrays = [shape.nodes * 4; 2];
        for kernel in [Kernel::Gibbs, Kernel::Msa] {
            let shared = dense_sizes(kernel, &shape).shared;
            assert_eq!(
                shared,
                [list.clone(), arrays.to_vec()].concat(),
                "{kernel:?}"
            );
        }
        assert_eq!(dense_sizes(Kernel::Sa, &shape).shared, list);
        let runnable = runnable_footprint(Kernel::Gibbs, &shape, &ROOMY).unwrap();
        assert_eq!(runnable.storage, Storage::Dense);
        assert_eq!(
            runnable.shared_bytes,
            sum(&dense_sizes(Kernel::Gibbs, &shape).shared)
        );
    }

    /// A unit model's dense matrix holds two `uint` bit planes per 32
    /// columns, 2 bits per cell (`sampler::upload_dense_inputs`), and the
    /// dense tier prices the dense layout, not the CSR one.
    #[test]
    fn the_dense_tier_prices_the_matrix_that_runs() {
        let unit = ModelShape {
            nodes: 1_000,
            edges: 1_000 * 999 / 2,
            max_degree: 999,
            precision: Precision::Int32,
            dense_precision: Precision::Binary,
            betas: 64,
            reads: 64,
        };
        // 32 words of two planes per row of 1,000 columns.
        assert_eq!(
            dense_sizes(Kernel::Msa, &unit).per_instance[0],
            1_000 * 64 * 4
        );
        let scaled = ModelShape {
            dense_precision: Precision::Int8Scaled,
            ..unit
        };
        let sizes = dense_sizes(Kernel::Sa, &scaled);
        assert_eq!(sizes.per_instance[0], 1_000 * 1_000);
        // The fields and the SA deltas follow the matrix: one byte per
        // field, four per `int` delta.
        assert_eq!(sizes.per_instance[1], 1_000);
        assert!(sizes
            .per_instance
            .contains(&(simd_rounded_reads(64) * 1_000 * 4)));
    }

    #[test]
    fn a_complete_model_routes_to_dense() {
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            let fp = footprint(kernel, &complete_shape(), &ROOMY).unwrap();
            assert_eq!(fp.storage, Storage::Dense, "{kernel:?}");
        }
    }

    #[test]
    fn the_cheaper_route_wins() {
        // Between the node cap and a dense-friendly size both tiers admit
        // SA and Gibbs, so the choice is a pure cost comparison.
        let shapes = [unit_shape(), sparse_shape(), complete_shape()];
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            for shape in shapes {
                let chosen = footprint(kernel, &shape, &ROOMY).unwrap();
                for storage in [Storage::Threadgroup, Storage::Spill, Storage::Dense] {
                    if !admits(kernel, storage, &shape) {
                        continue;
                    }
                    let sizes = match storage {
                        Storage::Threadgroup => threadgroup_sizes(kernel, &shape),
                        Storage::Spill => spill_sizes(kernel, &shape),
                        Storage::Dense => dense_sizes(kernel, &shape),
                    };
                    let bytes: usize = sizes.per_instance.iter().sum();
                    assert!(
                        chosen.per_instance_bytes <= bytes,
                        "{kernel:?} {storage:?}: {} > {bytes}",
                        chosen.per_instance_bytes
                    );
                }
            }
        }
    }

    /// A complete graph on `n` nodes, each pair once.
    fn complete_graph(n: usize) -> IsingGraph {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
            .collect();
        IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges)
    }

    /// `graph` plus one more coupler.
    fn with_edge(graph: &IsingGraph, edge: (usize, usize)) -> IsingGraph {
        let mut edges = graph.edges.to_vec();
        edges.push(edge);
        let mut j = graph.j.to_vec();
        j.push(1.0);
        IsingGraph::new(graph.h.to_vec(), j, edges)
    }

    fn shape_of(graph: &IsingGraph) -> ModelShape {
        let mut degree = vec![0usize; graph.num_nodes()];
        for &(u, v) in graph.edges.iter() {
            degree[u] += 1;
            degree[v] += 1;
        }
        ModelShape {
            nodes: graph.num_nodes(),
            edges: graph.edges.len(),
            max_degree: degree.into_iter().max().unwrap_or(0),
            precision: Precision::Int8,
            dense_precision: Precision::Int8,
            betas: 64,
            reads: 64,
        }
    }

    #[test]
    fn a_repeated_coupler_stays_sparse() {
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            let graph = complete_graph(64);
            let shape = shape_of(&graph);
            let clean = routed_footprint(kernel, &graph, &shape, &ROOMY).unwrap();
            assert_eq!(clean.storage, Storage::Dense, "{kernel:?}");
            // The same pair in the other orientation is still a repeat.
            let graph = with_edge(&graph, (1, 0));
            let shape = shape_of(&graph);
            let fp = routed_footprint(kernel, &graph, &shape, &ROOMY).unwrap();
            assert_ne!(fp.storage, Storage::Dense, "{kernel:?}");
        }
    }

    #[test]
    fn a_self_loop_stays_sparse() {
        let graph = with_edge(&complete_graph(64), (5, 5));
        let shape = shape_of(&graph);
        let fp = routed_footprint(Kernel::Sa, &graph, &shape, &ROOMY).unwrap();
        assert_ne!(fp.storage, Storage::Dense);
        assert!(!crate::topology::dense_candidate(&graph));
    }

    #[test]
    fn msa_above_degree_62_routes_dense() {
        let graph = complete_graph(80);
        let shape = shape_of(&graph);
        assert!(shape.max_degree > MSA_SPARSE_MAX_DEG);
        let fp = footprint(Kernel::Msa, &shape, &ROOMY).unwrap();
        assert_eq!(fp.storage, Storage::Dense);
        let fp = routed_footprint(Kernel::Msa, &graph, &shape, &ROOMY).unwrap();
        assert_eq!(fp.storage, Storage::Dense);
        // A repeated coupler keeps the model off dense, so no tier admits it.
        let repeated = with_edge(&graph, (1, 0));
        assert_eq!(
            routed_footprint(Kernel::Msa, &repeated, &shape_of(&repeated), &ROOMY),
            Err(CapacityError::Degree {
                max: MSA_SPARSE_MAX_DEG
            })
        );
    }

    #[test]
    fn the_sampler_prices_msa_dense_and_routes_non_candidates_off_it() {
        let graph = complete_graph(80);
        let shape = shape_of(&graph);
        assert_eq!(
            runnable_footprint(Kernel::Msa, &shape, &ROOMY)
                .unwrap()
                .storage,
            Storage::Dense
        );
        assert_eq!(
            routed_runnable_footprint(Kernel::Msa, &graph, &shape, &ROOMY)
                .unwrap()
                .storage,
            Storage::Dense
        );
        let repeated = with_edge(&graph, (1, 0));
        assert_eq!(
            routed_runnable_footprint(Kernel::Msa, &repeated, &shape_of(&repeated), &ROOMY),
            Err(CapacityError::Degree {
                max: MSA_SPARSE_MAX_DEG
            })
        );
        // SA and Gibbs run dense too, and a repeated coupler keeps them on
        // CSR.
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            assert_eq!(
                routed_runnable_footprint(kernel, &graph, &shape, &ROOMY)
                    .unwrap()
                    .storage,
                Storage::Dense,
                "{kernel:?}"
            );
            assert_ne!(
                routed_runnable_footprint(kernel, &repeated, &shape_of(&repeated), &ROOMY)
                    .unwrap()
                    .storage,
                Storage::Dense,
                "{kernel:?}"
            );
        }
    }

    #[test]
    fn sparse_models_pay_no_pair_scan() {
        let n = 2_000;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| (1..=10).map(move |o| (u, (u + o) % n)))
            .collect();
        let graph = IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges);
        let shape = shape_of(&graph);
        crate::topology::reset_pair_scan_count();
        let fp = routed_footprint(Kernel::Sa, &graph, &shape, &ROOMY).unwrap();
        assert_ne!(fp.storage, Storage::Dense);
        assert_eq!(crate::topology::pair_scan_count(), 0);
        let complete = complete_graph(64);
        routed_footprint(Kernel::Sa, &complete, &shape_of(&complete), &ROOMY).unwrap();
        assert_eq!(crate::topology::pair_scan_count(), 1);
    }

    #[test]
    fn a_model_over_memory_is_refused() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 20,
            max_buffer_bytes: 1 << 20,
        };
        for kernel in KERNELS {
            let err = footprint(kernel, &sparse_shape(), &limits).unwrap_err();
            assert!(matches!(err, CapacityError::Memory { .. }), "{kernel:?}");
        }
    }

    #[test]
    fn a_buffer_over_the_limit_is_refused() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 40,
            max_buffer_bytes: 1 << 10,
        };
        let err = footprint(Kernel::Sa, &unit_shape(), &limits).unwrap_err();
        assert!(matches!(err, CapacityError::Buffer { .. }), "{err:?}");
    }

    #[test]
    fn more_nodes_than_the_kernel_can_index_are_refused() {
        let mut shape = unit_shape();
        shape.nodes = MAX_NODES + 1;
        let err = footprint(Kernel::Sa, &shape, &ROOMY).unwrap_err();
        assert_eq!(err, CapacityError::Nodes { max: MAX_NODES });
    }

    #[test]
    fn instances_follow_the_stream_and_the_resident_bound() {
        assert_eq!(
            instances(Kernel::Sa, 100),
            2 * batch_size_for_reads(Kernel::Sa, 100)
        );
        assert_eq!(
            instances(Kernel::Gibbs, 100),
            2 * batch_size_for_reads(Kernel::Gibbs, 100)
        );
        assert_eq!(instances(Kernel::Msa, 100), 40);
    }

    /// A stream batch holds at most `batch_size_for_reads` problems, and the
    /// stream keeps `STREAM_IN_FLIGHT` batches on the GPU, so a footprint
    /// that fits at `instances` covers every batch in flight at every read
    /// count. One sweep of a capped batch also stays inside the cap.
    #[test]
    fn the_priced_instances_cover_the_stream_batches_in_flight() {
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            for reads in accepted_reads() {
                let batch = batch_size_for_reads(kernel, reads);
                assert_eq!(
                    instances(kernel, reads),
                    STREAM_IN_FLIGHT * batch,
                    "{kernel:?} {reads}"
                );
                let graph = complete_graph(8);
                let params = quip_solver_core::SampleParams {
                    num_reads: reads,
                    num_sweeps: 64,
                    ..Default::default()
                };
                let cap = crate::sampler::one_sweep_problem_cap(
                    &graph,
                    &params,
                    kernel,
                    STREAM_IN_FLIGHT,
                    batch,
                );
                assert!(
                    cap.is_none_or(|cap| cap <= batch),
                    "{kernel:?} {reads}: {cap:?}"
                );
            }
        }
    }

    #[test]
    fn total_adds_shared_bytes_once_per_batch() {
        let fp = Footprint {
            storage: Storage::Threadgroup,
            per_instance_bytes: 10,
            shared_bytes: 7,
        };
        assert_eq!(fp.total(3, 1), 37);
        assert_eq!(fp.total(3, 2), 44);
    }

    /// Each stream batch in flight allocates its own beta schedule and colour
    /// list, so SA and Gibbs count the shared buffers twice. The multi-spin
    /// batch path runs one batch at a time.
    #[test]
    fn the_shared_buffers_count_once_per_batch_in_flight() {
        let shape = unit_shape();
        for kernel in KERNELS {
            let sizes = threadgroup_sizes(kernel, &shape);
            let once =
                sum(&sizes.per_instance) * instances(kernel, shape.reads) + sum(&sizes.shared);
            // `usable_bytes` is 4/5 of the memory.
            let limits = DeviceLimits {
                memory_bytes: once / 4 * 5 + 5,
                max_buffer_bytes: 1 << 40,
            };
            assert!(limits.usable_bytes() >= once, "{kernel:?}");
            assert!(
                limits.usable_bytes() < once + sum(&sizes.shared),
                "{kernel:?}"
            );
            let fits = footprint(kernel, &shape, &limits);
            if kernel == Kernel::Msa {
                assert_eq!(fits.map(|fp| fp.storage), Ok(Storage::Threadgroup));
            } else {
                assert!(
                    matches!(fits, Err(CapacityError::Memory { .. })),
                    "{kernel:?}: {fits:?}"
                );
            }
        }
    }

    #[test]
    fn dense_carries_at_least_the_persistent_buffers() {
        for kernel in KERNELS {
            for shape in [unit_shape(), complete_shape()] {
                let persistent = sum(&persistent_buffers(kernel, &shape));
                assert!(persistent > 0, "{kernel:?}");
                let dense = sum(&dense_sizes(kernel, &shape).per_instance);
                let threadgroup = sum(&threadgroup_sizes(kernel, &shape).per_instance);
                assert!(dense >= persistent, "{kernel:?}: {dense} < {persistent}");
                assert!(threadgroup >= persistent, "{kernel:?}");
            }
        }
    }

    #[test]
    fn spill_counts_each_buffer_once() {
        for kernel in KERNELS {
            let shape = sparse_shape();
            let spill = spill_sizes(kernel, &shape);
            let threadgroup = threadgroup_sizes(kernel, &shape);
            assert_eq!(spill.per_instance, threadgroup.per_instance, "{kernel:?}");
            assert_eq!(spill.shared, threadgroup.shared, "{kernel:?}");
        }
    }

    #[test]
    fn memory_is_checked_before_the_buffer_limit() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 20,
            max_buffer_bytes: 1 << 10,
        };
        let err = footprint(Kernel::Sa, &sparse_shape(), &limits).unwrap_err();
        assert!(matches!(err, CapacityError::Memory { .. }), "{err:?}");
    }

    #[test]
    fn a_shared_buffer_over_the_limit_is_counted() {
        let sizes = Sizes {
            per_instance: vec![10, 20],
            shared: vec![5, 1_000],
        };
        // Per-model buffers are multiplied by the tiling; shared ones are not.
        assert_eq!(largest_buffer(&sizes, 4), 1_000);
        assert_eq!(largest_buffer(&sizes, 100), 2_000);
    }

    #[test]
    fn a_shared_buffer_alone_over_the_limit_returns_buffer() {
        let shape = ModelShape {
            nodes: 10,
            edges: 10,
            max_degree: 3,
            precision: Precision::Int8,
            dense_precision: Precision::Int8,
            betas: 10_000_000,
            reads: 32,
        };
        let limits = DeviceLimits {
            memory_bytes: 1 << 40,
            max_buffer_bytes: 1 << 20,
        };
        for kernel in KERNELS {
            let err = footprint(kernel, &shape, &limits).unwrap_err();
            assert_eq!(
                err,
                CapacityError::Buffer {
                    needed: 40_000_000,
                    max: 1 << 20
                },
                "{kernel:?}"
            );
        }
    }

    #[test]
    fn a_costlier_tier_is_used_when_the_cheapest_fails_the_buffer_limit() {
        // Dense is the cheaper tier (a bit matrix of 1,008,000 bytes against
        // 1.2 MB of CSR), but its matrix is the largest single buffer. A
        // limit between the two largest buffers refuses dense only.
        let shape = ModelShape {
            nodes: 2_000,
            edges: 120_000,
            max_degree: 120,
            precision: Precision::Binary,
            dense_precision: Precision::Binary,
            betas: 64,
            reads: 100,
        };
        let kernel = Kernel::Sa;
        let tiling = models_per_buffer(kernel, shape.reads);
        let dense = dense_sizes(kernel, &shape);
        let threadgroup = threadgroup_sizes(kernel, &shape);
        assert!(sum(&dense.per_instance) < sum(&threadgroup.per_instance));
        let dense_largest = largest_buffer(&dense, tiling);
        let threadgroup_largest = largest_buffer(&threadgroup, tiling);
        assert!(dense_largest > threadgroup_largest);
        let limits = DeviceLimits {
            memory_bytes: 1 << 40,
            max_buffer_bytes: threadgroup_largest,
        };
        assert_eq!(
            footprint(kernel, &shape, &limits).unwrap().storage,
            Storage::Threadgroup
        );
        let roomy = DeviceLimits {
            max_buffer_bytes: dense_largest,
            ..limits
        };
        assert_eq!(
            footprint(kernel, &shape, &roomy).unwrap().storage,
            Storage::Dense
        );
        let tight = DeviceLimits {
            max_buffer_bytes: 1,
            ..limits
        };
        assert_eq!(
            footprint(kernel, &shape, &tight).unwrap_err(),
            CapacityError::Buffer {
                needed: dense_largest,
                max: 1
            }
        );
    }

    #[test]
    fn msa_col_ind_is_in_the_shared_buffers() {
        let shape = unit_shape();
        let nnz_bytes = shape.edges * 2 * 4;
        let sizes = threadgroup_sizes(Kernel::Msa, &shape);
        assert!(sizes.shared.contains(&nnz_bytes));
    }

    /// 64 GiB of planning memory, near a large Apple Silicon host.
    const BIG: DeviceLimits = DeviceLimits {
        memory_bytes: 64 << 30,
        max_buffer_bytes: 64 << 30,
    };

    fn shape_for(connectivity: Connectivity, nodes: usize, reads: usize) -> ModelShape {
        let (edges, max_degree) = connectivity.size(nodes);
        ModelShape {
            nodes,
            edges,
            max_degree,
            precision: Precision::Binary,
            dense_precision: Precision::Binary,
            betas: 64,
            reads,
        }
    }

    #[test]
    fn max_model_is_the_last_shape_that_fits() {
        for kernel in KERNELS {
            let connectivity = Connectivity::Degree(20);
            let reads = MAX_READS;
            let (shape, fp) = max_model(kernel, connectivity, Precision::Binary, &BIG, reads)
                .unwrap_or_else(|| panic!("{kernel:?}: nothing fits"));
            assert_eq!(footprint(kernel, &shape, &BIG), Ok(fp), "{kernel:?}");
            // Memory is the binding limit here: the buffer limit equals the
            // planning memory, and the total always exceeds its largest buffer.
            let bigger = shape_for(connectivity, shape.nodes + 1, reads);
            let err = footprint(kernel, &bigger, &BIG).unwrap_err();
            assert!(
                matches!(err, CapacityError::Memory { .. }),
                "{kernel:?}: {err:?}"
            );
        }
    }

    /// The advertised corners are admitted at every read count and layout,
    /// and one node more is refused: `max_nodes` with `max_edges` couplers,
    /// and the complete model of `max_edges` couplers.
    #[test]
    fn the_advertised_corners_are_the_last_admitted_shapes() {
        for kernel in KERNELS {
            let nodes = advertised_nodes(kernel, &BIG, None).unwrap();
            let couplers = advertised_couplers(kernel, &BIG);
            assert!(session_admits(kernel, nodes, couplers, &BIG), "{kernel:?}");
            assert!(
                !session_admits(kernel, nodes + 1, couplers, &BIG),
                "{kernel:?}"
            );
            let complete = largest_fitting(DENSE_SEARCH_NODES, |n| n * (n - 1) / 2 <= couplers);
            assert_eq!(complete * (complete - 1) / 2, couplers, "{kernel:?}");
            let past = complete + 1;
            assert!(
                !session_admits(kernel, past, past * complete / 2, &BIG),
                "{kernel:?}"
            );
        }
    }

    /// A count with more couplers than a simple graph holds is never
    /// admitted, and neither is an empty model.
    #[test]
    fn session_admits_only_simple_graphs() {
        for kernel in KERNELS {
            assert!(session_admits(kernel, 4, 6, &BIG), "{kernel:?}");
            assert!(!session_admits(kernel, 4, 7, &BIG), "{kernel:?}");
            assert!(!session_admits(kernel, 0, 0, &BIG), "{kernel:?}");
        }
    }

    #[test]
    fn a_cap_above_the_limit_names_the_limit() {
        for kernel in KERNELS {
            let limit = advertised_nodes(kernel, &BIG, None).unwrap();
            assert!(limit > 0, "{kernel:?}");
            assert_eq!(
                advertised_nodes(kernel, &BIG, Some(usize::MAX)),
                Err(CapacityError::Nodes { max: limit }),
                "{kernel:?}"
            );
            assert_eq!(advertised_nodes(kernel, &BIG, Some(limit)), Ok(limit));
            assert_eq!(advertised_nodes(kernel, &BIG, Some(1)), Ok(1));
        }
    }

    #[test]
    fn edges_past_32_bit_indexing_are_refused_whatever_the_memory() {
        let mut shape = unit_shape();
        shape.edges = u32::MAX as usize / 2 + 1;
        for kernel in KERNELS {
            let err = footprint(kernel, &shape, &ROOMY).unwrap_err();
            assert!(
                matches!(err, CapacityError::Buffer { .. }),
                "{kernel:?}: {err:?}"
            );
        }
    }

    /// Planning memory far past any model here, so only an index bound can
    /// refuse one.
    const HUGE: DeviceLimits = DeviceLimits {
        memory_bytes: 1 << 60,
        max_buffer_bytes: 1 << 60,
    };

    fn edgeless(nodes: usize, precision: Precision) -> ModelShape {
        ModelShape {
            nodes,
            edges: 0,
            max_degree: 0,
            precision,
            dense_precision: precision,
            betas: 64,
            reads: 256,
        }
    }

    /// The SA global build indexes entry `i` of thread `t` as
    /// `i * num_threads + t` in 32 bits (`sampler::encode_batch_storage`),
    /// so one problem of 256 reads holds at most `u32::MAX / 256` nodes.
    #[test]
    fn the_capacity_model_mirrors_the_sa_global_index_guard() {
        let last = u32::MAX as usize / 256;
        let fits = footprint(Kernel::Sa, &edgeless(last, Precision::Int32), &HUGE).unwrap();
        assert_eq!(fits.storage, Storage::Spill);
        assert_eq!(
            footprint(Kernel::Sa, &edgeless(last + 1, Precision::Int32), &HUGE),
            Err(CapacityError::Index {
                entries: (last + 1) * 256,
                max: u32::MAX as usize
            })
        );
    }

    /// The multi-spin kernel indexes each threadgroup's words as
    /// `threadgroup * N` in 32 bits: 8 words of 256 reads. Its packed output
    /// index, 32 bytes per node at 256 reads, binds first.
    #[test]
    fn the_capacity_model_mirrors_the_multi_spin_index_guard() {
        let last = u32::MAX as usize / 8;
        let output_last = (u32::MAX as usize / 256) * 8;
        footprint(Kernel::Msa, &edgeless(output_last, Precision::Int32), &HUGE).unwrap();
        assert!(matches!(
            footprint(Kernel::Msa, &edgeless(last + 1, Precision::Int32), &HUGE),
            Err(CapacityError::Index { .. })
        ));
    }

    /// The chromatic Gibbs builds index the RNG streams as
    /// `(sample * 256 + thread) * 4` in 32 bits; the shared and dense builds
    /// also index the spins as `sample * N`. The spill build addresses its
    /// spins in 64 bits.
    #[test]
    fn the_gibbs_index_guard_bounds_the_rng_streams_and_the_shared_spins() {
        let shape = edgeless(4_000, Precision::Int8);
        let samples = u32::MAX as usize / (256 * 4);
        let problems = samples / 256;
        assert_eq!(
            check_indices(Kernel::Gibbs, Storage::Spill, &shape, problems),
            Ok(())
        );
        assert!(check_indices(Kernel::Gibbs, Storage::Spill, &shape, problems + 1).is_err());
        // `sample * N` binds first on the shared build.
        let shared = u32::MAX as usize / (256 * 4_000);
        assert_eq!(
            check_indices(Kernel::Gibbs, Storage::Threadgroup, &shape, shared),
            Ok(())
        );
        assert!(check_indices(Kernel::Gibbs, Storage::Threadgroup, &shape, shared + 1).is_err());
    }

    /// The SA and Gibbs kernels read their tiled CSR copies through `int`
    /// offsets (`sampler::upload_inputs`); the multi-spin kernel reads one
    /// untiled copy.
    #[test]
    fn the_tiled_csr_offsets_stay_in_int_range() {
        let shape = ModelShape {
            edges: 1_000_000,
            max_degree: 1_000,
            reads: 32,
            ..edgeless(2_000, Precision::Int8)
        };
        let copies = i32::MAX as usize / 2_000_000;
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            assert_eq!(
                check_indices(kernel, Storage::Spill, &shape, copies),
                Ok(()),
                "{kernel:?}"
            );
            assert!(
                check_indices(kernel, Storage::Spill, &shape, copies + 1).is_err(),
                "{kernel:?}"
            );
        }
    }

    /// The multi-spin kernel reads one shared CSR structure but tiles the
    /// couplings per problem through `int` offsets, and indexes its packed
    /// output as `(problem * reads + read) * packed_size + byte` in 32 bits.
    #[test]
    fn the_multi_spin_coupling_offsets_and_output_index_are_32_bit() {
        let shape = ModelShape {
            edges: 1_000_000,
            max_degree: 1_000,
            reads: 32,
            ..edgeless(2_000, Precision::Int32)
        };
        let copies = i32::MAX as usize / 2_000_000;
        assert_eq!(
            check_indices(Kernel::Msa, Storage::Spill, &shape, copies),
            Ok(())
        );
        assert!(check_indices(Kernel::Msa, Storage::Spill, &shape, copies + 1).is_err());
        // One problem of 256 reads: the output index binds before the
        // per-word state at `256 * ceil(N / 8)` entries.
        let packed = |nodes: usize| 256 * nodes.div_ceil(8);
        let last = (u32::MAX as usize / 256) * 8;
        assert!(packed(last) <= u32::MAX as usize);
        assert!(packed(last + 8) > u32::MAX as usize);
        let at = |nodes| {
            check_indices(
                Kernel::Msa,
                Storage::Spill,
                &edgeless(nodes, Precision::Int32),
                1,
            )
        };
        assert_eq!(at(last), Ok(()));
        assert!(at(last + 8).is_err());
    }

    /// A dense row of an `n`-node model is indexed in 32 bits: `n` cells,
    /// or the `2 * ceil(n / 32)` words of a bit matrix.
    #[test]
    fn the_dense_row_index_is_32_bit() {
        let at = |nodes, precision| {
            check_indices(Kernel::Msa, Storage::Dense, &edgeless(nodes, precision), 1)
        };
        assert_eq!(at(65_535, Precision::Int32), Ok(()));
        assert!(at(65_536, Precision::Int32).is_err());
        assert_eq!(at(65_536, Precision::Binary), Ok(()));
    }

    #[test]
    fn fit_is_monotone_in_the_node_count() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 25,
            max_buffer_bytes: 1 << 25,
        };
        for kernel in KERNELS {
            let reads = MAX_READS;
            for connectivity in [Connectivity::Degree(20), Connectivity::Complete] {
                let mut failed = false;
                for nodes in (1..40_000).step_by(97) {
                    let fits =
                        footprint(kernel, &shape_for(connectivity, nodes, reads), &limits).is_ok();
                    assert!(!(failed && fits), "{kernel:?} {connectivity:?} {nodes}");
                    failed |= !fits;
                }
                assert!(
                    failed,
                    "{kernel:?} {connectivity:?} never reached the limit"
                );
            }
        }
    }

    #[test]
    fn max_model_is_none_when_one_node_does_not_fit() {
        let tiny = DeviceLimits {
            memory_bytes: 16,
            max_buffer_bytes: 16,
        };
        for kernel in KERNELS {
            let reads = MAX_READS;
            for connectivity in [Connectivity::Degree(20), Connectivity::Complete] {
                assert!(
                    max_model(kernel, connectivity, Precision::Binary, &tiny, reads).is_none(),
                    "{kernel:?} {connectivity:?}"
                );
            }
            // A zero `max_nodes` would advertise no bound, so the limit is
            // an error.
            assert_eq!(
                advertised_nodes(kernel, &tiny, None),
                Err(CapacityError::Nodes { max: 0 })
            );
            assert_eq!(advertised_couplers(kernel, &tiny), 0);
        }
    }
}
