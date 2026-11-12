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
//!   same buffers. SA and Gibbs have no spill kernel yet; their formulas
//!   follow the intended layout and later tasks correct them.
//! - [`Storage::Dense`]: an `N * N` coupling matrix instead of CSR. Not yet
//!   implemented, and no multi-spin model is admitted to it until a dense
//!   multi-spin kernel exists.
//!
//! Every formula sits in one small function per kernel, so a later task can
//! correct one without touching the routing.

use crate::sampler::{
    simd_rounded_reads, Kernel, GIBBS_MAX_NODES, MSA_LANES, MSA_MAX_NODES, MSA_SPARSE_MAX_DEG,
    MSA_THREADS, MSA_UNIT_MAX_DEG, SA_MAX_NODES,
};
use crate::streaming::batch_size_for_reads;

/// Largest node count the 32-bit kernel indexing can address.
pub const MAX_NODES: usize = i32::MAX as usize;

/// Largest CSR entry count (`2 * edges`) the 32-bit kernel indexing can
/// address.
const MAX_NNZ: usize = u32::MAX as usize;
/// Bytes of one `col_ind` entry.
const COL_IND_ENTRY_BYTES: usize = 4;
/// Beta entries the advertised limits plan against.
const ADVERTISED_BETAS: usize = 64;

/// Memory headroom numerator, as CUDA's `MEMORY_HEADROOM_NUM`.
const HEADROOM_NUM: usize = 4;
/// Memory headroom denominator, as CUDA's `MEMORY_HEADROOM_DEN`.
const HEADROOM_DEN: usize = 5;
/// Threadgroup memory of every Apple GPU, in bytes. No opt-in raises it.
const TG_MEMORY_BYTES: usize = 32_768;
/// Threads per chromatic-Gibbs threadgroup (`kernels/gibbs.metal`).
const GIBBS_GROUP_THREADS: usize = 256;
/// Bytes of xoshiro128** state per random stream (four `uint`).
const RNG_BYTES_PER_STREAM: usize = 16;

/// Memory limits of the device a miner runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Memory the miner may plan against, in bytes.
    pub memory_bytes: usize,
    /// Threadgroup memory per threadgroup, in bytes.
    pub tg_memory_bytes: usize,
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

/// Limits of this host: half of `hw.memsize` as planning memory, 32,768
/// bytes of threadgroup memory, and a buffer limit equal to the planning
/// memory.
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
        tg_memory_bytes: TG_MEMORY_BYTES,
        max_buffer_bytes: memory_bytes,
    }
}

/// Numeric type of the couplings and fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// One bit per coupling in a dense matrix; one byte in CSR.
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

    /// Bytes of an `n * n` dense coupling matrix.
    const fn dense_bytes(self, cells: usize) -> usize {
        match self {
            Self::Binary => cells.div_ceil(8),
            Self::Int8 | Self::Int8Scaled => cells,
            Self::Int32 | Self::Float32 => cells.saturating_mul(4),
        }
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
    /// Coefficient type.
    pub precision: Precision,
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
    /// Bytes for `instances` models at once.
    #[must_use]
    pub fn total(&self, instances: usize) -> usize {
        self.per_instance_bytes
            .saturating_mul(instances)
            .saturating_add(self.shared_bytes)
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
    /// The degree exceeds what every admitting tier supports.
    #[error("degree above {max} is not supported")]
    Degree {
        /// Largest admitted degree.
        max: usize,
    },
}

/// Models whose buffers exist at once.
///
/// SA and Gibbs return twice [`batch_size_for_reads`], because the stream
/// keeps two batches in flight. MSA returns the resident runner's bound on
/// queued, preparing, and ready jobs (`PREP_BOUND`, "At most 40 jobs" in
/// `src/resident.rs`).
#[must_use]
pub fn instances(kernel: Kernel, reads: usize) -> usize {
    match kernel {
        Kernel::Sa | Kernel::Gibbs => 2 * batch_size_for_reads(kernel, reads),
        Kernel::Msa => crate::resident::PREP_BOUND,
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
        // No dense multi-spin kernel exists, so MSA refuses this tier and a
        // model past the sparse degree is a `CapacityError::Degree`.
        Storage::Dense => kernel != Kernel::Msa,
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
/// the same buffers and adds none (`sampler::encode_batch_storage`). SA and
/// Gibbs have no spill kernel yet and are sized the same way.
fn spill_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    threadgroup_sizes(kernel, shape)
}

/// Dense-tier sizes: an `N * N` coupling matrix at the precision's width
/// (1 bit, 1, 4, 4 bytes) replaces the CSR structure and couplings. The
/// fields, the outputs, and the carry-over buffers remain.
fn dense_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    let mut per_instance = vec![
        shape
            .precision
            .dense_bytes(shape.nodes.saturating_mul(shape.nodes)),
        shape.nodes.saturating_mul(shape.precision.coef_bytes()),
    ];
    per_instance.extend(output_buffers(shape));
    per_instance.extend(persistent_buffers(kernel, shape));
    Sizes {
        per_instance,
        shared: common_shared(shape),
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
/// - [`CapacityError::Memory`] when `total(instances)` exceeds
///   `limits.usable_bytes()`. Of the per-tier checks this runs first; the
///   node and `col_ind` bounds above run before any tier.
/// - [`CapacityError::Buffer`] when the largest buffer exceeds
///   `limits.max_buffer_bytes`. A per-model buffer counts for the
///   [`models_per_buffer`] models it holds; a shared buffer counts once.
pub fn footprint(
    kernel: Kernel,
    shape: &ModelShape,
    limits: &DeviceLimits,
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
    .filter(|(storage, _)| admits(kernel, *storage, shape))
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
        let needed = footprint.total(in_flight);
        let error = if needed > limits.usable_bytes() {
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

/// Reads the advertised limits plan against: the kernel's read cap, so a
/// limit holds for every read count the miner accepts.
fn advertised_reads(kernel: Kernel) -> usize {
    crate::streaming::max_reads(kernel) as usize
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

/// The most nodes a miner advertises: the largest Binary model with no
/// edges. `cap` lowers it.
///
/// # Errors
///
/// [`CapacityError::Nodes`] naming the limit when `cap` exceeds it.
pub fn advertised_nodes(
    kernel: Kernel,
    limits: &DeviceLimits,
    cap: Option<usize>,
) -> Result<usize, CapacityError> {
    let limit = max_model(
        kernel,
        Connectivity::Degree(0),
        Precision::Binary,
        limits,
        advertised_reads(kernel),
    )
    .map_or(0, |(shape, _)| shape.nodes);
    match cap {
        Some(cap) if cap > limit => Err(CapacityError::Nodes { max: limit }),
        Some(cap) => Ok(cap),
        None => Ok(limit),
    }
}

/// The most couplers a miner advertises: the couplers of the largest
/// complete Binary model, or 0 when none fits.
#[must_use]
pub fn advertised_couplers(kernel: Kernel, limits: &DeviceLimits) -> usize {
    max_model(
        kernel,
        Connectivity::Complete,
        Precision::Binary,
        limits,
        advertised_reads(kernel),
    )
    .map_or(0, |(shape, _)| shape.edges)
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
        tg_memory_bytes: TG_MEMORY_BYTES,
        max_buffer_bytes: 1 << 40,
    };

    fn unit_shape() -> ModelShape {
        ModelShape {
            nodes: 4_577,
            edges: 41_515,
            max_degree: 20,
            precision: Precision::Binary,
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
            betas: 64,
            reads: 100,
        }
    }

    #[test]
    fn usable_bytes_keeps_one_fifth_headroom() {
        let limits = DeviceLimits {
            memory_bytes: 1000,
            tg_memory_bytes: TG_MEMORY_BYTES,
            max_buffer_bytes: 1000,
        };
        assert_eq!(limits.usable_bytes(), 800);
    }

    #[test]
    fn usable_bytes_does_not_overflow_on_huge_memory() {
        let limits = DeviceLimits {
            memory_bytes: usize::MAX,
            tg_memory_bytes: TG_MEMORY_BYTES,
            max_buffer_bytes: usize::MAX,
        };
        assert!(limits.usable_bytes() > usize::MAX / 5 * 4 - 1);
    }

    #[test]
    fn host_limits_read_half_of_physical_memory() {
        let limits = host_limits();
        assert!(limits.memory_bytes > 0);
        assert_eq!(limits.tg_memory_bytes, 32_768);
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

    #[test]
    fn msa_above_sparse_degree_is_refused() {
        let mut shape = sparse_shape();
        shape.max_degree = MSA_SPARSE_MAX_DEG + 1;
        shape.nodes = 1_000;
        shape.edges = 40_000;
        assert_eq!(
            footprint(Kernel::Msa, &shape, &ROOMY),
            Err(CapacityError::Degree {
                max: MSA_SPARSE_MAX_DEG
            })
        );
        assert_eq!(
            footprint(Kernel::Msa, &complete_shape(), &ROOMY),
            Err(CapacityError::Degree {
                max: MSA_SPARSE_MAX_DEG
            })
        );
        shape.max_degree = MSA_SPARSE_MAX_DEG;
        let fp = footprint(Kernel::Msa, &shape, &ROOMY).unwrap();
        assert_ne!(fp.storage, Storage::Dense);
    }

    #[test]
    fn msa_never_admits_dense_storage() {
        for shape in [unit_shape(), sparse_shape(), complete_shape()] {
            assert!(!admits(Kernel::Msa, Storage::Dense, &shape), "{shape:?}");
        }
        assert!(admits(Kernel::Sa, Storage::Dense, &complete_shape()));
    }

    #[test]
    fn a_model_over_memory_is_refused() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 20,
            tg_memory_bytes: TG_MEMORY_BYTES,
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
            tg_memory_bytes: TG_MEMORY_BYTES,
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

    #[test]
    fn total_adds_shared_bytes_once() {
        let fp = Footprint {
            storage: Storage::Threadgroup,
            per_instance_bytes: 10,
            shared_bytes: 7,
        };
        assert_eq!(fp.total(3), 37);
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
            tg_memory_bytes: TG_MEMORY_BYTES,
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
            betas: 10_000_000,
            reads: 32,
        };
        let limits = DeviceLimits {
            memory_bytes: 1 << 40,
            tg_memory_bytes: TG_MEMORY_BYTES,
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
        // Dense is the cheaper tier (matrix 500,000 bytes against 563,000
        // bytes of CSR), but its matrix is the largest single buffer. A
        // limit between the two largest buffers refuses dense only.
        let shape = ModelShape {
            nodes: 2_000,
            edges: 55_555,
            max_degree: 100,
            precision: Precision::Binary,
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
            tg_memory_bytes: TG_MEMORY_BYTES,
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
        tg_memory_bytes: TG_MEMORY_BYTES,
        max_buffer_bytes: 64 << 30,
    };

    fn shape_for(connectivity: Connectivity, nodes: usize, reads: usize) -> ModelShape {
        let (edges, max_degree) = connectivity.size(nodes);
        ModelShape {
            nodes,
            edges,
            max_degree,
            precision: Precision::Binary,
            betas: 64,
            reads,
        }
    }

    #[test]
    fn max_model_is_the_last_shape_that_fits() {
        for kernel in KERNELS {
            let connectivity = Connectivity::Degree(20);
            let reads = advertised_reads(kernel);
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

    #[test]
    fn a_model_near_both_limits_is_refused() {
        for kernel in KERNELS {
            let nodes = advertised_nodes(kernel, &BIG, None).unwrap();
            let couplers = advertised_couplers(kernel, &BIG);
            let shape = ModelShape {
                nodes,
                edges: couplers,
                // The fewest neighbours the busiest node needs to carry
                // `couplers` edges over `nodes` nodes.
                max_degree: (2 * couplers).div_ceil(nodes).min(nodes - 1),
                precision: Precision::Binary,
                betas: 64,
                reads: advertised_reads(kernel),
            };
            let err = footprint(kernel, &shape, &BIG).unwrap_err();
            assert!(
                matches!(err, CapacityError::Memory { .. }),
                "{kernel:?}: {err:?}"
            );
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

    #[test]
    fn fit_is_monotone_in_the_node_count() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 25,
            tg_memory_bytes: TG_MEMORY_BYTES,
            max_buffer_bytes: 1 << 25,
        };
        for kernel in KERNELS {
            let reads = advertised_reads(kernel);
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
            tg_memory_bytes: TG_MEMORY_BYTES,
            max_buffer_bytes: 16,
        };
        for kernel in KERNELS {
            let reads = advertised_reads(kernel);
            for connectivity in [Connectivity::Degree(20), Connectivity::Complete] {
                assert!(
                    max_model(kernel, connectivity, Precision::Binary, &tiny, reads).is_none(),
                    "{kernel:?} {connectivity:?}"
                );
            }
            assert_eq!(advertised_nodes(kernel, &tiny, None), Ok(0));
            assert_eq!(advertised_couplers(kernel, &tiny), 0);
        }
    }
}
