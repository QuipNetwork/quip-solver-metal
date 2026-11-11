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
//!   at most [`MSA_MAX_DEG`]. The sizes come from the buffers that
//!   `sampler::encode_batch` allocates.
//! - [`Storage::Spill`]: the threadgroup-tier buffers, plus a device buffer
//!   for the spin state of the nodes past the threadgroup cap. This kernel
//!   does not exist yet; the formulas follow the intended layout and later
//!   tasks correct them.
//! - [`Storage::Dense`]: an `N * N` coupling matrix instead of CSR. Also not
//!   yet implemented.
//!
//! Every formula sits in one small function per kernel, so a later task can
//! correct one without touching the routing.

use crate::sampler::{
    simd_rounded_reads, Kernel, GIBBS_MAX_NODES, MSA_LANES, MSA_MAX_DEG, MSA_MAX_NODES,
    MSA_THREADS, SA_MAX_NODES,
};
use crate::streaming::batch_size_for_reads;

/// Highest degree the sparse multi-spin kernel admits. Above it only
/// [`Storage::Dense`] admits.
pub const MSA_SPARSE_MAX_DEGREE: usize = 62;

/// Largest node count the 32-bit kernel indexing can address.
pub const MAX_NODES: usize = i32::MAX as usize;

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
    /// 32-bit float coefficients.
    Float32,
}

impl Precision {
    /// Bytes per coefficient in the CSR value and field buffers.
    const fn sparse_bytes(self) -> usize {
        match self {
            Self::Binary | Self::Int8 => 1,
            Self::Int32 | Self::Float32 => 4,
        }
    }

    /// Bytes per entry of the `delta_energy` array: one for the narrow
    /// representation, four for the wide one (`new_sa_persistent`).
    const fn delta_bytes(self) -> usize {
        self.sparse_bytes()
    }

    /// Bytes of an `n * n` dense coupling matrix.
    const fn dense_bytes(self, cells: usize) -> usize {
        match self {
            Self::Binary => cells.div_ceil(8),
            Self::Int8 => cells,
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

/// One tier's buffers: sizes of the per-model buffers and the shared bytes.
struct Sizes {
    /// Each per-model buffer, in bytes.
    per_instance: Vec<usize>,
    /// Bytes shared by all models.
    shared: usize,
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
            shape.nodes <= node_cap(kernel)
                && (kernel != Kernel::Msa || shape.max_degree <= MSA_MAX_DEG)
        }
        Storage::Spill => kernel != Kernel::Msa || shape.max_degree <= MSA_SPARSE_MAX_DEGREE,
        Storage::Dense => true,
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
    let width = shape.precision.sparse_bytes();
    [
        shape.nodes.saturating_add(1).saturating_mul(4),
        nnz.saturating_mul(4),
        nnz.saturating_mul(width),
        shape.nodes.saturating_mul(width),
    ]
}

/// Bytes every kernel shares: the beta schedule and the colour-class node
/// list (`encode_batch`, `build_cached_topology`).
fn common_shared(shape: &ModelShape) -> usize {
    shape
        .betas
        .saturating_mul(4)
        .saturating_add(shape.nodes.saturating_mul(4))
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
    match kernel {
        Kernel::Sa => {
            per_instance.extend(csr);
            per_instance.extend(sa_persistent(shape));
        }
        Kernel::Gibbs => {
            per_instance.extend(csr);
            per_instance.extend(gibbs_persistent(shape));
        }
        Kernel::Msa => {
            per_instance.extend(&csr[2..]);
            shared = shared.saturating_add(csr[0]).saturating_add(csr[1]);
            per_instance.extend(msa_persistent(shape));
        }
    }
    Sizes {
        per_instance,
        shared,
    }
}

/// Bytes of the spin state for the `nodes - K` nodes past the threadgroup
/// cap `K`, held in a device buffer by the spill and dense tiers. This is
/// the intended layout of kernels that do not exist yet.
///
/// - MSA: `(nodes - K) * words * 4`, with `words = reads / 32`.
/// - SA: per read, the packed state `ceil((nodes - K) / 8)` plus
///   `delta_energy` at `(nodes - K) * delta_bytes`.
/// - Gibbs: one byte per spin per sample, `reads * (nodes - K)`.
fn spill_state_bytes(kernel: Kernel, shape: &ModelShape) -> usize {
    let extra = shape.nodes.saturating_sub(node_cap(kernel));
    let reads = simd_rounded_reads(shape.reads);
    match kernel {
        Kernel::Msa => extra.saturating_mul(reads / MSA_LANES).saturating_mul(4),
        Kernel::Sa => reads.saturating_mul(
            extra
                .div_ceil(8)
                .saturating_add(extra.saturating_mul(shape.precision.delta_bytes())),
        ),
        Kernel::Gibbs => reads.saturating_mul(extra),
    }
}

/// Spill-tier sizes: the threadgroup-tier buffers plus the spill state.
fn spill_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    let mut sizes = threadgroup_sizes(kernel, shape);
    sizes.per_instance.push(spill_state_bytes(kernel, shape));
    sizes
}

/// Dense-tier sizes: an `N * N` coupling matrix at the precision's width
/// (1 bit, 1, 4, 4 bytes) replaces the CSR structure and couplings. The
/// fields, the outputs, and the spill-style spin state remain.
fn dense_sizes(kernel: Kernel, shape: &ModelShape) -> Sizes {
    let mut per_instance = vec![
        shape
            .precision
            .dense_bytes(shape.nodes.saturating_mul(shape.nodes)),
        shape.nodes.saturating_mul(shape.precision.sparse_bytes()),
        spill_state_bytes(kernel, shape),
    ];
    per_instance.extend(output_buffers(shape));
    Sizes {
        per_instance,
        shared: common_shared(shape),
    }
}

/// Choose the cheapest tier that admits `shape` and check that it fits.
///
/// Ties go to the earlier tier in the order threadgroup, spill, dense.
///
/// # Errors
///
/// - [`CapacityError::Nodes`] when `shape.nodes` exceeds [`MAX_NODES`].
/// - [`CapacityError::Buffer`] when the largest buffer, counted for the
///   models of one in-flight batch (half of [`instances`]), exceeds
///   `limits.max_buffer_bytes`.
/// - [`CapacityError::Memory`] when `total(instances)` exceeds
///   `limits.usable_bytes()`.
pub fn footprint(
    kernel: Kernel,
    shape: &ModelShape,
    limits: &DeviceLimits,
) -> Result<Footprint, CapacityError> {
    if shape.nodes > MAX_NODES {
        return Err(CapacityError::Nodes { max: MAX_NODES });
    }
    let candidates = [
        (Storage::Threadgroup, threadgroup_sizes(kernel, shape)),
        (Storage::Spill, spill_sizes(kernel, shape)),
        (Storage::Dense, dense_sizes(kernel, shape)),
    ];
    let mut best: Option<(Footprint, usize)> = None;
    for (storage, sizes) in candidates {
        if !admits(kernel, storage, shape) {
            continue;
        }
        let per_instance_bytes = sizes
            .per_instance
            .iter()
            .fold(0usize, |sum, &bytes| sum.saturating_add(bytes));
        if best.is_some_and(|(chosen, _)| chosen.per_instance_bytes <= per_instance_bytes) {
            continue;
        }
        let largest = sizes.per_instance.iter().copied().max().unwrap_or(0);
        best = Some((
            Footprint {
                storage,
                per_instance_bytes,
                shared_bytes: sizes.shared,
            },
            largest,
        ));
    }
    let Some((chosen, largest)) = best else {
        return Err(CapacityError::Degree {
            max: MSA_SPARSE_MAX_DEGREE,
        });
    };
    let in_flight = instances(kernel, shape.reads);
    let buffer = largest.saturating_mul((in_flight / 2).max(1));
    if buffer > limits.max_buffer_bytes {
        return Err(CapacityError::Buffer {
            needed: buffer,
            max: limits.max_buffer_bytes,
        });
    }
    let needed = chosen.total(in_flight);
    if needed > limits.usable_bytes() {
        return Err(CapacityError::Memory {
            needed,
            usable: limits.usable_bytes(),
        });
    }
    Ok(chosen)
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
        for kernel in KERNELS {
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
    fn msa_above_sparse_degree_admits_only_dense() {
        let mut shape = sparse_shape();
        shape.max_degree = MSA_SPARSE_MAX_DEGREE + 1;
        shape.nodes = 1_000;
        shape.edges = 40_000;
        let fp = footprint(Kernel::Msa, &shape, &ROOMY).unwrap();
        assert_eq!(fp.storage, Storage::Dense);
        shape.max_degree = MSA_SPARSE_MAX_DEGREE;
        let fp = footprint(Kernel::Msa, &shape, &ROOMY).unwrap();
        assert_ne!(fp.storage, Storage::Dense);
    }

    #[test]
    fn a_model_over_memory_is_refused() {
        let limits = DeviceLimits {
            memory_bytes: 1 << 20,
            tg_memory_bytes: TG_MEMORY_BYTES,
            max_buffer_bytes: usize::MAX,
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
    fn spill_state_counts_only_the_nodes_past_the_cap() {
        let mut shape = unit_shape();
        shape.reads = 64;
        shape.nodes = MSA_MAX_NODES + 10;
        assert_eq!(spill_state_bytes(Kernel::Msa, &shape), 10 * 2 * 4);
        shape.nodes = SA_MAX_NODES;
        assert_eq!(spill_state_bytes(Kernel::Sa, &shape), 0);
        shape.nodes = GIBBS_MAX_NODES + 10;
        assert_eq!(spill_state_bytes(Kernel::Gibbs, &shape), 64 * 10);
    }
}
