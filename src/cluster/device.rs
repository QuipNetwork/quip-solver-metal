// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The Otsuka cluster anneal on one Metal device. Component updates are a
//! heuristic. Port of the CUDA `src/cluster/device.rs`; the kernel is
//! `kernels/cluster.metal`.
//!
//! Each call uploads the prepared arrays, runs the sweeps in command buffers
//! that [`crate::sampler::cluster_plan`] sizes, then repeats single greedy
//! passes, one command buffer each, until no spin flips. `stop` is checked
//! before every command buffer.

use super::{PrepareError, PreparedGraph, SpinError};
use crate::capacity::{CapacityError, DeviceLimits, CLUSTER_THREADS};
use crate::metal_device::{compile_precise_pipeline, MetalError};
use metal::{ComputePipelineState, Device, MTLResourceOptions, MTLSize};

const SOURCE: &str = include_str!("../../kernels/cluster.metal");

/// Cluster work per node update, as a multiple of the SA spin-update rate,
/// for chunk sizing. Measured at 4,577 nodes and 1 to 256 anneals on an
/// M4 Max under load in `docs/perf/2026-11-12-otsuka-metal.md`: at 1.0 a
/// chunk ran 3.1 times its plan at 256 anneals. At 7.0 the longest of three
/// runs measured 305 ms.
const CLUSTER_COST_FACTOR: f64 = 7.0;

/// Greedy passes of the final descent at most, as the CUDA mode-1 descent.
const MAX_DESCENT_PASSES: usize = 256;

/// Largest index the kernel computes in `uint`.
const MAX_UINT_INDEX: usize = u32::MAX as usize;

/// A cluster schedule and launch. The research defaults follow
/// `run_cluster.py:39-46`; weights belong to the prepared graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterParams {
    /// Total cluster sweeps, in `1..=i32::MAX`.
    pub sweeps: usize,
    /// Period of colored single-spin sweeps. Zero disables these sweeps.
    pub single_every: usize,
    /// Random groups for parallel component decisions. Must be positive.
    pub rounds: usize,
    /// Positive starting beta.
    pub beta_min: f32,
    /// Optional intermediate beta. Zero disables the knee.
    pub beta_knee: f32,
    /// Fraction of the schedule at the knee, in `[0, 1)`. Zero disables the
    /// knee.
    pub knee_frac: f32,
    /// Positive final beta.
    pub beta_max: f32,
    /// Concurrent anneals, one threadgroup each. Must be positive.
    pub anneals: usize,
    /// Seed of the initial spins and of every thread's random state.
    pub seed: u64,
    /// Plans command buffers at this many updates per second instead of the
    /// measured rate, for tests. The one-sweep ceiling then does not apply.
    #[doc(hidden)]
    pub planning_rate: Option<f64>,
}

impl Default for ClusterParams {
    fn default() -> Self {
        Self {
            sweeps: 65536,
            single_every: 1,
            rounds: 2,
            beta_min: 1.0,
            beta_knee: 0.0,
            knee_frac: 0.0,
            beta_max: 12.0,
            anneals: 1,
            seed: 1,
            planning_rate: None,
        }
    }
}

/// A refusal or failure of a cluster anneal.
#[derive(Debug, thiserror::Error)]
pub enum ClusterDeviceError {
    /// `stop` returned true before a command buffer.
    #[error("cluster job cancelled")]
    Cancelled,
    /// A schedule value is invalid.
    #[error("invalid cluster parameter: {0}")]
    Parameter(&'static str),
    /// Preparation rejected the launch.
    #[error(transparent)]
    Prepare(#[from] PrepareError),
    /// The launch does not fit the device, refused before any allocation.
    #[error(transparent)]
    Capacity(#[from] CapacityError),
    /// One sweep of every anneal is planned past the one-chunk limit.
    #[error("one cluster sweep is planned at {planned_ms:.1} ms, past the {max_ms:.0} ms limit of one command buffer")]
    SweepTooLong {
        /// Planned milliseconds of one sweep.
        planned_ms: f64,
        /// The limit.
        max_ms: f64,
    },
    /// Opening the device or compiling the kernel failed.
    #[error(transparent)]
    Metal(#[from] MetalError),
    /// A command buffer did not complete.
    #[error("cluster command buffer ended with status {0}")]
    CommandBuffer(String),
    /// The device returned a spin that is not a sign.
    #[error(transparent)]
    Spin(#[from] SpinError),
}

/// The kernel's `Params`, passed by value. Seven `int` then six `float`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ClusterArgs {
    n: i32,
    tiles: i32,
    pairs: i32,
    colors: i32,
    sweeps: i32,
    single_every: i32,
    rounds: i32,
    beta_min: f32,
    beta_knee: f32,
    knee_frac: f32,
    beta_max: f32,
    tile_w: f32,
    dimer_w: f32,
}

/// The kernel's `Window`: sweeps `sweep0..sweep0 + nsweeps`, or one greedy
/// pass when `final_pass` is nonzero.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Window {
    sweep0: i32,
    nsweeps: i32,
    final_pass: i32,
}

impl ClusterParams {
    /// Check the schedule and the kernel integer widths.
    ///
    /// # Errors
    /// A parameter error for an invalid count or beta, or
    /// [`PrepareError::EmptyLaunch`] for zero anneals.
    pub fn validate(&self) -> Result<(), ClusterDeviceError> {
        let int =
            |v| i32::try_from(v).map_err(|_| ClusterDeviceError::Parameter("count exceeds i32"));
        if self.anneals == 0 {
            return Err(PrepareError::EmptyLaunch.into());
        }
        if self.sweeps == 0 || self.rounds == 0 {
            return Err(ClusterDeviceError::Parameter(
                "sweeps and rounds must be positive",
            ));
        }
        int(self.sweeps)?;
        int(self.single_every)?;
        int(self.rounds)?;
        if ![self.beta_min, self.beta_knee, self.knee_frac, self.beta_max]
            .iter()
            .all(|v| v.is_finite())
            || self.beta_min <= 0.0
            || self.beta_max <= 0.0
            || self.beta_knee < 0.0
            || !(0.0..1.0).contains(&self.knee_frac)
        {
            return Err(ClusterDeviceError::Parameter(
                "betas must be finite and positive; knee fraction must be in [0, 1)",
            ));
        }
        if self
            .planning_rate
            .is_some_and(|rate| !rate.is_finite() || rate <= 0.0)
        {
            return Err(ClusterDeviceError::Parameter(
                "the planning rate must be finite and positive",
            ));
        }
        Ok(())
    }

    fn args(&self, prepared: &PreparedGraph) -> Result<ClusterArgs, ClusterDeviceError> {
        self.validate()?;
        let int =
            |v| i32::try_from(v).map_err(|_| ClusterDeviceError::Parameter("count exceeds i32"));
        let layout = prepared.layout();
        // The kernel's shared-weight fast path: every prepared tile and pair
        // carries the graph's one weight.
        #[expect(
            clippy::cast_precision_loss,
            reason = "milli weights are at most 1000, exact in f32"
        )]
        let weight = |w: Option<&i32>| w.copied().unwrap_or(1000) as f32;
        Ok(ClusterArgs {
            n: layout.header[0],
            tiles: layout.header[1],
            pairs: layout.header[2],
            colors: layout.header[5],
            sweeps: int(self.sweeps)?,
            single_every: int(self.single_every)?,
            rounds: int(self.rounds)?,
            beta_min: self.beta_min,
            beta_knee: self.beta_knee,
            knee_frac: self.knee_frac,
            beta_max: self.beta_max,
            tile_w: weight(layout.tile_w.first()),
            dimer_w: weight(layout.link_w.first()),
        })
    }
}

/// Timing of one command buffer.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct ChunkTiming {
    /// Cluster sweeps of this command buffer; zero for a greedy pass.
    pub sweeps: usize,
    /// Planned milliseconds: the sweeps at the planned rate, or one sweep
    /// for a greedy pass.
    pub planned_ms: f64,
    /// Measured GPU milliseconds.
    pub gpu_ms: f64,
}

/// The state after an anneal, for tests and measurements.
#[doc(hidden)]
#[derive(Debug)]
pub struct ClusterTrace {
    /// Spins in original node order, one vector per anneal.
    pub spins: Vec<Vec<i8>>,
    /// Both random words of every thread, anneal-major.
    pub rng: Vec<u32>,
    /// Every command buffer, in order.
    pub chunks: Vec<ChunkTiming>,
}

/// The cluster pipeline and a command queue on one Metal device. Opening it
/// compiles only the cluster kernel.
pub struct ClusterDevice {
    device: Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
    limits: DeviceLimits,
}

impl std::fmt::Debug for ClusterDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterDevice")
            .field("device", &self.device.name())
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl ClusterDevice {
    /// Open device `device_index` in `Device::all()` order and compile the
    /// cluster kernel. Memory limits come from the host's unified memory.
    ///
    /// # Errors
    /// [`MetalError::NoDevice`], a compile or pipeline error, or a parameter
    /// error when the pipeline admits fewer than 512 threads.
    pub fn open(device_index: usize) -> Result<Self, ClusterDeviceError> {
        let device = Device::all()
            .into_iter()
            .nth(device_index)
            .ok_or(MetalError::NoDevice(device_index))?;
        let pipeline = compile_precise_pipeline(&device, SOURCE, "cluster_anneal")?;
        if usize::try_from(pipeline.max_total_threads_per_threadgroup())
            .is_ok_and(|max| max < CLUSTER_THREADS)
        {
            return Err(ClusterDeviceError::Parameter(
                "the cluster pipeline admits fewer than 512 threads",
            ));
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipeline,
            limits: *crate::sampler::host_device_limits(),
        })
    }

    /// Anneal `params.anneals` copies of `prepared` and descend each to a
    /// single-flip local minimum. Every call starts from `params.seed`.
    ///
    /// # Errors
    /// Refuses invalid parameters and launches that do not fit before any
    /// allocation. Returns [`ClusterDeviceError::Cancelled`] when `stop`
    /// returns true before a command buffer, and every device failure.
    pub fn anneal(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
        stop: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<i8>>, ClusterDeviceError> {
        self.run(prepared, params, stop, true)
            .map(|trace| trace.spins)
    }

    /// [`Self::anneal`] with the random state and every command buffer's
    /// timing. `final_descent` false stops before the greedy passes.
    ///
    /// # Errors
    /// The errors of [`Self::anneal`].
    #[doc(hidden)]
    pub fn trace_for_test(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
        final_descent: bool,
    ) -> Result<ClusterTrace, ClusterDeviceError> {
        self.run(prepared, params, &|| false, final_descent)
    }

    fn run(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
        stop: &dyn Fn() -> bool,
        final_descent: bool,
    ) -> Result<ClusterTrace, ClusterDeviceError> {
        let args = params.args(prepared)?;
        let n = prepared.original_graph().num_nodes();
        check_capacity(prepared, params.anneals, &self.limits)?;
        let plan = crate::sampler::cluster_plan(
            params.anneals,
            n,
            params.sweeps,
            CLUSTER_COST_FACTOR,
            params.planning_rate,
        );
        if params.planning_rate.is_none() && plan.sweep_ms > crate::sampler::ONE_SWEEP_MAX_MS {
            return Err(ClusterDeviceError::SweepTooLong {
                planned_ms: plan.sweep_ms,
                max_ms: crate::sampler::ONE_SWEEP_MAX_MS,
            });
        }
        if stop() {
            return Err(ClusterDeviceError::Cancelled);
        }
        let buffers = self.upload(prepared, params)?;
        let mut chunks = Vec::with_capacity(plan.windows.len());
        for &(start, count) in &plan.windows {
            if stop() {
                return Err(ClusterDeviceError::Cancelled);
            }
            let window = Window {
                sweep0: window_int(start)?,
                nsweeps: window_int(count)?,
                final_pass: 0,
            };
            let gpu_ms = self.dispatch(&buffers, &args, window, params.anneals)?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "a sweep count is at most i32::MAX"
            )]
            let planned_ms = plan.sweep_ms * count as f64;
            chunks.push(ChunkTiming {
                sweeps: count,
                planned_ms,
                gpu_ms,
            });
        }
        if final_descent {
            for _ in 0..MAX_DESCENT_PASSES {
                if stop() {
                    return Err(ClusterDeviceError::Cancelled);
                }
                write_slice(&buffers.changed, &[0_i32]);
                let window = Window {
                    sweep0: 0,
                    nsweeps: 0,
                    final_pass: 1,
                };
                let gpu_ms = self.dispatch(&buffers, &args, window, params.anneals)?;
                chunks.push(ChunkTiming {
                    sweeps: 0,
                    planned_ms: plan.sweep_ms,
                    gpu_ms,
                });
                // A pass without a flip leaves every later pass unchanged.
                if read_slice::<i32>(&buffers.changed, 1).first() == Some(&0) {
                    break;
                }
            }
        }
        let states = read_slice::<i8>(&buffers.spins, params.anneals * n);
        let spins = states
            .chunks_exact(n)
            .map(|s| prepared.unpermute(s))
            .collect::<Result<_, _>>()?;
        let rng = read_slice::<u32>(&buffers.rng, params.anneals * CLUSTER_THREADS * 2);
        Ok(ClusterTrace { spins, rng, chunks })
    }

    fn upload(
        &self,
        prepared: &PreparedGraph,
        params: &ClusterParams,
    ) -> Result<Buffers, ClusterDeviceError> {
        let d = prepared.layout();
        let n = prepared.original_graph().num_nodes();
        let (spins, rng) = initial_state(prepared, params.anneals, params.seed)?;
        let zeros = |bytes: usize| self.zeroed(bytes);
        Ok(Buffers {
            uploads: [
                self.upload_slice(&d.tiles),
                self.upload_slice(&d.gauges),
                self.upload_slice(&d.pairs),
                self.upload_slice(&d.weak_deg),
                self.upload_slice(&d.weak_ell),
                self.upload_slice(&d.all_deg),
                self.upload_slice(&d.all_ell),
                self.upload_slice(&d.color_off),
                self.upload_slice(&d.h),
                self.upload_slice(&d.tile_w),
                self.upload_slice(&d.link_w),
            ],
            spins: self.upload_slice(&spins),
            rng: self.upload_slice(&rng),
            parent: zeros(params.anneals * n * 4),
            acc: zeros(params.anneals * n * 4),
            changed: zeros(4),
        })
    }

    /// A shared buffer holding `data`, or one zeroed element when it is empty.
    fn upload_slice<T: Copy>(&self, data: &[T]) -> metal::Buffer {
        if data.is_empty() {
            return self.zeroed(std::mem::size_of::<T>());
        }
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn zeroed(&self, bytes: usize) -> metal::Buffer {
        let buffer = self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        // SAFETY: `contents` points at `bytes` writable bytes of a new shared
        // buffer that no command buffer uses yet.
        unsafe { std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes) };
        buffer
    }

    /// Encode, commit, and wait for one command buffer. Returns its GPU
    /// milliseconds.
    fn dispatch(
        &self,
        buffers: &Buffers,
        args: &ClusterArgs,
        window: Window,
        anneals: usize,
    ) -> Result<f64, ClusterDeviceError> {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, buffer) in buffers.uploads.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        set_value(encoder, 11, args);
        set_value(encoder, 12, &window);
        encoder.set_buffer(13, Some(&buffers.spins), 0);
        encoder.set_buffer(14, Some(&buffers.rng), 0);
        encoder.set_buffer(15, Some(&buffers.parent), 0);
        encoder.set_buffer(16, Some(&buffers.acc), 0);
        encoder.set_buffer(17, Some(&buffers.changed), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(anneals as u64, 1, 1),
            MTLSize::new(CLUSTER_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let status = cmd.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(ClusterDeviceError::CommandBuffer(format!("{status:?}")));
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "GPU microseconds of one command buffer are far below 2^52"
        )]
        let gpu_ms = crate::sampler::gpu_time_us(cmd) as f64 / 1000.0;
        Ok(gpu_ms)
    }
}

/// The buffers of one anneal call, bound at the kernel's indices.
struct Buffers {
    /// Buffers 0 through 10: the prepared arrays in `DeviceLayout` order.
    uploads: [metal::Buffer; 11],
    /// Buffer 13: `n` spins per anneal.
    spins: metal::Buffer,
    /// Buffer 14: two words per thread.
    rng: metal::Buffer,
    /// Buffer 15: `n` union-find parents per anneal.
    parent: metal::Buffer,
    /// Buffer 16: `n` component sums per anneal.
    acc: metal::Buffer,
    /// Buffer 17: set when a greedy pass flips a spin.
    changed: metal::Buffer,
}

fn window_int(value: usize) -> Result<i32, ClusterDeviceError> {
    i32::try_from(value).map_err(|_| ClusterDeviceError::Parameter("sweep exceeds i32"))
}

fn set_value<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

fn write_slice<T: Copy>(buffer: &metal::Buffer, values: &[T]) {
    // SAFETY: the buffer is shared storage of at least `values.len()`
    // elements of `T`, and no command buffer uses it now.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast(), values.len());
    }
}

fn read_slice<T: Copy>(buffer: &metal::Buffer, len: usize) -> Vec<T> {
    // SAFETY: the buffer is shared storage of at least `len` initialized
    // elements of `T`, and every command buffer that wrote it has completed.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), len) }.to_vec()
}

/// The bytes of every buffer one call allocates, in allocation order: the
/// eleven uploads (one element at least), then spins, random state,
/// `parent`, `acc`, and the descent flag. Saturates on overflow.
fn buffer_bytes(prepared: &PreparedGraph, anneals: usize) -> [usize; 16] {
    let d = prepared.layout();
    let n = prepared.original_graph().num_nodes();
    let ints = |len: usize| len.max(1).saturating_mul(4);
    let per_anneal = anneals.saturating_mul(n);
    [
        ints(d.tiles.len()),
        d.gauges.len().max(1),
        ints(d.pairs.len()),
        ints(d.weak_deg.len()),
        ints(d.weak_ell.len()),
        ints(d.all_deg.len()),
        ints(d.all_ell.len()),
        ints(d.color_off.len()),
        ints(d.h.len()),
        ints(d.tile_w.len()),
        ints(d.link_w.len()),
        per_anneal,
        anneals.saturating_mul(CLUSTER_THREADS).saturating_mul(8),
        per_anneal.saturating_mul(4),
        per_anneal.saturating_mul(4),
        4,
    ]
}

/// Refuse a launch of `anneals` anneals of `prepared` before any buffer
/// exists: a `uint` index of the kernel past its range, the total past the
/// usable memory, or one buffer past the single-buffer limit.
///
/// The kernel's `uint` indices: each anneal's region starts at
/// `anneal * n` in the spins, `parent`, and `acc`; each thread's random
/// state is `anneal * 512 + thread`; an ELL slot is `slot * n + node`.
/// Preparation bounds the `int` tile, pair, colour, and ELL counts.
fn check_capacity(
    prepared: &PreparedGraph,
    anneals: usize,
    limits: &DeviceLimits,
) -> Result<(), ClusterDeviceError> {
    if anneals == 0 {
        return Err(PrepareError::EmptyLaunch.into());
    }
    let d = prepared.layout();
    let n = prepared.original_graph().num_nodes();
    if let Some(entries) = [
        anneals.saturating_mul(n),
        anneals.saturating_mul(CLUSTER_THREADS),
        d.weak_ell.len(),
        d.all_ell.len(),
    ]
    .into_iter()
    .filter(|&e| e > MAX_UINT_INDEX)
    .max()
    {
        return Err(CapacityError::Index {
            entries,
            max: MAX_UINT_INDEX,
        }
        .into());
    }
    let needed = prepared.footprint().total(anneals, 1);
    if needed > limits.usable_bytes() {
        return Err(CapacityError::Memory {
            needed,
            usable: limits.usable_bytes(),
        }
        .into());
    }
    let largest = buffer_bytes(prepared, anneals)
        .into_iter()
        .max()
        .unwrap_or(0);
    if largest > limits.max_buffer_bytes {
        return Err(CapacityError::Buffer {
            needed: largest,
            max: limits.max_buffer_bytes,
        }
        .into());
    }
    Ok(())
}

/// Initial device-order spins and two random words per thread
/// (`gpucluster.cu:311-319`). Each splitmix64 value, made odd, splits into
/// its low and high words, so the low word, and the state, is never zero.
/// Wrapping arithmetic preserves the unsigned C++ operations.
fn initial_state(
    prepared: &PreparedGraph,
    anneals: usize,
    seed: u64,
) -> Result<(Vec<i8>, Vec<u32>), ClusterDeviceError> {
    let count = |width: usize| {
        anneals
            .checked_mul(width)
            .ok_or(ClusterDeviceError::Parameter("allocation size overflow"))
    };
    let spins = count(prepared.original_graph().num_nodes())?;
    let threads = count(CLUSTER_THREADS)?;
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x0123_4567);
    let mut rng = Vec::with_capacity(threads * 2);
    for _ in 0..threads {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let z = (z ^ (z >> 31)) | 1;
        rng.push(u32::try_from(z & 0xFFFF_FFFF).unwrap_or(1));
        rng.push(u32::try_from(z >> 32).unwrap_or(0));
    }
    let mut y = seed ^ 0xD1B5_4A32_D192_ED03;
    let spins = (0..spins)
        .map(|_| {
            y ^= y << 13;
            y ^= y >> 7;
            y ^= y << 17;
            if (y >> 33) & 1 == 1 {
                1
            } else {
                -1
            }
        })
        .collect();
    Ok((spins, rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{Model, Motifs, Weights};

    fn limits() -> DeviceLimits {
        DeviceLimits {
            memory_bytes: 1 << 30,
            max_buffer_bytes: 1 << 30,
        }
    }

    fn single_node() -> PreparedGraph {
        PreparedGraph::new(
            &Model {
                h_milli: &[0],
                edges: &[],
                j_milli: &[],
            },
            &Motifs {
                tiles: &[],
                gauges: &[],
                pairs: &[],
            },
            Weights::FULL,
            &limits(),
            1,
        )
        .unwrap()
    }

    #[test]
    fn args_match_the_kernel_layout() {
        assert_eq!(std::mem::size_of::<ClusterArgs>(), 52);
        assert_eq!(std::mem::align_of::<ClusterArgs>(), 4);
        assert_eq!(std::mem::offset_of!(ClusterArgs, rounds), 24);
        assert_eq!(std::mem::offset_of!(ClusterArgs, beta_min), 28);
        assert_eq!(std::mem::offset_of!(ClusterArgs, dimer_w), 48);
        assert_eq!(std::mem::size_of::<Window>(), 12);
        assert!(SOURCE.contains("#define CLUSTER_THREADS 512"));
        assert_eq!(CLUSTER_THREADS, 512);
    }

    #[test]
    fn initialization_matches_the_research_seed_split_into_words() {
        let (spins, rng) = initial_state(&single_node(), 2, 1).unwrap();
        assert_eq!(spins, [-1, -1]);
        // The CUDA 64-bit states of seed 1, low word first.
        assert_eq!(
            &rng[..4],
            [0x50dc_981b, 0xc05a_6778, 0x9483_70bd, 0x9e32_cdf7]
        );
        assert_eq!(rng.len(), 2 * 512 * 2);
    }

    #[test]
    fn schedule_defaults_and_validation() {
        let q = ClusterParams::default();
        assert_eq!((q.sweeps, q.single_every, q.rounds), (65536, 1, 2));
        assert_eq!(
            (q.beta_min, q.beta_knee, q.knee_frac, q.beta_max),
            (1.0, 0.0, 0.0, 12.0)
        );
        q.validate().unwrap();
        for bad in [
            ClusterParams {
                sweeps: 0,
                ..q.clone()
            },
            ClusterParams {
                rounds: 0,
                ..q.clone()
            },
            ClusterParams {
                anneals: 0,
                ..q.clone()
            },
            ClusterParams {
                beta_min: f32::NAN,
                ..q.clone()
            },
            ClusterParams {
                beta_max: f32::INFINITY,
                ..q.clone()
            },
            ClusterParams {
                sweeps: usize::MAX,
                ..q.clone()
            },
            ClusterParams {
                single_every: usize::MAX,
                ..q.clone()
            },
            ClusterParams {
                rounds: usize::MAX,
                ..q.clone()
            },
            ClusterParams {
                knee_frac: 1.0,
                ..q.clone()
            },
            ClusterParams {
                planning_rate: Some(0.0),
                ..q.clone()
            },
            ClusterParams { beta_min: 0.0, ..q },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn allocations_match_the_footprint() {
        let p = single_node();
        for anneals in [1, 3, 64] {
            let total: usize = buffer_bytes(&p, anneals).iter().sum();
            assert_eq!(total, p.footprint().total(anneals, 1));
        }
    }

    #[test]
    fn a_buffer_past_the_single_buffer_limit_is_refused() {
        let p = single_node();
        // The random state, 512 threads of 8 bytes per anneal, is the largest.
        let largest = 3 * 512 * 8;
        assert_eq!(buffer_bytes(&p, 3).into_iter().max(), Some(largest));
        let fits = DeviceLimits {
            max_buffer_bytes: largest,
            ..limits()
        };
        check_capacity(&p, 3, &fits).unwrap();
        let over = DeviceLimits {
            max_buffer_bytes: largest - 1,
            ..limits()
        };
        assert!(matches!(
            check_capacity(&p, 3, &over),
            Err(ClusterDeviceError::Capacity(CapacityError::Buffer {
                needed,
                max
            })) if needed == largest && max == largest - 1
        ));
    }

    #[test]
    fn memory_and_index_limits_are_refused() {
        let p = single_node();
        let total = p.footprint().total(2, 1);
        let tight = DeviceLimits {
            memory_bytes: total.div_ceil(4) * 5,
            max_buffer_bytes: 1 << 30,
        };
        check_capacity(&p, 2, &tight).unwrap();
        assert!(matches!(
            check_capacity(&p, 3, &tight),
            Err(ClusterDeviceError::Capacity(CapacityError::Memory { .. }))
        ));
        let anneals = MAX_UINT_INDEX / CLUSTER_THREADS + 1;
        assert!(matches!(
            check_capacity(&p, anneals, &limits()),
            Err(ClusterDeviceError::Capacity(CapacityError::Index { entries, .. }))
                if entries == anneals * CLUSTER_THREADS
        ));
        assert!(matches!(
            check_capacity(&p, 0, &limits()),
            Err(ClusterDeviceError::Prepare(PrepareError::EmptyLaunch))
        ));
    }
}
