// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

use crate::metal_device::MetalDevice;
use crate::sampler::{self, BufferPool, CachedTopology, Kernel, SampleError, MSA_THREADS};
use crate::topology::SelfFeedingTopology;
use crate::topology::{fill_couplings, fill_h_j_matching};
use crate::{IsingGraph, SampleParams, SamplerResult};
use metal::{MTLCommandBufferStatus, MTLSize};
use std::sync::{Arc, OnceLock};

fn decode_pool() -> Result<&'static rayon::ThreadPool, SampleError> {
    static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
    // Bound decode fan-out independently of the global Rayon pool. Four
    // decode workers plus four preparers leave four of the M4 Max's twelve
    // performance cores for the runner, producers and other work. Byte-table
    // expansion reduces the work on the slowest decode worker. Streaming
    // keeps its existing pool and unpacker.
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|index| format!("resident-decode-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
    .as_ref()
    .map_err(|error| SampleError::Driver(format!("start resident decode pool: {error}")))
}

fn unpack_slot_spins(packed: &[i8], n: usize) -> Vec<i8> {
    const SPINS: [[i8; 8]; 256] = {
        let mut table = [[1; 8]; 256];
        let mut byte = 0;
        while byte < 256 {
            let mut bit = 0;
            while bit < 8 {
                if byte & (1 << bit) != 0 {
                    table[byte][bit] = -1;
                }
                bit += 1;
            }
            byte += 1;
        }
        table
    };
    let mut spins = vec![0; n];
    let (chunks, tail) = spins.as_chunks_mut::<8>();
    for (chunk, &byte) in chunks.iter_mut().zip(packed) {
        *chunk = SPINS[byte as u8 as usize];
    }
    if !tail.is_empty() {
        tail.copy_from_slice(&SPINS[packed[n / 8] as u8 as usize][..n % 8]);
    }
    spins
}

pub(crate) struct SlotJob {
    pub(crate) graph: IsingGraph,
    /// Shared across jobs with the same schedule, so a slot that already
    /// holds it skips the upload.
    pub(crate) schedule: Arc<[f32]>,
    pub(crate) checkpoints: Vec<usize>,
    /// Index of the first leg that starts a fresh anneal. Earlier legs
    /// continue the previous leg's spins.
    pub(crate) fresh_from: usize,
    pub(crate) seed: u64,
    /// Whether this job gets observe-only readback points between its
    /// cascade checkpoints (see [`OBSERVE_INTERVAL`]). Every salt sets this,
    /// even one with no target yet: the lease's live target can appear or
    /// change at any point while the unit is running, and an observe point
    /// checks the *current* target, not the one at admission. Stream jobs,
    /// which never carry a target, leave it false so no extra output steps
    /// are ever built for them.
    pub(crate) observe: bool,
}

impl SlotJob {
    #[cfg(test)]
    fn validate(&self, sched_stride: usize) -> Result<(), SampleError> {
        if self.schedule.len() > sched_stride {
            return Err(SampleError::TooLarge("schedule exceeds slot stride".into()));
        }
        if self.graph.j.len() != self.graph.edges.len()
            || !sampler::device_energy_exact(&self.graph)
        {
            return Err(SampleError::TooLarge(
                "slot jobs require exact device-energy coefficients".into(),
            ));
        }
        validate_schedule(&self.schedule, &self.checkpoints)
    }
}

pub(crate) fn validate_schedule(
    schedule: &[f32],
    checkpoints: &[usize],
) -> Result<(), SampleError> {
    if schedule.is_empty()
        || schedule.iter().any(|b| !b.is_finite() || *b < 0.0)
        || checkpoints.first().is_none_or(|&p| p == 0)
        || checkpoints.last() != Some(&schedule.len())
        || checkpoints.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(SampleError::Driver(
            "invalid slot schedule or checkpoints".into(),
        ));
    }
    Ok(())
}

/// A shared edge list. Equal storage identifies a topology without comparing
/// its edges.
pub(crate) type Edges = Arc<[(usize, usize)]>;

/// Host allocations only. The coefficients belong to the job they were
/// prepared from; admission pairs them with that job's graph.
pub(crate) struct PreparedInputs {
    nodes: usize,
    /// The edge list the job's edges were verified against.
    edges: Edges,
    couplings: Vec<i8>,
    fields: Vec<i8>,
}

impl PreparedInputs {
    pub(crate) fn new(
        graph: &IsingGraph,
        topology: &SelfFeedingTopology,
        edges: &Edges,
    ) -> Option<Self> {
        let (couplings, fields) = fill_h_j_matching(topology, edges, graph)?;
        Some(Self {
            nodes: graph.num_nodes(),
            edges: Arc::clone(edges),
            couplings,
            fields,
        })
    }

    /// Inputs from a lease draw in whole device units. The caller verified
    /// once that `edges` is the lease topology and that it meets
    /// [`fill_couplings`]'s precondition.
    pub(crate) fn from_units(
        topology: &SelfFeedingTopology,
        edges: &Edges,
        fields: Vec<i8>,
        j_units: &[i8],
    ) -> Self {
        Self {
            nodes: fields.len(),
            edges: Arc::clone(edges),
            couplings: fill_couplings(topology, j_units),
            fields,
        }
    }

    pub(crate) fn edges(&self) -> &Edges {
        &self.edges
    }

    #[cfg(test)]
    pub(crate) fn coefficients(&self) -> (&[i8], &[i8]) {
        (&self.couplings, &self.fields)
    }
}

pub(crate) type SlotId = usize;

fn device_time_share(total: u64, count: usize, index: usize) -> u64 {
    total / count as u64 + u64::from((index as u64) < total % count as u64)
}

fn checked_read_pointer<T>(
    contents: *mut std::ffi::c_void,
    length: u64,
    count: usize,
) -> Result<*const T, SampleError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| SampleError::Driver("slot read size overflow".into()))?;
    if contents.is_null() || length < bytes as u64 {
        return Err(SampleError::Driver(
            "slot read buffer is null or too short".into(),
        ));
    }
    Ok(contents.cast())
}

/// One completed step command: its absolute GPU interval, in seconds, and
/// how many slots it advanced.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepSpan {
    /// Host time the command was committed, on the GPU timestamps' clock.
    pub(crate) committed: f64,
    pub(crate) start: f64,
    pub(crate) end: f64,
    pub(crate) slots: usize,
}

pub(crate) struct Checkpoint {
    pub(crate) slot: SlotId,
    pub(crate) index: usize,
    pub(crate) last: bool,
    pub(crate) best: i64,
    /// An observe-only readback: `index`/`last` describe the *next* real
    /// checkpoint and must not be treated as one. No gate, no stage advance,
    /// no controller stats.
    pub(crate) observe: bool,
    /// Sweeps of the current anneal completed as of this readback.
    pub(crate) position: usize,
}

struct ResidentJob {
    job: SlotJob,
    position: usize,
    next_checkpoint: usize,
    has_output: bool,
    device_us: u64,
}

impl ResidentJob {
    /// Schedule position where leg `leg` starts: each checkpoint ends one leg.
    fn leg_start(&self, leg: usize) -> usize {
        leg.checked_sub(1)
            .map_or(0, |prev| self.job.checkpoints[prev])
    }

    /// Schedule position where the anneal that leg `leg` belongs to started.
    fn anneal_start(&self, leg: usize) -> usize {
        if leg >= self.job.fresh_from {
            self.leg_start(leg)
        } else {
            0
        }
    }
}

pub(crate) struct SlotPool {
    decode: &'static rayon::ThreadPool,
    cached: Arc<CachedTopology>,
    pool: Arc<BufferPool>,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    // Buffer indices match the slot kernel ABI. Storage is rented only in new.
    buffers: Vec<(u64, metal::Buffer)>,
    slots: Vec<Option<ResidentJob>>,
    /// The schedule each slot's region of buffer 9 holds.
    held: Vec<Option<Arc<[f32]>>>,
    /// Edge storage already found equal to this pool's topology.
    verified: Option<Edges>,
    steps: Vec<SlotStep>,
    command: Option<metal::CommandBuffer>,
    /// The span of the last command `take_checkpoints` retired.
    span: Option<StepSpan>,
    /// Host time of the last commit, for [`StepSpan::committed`].
    committed: f64,
    faulted: bool,
    num_reads: usize,
    words: usize,
    threads: usize,
    sched_stride: usize,
}

impl SlotPool {
    pub(crate) fn new(
        device: &MetalDevice,
        graph: &IsingGraph,
        num_reads: usize,
        capacity: usize,
        sched_stride: usize,
    ) -> Result<Self, SampleError> {
        let decode = decode_pool()?;
        // Graph checks only: a schedule of fresh legs is longer than any one
        // anneal, and the 32-bit region checks below bound `sched_stride`.
        let params = SampleParams {
            num_reads,
            num_sweeps: 1,
            ..Default::default()
        };
        sampler::validate_batch(&[graph], &params, Kernel::Msa)?;
        if graph.num_nodes() == 0 || num_reads == 0 || capacity == 0 || sched_stride == 0 {
            return Err(SampleError::TooLarge(
                "slot pool dimensions must be nonzero".into(),
            ));
        }
        let state_bytes = (graph.num_nodes() * 4).div_ceil(16) * 16;
        let need = state_bytes + sampler::MSA_STATIC_TG_BYTES;
        if need > device.device.max_threadgroup_memory_length() as usize {
            return Err(SampleError::TooLarge(format!(
                "slot pool needs {need} B of threadgroup memory"
            )));
        }
        let cached = device.topology_cache.get_or_build(device, graph, true);
        let words = num_reads.div_ceil(sampler::MSA_LANES);
        let threads = MSA_THREADS
            .min(device.msa_slots.max_total_threads_per_threadgroup() as usize)
            .max(1);
        // Bound every dimension and offset used by the kernel's 32-bit ABI.
        let region = |factors: &[usize]| -> Result<u64, SampleError> {
            let bytes = factors
                .iter()
                .try_fold(capacity, |n, &v| n.checked_mul(v))
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| {
                    SampleError::TooLarge("slot buffer exceeds 32-bit indexing".into())
                })?;
            Ok(bytes.max(4) as u64)
        };
        let sizes = [
            (2, region(&[cached.topo.nnz])?),
            (9, region(&[sched_stride, 4])?),
            (10, region(&[num_reads, cached.n.div_ceil(8)])?),
            (11, region(&[num_reads, 4])?),
            (15, region(&[cached.n])?),
            (23, region(&[words, cached.n, 4])?),
            (24, region(&[words, threads, 4, 4])?),
            (25, region(&[std::mem::size_of::<SlotStep>()])?),
        ];
        let buffers = sizes
            .into_iter()
            .map(|(index, bytes)| (index, device.buffer_pool.take(&device.device, bytes)))
            .collect();
        Ok(Self {
            decode,
            cached,
            pool: Arc::clone(&device.buffer_pool),
            queue: device.queue.clone(),
            pipeline: device.msa_slots.clone(),
            buffers,
            slots: (0..capacity).map(|_| None).collect(),
            held: (0..capacity).map(|_| None).collect(),
            verified: None,
            steps: Vec::with_capacity(capacity),
            command: None,
            span: None,
            committed: 0.0,
            faulted: false,
            num_reads,
            words,
            threads,
            sched_stride,
        })
    }

    /// Whether inputs prepared against `edges` fit this pool. Admission checks
    /// this itself, so callers targeting this pool need not call it first.
    /// Storage already verified skips the edge compare, which reads the whole
    /// edge list and costs more than the rest of admission.
    pub(crate) fn matches_prepared(
        &mut self,
        edges: &Edges,
        nodes: usize,
        num_reads: usize,
    ) -> bool {
        if self.cached.n != nodes || self.num_reads != num_reads {
            return false;
        }
        if self
            .verified
            .as_ref()
            .is_some_and(|v| Arc::ptr_eq(v, edges))
        {
            return true;
        }
        let same = self.cached.edges[..] == edges[..];
        if same {
            self.verified = Some(Arc::clone(edges));
        }
        same
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }
    /// The span of the last retired step command, once.
    pub(crate) fn take_span(&mut self) -> Option<StepSpan> {
        self.span.take()
    }
    pub(crate) fn live(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// A submitted command stays in flight until `take_checkpoints` retires it.
    pub(crate) fn in_flight(&self) -> bool {
        self.command.is_some()
    }

    fn idle(&self) -> Result<(), SampleError> {
        if self.in_flight() || self.faulted {
            return Err(SampleError::Driver(
                "slot pool is in flight or faulted".into(),
            ));
        }
        Ok(())
    }

    fn buffer(&self, index: u64) -> Result<&metal::BufferRef, SampleError> {
        self.buffers
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| b.as_ref())
            .ok_or_else(|| SampleError::Driver(format!("missing slot buffer {index}")))
    }

    fn read_pointer<T>(&self, index: u64, count: usize) -> Result<*const T, SampleError> {
        let buffer = self.buffer(index)?;
        checked_read_pointer(buffer.contents(), buffer.length(), count)
    }

    fn write<T: Copy>(&self, index: u64, offset: usize, values: &[T]) -> Result<(), SampleError> {
        let buf = self.buffer(index)?;
        let bytes = std::mem::size_of_val(values);
        let offset = offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| SampleError::Driver("slot write offset overflow".into()))?;
        let contents = buf.contents().cast::<u8>();
        if contents.is_null()
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > buf.length() as usize)
        {
            return Err(SampleError::Driver(
                "slot buffer write exceeds storage".into(),
            ));
        }
        // SAFETY: the caller holds the idle pool exclusively. The checked range
        // lies in shared storage and values is a separate host allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                contents.add(offset),
                bytes,
            );
        }
        Ok(())
    }

    /// Validate topology exactly during coefficient construction. Callers can
    /// admit directly without a prior matches query or a topology hash.
    #[cfg(test)]
    pub(crate) fn admit(&mut self, job: SlotJob) -> Result<SlotId, SampleError> {
        self.idle()?;
        job.validate(self.sched_stride)?;
        let (couplings, fields) =
            fill_h_j_matching(&self.cached.topo, &self.cached.edges, &job.graph)
                .ok_or_else(|| SampleError::Driver("job topology differs from slot pool".into()))?;
        self.upload(job, &couplings, &fields)
    }

    /// Validation and quantization have already run on a preparation worker.
    /// `graph` is the graph of the job `inputs` were prepared from.
    pub(crate) fn admit_prepared(
        &mut self,
        inputs: PreparedInputs,
        graph: IsingGraph,
        schedule: crate::cascade::PreparedSchedule,
        seed: u64,
        observe: bool,
    ) -> Result<SlotId, SampleError> {
        self.idle()?;
        let crate::cascade::PreparedSchedule {
            betas: schedule,
            checkpoints,
            fresh_from,
            ..
        } = schedule;
        if schedule.len() > self.sched_stride {
            return Err(SampleError::TooLarge("schedule exceeds slot stride".into()));
        }
        if graph.num_nodes() != inputs.nodes
            || graph.edges.len() != inputs.edges.len()
            || !self.matches_prepared(&inputs.edges, inputs.nodes, self.num_reads)
        {
            return Err(SampleError::Driver(
                "job topology differs from slot pool".into(),
            ));
        }
        let job = SlotJob {
            graph,
            schedule,
            checkpoints,
            fresh_from,
            seed,
            observe,
        };
        self.upload(job, &inputs.couplings, &inputs.fields)
    }

    fn upload(
        &mut self,
        job: SlotJob,
        couplings: &[i8],
        fields: &[i8],
    ) -> Result<SlotId, SampleError> {
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| SampleError::TooLarge("slot pool is full".into()))?;
        self.write(2, slot * self.cached.topo.nnz, couplings)?;
        self.write(15, slot * self.cached.n, fields)?;
        let held = self.held[slot]
            .as_ref()
            .is_some_and(|held| Arc::ptr_eq(held, &job.schedule));
        if !held {
            self.write(9, slot * self.sched_stride, &job.schedule)?;
            self.held[slot] = Some(Arc::clone(&job.schedule));
        }
        self.slots[slot] = Some(ResidentJob {
            job,
            position: 0,
            next_checkpoint: 0,
            has_output: false,
            device_us: 0,
        });
        Ok(slot)
    }

    pub(crate) fn release(&mut self, slot: SlotId) -> Result<(), SampleError> {
        self.idle()?;
        let entry = self
            .slots
            .get_mut(slot)
            .filter(|s| s.is_some())
            .ok_or_else(|| SampleError::Driver("slot is not live".into()))?;
        *entry = None;
        Ok(())
    }

    /// Advance every live job, stopping each at its next checkpoint. Release
    /// jobs at their final checkpoint before submitting another step. A job
    /// in a fresh leg (see [`SlotJob::fresh_from`]) advances up to
    /// `deep_slice` sweeps, every other job up to `slice`.
    pub(crate) fn commit_step(
        &mut self,
        slice: usize,
        deep_slice: usize,
    ) -> Result<bool, SampleError> {
        self.idle()?;
        if self.live() == 0 {
            return Ok(false);
        }
        if slice == 0 || deep_slice == 0 {
            return Err(SampleError::Driver("slot slice must be nonzero".into()));
        }
        let groups = self.live() * self.words;
        let slice = sampler::msa_step_limit(self.cached.n, groups, slice);
        let deep_slice = sampler::msa_step_limit(self.cached.n, groups, deep_slice);
        self.steps.clear();
        for (slot, resident) in self.slots.iter().enumerate() {
            let Some(r) = resident else {
                continue;
            };
            let Some(&checkpoint) = r.job.checkpoints.get(r.next_checkpoint) else {
                return Err(SampleError::Driver(
                    "release final-checkpoint slots before stepping".into(),
                ));
            };
            // An observe job also stops no later than the next multiple of
            // OBSERVE_INTERVAL, so its best energy reaches the host at least
            // that often even when the next real checkpoint is far away.
            // This only ever shrinks the step; it never changes which betas
            // run or their order, so the anneal itself is unaffected.
            let leg = r.next_checkpoint;
            let leg_start = r.leg_start(leg);
            let checkpoint_room = checkpoint - r.position;
            let mut room = checkpoint_room;
            let mut observe_stop = false;
            if r.job.observe {
                let next_observe = (r.position / OBSERVE_INTERVAL + 1) * OBSERVE_INTERVAL;
                let observe_room = next_observe - r.position;
                if observe_room < room {
                    room = observe_room;
                    observe_stop = true;
                }
            }
            let limit = if leg >= r.job.fresh_from {
                deep_slice
            } else {
                slice
            };
            let count = limit.min(room);
            let mut flags = if count == checkpoint_room {
                SLOT_WRITE_OUTPUT
            } else if observe_stop && count == room {
                SLOT_WRITE_OUTPUT | SLOT_OBSERVE
            } else {
                0
            };
            if leg > 0 && leg >= r.job.fresh_from && r.position == leg_start {
                flags |= SLOT_RESTART;
            }
            self.steps.push(SlotStep {
                slot: slot as u32,
                beta_start: r.position as i32,
                beta_count: count as i32,
                num_betas: r.job.schedule.len() as i32,
                seed: slot_seed(leg_seed(r.job.seed, leg)),
                flags,
            });
        }
        self.write(25, 0, &self.steps)?;
        let command = self.queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.cached.row), 0);
        encoder.set_buffer(1, Some(&self.cached.col), 0);
        for (index, buffer) in &self.buffers {
            encoder.set_buffer(*index, Some(buffer), 0);
        }
        for (i, buffer) in self.cached.colors.iter().enumerate() {
            encoder.set_buffer(16 + i as u64, Some(buffer), 0);
        }
        for (index, value) in [
            (3, 0),
            (4, self.cached.topo.nnz as i32),
            (5, self.cached.n as i32),
            (6, self.sched_stride as i32),
            (7, 1),
            (8, 0),
            (12, (self.steps.len() * self.words) as i32),
            (13, self.capacity() as i32),
            (14, self.num_reads as i32),
            (19, self.words as i32),
            (20, self.cached.topo.colors.num_colors),
            (21, 0),
            (22, 0),
        ] {
            encoder.set_bytes(index, 4, (&value as *const i32).cast());
        }
        encoder.set_threadgroup_memory_length(0, ((self.cached.n * 4).div_ceil(16) * 16) as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new((self.steps.len() * self.words) as u64, 1, 1),
            MTLSize::new(self.threads as u64, 1, 1),
        );
        encoder.end_encoding();
        self.committed = sampler::host_seconds();
        command.commit();
        self.command = Some(command);
        Ok(true)
    }

    pub(crate) fn wait(&self) {
        if let Some(command) = &self.command {
            command.wait_until_completed();
        }
    }

    pub(crate) fn take_checkpoints(&mut self) -> Result<Vec<Checkpoint>, SampleError> {
        let Some(command) = &self.command else {
            self.idle()?;
            return Ok(Vec::new());
        };
        let status = command.status();
        if status == MTLCommandBufferStatus::Error {
            self.faulted = true;
            self.command = None;
            return Err(SampleError::Driver("slot command buffer failed".into()));
        }
        if status != MTLCommandBufferStatus::Completed {
            return Err(SampleError::Driver(
                "slot command buffer has not completed".into(),
            ));
        }
        let device_us = sampler::gpu_time_us(command);
        self.span = sampler::gpu_span(command).map(|(start, end)| StepSpan {
            committed: self.committed,
            start,
            end,
            slots: self.steps.len(),
        });
        let energies = self.read_pointer::<i32>(11, self.capacity() * self.num_reads)?;
        self.command = None;
        let output_count = self
            .steps
            .iter()
            .filter(|step| step.flags & SLOT_WRITE_OUTPUT != 0)
            .count();
        let mut checkpoints = Vec::with_capacity(output_count);
        for (step_index, step) in self.steps.iter().enumerate() {
            let slot = step.slot as usize;
            let Some(r) = self.slots[slot].as_mut() else {
                continue;
            };
            r.position += step.beta_count as usize;
            // Attribute equal shares, distributing the integer remainder so
            // per-job totals reconcile exactly with command-buffer time.
            r.device_us = r.device_us.saturating_add(device_time_share(
                device_us,
                self.steps.len(),
                step_index,
            ));
            if step.flags & SLOT_WRITE_OUTPUT != 0 {
                // SAFETY: the completed output step initialized this live
                // slot's region. read_pointer checked the mapping and byte
                // length. No host mutation or GPU write overlaps this copy.
                let slot_energies = unsafe {
                    std::slice::from_raw_parts(energies.add(slot * self.num_reads), self.num_reads)
                }
                .to_vec();
                let best = slot_energies
                    .iter()
                    .copied()
                    .min()
                    .map(i64::from)
                    .ok_or_else(|| SampleError::Driver("slot has no energies".into()))?;
                let observe = step.flags & SLOT_OBSERVE != 0;
                // Either kind of output write leaves the buffers valid for
                // reads_many. An observe readback does not advance
                // next_checkpoint: it is not a gate, so the next real
                // checkpoint stays exactly where the cascade schedule put
                // it, and `last` is always false.
                r.has_output = true;
                let (index, last) = if observe {
                    (r.next_checkpoint, false)
                } else {
                    let index = r.next_checkpoint;
                    r.next_checkpoint += 1;
                    (index, r.next_checkpoint == r.job.checkpoints.len())
                };
                checkpoints.push(Checkpoint {
                    slot,
                    index,
                    last,
                    best,
                    observe,
                    position: r.position - r.anneal_start(index),
                });
            }
        }
        Ok(checkpoints)
    }

    /// Read the latest checkpoint, which remains valid between output steps.
    #[cfg(test)]
    pub(crate) fn reads(
        &self,
        slot: SlotId,
        num_reads: usize,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.reads_many(&[slot], num_reads)?
            .pop()
            .ok_or_else(|| SampleError::Driver("missing slot read result".into()))
    }

    /// Harvest checkpoint outputs in requested slot order with one dense
    /// decode across jobs. Repeated slots produce repeated results.
    pub(crate) fn reads_many(
        &self,
        slots: &[SlotId],
        num_reads: usize,
    ) -> Result<Vec<Vec<SamplerResult>>, SampleError> {
        self.idle()?;
        if num_reads > self.num_reads {
            return Err(SampleError::TooLarge(
                "read count exceeds slot capacity".into(),
            ));
        }
        let graphs: Vec<_> = slots
            .iter()
            .map(|&slot| {
                self.slots
                    .get(slot)
                    .and_then(Option::as_ref)
                    .filter(|r| r.has_output)
                    .map(|r| &r.job.graph)
                    .ok_or_else(|| SampleError::Driver("slot has no checkpoint output".into()))
            })
            .collect::<Result<_, _>>()?;
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let packed_size = self.cached.n.div_ceil(8);
        let count = slots
            .len()
            .checked_mul(num_reads)
            .filter(|&count| {
                count
                    .checked_mul(packed_size.max(4))
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| {
                SampleError::TooLarge("slot read output exceeds host capacity".into())
            })?;
        let mut packed = Vec::with_capacity(count * packed_size);
        let mut energies = Vec::with_capacity(count);
        let sample_ptr =
            self.read_pointer::<i8>(10, self.capacity() * self.num_reads * packed_size)?;
        let energy_ptr = self.read_pointer::<i32>(11, self.capacity() * self.num_reads)?;
        let mut first = 0;
        while first < slots.len() {
            let mut end = first + 1;
            // Full-read contiguous slots use one bulk copy per output buffer.
            // Sparse slots and read prefixes are gathered into the same dense layout.
            if num_reads == self.num_reads {
                while end < slots.len() && slots[end] == slots[end - 1] + 1 {
                    end += 1;
                }
            }
            let read_count = (end - first) * num_reads;
            let read_offset = slots[first] * self.num_reads;
            // SAFETY: all requested slots have completed checkpoint output and
            // idle() excludes GPU writes. Only validated slot regions are read,
            // including when uninitialized or released slots lie between them.
            // read_pointer checked non-null mappings and full buffer lengths.
            unsafe {
                packed.extend_from_slice(std::slice::from_raw_parts(
                    sample_ptr.add(read_offset * packed_size),
                    read_count * packed_size,
                ));
                energies.extend_from_slice(std::slice::from_raw_parts(
                    energy_ptr.add(read_offset),
                    read_count,
                ));
            }
            first = end;
        }
        let n = self.cached.n;
        // Byte expansion avoids per-spin shifts. Keep the same energy audit.
        // Bound both ends of leaf size: a minimum alone can leave 160 reads
        // on one worker, whose remaining sequential work cannot be stolen.
        let decoded = self.decode.install(|| {
            use rayon::prelude::*;
            (0..count)
                .into_par_iter()
                .with_min_len(16)
                .with_max_len(32)
                .map(|index| {
                    let start = index * packed_size;
                    SamplerResult {
                        spins: unpack_slot_spins(&packed[start..start + packed_size], n),
                        energy_milli: i64::from(energies[index]),
                    }
                })
                .collect::<Vec<_>>()
        });
        let mut decoded = decoded.into_iter();
        let reads: Vec<Vec<_>> = graphs
            .iter()
            .map(|_| decoded.by_ref().take(num_reads).collect())
            .collect();
        sampler::audit_device_energies(&reads, &graphs)?;
        Ok(reads)
    }

    pub(crate) fn device_us(&self, slot: SlotId) -> u64 {
        self.slots
            .get(slot)
            .and_then(Option::as_ref)
            .map_or(0, |r| r.device_us)
    }

    /// Cumulative sweeps completed by a live job. Lets a test drive a job up
    /// to an exact observe or checkpoint boundary without guessing how many
    /// `commit_step` calls the device's own dispatch budget will need.
    #[cfg(test)]
    pub(crate) fn position(&self, slot: SlotId) -> usize {
        self.slots[slot].as_ref().expect("slot is live").position
    }
}

impl Drop for SlotPool {
    fn drop(&mut self) {
        self.wait();
        for (_, buffer) in self.buffers.drain(..) {
            self.pool.give(buffer);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotStep {
    pub(crate) slot: u32,
    pub(crate) beta_start: i32,
    pub(crate) beta_count: i32,
    pub(crate) num_betas: i32,
    pub(crate) seed: u32,
    pub(crate) flags: u32,
}

pub(crate) const SLOT_WRITE_OUTPUT: u32 = 1;
/// Set together with [`SLOT_WRITE_OUTPUT`] on a step that stopped at an
/// observe boundary rather than a real cascade checkpoint. The device
/// kernel ignores this bit; it is a host-side annotation read back in
/// [`SlotPool::take_checkpoints`].
pub(crate) const SLOT_OBSERVE: u32 = 2;
/// Set on the first step of every fresh leg after the first (see
/// [`SlotJob::fresh_from`]). The device draws new spins and a new RNG stream
/// from the step's seed, as at position 0, instead of continuing the previous
/// leg's spins.
pub(crate) const SLOT_RESTART: u32 = 4;
/// Longest gap, in sweeps, between output write-backs for a job admitted
/// with `SlotJob::observe` set. Deep cascade stages can otherwise go up
/// to `CHAIN_GATES.full_sweeps` sweeps between real checkpoints; this
/// bounds how stale a live unit's best energy and live-target check can be.
pub(crate) const OBSERVE_INTERVAL: usize = 65_536;

pub(crate) fn slot_seed(seed: u64) -> u32 {
    ((seed ^ (seed >> 32)) as u32).max(1)
}

/// The job seed of leg `leg`: the job's own seed for the first leg, and an
/// independent `SplitMix64` draw for each later one.
pub(crate) fn leg_seed(seed: u64, leg: usize) -> u64 {
    if leg == 0 {
        return seed;
    }
    let mut z = seed ^ (leg as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SlotJob::fresh_from` for a job whose legs all continue one anneal.
    const CONTINUING: usize = usize::MAX;

    #[test]
    fn slot_unpack_matches_every_byte_and_partial_tail() {
        for byte in 0..=255u8 {
            for n in 0usize..=33 {
                let packed = vec![byte as i8; n.div_ceil(8)];
                assert_eq!(
                    unpack_slot_spins(&packed, n),
                    sampler::unpack_spins(&packed, n)
                );
            }
        }
    }

    #[test]
    fn device_time_shares_conserve_command_time() {
        for count in [1, 2, 20, 40] {
            for total in [0, 1, 19, 20, 21, 1250243] {
                let shares: Vec<_> = (0..count)
                    .map(|index| device_time_share(total, count, index))
                    .collect();
                assert_eq!(shares.iter().sum::<u64>(), total);
                assert!(shares.iter().max().unwrap() - shares.iter().min().unwrap() <= 1);
            }
        }
    }

    #[test]
    fn prepared_coefficients_match_slot_quantization() {
        let graph = advantage2_system1(7);
        let topology = SelfFeedingTopology::build_with_advantage2_coloring(&graph);
        let host_topology = SelfFeedingTopology::build(&graph);
        let edges = Arc::clone(&graph.edges);
        let inputs = PreparedInputs::new(&graph, &host_topology, &edges).unwrap();
        let (couplings, fields) = fill_h_j_matching(&topology, &graph.edges, &graph).unwrap();
        assert_eq!(inputs.couplings, couplings);
        assert_eq!(inputs.fields, fields);
        let mut other = graph;
        let mut swapped = other.edges.to_vec();
        swapped.swap(0, 1);
        other.edges = swapped.into();
        assert!(PreparedInputs::new(&other, &topology, &edges).is_none());
    }
    use crate::metal_device::MetalDevice;
    use crate::sampler::{build_beta_schedule, energy_milli, unpack_spins, MSA_THREADS};
    use crate::topology::{fill_h_j, SelfFeedingTopology};
    use crate::IsingGraph;
    use metal::{MTLCommandBufferStatus, MTLSize};

    const READS: usize = 64;
    const WORDS: usize = 2;

    #[test]
    fn checked_read_pointer_rejects_null_short_and_overflowing_buffers() {
        let mut values = [1i32, 2];
        let ptr = values.as_mut_ptr().cast();
        assert_eq!(
            checked_read_pointer::<i32>(ptr, 8, 2).unwrap(),
            values.as_ptr()
        );
        checked_read_pointer::<i32>(std::ptr::null_mut(), 8, 2).unwrap_err();
        checked_read_pointer::<i32>(ptr, 7, 2).unwrap_err();
        checked_read_pointer::<i32>(ptr, u64::MAX, usize::MAX).unwrap_err();
        checked_read_pointer::<i8>(ptr, u64::MAX, isize::MAX as usize + 1).unwrap_err();
    }

    fn job(seed: u64, sweeps: usize, checkpoints: Vec<usize>) -> SlotJob {
        let graph = advantage2_system1(seed);
        let schedule = build_beta_schedule(
            &graph,
            sweeps,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        SlotJob {
            graph,
            schedule: schedule.into(),
            checkpoints,
            fresh_from: CONTINUING,
            seed,
            observe: false,
        }
    }

    fn observing_job(seed: u64, sweeps: usize, checkpoints: Vec<usize>) -> SlotJob {
        SlotJob {
            observe: true,
            ..job(seed, sweeps, checkpoints)
        }
    }

    fn finish_step(pool: &mut SlotPool, slice: usize) -> Vec<Checkpoint> {
        assert!(pool.commit_step(slice, slice).unwrap());
        pool.wait();
        pool.take_checkpoints().unwrap()
    }

    #[test]
    fn pool_matches_the_raw_dispatch() {
        let Some(device) = device() else {
            return;
        };
        let jobs: Vec<_> = [7, 19, 43]
            .into_iter()
            .map(|s| job(s, 64, vec![64]))
            .collect();
        let graphs: Vec<_> = jobs.iter().map(|j| j.graph.clone()).collect();
        let schedules: Vec<_> = jobs.iter().map(|j| j.schedule.to_vec()).collect();
        let steps: Vec<_> = [0, 32]
            .into_iter()
            .map(|start| {
                jobs.iter()
                    .enumerate()
                    .map(|(i, j)| step(i as u32, j.seed, start, 32, 64))
                    .collect()
            })
            .collect();
        let reference = dispatch_slots(&device, &graphs, &schedules, &steps);
        let mut pool = SlotPool::new(&device, &graphs[0], READS, 3, 64).unwrap();
        assert_eq!(pool.capacity(), 3);
        let nodes = graphs[1].num_nodes();
        let edges = Arc::clone(&graphs[1].edges);
        assert!(pool.matches_prepared(&edges, nodes, READS));
        // The second query takes the verified-storage path.
        assert!(pool.matches_prepared(&edges, nodes, READS));
        assert!(!pool.matches_prepared(&edges, nodes, READS + 1));
        let mut swapped = graphs[1].edges.to_vec();
        swapped.swap(0, 1);
        assert!(!pool.matches_prepared(&Edges::from(swapped.as_slice()), nodes, READS));
        for j in jobs {
            pool.admit(j).unwrap();
        }
        let mut expected_us = [0; 3];
        for index in 0..2 {
            assert!(pool.commit_step(32, 32).unwrap());
            pool.wait();
            let command_us = sampler::gpu_time_us(pool.command.as_ref().unwrap());
            for (slot, us) in expected_us.iter_mut().enumerate() {
                *us += device_time_share(command_us, 3, slot);
            }
            assert_eq!(
                pool.take_checkpoints().unwrap().len(),
                if index == 0 { 0 } else { 3 }
            );
        }
        // SAFETY: both completed steps wrote every live slot's output. The
        // reference lengths are the exact allocated sample and energy counts.
        unsafe {
            let energies = std::slice::from_raw_parts(
                pool.buffer(11).unwrap().contents().cast::<i32>(),
                reference.0.len(),
            );
            let samples = std::slice::from_raw_parts(
                pool.buffer(10).unwrap().contents().cast::<i8>(),
                reference.1.len(),
            );
            assert_eq!(energies, reference.0);
            assert_eq!(samples, reference.1);
        }
        let all_reads = pool.reads_many(&[0, 1, 2], READS).unwrap();
        for (slot, graph) in graphs.iter().enumerate() {
            let packed_size = graph.h.len().div_ceil(8);
            for (r, read) in all_reads[slot].iter().enumerate() {
                let idx = slot * READS + r;
                assert_eq!(read.energy_milli, i64::from(reference.0[idx]));
                assert_eq!(
                    read.spins,
                    unpack_spins(
                        &reference.1[idx * packed_size..(idx + 1) * packed_size],
                        graph.h.len()
                    )
                );
            }
            assert_eq!(pool.device_us(slot), expected_us[slot]);
        }
    }

    #[test]
    fn checkpoints_fire_at_each_boundary_and_last_marks_the_end() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 1, 160).unwrap();
        pool.admit(job(7, 160, vec![32, 96, 160])).unwrap();
        for i in 0..5 {
            let checkpoints = finish_step(&mut pool, 32);
            if i % 2 == 0 {
                assert_eq!(checkpoints.len(), 1);
                let cp = &checkpoints[0];
                assert_eq!((cp.slot, cp.index, cp.last), (0, i / 2, i == 4));
                assert_eq!(
                    cp.best,
                    pool.reads(0, READS)
                        .unwrap()
                        .iter()
                        .map(|r| r.energy_milli)
                        .min()
                        .unwrap()
                );
            } else {
                assert!(checkpoints.is_empty());
            }
            assert!(pool.take_checkpoints().unwrap().is_empty());
        }
        pool.release(0).unwrap();
        assert_eq!(pool.live(), 0);
        assert!(!pool.commit_step(32, 32).unwrap());
    }

    #[test]
    fn slot_keeps_a_held_schedule_and_rewrites_a_different_one() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let shared: Arc<[f32]> = build_beta_schedule(
            &graph,
            32,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0
        .into();
        let halved: Arc<[f32]> = shared.iter().map(|b| b * 0.5).collect();
        let with = |seed: u64, schedule: &Arc<[f32]>| SlotJob {
            graph: graph.clone(),
            schedule: Arc::clone(schedule),
            checkpoints: vec![32],
            fresh_from: CONTINUING,
            seed,
            observe: false,
        };
        let run = |pool: &mut SlotPool, job: SlotJob| {
            assert_eq!(pool.admit(job).unwrap(), 0);
            finish_step(pool, 32);
            let reads = pool.reads(0, READS).unwrap();
            pool.release(0).unwrap();
            reads
        };
        let mut pool = SlotPool::new(&device, &graph, READS, 1, 32).unwrap();
        run(&mut pool, with(7, &shared));
        for schedule in [&shared, &halved] {
            let reused = run(&mut pool, with(19, schedule));
            let mut fresh = SlotPool::new(&device, &graph, READS, 1, 32).unwrap();
            let copy: Arc<[f32]> = schedule.to_vec().into();
            let expected = run(&mut fresh, with(19, &copy));
            for (a, b) in reused.iter().zip(&expected) {
                assert_eq!(a.spins, b.spins);
                assert_eq!(a.energy_milli, b.energy_milli);
            }
        }
    }

    #[test]
    fn released_slot_starts_fresh() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, READS, 1, 64).unwrap();
        pool.admit(job(7, 64, vec![32, 64])).unwrap();
        finish_step(&mut pool, 32);
        pool.release(0).unwrap();
        assert_eq!(pool.admit(job(19, 32, vec![32])).unwrap(), 0);
        pool.reads(0, READS).unwrap_err();
        finish_step(&mut pool, 32);
        let mut fresh = SlotPool::new(&device, &graph, READS, 1, 64).unwrap();
        fresh.admit(job(19, 32, vec![32])).unwrap();
        finish_step(&mut fresh, 32);
        for (a, b) in pool
            .reads(0, READS)
            .unwrap()
            .iter()
            .zip(fresh.reads(0, READS).unwrap())
        {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
    }

    #[test]
    fn admit_and_release_refuse_while_in_flight() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 2, 32).unwrap();
        pool.admit(job(7, 32, vec![32])).unwrap();
        assert!(pool.commit_step(32, 32).unwrap());
        assert!(pool.in_flight());
        pool.admit(job(19, 32, vec![32])).unwrap_err();
        assert!(pool.release(0).is_err());
        pool.reads(0, READS).unwrap_err();
        pool.commit_step(32, 32).unwrap_err();
        assert_eq!(pool.live(), 1);
        pool.wait();
        assert_eq!(pool.take_checkpoints().unwrap().len(), 1);
        assert!(!pool.in_flight());
    }

    #[test]
    fn over_long_schedule_is_too_large() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 1, 64).unwrap();
        assert!(matches!(
            pool.admit(job(7, 65, vec![65])),
            Err(crate::sampler::SampleError::TooLarge(_))
        ));
        assert_eq!(pool.live(), 0);
    }

    #[test]
    fn reads_carry_device_energies_equal_to_host_scoring() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, READS, 1, 32).unwrap();
        pool.admit(job(7, 32, vec![32])).unwrap();
        finish_step(&mut pool, 32);
        for read in pool.reads(0, READS).unwrap() {
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }

    #[test]
    fn invalid_jobs_fail_validation_without_a_device() {
        let mut j = job(7, 64, vec![32, 64]);
        j.validate(64).unwrap();
        assert!(matches!(j.validate(63), Err(SampleError::TooLarge(_))));
        for checkpoints in [
            vec![],
            vec![0, 64],
            vec![32, 32, 64],
            vec![64, 32],
            vec![32],
        ] {
            j.checkpoints = checkpoints;
            j.validate(64).unwrap_err();
        }
        j.checkpoints = vec![64];
        Arc::make_mut(&mut j.schedule)[0] = f32::NAN;
        j.validate(64).unwrap_err();
        Arc::make_mut(&mut j.schedule)[0] = -1.0;
        j.validate(64).unwrap_err();
        Arc::make_mut(&mut j.schedule)[0] = 0.1;
        j.graph.h[0] = 0.5;
        j.validate(64).unwrap_err();
    }

    #[test]
    fn pool_clamps_steps_and_preserves_checkpoint_reads() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, 33, 1, 64).unwrap();
        pool.admit(job(7, 64, vec![5, 64])).unwrap();
        pool.admit(job(19, 32, vec![32])).unwrap_err();
        pool.commit_step(0, 0).unwrap_err();
        pool.release(1).unwrap_err();
        let cp = finish_step(&mut pool, 32);
        assert_eq!((cp[0].index, cp[0].last), (0, false));
        let before = pool.reads(0, 33).unwrap();
        pool.reads(0, 34).unwrap_err();
        assert_eq!(pool.reads(0, 1).unwrap().len(), 1);
        assert!(finish_step(&mut pool, 32).is_empty());
        for (a, b) in before.iter().zip(pool.reads(0, 33).unwrap()) {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
        let cp = finish_step(&mut pool, 32);
        assert_eq!((cp[0].index, cp[0].last), (1, true));
        for read in pool.reads(0, 33).unwrap() {
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        pool.commit_step(32, 32).unwrap_err();
    }

    #[test]
    fn each_leg_after_a_checkpoint_is_a_fresh_anneal() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let first = build_beta_schedule(
            &graph,
            16,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let second = build_beta_schedule(
            &graph,
            48,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let legs: Vec<f32> = first.iter().chain(&second).copied().collect();
        let mut pool = SlotPool::new(&device, &graph, 64, 2, legs.len()).unwrap();
        let two_legs = pool
            .admit(SlotJob {
                graph: graph.clone(),
                schedule: legs.into(),
                checkpoints: vec![16, 64],
                fresh_from: 1,
                seed: 7,
                observe: false,
            })
            .unwrap();
        // The second leg alone, as its own job with that leg's seed.
        let alone = pool
            .admit(SlotJob {
                graph,
                schedule: second.into(),
                checkpoints: vec![48],
                fresh_from: 1,
                seed: leg_seed(7, 1),
                observe: false,
            })
            .unwrap();
        let mut finished = std::collections::HashMap::new();
        while pool.live() > 0 {
            for checkpoint in finish_step(&mut pool, 8) {
                if checkpoint.last {
                    assert_eq!(checkpoint.position, 48, "positions count within the leg");
                    let reads = pool.reads(checkpoint.slot, 64).unwrap();
                    pool.release(checkpoint.slot).unwrap();
                    finished.insert(checkpoint.slot, reads);
                }
            }
        }
        for (a, b) in finished[&two_legs].iter().zip(&finished[&alone]) {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
    }

    #[test]
    fn fresh_legs_step_at_the_deep_slice() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, READS, 2, 256).unwrap();
        let continuing = pool.admit(job(7, 256, vec![256])).unwrap();
        let fresh = pool
            .admit(SlotJob {
                fresh_from: 0,
                ..job(19, 256, vec![256])
            })
            .unwrap();
        assert!(pool.commit_step(8, 16).unwrap());
        let count = |slot: SlotId| {
            pool.steps
                .iter()
                .find(|step| step.slot as usize == slot)
                .unwrap()
                .beta_count
        };
        assert_eq!(count(continuing), 8);
        assert_eq!(count(fresh), 16);
        pool.wait();
    }

    #[test]
    fn leg_seeds_are_distinct_and_keep_the_job_seed_first() {
        assert_eq!(leg_seed(42, 0), 42);
        let seeds: std::collections::HashSet<_> = (0..16).map(|leg| leg_seed(42, leg)).collect();
        assert_eq!(seeds.len(), 16);
    }

    #[test]
    fn reads_many_preserves_requested_slot_and_read_order() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), 33, 4, 32).unwrap();
        for seed in [7, 19, 43] {
            pool.admit(job(seed, 32, vec![32])).unwrap();
        }
        pool.reads_many(&[0], 33).unwrap_err();
        assert!(pool.commit_step(32, 32).unwrap());
        pool.reads_many(&[0], 33).unwrap_err();
        pool.wait();
        assert_eq!(pool.take_checkpoints().unwrap().len(), 3);
        let dense = pool.reads_many(&[0, 1, 2], 33).unwrap();
        pool.release(1).unwrap();
        let sparse = pool.reads_many(&[2, 0, 2], 17).unwrap();
        for (reads, slot) in sparse.iter().zip([2, 0, 2]) {
            assert_eq!(reads.len(), 17);
            for (a, b) in reads.iter().zip(&dense[slot]) {
                assert_eq!(a.spins, b.spins);
                assert_eq!(a.energy_milli, b.energy_milli);
            }
        }
        assert!(pool.reads_many(&[], 33).unwrap().is_empty());
        assert!(pool.reads_many(&[0], 0).unwrap()[0].is_empty());
        pool.reads_many(&[0], 34).unwrap_err();
        pool.reads_many(&[0, 1], 33).unwrap_err();
        pool.reads_many(&[3], 33).unwrap_err();
        pool.reads_many(&[4], 33).unwrap_err();
    }

    fn device() -> Option<MetalDevice> {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping slot device test: no Metal device");
            }
            return None;
        }
        Some(MetalDevice::open(0).expect("Metal device and pipelines"))
    }

    fn step(slot: u32, seed: u64, start: i32, count: i32, total: i32) -> SlotStep {
        SlotStep {
            slot,
            beta_start: start,
            beta_count: count,
            num_betas: total,
            seed: slot_seed(seed),
            flags: if start + count >= total {
                SLOT_WRITE_OUTPUT
            } else {
                0
            },
        }
    }

    fn slot_coefficients(topo: &SelfFeedingTopology, graphs: &[IsingGraph]) -> (Vec<i8>, Vec<i8>) {
        let mut h = Vec::new();
        let mut j = Vec::new();
        for graph in graphs {
            assert_eq!(graph.edges, graphs[0].edges);
            let (couplings, fields) = fill_h_j(topo, graph);
            assert_eq!(fields.len(), topo.n);
            assert_eq!(couplings.len(), topo.nnz);
            h.extend(fields);
            j.extend(couplings);
        }
        (h, j)
    }

    #[test]
    fn slot_coefficients_match_field_and_csr_strides() {
        let graphs = [
            IsingGraph::new(vec![1.0, 0.0, -1.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]),
            IsingGraph::new(vec![0.0, -1.0, 1.0], vec![-1.0, 1.0], vec![(0, 1), (1, 2)]),
        ];
        let topo = SelfFeedingTopology::build(&graphs[0]);
        let (h, j) = slot_coefficients(&topo, &graphs);
        assert_eq!(h.len(), graphs.len() * topo.n);
        assert_eq!(j.len(), graphs.len() * topo.nnz);
        assert_eq!(h, [1, 0, -1, 0, -1, 1]);
        assert_eq!(j, [1, 1, -1, -1, -1, -1, 1, 1]);
    }

    // Each inner slice is one dispatch in a separate command buffer. Storage
    // survives across slices so continuation steps exercise the device state.
    fn dispatch_slots(
        device: &MetalDevice,
        graph_per_slot: &[IsingGraph],
        schedules: &[Vec<f32>],
        steps: &[Vec<SlotStep>],
    ) -> (Vec<i32>, Vec<i8>) {
        let topo = SelfFeedingTopology::build_with_advantage2_coloring(&graph_per_slot[0]);
        let slots = graph_per_slot.len();
        let max_threads = device.msa_slots.max_total_threads_per_threadgroup() as usize;
        let threads_per_group = MSA_THREADS.min(max_threads).max(1);
        let packed_size = topo.n.div_ceil(8);
        let sched_stride = schedules.iter().map(Vec::len).max().unwrap();
        let (h, j) = slot_coefficients(&topo, graph_per_slot);
        let mut schedule = vec![0.0f32; slots * sched_stride];
        for (slot, slot_schedule) in schedules.iter().enumerate() {
            schedule[slot * sched_stride..slot * sched_stride + slot_schedule.len()]
                .copy_from_slice(slot_schedule);
        }
        let samples = device.new_buffer_from_slice(&vec![85i8; slots * READS * packed_size]);
        let energies = device.new_buffer_from_slice(&vec![i32::MIN; slots * READS]);
        let state = device.new_buffer_from_slice(&vec![0u32; slots * WORDS * topo.n]);
        let rng = device.new_buffer_from_slice(&vec![0u32; slots * WORDS * threads_per_group * 4]);
        let buffers = [
            (0, device.new_buffer_from_slice(&topo.row_ptr)),
            (1, device.new_buffer_from_slice(&topo.col_ind)),
            (2, device.new_buffer_from_slice(&j)),
            (3, device.new_buffer_from_slice(&[0i32])),
            (9, device.new_buffer_from_slice(&schedule)),
            (15, device.new_buffer_from_slice(&h)),
            (16, device.new_buffer_from_slice(&topo.colors.starts)),
            (17, device.new_buffer_from_slice(&topo.colors.counts)),
            (18, device.new_buffer_from_slice(&topo.colors.nodes)),
        ];
        for dispatch in steps {
            let table = device.new_buffer_from_slice(dispatch);
            let command = device.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&device.msa_slots);
            for (index, buffer) in &buffers {
                encoder.set_buffer(*index, Some(buffer), 0);
            }
            for (index, buffer) in [
                (10, &samples),
                (11, &energies),
                (23, &state),
                (24, &rng),
                (25, &table),
            ] {
                encoder.set_buffer(index, Some(buffer), 0);
            }
            for (index, value) in [
                (4, topo.nnz as i32),
                (5, topo.n as i32),
                (6, sched_stride as i32),
                (7, 1),
                (8, 0),
                (12, (dispatch.len() * WORDS) as i32),
                (13, slots as i32),
                (14, READS as i32),
                (19, WORDS as i32),
                (20, topo.colors.num_colors),
                (21, 0),
                (22, 0),
            ] {
                encoder.set_bytes(index, 4, (&value as *const i32).cast());
            }
            encoder.set_threadgroup_memory_length(0, ((topo.n * 4).div_ceil(16) * 16) as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new((dispatch.len() * WORDS) as u64, 1, 1),
                MTLSize::new(threads_per_group as u64, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        }
        // SAFETY: shared buffers hold these initialized element counts, and all
        // commands have completed before the host reads or drops the buffers.
        unsafe {
            (
                std::slice::from_raw_parts(energies.contents().cast::<i32>(), slots * READS)
                    .to_vec(),
                std::slice::from_raw_parts(
                    samples.contents().cast::<i8>(),
                    slots * READS * packed_size,
                )
                .to_vec(),
            )
        }
    }

    #[test]
    fn slot_step_layout_matches_msl() {
        assert_eq!(std::mem::size_of::<SlotStep>(), 24);
        assert_eq!(std::mem::align_of::<SlotStep>(), 4);
        assert_eq!(std::mem::offset_of!(SlotStep, slot), 0);
        assert_eq!(std::mem::offset_of!(SlotStep, beta_start), 4);
        assert_eq!(std::mem::offset_of!(SlotStep, beta_count), 8);
        assert_eq!(std::mem::offset_of!(SlotStep, num_betas), 12);
        assert_eq!(std::mem::offset_of!(SlotStep, seed), 16);
        assert_eq!(std::mem::offset_of!(SlotStep, flags), 20);
    }

    #[test]
    fn slot_seed_is_never_zero() {
        assert_eq!(slot_seed(0), 1);
        assert_eq!(slot_seed(1 << 32 | 1), 1);
        assert_ne!(slot_seed(7), 0);
    }

    #[test]
    fn slot_energies_equal_host_energy_of_packed_samples() {
        let Some(device) = device() else {
            return;
        };
        let graphs: Vec<_> = [7, 19, 43].into_iter().map(advantage2_system1).collect();
        let schedules: Vec<_> = graphs
            .iter()
            .map(|g| {
                build_beta_schedule(g, 32, 1, None, quip_solver_core::BetaSchedule::Geometric).0
            })
            .collect();
        let steps = vec![vec![
            step(2, 43, 0, 32, 32),
            step(0, 7, 0, 32, 32),
            step(1, 19, 0, 32, 32),
        ]];
        let (energies, samples) = dispatch_slots(&device, &graphs, &schedules, &steps);
        for (slot, graph) in graphs.iter().enumerate() {
            let packed_size = graph.h.len().div_ceil(8);
            for read in 0..READS {
                let idx = slot * READS + read;
                let spins = unpack_spins(
                    &samples[idx * packed_size..(idx + 1) * packed_size],
                    graph.h.len(),
                );
                assert_eq!(
                    i64::from(energies[idx]),
                    energy_milli(&spins, &graph.h, &graph.j, &graph.edges)
                );
            }
        }
    }

    #[test]
    fn a_job_gives_the_same_samples_in_any_slot_and_beside_any_neighbours() {
        let Some(device) = device() else {
            return;
        };
        let x = advantage2_system1(7);
        let sched =
            build_beta_schedule(&x, 32, 1, None, quip_solver_core::BetaSchedule::Geometric).0;
        let alone = dispatch_slots(
            &device,
            std::slice::from_ref(&x),
            std::slice::from_ref(&sched),
            &[vec![step(0, 99, 0, 32, 32)]],
        );
        let graphs = vec![advantage2_system1(19), advantage2_system1(43), x];
        let schedules = vec![
            build_beta_schedule(
                &graphs[0],
                16,
                1,
                None,
                quip_solver_core::BetaSchedule::Geometric,
            )
            .0,
            build_beta_schedule(
                &graphs[1],
                48,
                1,
                None,
                quip_solver_core::BetaSchedule::Geometric,
            )
            .0,
            sched,
        ];
        let beside = dispatch_slots(
            &device,
            &graphs,
            &schedules,
            &[vec![
                step(2, 99, 0, 32, 32),
                step(0, 11, 0, 16, 16),
                step(1, 22, 0, 48, 48),
            ]],
        );
        assert_eq!(alone.0, beside.0[2 * READS..]);
        assert_eq!(alone.1, beside.1[2 * alone.1.len()..]);
    }

    #[test]
    fn slicing_does_not_change_the_result() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let schedule = build_beta_schedule(
            &graph,
            64,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let run = |count| {
            let steps: Vec<_> = (0..64)
                .step_by(count)
                .map(|start| vec![step(0, 99, start, count as i32, 64)])
                .collect();
            dispatch_slots(
                &device,
                std::slice::from_ref(&graph),
                std::slice::from_ref(&schedule),
                &steps,
            )
        };
        let full = run(64);
        assert_eq!(full, run(32));
        assert_eq!(full, run(16));
        assert_eq!(full, run(4));
    }

    #[test]
    fn large_first_checkpoint_has_bounded_steps_and_identical_reads() {
        let device = MetalDevice::open(0).unwrap();
        let graph = advantage2_system1(7);
        let limit = sampler::msa_step_limit(graph.num_nodes(), WORDS, sampler::MAX_SWEEPS);
        let first = limit + 17;
        let total = first + 31;
        assert!(total <= sampler::MAX_SWEEPS);
        let schedule = build_beta_schedule(
            &graph,
            total,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let mut pool = SlotPool::new(&device, &graph, READS, 1, total).unwrap();
        pool.admit(SlotJob {
            graph: graph.clone(),
            schedule: schedule.clone().into(),
            checkpoints: vec![first, total],
            fresh_from: CONTINUING,
            seed: 99,
            observe: false,
        })
        .unwrap();
        let mut position = 0;
        let mut commands = 0;
        let mut reference_steps = Vec::new();
        for (index, checkpoint) in [first, total].into_iter().enumerate() {
            let start = if index == 0 { 0 } else { first };
            let mut reference_step = step(
                0,
                99,
                start as i32,
                (checkpoint - start) as i32,
                total as i32,
            );
            reference_step.flags = SLOT_WRITE_OUTPUT;
            reference_steps.push(vec![reference_step]);
            loop {
                let checkpoints = finish_step(&mut pool, first);
                let actual = pool.steps[0];
                assert_eq!(actual.beta_start as usize, position);
                assert!((1..=limit).contains(&(actual.beta_count as usize)));
                position += actual.beta_count as usize;
                commands += 1;
                if position < checkpoint {
                    assert!(checkpoints.is_empty());
                    assert_eq!(actual.flags, 0);
                    continue;
                }
                assert_eq!(position, checkpoint);
                assert_eq!(checkpoints.len(), 1);
                assert_eq!(
                    (checkpoints[0].index, checkpoints[0].last),
                    (index, index == 1)
                );
                let (energies, samples) = dispatch_slots(
                    &device,
                    std::slice::from_ref(&graph),
                    std::slice::from_ref(&schedule),
                    &reference_steps,
                );
                let packed_size = graph.num_nodes().div_ceil(8);
                for (read, actual) in pool.reads(0, READS).unwrap().iter().enumerate() {
                    assert_eq!(actual.energy_milli, i64::from(energies[read]));
                    assert_eq!(
                        actual.spins,
                        unpack_spins(
                            &samples[read * packed_size..(read + 1) * packed_size],
                            graph.num_nodes()
                        )
                    );
                }
                break;
            }
        }
        assert!(commands > 2);
        assert_eq!(
            &pool.slots[0].as_ref().unwrap().job.schedule[..],
            &schedule[..]
        );
    }

    #[test]
    fn observe_points_write_output_without_becoming_checkpoints() {
        let Some(device) = device() else {
            return;
        };
        // Far enough past two OBSERVE_INTERVAL boundaries that the job must
        // stop for an observe readback before it ever reaches the one real
        // checkpoint at the end of the schedule.
        let checkpoint = 2 * OBSERVE_INTERVAL + 17;
        let mut pool =
            SlotPool::new(&device, &advantage2_system1(7), READS, 1, checkpoint).unwrap();
        pool.admit(observing_job(7, checkpoint, vec![checkpoint]))
            .unwrap();
        let mut observed = 0;
        let mut next_boundary = OBSERVE_INTERVAL;
        loop {
            let checkpoints = finish_step(&mut pool, checkpoint);
            let position = pool.position(0);
            // The device's own dispatch budget (msa_step_limit) can require
            // several commit_step calls to cross one OBSERVE_INTERVAL: most
            // of those calls write no output at all.
            if checkpoints.is_empty() {
                assert!(position < checkpoint);
                continue;
            }
            assert_eq!(checkpoints.len(), 1);
            let cp = &checkpoints[0];
            if position < checkpoint {
                // An observe stop is not a checkpoint: the next real
                // checkpoint (index 0, the only one in this schedule) has
                // not advanced, and it is never reported as the last one.
                assert!(cp.observe);
                assert_eq!(cp.index, 0);
                assert!(!cp.last);
                assert_eq!(cp.position, position);
                assert_eq!(
                    position, next_boundary,
                    "an observe stop lands exactly on the next OBSERVE_INTERVAL boundary"
                );
                next_boundary += OBSERVE_INTERVAL;
                observed += 1;
                continue;
            }
            assert_eq!(position, checkpoint);
            assert!(!cp.observe);
            assert_eq!((cp.index, cp.last), (0, true));
            break;
        }
        assert_eq!(
            observed, 2,
            "expected one observe readback per OBSERVE_INTERVAL boundary before the checkpoint"
        );
        pool.release(0).unwrap();

        // A plain (non-observe) job with the same schedule sees no output
        // writes at all until the one real checkpoint: no observe points
        // are ever created for it.
        let mut plain =
            SlotPool::new(&device, &advantage2_system1(7), READS, 1, checkpoint).unwrap();
        plain.admit(job(7, checkpoint, vec![checkpoint])).unwrap();
        loop {
            let checkpoints = finish_step(&mut plain, checkpoint);
            if plain.position(0) < checkpoint {
                assert!(checkpoints.is_empty());
                continue;
            }
            assert_eq!(checkpoints.len(), 1);
            assert!(!checkpoints[0].observe);
            break;
        }
    }

    #[test]
    fn observe_hit_before_any_checkpoint_can_be_read() {
        let Some(device) = device() else {
            return;
        };
        // The one real checkpoint sits past the first observe boundary, so
        // the observe stop is this slot's very first output write — no real
        // checkpoint has ever fired for it yet.
        let checkpoint = OBSERVE_INTERVAL + 17;
        let mut pool =
            SlotPool::new(&device, &advantage2_system1(7), READS, 1, checkpoint).unwrap();
        pool.admit(observing_job(7, checkpoint, vec![checkpoint]))
            .unwrap();
        loop {
            let checkpoints = finish_step(&mut pool, checkpoint);
            if checkpoints.is_empty() {
                assert!(pool.position(0) < checkpoint);
                continue;
            }
            assert_eq!(checkpoints.len(), 1);
            let cp = &checkpoints[0];
            assert!(cp.observe);
            assert_eq!(cp.position, OBSERVE_INTERVAL);
            break;
        }
        // Before has_output was set on an observe stop, this failed with
        // "slot has no checkpoint output" instead of returning the reads
        // the device already wrote.
        let reads = pool.reads(0, READS).unwrap();
        assert_eq!(reads.len(), READS);
    }

    #[test]
    fn observe_points_do_not_change_the_final_energies_or_spins() {
        let Some(device) = device() else {
            return;
        };
        // The anneal must be bit-identical with or without observe points:
        // they only add output write-backs at extra step boundaries, which
        // the kernel already tolerates (every step resumes spins and RNG
        // from device memory regardless of where the previous one stopped).
        let checkpoint = 2 * OBSERVE_INTERVAL + 17;
        let graph = advantage2_system1(11);
        let schedule: Arc<[f32]> = build_beta_schedule(
            &graph,
            checkpoint,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0
        .into();

        let mut observed_pool = SlotPool::new(&device, &graph, READS, 1, checkpoint).unwrap();
        observed_pool
            .admit(SlotJob {
                graph: graph.clone(),
                schedule: Arc::clone(&schedule),
                checkpoints: vec![checkpoint],
                fresh_from: CONTINUING,
                seed: 11,
                observe: true,
            })
            .unwrap();
        loop {
            let checkpoints = finish_step(&mut observed_pool, checkpoint);
            if checkpoints.iter().any(|c| !c.observe) {
                break;
            }
        }
        let observed_reads = observed_pool.reads(0, READS).unwrap();

        let mut plain_pool = SlotPool::new(&device, &graph, READS, 1, checkpoint).unwrap();
        plain_pool
            .admit(SlotJob {
                graph,
                schedule,
                checkpoints: vec![checkpoint],
                fresh_from: CONTINUING,
                seed: 11,
                observe: false,
            })
            .unwrap();
        loop {
            let checkpoints = finish_step(&mut plain_pool, checkpoint);
            if !checkpoints.is_empty() {
                break;
            }
        }
        let plain_reads = plain_pool.reads(0, READS).unwrap();

        assert_eq!(observed_reads.len(), plain_reads.len());
        for (a, b) in observed_reads.iter().zip(&plain_reads) {
            assert_eq!(a.energy_milli, b.energy_milli);
            assert_eq!(a.spins, b.spins);
        }
    }

    #[test]
    fn a_step_without_output_preserves_output_buffers() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let schedule = build_beta_schedule(
            &graph,
            32,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let mut no_output = step(0, 99, 0, 32, 32);
        no_output.flags = 0;
        let (energies, samples) =
            dispatch_slots(&device, &[graph], &[schedule], &[vec![no_output]]);
        assert!(energies.iter().all(|&e| e == i32::MIN));
        assert!(samples.iter().all(|&b| b == 85));
    }

    /// Deterministic xorshift64 for fixture generation.
    fn xorshift64(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Load compacted, zero-based fixture edges with seeded `J` in {-1, 1} and
    /// `h` in {-1, 0, 1}.
    fn advantage2_system1(seed: u64) -> IsingGraph {
        let mut s = seed | 1;
        let mut edges = Vec::with_capacity(41515);
        let mut j = Vec::with_capacity(41515);
        for line in include_str!("../tests/fixtures/advantage2-system1.edges").lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut nodes = line.split_whitespace();
            let u = nodes.next().expect("edge start").parse().expect("node id");
            let v = nodes.next().expect("edge end").parse().expect("node id");
            assert!(nodes.next().is_none(), "two node ids per edge");
            assert!(u < 4577 && v < 4577, "fixture node range");
            edges.push((u, v));
            j.push(if xorshift64(&mut s) & 1 == 0 {
                1.0
            } else {
                -1.0
            });
        }
        assert_eq!(edges.len(), 41515);
        let h = (0..4577)
            .map(|_| [-1.0, 0.0, 1.0][(xorshift64(&mut s) % 3) as usize])
            .collect();
        IsingGraph::new(h, j, edges)
    }

    /// Per-checkpoint best energies of the production cascade schedule
    /// ([`crate::cascade::segment_schedule`]) on recorded problems, with no
    /// gating: every job runs every leg. Used to set the chain gates.
    ///
    /// Ignored: needs a Metal device and a problem directory written by
    /// `scripts/testnet/regen`. Every problem must share one topology with
    /// zero fields, so one schedule serves all of them. Writes
    /// `qblock_id,k,best_<stage>...,best_<sweeps>`.
    ///
    /// ```text
    /// QUIP_TRACE_PROBLEMS=dir QUIP_TRACE_OUT=trace.csv QUIP_TRACE_STAGES=8,16,64 \
    ///   QUIP_TRACE_SWEEPS=256 QUIP_TRACE_SEEDS=30 \
    ///   cargo test --release --lib checkpoint_trace -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device and QUIP_TRACE_PROBLEMS"]
    #[expect(clippy::print_stderr, reason = "the trace reports progress on stderr")]
    fn checkpoint_trace() {
        #[derive(serde::Deserialize)]
        struct Entry {
            path: String,
            qblock_id: u64,
        }
        #[derive(serde::Deserialize)]
        struct Problem {
            h: Vec<f64>,
            j: Vec<f64>,
            edges: Vec<(usize, usize)>,
        }
        let var =
            |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_owned());
        let dir = std::env::var("QUIP_TRACE_PROBLEMS").expect("QUIP_TRACE_PROBLEMS");
        let out = std::env::var("QUIP_TRACE_OUT").expect("QUIP_TRACE_OUT");
        let stages: Vec<usize> = var("QUIP_TRACE_STAGES", "8,16,64")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let sweeps: usize = var("QUIP_TRACE_SWEEPS", "256").parse().unwrap();
        let seeds: u64 = var("QUIP_TRACE_SEEDS", "30").parse().unwrap();
        let slice: usize = var("QUIP_TRACE_SLICE", "4096").parse().unwrap();
        let index: Vec<Entry> =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/index.json")).unwrap())
                .unwrap();
        let mut jobs = Vec::new();
        for entry in &index {
            let p: Problem =
                serde_json::from_str(&std::fs::read_to_string(&entry.path).unwrap()).unwrap();
            assert!(
                p.h.iter().all(|&h| h == 0.0),
                "one schedule needs zero fields"
            );
            let graph = IsingGraph::new(p.h, p.j, p.edges);
            for k in 0..seeds {
                jobs.push((entry.qblock_id, k, graph.clone()));
            }
        }
        let trace_seed = |qblock: u64, k: u64| {
            let mut z = qblock.wrapping_mul(1_000_003).wrapping_add(k);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)).max(1)
        };
        let params = SampleParams {
            num_reads: READS,
            num_sweeps: sweeps,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: None,
            seed: 1,
        };
        let (schedule, checkpoints, fresh_from) =
            crate::cascade::segment_schedule(&jobs[0].2, &params, &stages);
        let schedule: Arc<[f32]> = schedule.into();
        let device = MetalDevice::open(0).expect("Metal device 0");
        let capacity = crate::streaming::batch_size_for_reads(crate::Kernel::Msa, READS);
        let mut pool = SlotPool::new(&device, &jobs[0].2, READS, capacity, schedule.len()).unwrap();
        let total = jobs.len();
        let mut owner: Vec<Option<(u64, u64)>> = vec![None; capacity];
        let mut rows: std::collections::BTreeMap<(u64, u64), Vec<i64>> =
            std::collections::BTreeMap::new();
        let mut pending = jobs.into_iter();
        let started = std::time::Instant::now();
        let mut done = 0usize;
        loop {
            while pool.live() < pool.capacity() {
                let Some((qblock, k, graph)) = pending.next() else {
                    break;
                };
                let slot = pool
                    .admit(SlotJob {
                        graph,
                        schedule: Arc::clone(&schedule),
                        checkpoints: checkpoints.clone(),
                        fresh_from,
                        seed: trace_seed(qblock, k),
                        observe: false,
                    })
                    .unwrap();
                owner[slot] = Some((qblock, k));
            }
            if pool.live() == 0 {
                break;
            }
            assert!(pool.commit_step(slice, slice).unwrap());
            pool.wait();
            for checkpoint in pool.take_checkpoints().unwrap() {
                let key = owner[checkpoint.slot].expect("live slot has an owner");
                rows.entry(key).or_default().push(checkpoint.best);
                if checkpoint.last {
                    pool.release(checkpoint.slot).unwrap();
                    owner[checkpoint.slot] = None;
                    done += 1;
                    if done.is_multiple_of(100) {
                        let secs = started.elapsed().as_secs_f64();
                        eprintln!("{done}/{total} jobs in {secs:.0} s");
                    }
                }
            }
        }
        let mut text = String::from("qblock_id,k");
        for stage in stages.iter().filter(|&&s| s < sweeps).chain([&sweeps]) {
            text.push_str(&format!(",best_{stage}"));
        }
        text.push('\n');
        for ((qblock, k), bests) in &rows {
            let cells: Vec<String> = bests.iter().map(i64::to_string).collect();
            text.push_str(&format!("{qblock},{k},{}\n", cells.join(",")));
        }
        std::fs::write(&out, text).unwrap();
        eprintln!("{total} jobs in {:.0} s", started.elapsed().as_secs_f64());
    }

    /// Screen throughput against deep-leg speed for several deep step sizes.
    ///
    /// Fills a pool of production capacity with screen slots running
    /// back-to-back fresh 8-sweep legs, each checkpoint standing for one
    /// screened salt, plus `QUIP_BENCH_DEEP` slots in one long fresh leg. For
    /// each `deep_slice` it reports screen legs per second and deep sweeps per
    /// second over `QUIP_BENCH_SECS` seconds.
    ///
    /// ```text
    /// cargo test --release --lib deep_slice_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device; a benchmark"]
    #[expect(clippy::print_stderr, reason = "the benchmark reports on stderr")]
    fn deep_slice_bench() {
        let var =
            |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_owned());
        let deep: usize = var("QUIP_BENCH_DEEP", "4").parse().unwrap();
        let secs: f64 = var("QUIP_BENCH_SECS", "8").parse().unwrap();
        let slices: Vec<usize> = var("QUIP_BENCH_SLICES", "8,16,32,64,128,256")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let device = MetalDevice::open(0).expect("Metal device 0");
        let graph = advantage2_system1(7);
        let leg = build_beta_schedule(
            &graph,
            8,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let legs = 40_000;
        let screen: Arc<[f32]> = leg.iter().copied().cycle().take(8 * legs).collect();
        let screen_checkpoints: Vec<usize> = (1..=legs).map(|k| 8 * k).collect();
        let deep_sweeps = 4_000_000;
        let deep_schedule: Arc<[f32]> = build_beta_schedule(
            &graph,
            deep_sweeps,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0
        .into();
        let capacity = crate::streaming::batch_size_for_reads(crate::Kernel::Msa, READS);
        for &deep_slice in &slices {
            let mut pool = SlotPool::new(&device, &graph, READS, capacity, deep_sweeps).unwrap();
            let mut deep_slots = Vec::new();
            for slot in 0..capacity {
                let deep_job = slot < deep;
                let id = pool
                    .admit(SlotJob {
                        graph: graph.clone(),
                        schedule: if deep_job {
                            Arc::clone(&deep_schedule)
                        } else {
                            Arc::clone(&screen)
                        },
                        checkpoints: if deep_job {
                            vec![deep_sweeps]
                        } else {
                            screen_checkpoints.clone()
                        },
                        fresh_from: if deep_job { 0 } else { 1 },
                        seed: slot as u64 + 1,
                        observe: false,
                    })
                    .unwrap();
                if deep_job {
                    deep_slots.push(id);
                }
            }
            // The screen slots' legs are fresh from the second on, so they
            // step at `slice`; the deep slots' only leg is fresh, so they step
            // at `deep_slice`. Warm up, then measure.
            for _ in 0..50 {
                assert!(pool.commit_step(8, deep_slice).unwrap());
                pool.wait();
                pool.take_checkpoints().unwrap();
            }
            let deep_start: usize = deep_slots.iter().map(|&s| pool.position(s)).sum();
            let started = std::time::Instant::now();
            let (mut steps, mut screened) = (0u64, 0u64);
            while started.elapsed().as_secs_f64() < secs {
                assert!(pool.commit_step(8, deep_slice).unwrap());
                pool.wait();
                screened += pool.take_checkpoints().unwrap().len() as u64;
                steps += 1;
            }
            let wall = started.elapsed().as_secs_f64();
            let deep_done: usize =
                deep_slots.iter().map(|&s| pool.position(s)).sum::<usize>() - deep_start;
            let per_deep = deep_done as f64 / deep.max(1) as f64 / wall;
            eprintln!(
                "deep_slice {deep_slice:>4}: {:>6.0} steps/s, {:>8.0} screen legs/s, \
                 {:>7.0} sweeps/s per deep slot, 2.1M-sweep unit in {:>5.0} s",
                steps as f64 / wall,
                screened as f64 / wall,
                per_deep,
                2_097_152.0 / per_deep.max(1.0)
            );
        }
    }
}
