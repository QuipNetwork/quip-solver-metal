// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Nonce-parallel screen: eight models of four reads in each 32-lane word.
//!
//! A [`ScreenBank`] holds up to `capacity` words and encodes one
//! `msa_screen` dispatch. Every word is a complete anneal of one of the
//! bank's stage schedules, so nothing persists between dispatches. Models
//! must have couplings of ±1, zero fields and no self-loops. A model is
//! given as its edge sign bits ([`sign_bits`]), or, for a bank built with a
//! [`Draw`], as its lease nonce ([`key_words`]): the kernel then draws the
//! couplings itself, as `quip_protocol::chacha8::draw_ising` does.

use crate::metal_device::MetalDevice;
use crate::sampler::{self, BufferPool, CachedTopology, SampleError, MSA_THREADS};
use crate::{IsingGraph, SamplerResult};
use metal::MTLSize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Models per word.
pub(crate) const GROUPS: usize = 8;
/// Reads per model: lane `r` is read `r / GROUPS` of model `r % GROUPS`.
pub(crate) const READS: usize = sampler::MSA_LANES / GROUPS;

/// Bit `e` of the result is set where edge `e` has a negative coupling.
#[cfg(test)]
pub(crate) fn sign_bits(j_units: &[i8]) -> Vec<u32> {
    let mut bits = vec![0u32; j_units.len().div_ceil(32)];
    for (word, chunk) in bits.iter_mut().zip(j_units.chunks(32)) {
        for (bit, &j) in chunk.iter().enumerate() {
            *word |= u32::from(j < 0) << bit;
        }
    }
    bits
}

/// The lease draw a bank runs on the device.
pub(crate) struct Draw {
    /// Keystream word of edge 0: the model's node count.
    pub(crate) first_word: u32,
    /// The allowed couplings in whole units, at most 32 of them.
    pub(crate) allowed_j: Vec<i8>,
}

/// The ChaCha8 key of a lease nonce, as eight little-endian words.
pub(crate) fn key_words(nonce: [u8; 32]) -> Vec<u32> {
    nonce
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect()
}

/// Matches `ScreenDraw` in `kernels/msa.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ScreenDraw {
    mode: u32,
    first_word: u32,
    allowed_len: u32,
    neg_table: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ScreenWord {
    sched: u32,
    num_betas: i32,
    seed: u32,
    pad: u32,
}

pub(crate) struct ScreenBank {
    cached: Arc<CachedTopology>,
    pool: Arc<BufferPool>,
    pipeline: metal::ComputePipelineState,
    edge_pos: metal::Buffer,
    // Buffer indices match the msa_screen ABI.
    buffers: Vec<(u64, metal::Buffer)>,
    /// Offset and length of each stage schedule in buffer 9.
    stages: Vec<(u32, i32)>,
    words: Vec<ScreenWord>,
    capacity: usize,
    /// Words per model row: sign bits, or a key in draw mode.
    row_words: usize,
    draw: ScreenDraw,
    threads: usize,
}

impl ScreenBank {
    /// `schedules[s]` is stage `s`'s anneal, one beta per sweep. With
    /// `draw`, model rows are keys and the device draws the couplings.
    pub(crate) fn new(
        device: &MetalDevice,
        graph: &IsingGraph,
        capacity: usize,
        schedules: &[&[f32]],
        draw: Option<&Draw>,
    ) -> Result<Self, SampleError> {
        let n = graph.num_nodes();
        if n == 0
            || capacity == 0
            || schedules.is_empty()
            || schedules.iter().any(|s| s.is_empty())
            || graph.edges.iter().any(|&(u, v)| u == v)
            || draw.is_some_and(|d| !(1..=32).contains(&d.allowed_j.len()))
        {
            return Err(SampleError::TooLarge(
                "screen bank needs nodes, words, stages and no self-loops".into(),
            ));
        }
        let state_bytes = (n * 4).div_ceil(16) * 16;
        if state_bytes + sampler::MSA_STATIC_TG_BYTES
            > device.device.max_threadgroup_memory_length() as usize
        {
            return Err(SampleError::TooLarge(
                "screen spin words exceed threadgroup memory".into(),
            ));
        }
        let cached = device.topology_cache.get_or_build(device, graph, true);
        let nnz = cached.topo.nnz;
        let edges = cached.topo.edge_pos.len();
        let row_words = if draw.is_some() {
            8
        } else {
            edges.div_ceil(32)
        };
        let draw = draw.map_or(ScreenDraw::default(), |d| ScreenDraw {
            mode: 1,
            first_word: d.first_word,
            allowed_len: d.allowed_j.len() as u32,
            neg_table: d
                .allowed_j
                .iter()
                .enumerate()
                .fold(0, |table, (k, &j)| table | (u32::from(j < 0) << k)),
        });
        let sched_len: usize = schedules.iter().map(|s| s.len()).sum();
        let region = |factors: &[usize]| -> Result<u64, SampleError> {
            factors
                .iter()
                .try_fold(1usize, |acc, &v| acc.checked_mul(v))
                .filter(|&bytes| bytes <= i32::MAX as usize)
                .map(|bytes| bytes.max(4) as u64)
                .ok_or_else(|| {
                    SampleError::TooLarge("screen buffer exceeds 32-bit indexing".into())
                })
        };
        let sizes = [
            (2, region(&[capacity, nnz])?),
            (3, region(&[capacity, GROUPS, row_words, 4])?),
            (9, region(&[sched_len, 4])?),
            (10, region(&[capacity, sampler::MSA_LANES, n.div_ceil(8)])?),
            (11, region(&[capacity, sampler::MSA_LANES, 4])?),
            (25, region(&[capacity, std::mem::size_of::<ScreenWord>()])?),
        ];
        let buffers: Vec<_> = sizes
            .into_iter()
            .map(|(index, bytes)| (index, device.buffer_pool.take(&device.device, bytes)))
            .collect();
        let pairs: Vec<[u32; 2]> = cached.topo.edge_pos.iter().map(|&(a, b)| [a, b]).collect();
        let edge_pos = device.device.new_buffer_with_data(
            pairs.as_ptr().cast(),
            (std::mem::size_of_val(pairs.as_slice()) as u64).max(8),
            metal::MTLResourceOptions::StorageModeShared,
        );
        let threads = MSA_THREADS
            .min(device.msa_screen.max_total_threads_per_threadgroup() as usize)
            .max(1);
        let mut bank = Self {
            cached,
            pool: Arc::clone(&device.buffer_pool),
            pipeline: device.msa_screen.clone(),
            edge_pos,
            buffers,
            stages: Vec::with_capacity(schedules.len()),
            words: Vec::with_capacity(capacity),
            capacity,
            row_words,
            draw,
            threads,
        };
        let mut offset = 0usize;
        for schedule in schedules {
            bank.write(9, offset, schedule)?;
            bank.stages.push((offset as u32, schedule.len() as i32));
            offset += schedule.len();
        }
        Ok(bank)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn len(&self) -> usize {
        self.words.len()
    }

    /// Words per model row, the length [`Self::push`] expects.
    pub(crate) fn row_words(&self) -> usize {
        self.row_words
    }

    /// Drop every loaded word.
    pub(crate) fn clear(&mut self) {
        self.words.clear();
    }

    fn buffer(&self, index: u64) -> Result<&metal::BufferRef, SampleError> {
        self.buffers
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| b.as_ref())
            .ok_or_else(|| SampleError::Driver(format!("missing screen buffer {index}")))
    }

    fn write<T: Copy>(&self, index: u64, offset: usize, values: &[T]) -> Result<(), SampleError> {
        let buf = self.buffer(index)?;
        let start = offset * std::mem::size_of::<T>();
        let bytes = std::mem::size_of_val(values);
        let contents = buf.contents().cast::<u8>();
        if contents.is_null() || start + bytes > buf.length() as usize {
            return Err(SampleError::Driver(
                "screen buffer write exceeds storage".into(),
            ));
        }
        // SAFETY: the range lies in shared storage the GPU is not using: the
        // caller loads words only while no command of this bank is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), contents.add(start), bytes);
        }
        Ok(())
    }

    /// Load one word of up to [`GROUPS`] models at `stage`. A missing model
    /// runs all-positive couplings and its results mean nothing. Returns the
    /// word's index.
    pub(crate) fn push(
        &mut self,
        stage: usize,
        seed: u32,
        models: &[&[u32]],
    ) -> Result<usize, SampleError> {
        let &(sched, num_betas) = self
            .stages
            .get(stage)
            .ok_or_else(|| SampleError::Driver("unknown screen stage".into()))?;
        if self.words.len() == self.capacity || models.len() > GROUPS {
            return Err(SampleError::TooLarge("screen bank is full".into()));
        }
        let word = self.words.len();
        let row = self.row_words;
        for g in 0..GROUPS {
            let at = (word * GROUPS + g) * row;
            match models.get(g) {
                Some(bits) if bits.len() == row => self.write(3, at, bits)?,
                Some(_) => {
                    return Err(SampleError::Driver(
                        "model row does not match the bank".into(),
                    ))
                }
                None => self.write(3, at, &vec![0u32; row])?,
            }
        }
        self.words.push(ScreenWord {
            sched,
            num_betas,
            seed: seed.max(1),
            pad: 0,
        });
        Ok(word)
    }

    /// Encode one dispatch of every loaded word.
    pub(crate) fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Result<(), SampleError> {
        if self.words.is_empty() {
            return Ok(());
        }
        self.write(25, 0, &self.words)?;
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.cached.row), 0);
        encoder.set_buffer(1, Some(&self.cached.col), 0);
        encoder.set_buffer(4, Some(&self.edge_pos), 0);
        for (index, buffer) in &self.buffers {
            encoder.set_buffer(*index, Some(buffer), 0);
        }
        for (i, buffer) in self.cached.colors.iter().enumerate() {
            encoder.set_buffer(16 + i as u64, Some(buffer), 0);
        }
        for (index, value) in [
            (5, self.cached.n as i32),
            (6, self.cached.topo.nnz as i32),
            (7, self.cached.topo.edge_pos.len() as i32),
            (8, self.row_words as i32),
            (12, self.words.len() as i32),
            (20, self.cached.topo.colors.num_colors),
        ] {
            encoder.set_bytes(index, 4, (&value as *const i32).cast());
        }
        encoder.set_bytes(
            13,
            std::mem::size_of::<ScreenDraw>() as u64,
            (&self.draw as *const ScreenDraw).cast(),
        );
        encoder.set_threadgroup_memory_length(0, ((self.cached.n * 4).div_ceil(16) * 16) as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(self.words.len() as u64, 1, 1),
            MTLSize::new(self.threads as u64, 1, 1),
        );
        Ok(())
    }

    fn read<T: Copy>(&self, index: u64, offset: usize, count: usize) -> Result<&[T], SampleError> {
        let buf = self.buffer(index)?;
        let contents = buf.contents().cast::<T>();
        if contents.is_null() || (offset + count) * std::mem::size_of::<T>() > buf.length() as usize
        {
            return Err(SampleError::Driver(
                "screen buffer read exceeds storage".into(),
            ));
        }
        // SAFETY: the range lies in shared storage; the caller reads only
        // after the bank's command completed and before the next load.
        Ok(unsafe { std::slice::from_raw_parts(contents.add(offset), count) })
    }

    /// The milli energies of model `group`'s reads in `word`.
    pub(crate) fn energies(&self, word: usize, group: usize) -> Result<[i64; READS], SampleError> {
        let lanes = self.read::<i32>(11, word * sampler::MSA_LANES, sampler::MSA_LANES)?;
        Ok(std::array::from_fn(|r| {
            i64::from(lanes[r * GROUPS + group])
        }))
    }

    /// The mask bytes the last dispatch built for `word`, one per CSR slot.
    #[cfg(test)]
    fn masks(&self, word: usize) -> Result<&[u8], SampleError> {
        let nnz = self.cached.topo.nnz;
        self.read::<u8>(2, word * nnz, nnz)
    }

    /// The spins of read `read` of model `group` in `word`.
    pub(crate) fn spins(
        &self,
        word: usize,
        group: usize,
        read: usize,
    ) -> Result<Vec<i8>, SampleError> {
        let packed = self.cached.n.div_ceil(8);
        let lane = word * sampler::MSA_LANES + read * GROUPS + group;
        let bytes = self.read::<i8>(10, lane * packed, packed)?;
        Ok(sampler::unpack_spins(bytes, self.cached.n))
    }
}

/// Words each of a [`Screener`]'s two banks holds: 2,560 models. A
/// 128-sweep word takes about 7 ms on one core; this many 16-sweep words
/// keep the other cores of an M4 Max busy for about as long.
pub(crate) const BANK_WORDS: usize = 320;
/// Longest wait of a model for the rest of its word.
const FLUSH: Duration = Duration::from_millis(2);

struct Entry<T> {
    item: T,
    bits: Vec<u32>,
    queued: Instant,
}

/// What one harvested model did at its screen stage.
pub(crate) enum Verdict<T> {
    /// Failed stage `stage` with best `best`; the model leaves the screen.
    /// `reads` holds its [`READS`] reads when the harvest asked for them.
    Failed {
        item: T,
        stage: usize,
        best: i64,
        reads: Option<Vec<SamplerResult>>,
    },
    /// Passed stage `stage` and waits for the next one.
    Kept { stage: usize },
    /// Passed the last stage with best `best`; the model leaves the screen.
    Passed { item: T, best: i64 },
}

/// Two alternating banks and the models waiting for each stage. Later
/// stages load first, so a kept model never waits behind new ones.
pub(crate) struct Screener<T> {
    banks: [ScreenBank; 2],
    /// Per loaded word of each bank: its stage and models.
    loaded: [Vec<(usize, Vec<Entry<T>>)>; 2],
    command: [Option<metal::CommandBuffer>; 2],
    /// Whether the bank's command holds only the screen dispatch.
    alone: [bool; 2],
    queues: Vec<VecDeque<Entry<T>>>,
    gates: Vec<i64>,
    seed: u32,
}

impl<T> Screener<T> {
    /// Stage `s` anneals `schedules[s]` and keeps a model whose best is at or
    /// below `gates[s]`.
    pub(crate) fn new(
        device: &MetalDevice,
        graph: &IsingGraph,
        schedules: &[&[f32]],
        gates: &[i64],
        draw: Option<&Draw>,
    ) -> Result<Self, SampleError> {
        if schedules.len() != gates.len() {
            return Err(SampleError::Driver("one gate per screen stage".into()));
        }
        Ok(Self {
            banks: [
                ScreenBank::new(device, graph, BANK_WORDS, schedules, draw)?,
                ScreenBank::new(device, graph, BANK_WORDS, schedules, draw)?,
            ],
            loaded: [Vec::new(), Vec::new()],
            command: [None, None],
            alone: [false; 2],
            queues: gates.iter().map(|_| VecDeque::new()).collect(),
            gates: gates.to_vec(),
            seed: 1,
        })
    }

    /// Whether a model row fits this screener.
    pub(crate) fn fits(&self, row: &[u32]) -> bool {
        row.len() == self.banks[0].row_words()
    }

    pub(crate) fn offer(&mut self, item: T, bits: Vec<u32>) {
        self.queues[0].push_back(Entry {
            item,
            bits,
            queued: Instant::now(),
        });
    }

    /// Models waiting for a stage, not counting loaded ones.
    pub(crate) fn queued(&self) -> usize {
        self.queues.iter().map(VecDeque::len).sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.queued() == 0 && self.loaded.iter().all(Vec::is_empty)
    }

    /// Load bank `bank` from the queues and return its word count. Fills
    /// whole words, and a partial one only when its oldest model has
    /// waited [`FLUSH`] or `flush` is set.
    pub(crate) fn load(&mut self, bank: usize, flush: bool) -> Result<usize, SampleError> {
        if self.command[bank].is_some() || !self.loaded[bank].is_empty() {
            return Ok(self.banks[bank].len());
        }
        self.banks[bank].clear();
        for stage in (0..self.queues.len()).rev() {
            while self.banks[bank].len() < self.banks[bank].capacity() {
                let queue = &mut self.queues[stage];
                let ready = queue.len() >= GROUPS
                    || queue
                        .front()
                        .is_some_and(|e| flush || e.queued.elapsed() >= FLUSH);
                if !ready {
                    break;
                }
                let take = queue.len().min(GROUPS);
                let entries: Vec<_> = queue.drain(..take).collect();
                let models: Vec<&[u32]> = entries.iter().map(|e| e.bits.as_slice()).collect();
                self.seed = self
                    .seed
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(0x7f4a_7c15)
                    | 1;
                self.banks[bank].push(stage, self.seed, &models)?;
                self.loaded[bank].push((stage, entries));
            }
        }
        Ok(self.banks[bank].len())
    }

    pub(crate) fn bank(&self, bank: usize) -> &ScreenBank {
        &self.banks[bank]
    }

    /// Record the command that holds bank `bank`'s dispatch.
    pub(crate) fn attach(&mut self, bank: usize, command: metal::CommandBuffer) {
        self.command[bank] = Some(command);
        self.alone[bank] = false;
    }

    /// Commit bank `bank`'s dispatch in a command of its own.
    pub(crate) fn commit(
        &mut self,
        bank: usize,
        queue: &metal::CommandQueue,
    ) -> Result<(), SampleError> {
        let command = queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        self.banks[bank].encode(encoder)?;
        encoder.end_encoding();
        command.commit();
        self.command[bank] = Some(command);
        self.alone[bank] = true;
        Ok(())
    }

    /// Wait for bank `bank`'s command and gate its models. A failed model
    /// for which `wants_reads` returns true carries its reads. Returns the
    /// verdicts and the command's GPU time when it held only the screen.
    pub(crate) fn harvest(
        &mut self,
        bank: usize,
        mut wants_reads: impl FnMut(&T, i64) -> bool,
    ) -> Result<(Vec<Verdict<T>>, u64), SampleError> {
        let Some(command) = self.command[bank].take() else {
            return Ok((Vec::new(), 0));
        };
        command.wait_until_completed();
        if command.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(SampleError::Driver("screen command buffer failed".into()));
        }
        let busy = if self.alone[bank] {
            sampler::gpu_time_us(&command)
        } else {
            0
        };
        let mut verdicts = Vec::new();
        for (word, (stage, entries)) in std::mem::take(&mut self.loaded[bank])
            .into_iter()
            .enumerate()
        {
            for (group, entry) in entries.into_iter().enumerate() {
                let energies = self.banks[bank].energies(word, group)?;
                let best = energies.into_iter().min().unwrap_or(i64::MAX);
                if best > self.gates[stage] {
                    let reads = if wants_reads(&entry.item, best) {
                        let mut reads = Vec::with_capacity(READS);
                        for (read, &energy_milli) in energies.iter().enumerate() {
                            reads.push(SamplerResult {
                                spins: self.banks[bank].spins(word, group, read)?,
                                energy_milli,
                            });
                        }
                        Some(reads)
                    } else {
                        None
                    };
                    verdicts.push(Verdict::Failed {
                        item: entry.item,
                        stage,
                        best,
                        reads,
                    });
                } else if stage + 1 < self.gates.len() {
                    verdicts.push(Verdict::Kept { stage });
                    self.queues[stage + 1].push_back(Entry {
                        queued: Instant::now(),
                        ..entry
                    });
                } else {
                    verdicts.push(Verdict::Passed {
                        item: entry.item,
                        best,
                    });
                }
            }
        }
        self.banks[bank].clear();
        Ok((verdicts, busy))
    }

    /// Every model still in the screen, after its command completes.
    pub(crate) fn drain(&mut self) -> Vec<T> {
        for command in self.command.iter_mut().filter_map(Option::take) {
            command.wait_until_completed();
        }
        let loaded = self.loaded.iter_mut().flat_map(std::mem::take);
        let loaded: Vec<_> = loaded.flat_map(|(_, entries)| entries).collect();
        let queued = self.queues.iter_mut().flat_map(std::mem::take);
        loaded.into_iter().chain(queued).map(|e| e.item).collect()
    }
}

impl Drop for ScreenBank {
    fn drop(&mut self) {
        for (_, buffer) in self.buffers.drain(..) {
            self.pool.give(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::energy_milli;

    fn xorshift64(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// The Advantage2 fixture without self-loops, zero fields and seeded ±1
    /// couplings.
    fn advantage2(seed: u64) -> IsingGraph {
        let edges: Vec<(usize, usize)> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut nodes = line.split_whitespace().map(|v| v.parse::<usize>().unwrap());
                (nodes.next().unwrap(), nodes.next().unwrap())
            })
            .filter(|&(u, v)| u != v)
            .collect();
        let mut s = seed | 1;
        let j = edges
            .iter()
            .map(|_| {
                if xorshift64(&mut s) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        IsingGraph::new(vec![0.0; 4577], j, edges)
    }

    fn units(graph: &IsingGraph) -> Vec<i8> {
        graph.j.iter().map(|&j| j as i8).collect()
    }

    fn run(device: &MetalDevice, bank: &ScreenBank) {
        let command = device.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        bank.encode(encoder).unwrap();
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
    }

    #[test]
    fn sign_bits_mark_negative_couplings() {
        let bits = sign_bits(&[
            -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, -1,
        ]);
        assert_eq!(bits, vec![0b1001, 1]);
    }

    /// Every lane's device energy is its decoded spins' energy under its own
    /// model, for a full word and for a partial one.
    #[test]
    fn lane_energies_match_each_lanes_model() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let device = MetalDevice::open(0).unwrap();
        let graphs: Vec<_> = (0..GROUPS as u64).map(|k| advantage2(k + 11)).collect();
        let bits: Vec<_> = graphs.iter().map(|g| sign_bits(&units(g))).collect();
        let schedule = sampler::build_beta_schedule(
            &graphs[0],
            16,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let mut bank = ScreenBank::new(&device, &graphs[0], 4, &[&schedule], None).unwrap();
        let all: Vec<&[u32]> = bits.iter().map(Vec::as_slice).collect();
        bank.push(0, 7, &all).unwrap();
        bank.push(0, 9, &all[..3]).unwrap();
        run(&device, &bank);
        for (word, models) in [(0, GROUPS), (1, 3)] {
            for (g, graph) in graphs.iter().enumerate().take(models) {
                let energies = bank.energies(word, g).unwrap();
                for (r, &energy) in energies.iter().enumerate() {
                    let spins = bank.spins(word, g, r).unwrap();
                    assert_eq!(
                        energy,
                        energy_milli(&spins, &graph.h, &graph.j, &graph.edges)
                    );
                    // 16 sweeps reach well below a random state.
                    assert!(
                        energy < -10_000_000,
                        "word {word} model {g} read {r}: {energy}"
                    );
                }
            }
        }
    }

    /// In draw mode the device builds, from each lease nonce, the same
    /// couplings as the host draw, bit for bit, and every lane's energy is
    /// its spins' energy under that model.
    #[test]
    fn device_draw_matches_the_host_draw() {
        if MetalDevice::device_count() == 0 {
            return;
        }
        let device = MetalDevice::open(0).unwrap();
        let topology = advantage2(1);
        let n = topology.num_nodes();
        let edges = topology.edges.len();
        let schedule = sampler::build_beta_schedule(
            &topology,
            16,
            1,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        let cached = device.topology_cache.get_or_build(&device, &topology, true);
        for allowed_j in [vec![-1i8, 1], vec![1, -1], vec![-1, 1, 1]] {
            let draw = Draw {
                first_word: n as u32,
                allowed_j: allowed_j.clone(),
            };
            let mut bank =
                ScreenBank::new(&device, &topology, 2, &[&schedule], Some(&draw)).unwrap();
            let nonces: Vec<[u8; 32]> = (0..11u8)
                .map(|k| std::array::from_fn(|i| k.wrapping_mul(37).wrapping_add(i as u8 * 11)))
                .collect();
            let keys: Vec<Vec<u32>> = nonces.iter().map(|&nonce| key_words(nonce)).collect();
            let rows: Vec<&[u32]> = keys.iter().map(Vec::as_slice).collect();
            bank.push(0, 5, &rows[..GROUPS]).unwrap();
            bank.push(0, 6, &rows[GROUPS..]).unwrap();
            run(&device, &bank);
            for (index, &nonce) in nonces.iter().enumerate() {
                let (word, group) = (index / GROUPS, index % GROUPS);
                let (h, j) = crate::draw::draw_units(nonce, n, edges, &[0], &allowed_j).unwrap();
                let masks = bank.masks(word).unwrap();
                for (e, &(ij, ji)) in cached.topo.edge_pos.iter().enumerate() {
                    let negative = j[e] < 0;
                    for slot in [ij, ji] {
                        let bit = (masks[slot as usize] >> group) & 1 == 1;
                        assert_eq!(bit, negative, "set {allowed_j:?} nonce {index} edge {e}");
                    }
                }
                let model = IsingGraph::new(
                    h.iter().map(|&v| f64::from(v)).collect(),
                    j.iter().map(|&v| f64::from(v)).collect(),
                    topology.edges.to_vec(),
                );
                for (r, &energy) in bank.energies(word, group).unwrap().iter().enumerate() {
                    let spins = bank.spins(word, group, r).unwrap();
                    assert_eq!(
                        energy,
                        energy_milli(&spins, &model.h, &model.j, &model.edges)
                    );
                }
            }
        }
    }

    /// Best-of-[`READS`] energies of one screen stage on recorded problems.
    ///
    /// Ignored: needs a Metal device and a problem directory in the
    /// `slots::tests::checkpoint_trace` format. Packs [`GROUPS`] problem runs
    /// per word with the production standard schedule of
    /// `QUIP_TRACE_SWEEPS` sweeps and writes `qblock_id,k,best_<sweeps>_r4`.
    ///
    /// ```text
    /// QUIP_TRACE_PROBLEMS=dir QUIP_TRACE_OUT=screen.csv QUIP_TRACE_SWEEPS=16 \
    ///   QUIP_TRACE_SEEDS=2 cargo test --release --lib screen_trace -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device and QUIP_TRACE_PROBLEMS"]
    #[expect(clippy::print_stderr, reason = "the trace reports progress on stderr")]
    fn screen_trace() {
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
        let sweeps: usize = var("QUIP_TRACE_SWEEPS", "16").parse().unwrap();
        let seeds: u64 = var("QUIP_TRACE_SEEDS", "2").parse().unwrap();
        let index: Vec<Entry> =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/index.json")).unwrap())
                .unwrap();
        let mut runs = Vec::new();
        let mut first: Option<IsingGraph> = None;
        for entry in &index {
            let p: Problem =
                serde_json::from_str(&std::fs::read_to_string(&entry.path).unwrap()).unwrap();
            assert!(
                p.h.iter().all(|&h| h == 0.0),
                "screen models have zero fields"
            );
            let j: Vec<i8> = p.j.iter().map(|&j| j as i8).collect();
            let bits = sign_bits(&j);
            if first.is_none() {
                first = Some(IsingGraph::new(p.h, p.j, p.edges));
            }
            for k in 0..seeds {
                runs.push((entry.qblock_id, k, bits.clone()));
            }
        }
        let graph = first.expect("at least one problem");
        let params = quip_solver_core::SampleParams {
            num_reads: 64,
            num_sweeps: sweeps,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: None,
            seed: 1,
        };
        let (schedule, _, _) = crate::cascade::segment_schedule(&graph, &params, &[]);
        let device = MetalDevice::open(0).expect("Metal device 0");
        let capacity = 256;
        let mut bank = ScreenBank::new(&device, &graph, capacity, &[&schedule], None).unwrap();
        let started = std::time::Instant::now();
        let mut text = format!("qblock_id,k,best_{sweeps}_r{READS}\n");
        for (batch, chunk) in runs.chunks(capacity * GROUPS).enumerate() {
            bank.clear();
            for (w, word) in chunk.chunks(GROUPS).enumerate() {
                let models: Vec<&[u32]> = word.iter().map(|(_, _, b)| b.as_slice()).collect();
                bank.push(
                    0,
                    (batch * capacity + w) as u32 * 2_654_435_761 + 1,
                    &models,
                )
                .unwrap();
            }
            run(&device, &bank);
            for (w, word) in chunk.chunks(GROUPS).enumerate() {
                for (g, (qblock, k, _)) in word.iter().enumerate() {
                    let best = bank.energies(w, g).unwrap().into_iter().min().unwrap();
                    text.push_str(&format!("{qblock},{k},{best}\n"));
                }
            }
        }
        std::fs::write(&out, text).unwrap();
        eprintln!(
            "{} runs in {:.1} s",
            runs.len(),
            started.elapsed().as_secs_f64()
        );
    }

    /// GPU-only screen rate: `QUIP_BENCH_WORDS` words of one stage per
    /// dispatch, back to back for `QUIP_BENCH_SECS` seconds, for each
    /// sweep count in `QUIP_BENCH_SWEEPS`.
    ///
    /// ```text
    /// cargo test --release --lib screen_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device; a benchmark"]
    #[expect(clippy::print_stderr, reason = "the benchmark reports on stderr")]
    fn screen_bench() {
        let var =
            |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_owned());
        let words: usize = var("QUIP_BENCH_WORDS", "160").parse().unwrap();
        let secs: f64 = var("QUIP_BENCH_SECS", "5").parse().unwrap();
        let device = MetalDevice::open(0).unwrap();
        let graphs: Vec<_> = (0..GROUPS as u64).map(|k| advantage2(k + 3)).collect();
        let bits: Vec<_> = graphs.iter().map(|g| sign_bits(&units(g))).collect();
        let models: Vec<&[u32]> = bits.iter().map(Vec::as_slice).collect();
        for sweeps in var("QUIP_BENCH_SWEEPS", "16,128").split(',') {
            let sweeps: usize = sweeps.parse().unwrap();
            let schedule = sampler::build_beta_schedule(
                &graphs[0],
                sweeps,
                1,
                None,
                quip_solver_core::BetaSchedule::Geometric,
            )
            .0;
            let mut bank = ScreenBank::new(&device, &graphs[0], words, &[&schedule], None).unwrap();
            for w in 0..words {
                bank.push(0, w as u32 + 1, &models).unwrap();
            }
            for _ in 0..5 {
                run(&device, &bank);
            }
            let started = std::time::Instant::now();
            let mut dispatches = 0u64;
            let mut gpu = 0.0;
            while started.elapsed().as_secs_f64() < secs {
                let command = device.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                bank.encode(encoder).unwrap();
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                gpu += sampler::gpu_time_us(command) as f64 / 1e6;
                dispatches += 1;
            }
            let wall = started.elapsed().as_secs_f64();
            let salts = (dispatches as usize * words * GROUPS) as f64;
            eprintln!(
                "{sweeps:>4} sweeps x {words} words: {:.2} ms per dispatch, {:.0} salts/s wall, \
                 {:.0} salts/s GPU",
                1000.0 * gpu / dispatches as f64,
                salts / wall,
                salts / gpu
            );
        }
    }
}
