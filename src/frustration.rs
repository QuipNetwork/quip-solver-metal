// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Frustration score: a prediction of a model's annealed energy from its
//! couplings alone, with no anneal.
//!
//! A cycle is frustrated when the product of its couplings is negative, so
//! no spin assignment satisfies every edge on it. The ground-state energy of
//! a model rises with the number of frustrated short cycles. On Advantage2,
//! a linear fit of the 1,024-sweep best energy of 3,000 fresh lease models
//! on the frustrated triangle and 4-cycle counts ranks won qblocks into the
//! best 2 percent of fresh models: all 255 winners of qblocks 154 to 408 and
//! all 67 of qblocks 409 to 475.

/// Energy units per frustrated 4-cycle, from the fit.
pub(crate) const QUAD_WEIGHT: f64 = 0.052_486_9;
/// Energy units per frustrated triangle, from the fit.
pub(crate) const TRI_WEIGHT: f64 = -0.195_177;

/// Every triangle and 4-cycle of a topology, as edge indices.
///
/// The 4-cycles are grouped by diagonal. A diagonal is a node pair `(a, c)`
/// with `a` the smallest node of its cycles, and its paths are the two-edge
/// paths `a - b - c` with `b > a`. Each two paths of a diagonal close one
/// 4-cycle, which is frustrated when exactly one of the two paths has an odd
/// number of negative couplings. A diagonal with `p` odd paths of `k`
/// therefore holds `p * (k - p)` frustrated 4-cycles, and loading its paths
/// once serves all of its cycles.
pub(crate) struct Cycles {
    pub(crate) triangles: Vec<[u32; 3]>,
    /// Each diagonal's paths as `(ab, bc)` edge pairs, two or more a
    /// diagonal, in order of the diagonal's smallest node.
    pub(crate) diagonals: Vec<Vec<[u32; 2]>>,
}

impl Cycles {
    /// Enumerate the cycles of the graph on `num_nodes` nodes and `edges`.
    /// Each cycle appears once: a 4-cycle under the diagonal through its
    /// smallest node, a triangle by its edge with the two smaller nodes.
    pub(crate) fn new(num_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let mut neighbours: Vec<Vec<(usize, u32)>> = vec![Vec::new(); num_nodes];
        for (index, &(u, v)) in edges.iter().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "topologies have far fewer than 2^32 edges"
            )]
            let index = index as u32;
            neighbours[u].push((v, index));
            neighbours[v].push((u, index));
        }
        for list in &mut neighbours {
            list.sort_unstable();
        }
        let edge = |a: usize, b: usize| -> Option<u32> {
            neighbours[a]
                .binary_search_by_key(&b, |&(node, _)| node)
                .ok()
                .map(|at| neighbours[a][at].1)
        };
        let mut triangles = Vec::new();
        for (ab, &(u, v)) in edges.iter().enumerate() {
            let (a, b) = (u.min(v), u.max(v));
            for &(c, bc) in &neighbours[b] {
                if c > b {
                    if let Some(ac) = edge(a, c) {
                        triangles.push([ab as u32, bc, ac]);
                    }
                }
            }
        }
        let mut diagonals = Vec::new();
        for a in 0..num_nodes {
            let mut paths: std::collections::BTreeMap<usize, Vec<[u32; 2]>> =
                std::collections::BTreeMap::new();
            for &(b, ab) in &neighbours[a] {
                if b <= a {
                    continue;
                }
                for &(c, bc) in &neighbours[b] {
                    if c > a {
                        paths.entry(c).or_default().push([ab, bc]);
                    }
                }
            }
            diagonals.extend(paths.into_values().filter(|list| list.len() >= 2));
        }
        Self {
            triangles,
            diagonals,
        }
    }

    /// Number of 4-cycles.
    #[cfg(test)]
    pub(crate) fn num_quads(&self) -> usize {
        self.diagonals
            .iter()
            .map(|d| d.len() * (d.len() - 1) / 2)
            .sum()
    }

    /// Frustrated `(triangles, quads)` of a model with couplings `j`.
    #[cfg(test)]
    pub(crate) fn counts(&self, j: &[i8]) -> (u32, u32) {
        let negative = |e: u32| u32::from(j[e as usize] < 0);
        let odd = |edges: &[u32]| edges.iter().map(|&e| negative(e)).sum::<u32>() & 1;
        let triangles = self.triangles.iter().map(|t| odd(t)).sum();
        let quads = self
            .diagonals
            .iter()
            .map(|paths| {
                let p = paths.iter().map(|path| odd(path)).sum::<u32>();
                p * (paths.len() as u32 - p)
            })
            .sum();
        (triangles, quads)
    }
}

/// Whether a model with couplings `j` passes stage F on `cycles`, the
/// chain gate [`crate::cascade::ChainGates::frustration`].
#[cfg(test)]
pub(crate) fn passes(cycles: &Cycles, j: &[i8]) -> bool {
    let (triangles, quads) = cycles.counts(j);
    score(triangles, quads) <= crate::cascade::CHAIN_GATES.frustration
}

/// Models per frustration threadgroup: one per SIMD lane.
pub(crate) const GROUP: usize = 32;
/// Groups a count threadgroup takes together; matches `FRUSTRATION_WIDTH`.
const WIDTH: usize = 4;
/// Paths a diagonal may hold; matches `FRUSTRATION_MAX_PATHS` in the kernel.
const MAX_PATHS: usize = 16;
/// Threads per draw threadgroup.
const DRAW_THREADS: u64 = 64;
/// Threads per count threadgroup.
const THREADS: u64 = 256;

/// Device buffers that count frustrated cycles of up to `capacity` lease
/// models per dispatch, drawing each model from its ChaCha8 key.
pub(crate) struct FrustrationBank {
    draw_pipeline: metal::ComputePipelineState,
    count_pipeline: metal::ComputePipelineState,
    diagonal_starts: metal::Buffer,
    paths: metal::Buffer,
    triangles: metal::Buffer,
    keys: metal::Buffer,
    signs: metal::Buffer,
    counts: metal::Buffer,
    num_diagonals: u32,
    num_triangles: u32,
    num_edges: u32,
    draw: crate::screen::ScreenDraw,
    capacity: usize,
    loaded: usize,
}

impl FrustrationBank {
    pub(crate) fn new(
        device: &crate::metal_device::MetalDevice,
        cycles: &Cycles,
        num_edges: usize,
        draw: &crate::screen::Draw,
        capacity: usize,
    ) -> Result<Self, crate::sampler::SampleError> {
        if num_edges > usize::from(u16::MAX) + 1
            || !(1..=32).contains(&draw.allowed_j.len())
            || cycles.diagonals.iter().any(|d| d.len() > MAX_PATHS)
        {
            return Err(crate::sampler::SampleError::TooLarge(
                "frustration needs edge indices under 65,536, 1 to 32 couplings and at most 32 \
                 paths a diagonal"
                    .into(),
            ));
        }
        let capacity = capacity.div_ceil(GROUP * WIDTH) * GROUP * WIDTH;
        // Shortest diagonals first, so neighbouring threads pair the same
        // number of paths.
        let mut order: Vec<&Vec<[u32; 2]>> = cycles.diagonals.iter().collect();
        order.sort_by_key(|d| d.len());
        let mut diagonal_starts = vec![0u32];
        let mut paths: Vec<[u16; 2]> = Vec::new();
        for diagonal in order {
            paths.extend(diagonal.iter().map(|&[ab, bc]| [ab as u16, bc as u16]));
            diagonal_starts.push(paths.len() as u32);
        }
        let triangles: Vec<[u16; 4]> = cycles
            .triangles
            .iter()
            .map(|&[a, b, c]| [a as u16, b as u16, c as u16, 0])
            .collect();
        let groups = (capacity / GROUP) as u64;
        Ok(Self {
            draw_pipeline: device.msa_frustration_draw.clone(),
            count_pipeline: device.msa_frustration_count.clone(),
            diagonal_starts: device.new_buffer_from_slice(&diagonal_starts),
            paths: device.new_buffer_from_slice(&paths),
            triangles: device.new_buffer_from_slice(&triangles),
            keys: device.new_zeroed_buffer(capacity as u64 * 32),
            signs: device.new_zeroed_buffer(groups * num_edges as u64 * 4),
            counts: device.new_zeroed_buffer(capacity as u64 * 8),
            num_diagonals: cycles.diagonals.len() as u32,
            num_triangles: cycles.triangles.len() as u32,
            num_edges: num_edges as u32,
            draw: draw.device(),
            capacity,
            loaded: 0,
        })
    }

    /// Load the ChaCha8 keys of the next dispatch, at most `capacity`.
    pub(crate) fn load(&mut self, keys: &[[u32; 8]]) {
        assert!(
            keys.len() <= self.capacity,
            "frustration keys exceed the bank"
        );
        // SAFETY: shared storage holds `capacity` keys, and the caller loads
        // only while no dispatch of this bank is in flight.
        let at = unsafe {
            std::slice::from_raw_parts_mut(self.keys.contents().cast::<[u32; 8]>(), self.capacity)
        };
        at[..keys.len()].copy_from_slice(keys);
        self.loaded = keys.len();
    }

    /// Encode one dispatch over the loaded keys.
    pub(crate) fn encode(&self, encoder: &metal::ComputeCommandEncoderRef) {
        if self.loaded == 0 {
            return;
        }
        let groups = self.loaded.div_ceil(GROUP) as u32;
        for (index, buffer) in [
            (0, &self.keys),
            (1, &self.signs),
            (4, &self.diagonal_starts),
            (5, &self.paths),
            (6, &self.triangles),
            (8, &self.counts),
        ] {
            encoder.set_buffer(index, Some(buffer), 0);
        }
        for (index, value) in [
            (2, self.num_diagonals),
            (3, self.num_triangles),
            (10, self.num_edges),
            (11, groups),
        ] {
            encoder.set_bytes(index, 4, (&value as *const u32).cast());
        }
        encoder.set_bytes(
            7,
            std::mem::size_of::<crate::screen::ScreenDraw>() as u64,
            (&self.draw as *const crate::screen::ScreenDraw).cast(),
        );
        encoder.set_compute_pipeline_state(&self.draw_pipeline);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(groups), 1, 1),
            metal::MTLSize::new(DRAW_THREADS, 1, 1),
        );
        encoder.memory_barrier_with_resources(&[&self.signs]);
        if self.num_diagonals + self.num_triangles > 0 {
            encoder.set_compute_pipeline_state(&self.count_pipeline);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(u64::from(groups).div_ceil(WIDTH as u64), 1, 1),
                metal::MTLSize::new(THREADS, 1, 1),
            );
        }
    }

    /// `(triangles, quads)` frustrated in each loaded model, after the
    /// dispatch completed.
    pub(crate) fn counts(&self) -> &[[u32; 2]] {
        // SAFETY: shared storage holds `capacity` count pairs, read after the
        // dispatch completed and before the next load.
        unsafe {
            std::slice::from_raw_parts(self.counts.contents().cast::<[u32; 2]>(), self.loaded)
        }
    }
}

/// Salts a prefilter dispatch scores.
const PREFILTER_CHUNK: usize = 32_768;

/// Prefilter lanes. Each visit waits on its lane's command, so the lanes in
/// flight, not the device, bound the rate: beside the live miner a lease
/// fed through 2 lanes ran at 0.9 million salts a second while a second
/// process scored 1.35 million on the same device.
const PREFILTER_LANES: usize = 4;

/// Stage F for lease salts: each salt's model is drawn and scored on the
/// device, and only salts at or below the gate go on to the screen.
/// [`PREFILTER_LANES`] banks take chunks in turn, so the host hashes one
/// chunk while the device runs the others.
pub(crate) struct Prefilter {
    queue: metal::CommandQueue,
    lanes: Vec<FrustrationBank>,
    gate: f64,
    chunk: usize,
}

/// A chunk in a prefilter lane: its first salt, its size, and its command.
struct Chunk {
    first: u64,
    len: usize,
    command: metal::CommandBuffer,
}

impl Prefilter {
    /// The prefilter for lease salts drawn on `view`, or `None` unless `view`
    /// is the chain topology with zero fields and couplings of ±1.
    pub(crate) fn for_view(
        device: &crate::metal_device::MetalDevice,
        view: &quip_solver_core::quip_protocol::lease::TopologyView,
    ) -> Option<Result<Self, crate::sampler::SampleError>> {
        let gates = &crate::cascade::CHAIN_GATES;
        let mut couplings = view.allowed_j_milli.clone();
        couplings.sort_unstable();
        if crate::cascade::topology_fingerprint(view.num_nodes, &view.edges) != gates.fingerprint
            || view.allowed_h_milli != [0]
            || couplings != [-1000, 1000]
        {
            return None;
        }
        let draw = crate::screen::Draw {
            first_word: u32::try_from(view.num_nodes).ok()?,
            allowed_j: view
                .allowed_j_milli
                .iter()
                .map(|&j| (j / 1000) as i8)
                .collect(),
        };
        let cycles = Cycles::new(view.num_nodes, &view.edges);
        Some(Self::new(
            device,
            &cycles,
            view.edges.len(),
            gates.frustration,
            &draw,
            PREFILTER_CHUNK,
        ))
    }

    fn new(
        device: &crate::metal_device::MetalDevice,
        cycles: &Cycles,
        num_edges: usize,
        gate: f64,
        draw: &crate::screen::Draw,
        chunk: usize,
    ) -> Result<Self, crate::sampler::SampleError> {
        Ok(Self {
            queue: device.device.new_command_queue(),
            lanes: (0..PREFILTER_LANES)
                .map(|_| FrustrationBank::new(device, cycles, num_edges, draw, chunk))
                .collect::<Result<_, _>>()?,
            gate,
            chunk,
        })
    }

    /// Score the salts `indices` with nonces from `nonce`, and call `each`
    /// with every salt index, in order, and whether the salt passed. Stops
    /// early when `each` returns false.
    pub(crate) fn run(
        &mut self,
        indices: std::ops::Range<u64>,
        nonce: impl Fn(u64) -> [u8; 32] + Sync,
        mut each: impl FnMut(u64, bool) -> bool,
    ) -> Result<(), crate::sampler::SampleError> {
        let mut start = indices.start;
        let mut chunks: Vec<Option<Chunk>> = Vec::with_capacity(self.lanes.len());
        for lane in 0..self.lanes.len() {
            chunks.push(self.commit(lane, &mut start, indices.end, &nonce));
        }
        // The lanes take chunks in turn, so chunks finish, and report, in
        // salt order.
        let mut lane = 0;
        while chunks.iter().any(Option::is_some) {
            if let Some(chunk) = chunks[lane].take() {
                let passed = self.finish(lane, &chunk)?;
                for (offset, &kept) in passed.iter().enumerate() {
                    if !each(chunk.first + offset as u64, kept) {
                        for chunk in chunks.iter().flatten() {
                            chunk.command.wait_until_completed();
                        }
                        return Ok(());
                    }
                }
                chunks[lane] = self.commit(lane, &mut start, indices.end, &nonce);
            }
            lane = (lane + 1) % chunks.len();
        }
        Ok(())
    }

    /// Hash the next chunk of salts from `start`, load it into the lane's
    /// bank and commit it, or `None` once `start` reaches `end`.
    fn commit(
        &mut self,
        lane: usize,
        start: &mut u64,
        end: u64,
        nonce: &(impl Fn(u64) -> [u8; 32] + Sync),
    ) -> Option<Chunk> {
        use rayon::prelude::*;
        if *start >= end {
            return None;
        }
        let first = *start;
        *start = (first + self.chunk as u64).min(end);
        let keys: Vec<[u32; 8]> = (first..*start)
            .into_par_iter()
            .map(|index| {
                let bytes = nonce(index);
                let words = bytes.as_chunks::<4>().0;
                std::array::from_fn(|w| u32::from_le_bytes(words[w]))
            })
            .collect();
        let bank = &mut self.lanes[lane];
        bank.load(&keys);
        let command = self.queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        bank.encode(encoder);
        encoder.end_encoding();
        command.commit();
        Some(Chunk {
            first,
            len: keys.len(),
            command,
        })
    }

    /// Wait for the chunk's command and report which of its salts are at or
    /// below the gate.
    fn finish(&self, lane: usize, chunk: &Chunk) -> Result<Vec<bool>, crate::sampler::SampleError> {
        chunk.command.wait_until_completed();
        if chunk.command.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(crate::sampler::SampleError::Driver(format!(
                "prefilter command ended {:?}",
                chunk.command.status()
            )));
        }
        let counts = self.lanes[lane].counts();
        debug_assert_eq!(counts.len(), chunk.len);
        Ok(counts
            .iter()
            .map(|&[triangles, quads]| score(triangles, quads) <= self.gate)
            .collect())
    }
}

/// Predicted energy, in units, up to a constant: lower ranks deeper.
pub(crate) fn score(triangles: u32, quads: u32) -> f64 {
    TRI_WEIGHT * f64::from(triangles) + QUAD_WEIGHT * f64::from(quads)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel's widths match the host's.
    #[test]
    fn kernel_width_matches() {
        let source = include_str!("../kernels/msa.metal");
        assert!(source.contains(&format!("#define FRUSTRATION_WIDTH {WIDTH}\n")));
        assert!(source.contains(&format!("#define FRUSTRATION_MAX_PATHS {MAX_PATHS}\n")));
    }

    /// K4 has four triangles and three 4-cycles.
    /// K4 has four triangles and three 4-cycles.
    #[test]
    fn k4_cycles_and_frustration() {
        let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let cycles = Cycles::new(4, &edges);
        assert_eq!(cycles.triangles.len(), 4);
        assert_eq!(cycles.num_quads(), 3);
        assert_eq!(cycles.counts(&[1; 6]), (0, 0));
        // One negative edge, (0, 1): two triangles and two 4-cycles hold it.
        assert_eq!(cycles.counts(&[-1, 1, 1, 1, 1, 1]), (2, 2));
    }

    /// The Advantage2 working graph the chain mines on.
    pub(crate) fn advantage2_edges() -> Vec<(usize, usize)> {
        let edges: Vec<(usize, usize)> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().parse().unwrap(),
                    words.next().unwrap().parse().unwrap(),
                )
            })
            .filter(|&edge| edge != (880, 2695))
            .collect();
        assert_eq!(edges.len(), 41_514);
        edges
    }

    fn test_nonce(index: u64) -> [u8; 32] {
        // A BLAKE3 nonce is uniform; splitmix64 words stand in for it.
        let mut state = index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5a5a;
        let mut nonce = [0u8; 32];
        for chunk in nonce.chunks_mut(8) {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
        }
        nonce
    }

    fn advantage2_bank(
        device: &crate::metal_device::MetalDevice,
        capacity: usize,
    ) -> (Vec<(usize, usize)>, Cycles, FrustrationBank) {
        let edges = advantage2_edges();
        let cycles = Cycles::new(4577, &edges);
        let draw = crate::screen::Draw {
            first_word: 4577,
            allowed_j: vec![-1, 1],
        };
        let bank = FrustrationBank::new(device, &cycles, edges.len(), &draw, capacity).unwrap();
        (edges, cycles, bank)
    }

    fn keys(range: std::ops::Range<u64>) -> Vec<[u32; 8]> {
        range
            .map(|i| crate::screen::key_words(test_nonce(i)).try_into().unwrap())
            .collect()
    }

    /// The device draw and count give the host's counts for every model,
    /// including a partial last group.
    #[test]
    fn device_counts_match_the_host() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let (edges, cycles, mut bank) = advantage2_bank(&device, 96);
        bank.load(&keys(0..70));
        let command = device.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        bank.encode(encoder);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        for (i, got) in bank.counts().iter().enumerate() {
            let (_, j) =
                crate::draw::draw_units(test_nonce(i as u64), 4577, edges.len(), &[0], &[-1, 1])
                    .unwrap();
            let (t, q) = cycles.counts(&j);
            assert_eq!(*got, [t, q], "model {i}");
        }
    }

    /// The prefilter keeps exactly the salts that pass stage F on the host,
    /// and reports every salt once, in order, across chunks and lanes.
    #[test]
    fn prefilter_keeps_the_salts_stage_f_passes() {
        if crate::metal_device::MetalDevice::device_count() == 0 {
            return;
        }
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let edges = advantage2_edges();
        let cycles = Cycles::new(4577, &edges);
        let draw = crate::screen::Draw {
            first_word: 4577,
            allowed_j: vec![-1, 1],
        };
        let gate = crate::cascade::CHAIN_GATES.frustration;
        let mut prefilter = Prefilter::new(&device, &cycles, edges.len(), gate, &draw, 96).unwrap();
        let salts = 0..1_000u64;
        let mut seen = Vec::new();
        prefilter
            .run(salts.clone(), test_nonce, |index, kept| {
                seen.push((index, kept));
                true
            })
            .unwrap();
        let expected: Vec<(u64, bool)> = salts
            .map(|index| {
                let (_, j) =
                    crate::draw::draw_units(test_nonce(index), 4577, edges.len(), &[0], &[-1, 1])
                        .unwrap();
                (index, passes(&cycles, &j))
            })
            .collect();
        assert_eq!(seen, expected);
        // The sample must exercise both outcomes of the gate.
        let passed = expected.iter().filter(|(_, kept)| *kept).count();
        assert!(passed > 0 && passed < expected.len(), "{passed} passed");
    }

    /// End-to-end stage F rate over `QUIP_PREFILTER_SALTS` test salts, with
    /// the production gate and chunk.
    ///
    /// ```text
    /// QUIP_PREFILTER_SALTS=2000000 \
    ///   cargo test --release --lib prefilter_rate -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device; a benchmark"]
    #[expect(clippy::print_stderr, reason = "the benchmark reports on stderr")]
    fn prefilter_rate() {
        let salts: u64 =
            std::env::var("QUIP_PREFILTER_SALTS").map_or(1_000_000, |v| v.parse().unwrap());
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let edges = advantage2_edges();
        let cycles = Cycles::new(4577, &edges);
        let draw = crate::screen::Draw {
            first_word: 4577,
            allowed_j: vec![-1, 1],
        };
        let gate = crate::cascade::CHAIN_GATES.frustration;
        let mut prefilter =
            Prefilter::new(&device, &cycles, edges.len(), gate, &draw, PREFILTER_CHUNK).unwrap();
        let started = std::time::Instant::now();
        let mut kept = 0u64;
        prefilter
            .run(0..salts, test_nonce, |_, pass| {
                kept += u64::from(pass);
                true
            })
            .unwrap();
        let wall = started.elapsed().as_secs_f64();
        eprintln!(
            "{salts} salts: {:.0} salts/s, {kept} kept ({:.3}%)",
            salts as f64 / wall,
            100.0 * kept as f64 / salts as f64
        );
    }

    /// Device frustration rate over `QUIP_FRUSTRATION_BANK` models a dispatch.
    /// `QUIP_FRUSTRATION_DRAW_ONLY` skips the count, and
    /// `QUIP_FRUSTRATION_TRI_ONLY` counts triangles only.
    ///
    /// ```text
    /// cargo test --release --lib frustration_rate -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Metal device; a benchmark"]
    #[expect(clippy::print_stderr, reason = "the benchmark reports on stderr")]
    fn frustration_rate() {
        let capacity: usize =
            std::env::var("QUIP_FRUSTRATION_BANK").map_or(16_384, |v| v.parse().unwrap());
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let (_, _, mut bank) = advantage2_bank(&device, capacity);
        if std::env::var("QUIP_FRUSTRATION_DRAW_ONLY").is_ok() {
            (bank.num_diagonals, bank.num_triangles) = (0, 0);
        }
        if std::env::var("QUIP_FRUSTRATION_TRI_ONLY").is_ok() {
            bank.num_diagonals = 0;
        }
        bank.load(&keys(0..capacity as u64));
        let rounds = 10;
        let started = std::time::Instant::now();
        let mut gpu = 0.0;
        for _ in 0..rounds {
            let command = device.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            bank.encode(encoder);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            gpu += crate::sampler::gpu_time_us(command) as f64 / 1e6;
        }
        let wall = started.elapsed().as_secs_f64();
        let models = (rounds * capacity) as f64;
        eprintln!(
            "{capacity} models a dispatch: {:.0} models/s wall, {:.0} models/s GPU",
            models / wall,
            models / gpu
        );
    }

    /// Frustration counts of `QUIP_FRUSTRATION_NONCES` random lease nonces,
    /// one CSV row each, to `QUIP_FRUSTRATION_OUT`.
    ///
    /// ```text
    /// QUIP_FRUSTRATION_NONCES=100000 QUIP_FRUSTRATION_OUT=counts.csv \
    ///   cargo test --release --lib frustration_sample -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "a calibration sample"]
    fn frustration_sample() {
        use rayon::prelude::*;
        let edges = advantage2_edges();
        let cycles = Cycles::new(4577, &edges);
        assert_eq!(
            (cycles.triangles.len(), cycles.num_quads()),
            (34_093, 893_326)
        );
        let total: u64 =
            std::env::var("QUIP_FRUSTRATION_NONCES").map_or(10_000, |v| v.parse().unwrap());
        let rows: Vec<String> = (0..total)
            .into_par_iter()
            .map(|index| {
                let nonce = test_nonce(index);
                let (_, j) =
                    crate::draw::draw_units(nonce, 4577, edges.len(), &[0], &[-1, 1]).unwrap();
                let (t, q) = cycles.counts(&j);
                format!("{index},{t},{q}")
            })
            .collect();
        std::fs::write(
            std::env::var("QUIP_FRUSTRATION_OUT").unwrap(),
            format!("index,frus_tri,frus_quad\n{}\n", rows.join("\n")),
        )
        .unwrap();
    }
}
