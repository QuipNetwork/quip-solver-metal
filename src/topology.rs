//! CSR topology + chromatic color-blocks + int8 quantization for the
//! self-feeding kernels.
//!
//! Mirrors `GPU/sampler_utils.py::build_csr_structure_from_edges` /
//! `build_edge_position_index` / `compute_color_blocks`, but computes a
//! generic greedy coloring by default (`build_with_advantage2_coloring` uses
//! the MSA four-colouring when the graph is Advantage2). The candidate maps the
//! audited Advantage2 System 1 compact labels back to physical labels and
//! validates its four-color partition against every supplied edge.
//!
//! Consensus `h`/`J` are constrained to small integers by protocol design
//! (`DEFAULT_ALLOWED_H = {-1,0,1}`, `DEFAULT_ALLOWED_J = {-1,1}`, milli
//! units; see `shared/quantum_proof_of_work.py`), so the int8 cast the
//! original kernel relies on is lossless for real jobs.

use quip_solver_core::IsingGraph;

#[cfg(test)]
thread_local! {
    static PAIR_SCAN_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_pair_scan_count() {
    PAIR_SCAN_COUNT.set(0);
}

#[cfg(test)]
pub(crate) fn pair_scan_count() -> usize {
    PAIR_SCAN_COUNT.get()
}

/// Whether `graph` can use an `N * N` dense coupling matrix: no self-loop,
/// and no pair of nodes coupled twice, in either orientation. A dense matrix
/// holds one coefficient per pair, so a repeat would lose a term that CSR
/// keeps.
///
/// The scan allocates one bit per matrix cell. Callers run it only when
/// dense storage is already the cheaper choice, so the bitset stays small
/// beside the CSR it replaces.
pub(crate) fn dense_candidate(graph: &IsingGraph) -> bool {
    #[cfg(test)]
    PAIR_SCAN_COUNT.set(PAIR_SCAN_COUNT.get() + 1);
    let n = graph.num_nodes();
    let Some(cells) = n.checked_mul(n) else {
        return false;
    };
    let mut seen = vec![0u8; cells.div_ceil(8)];
    for &(u, v) in graph.edges.iter() {
        if u == v || u >= n || v >= n {
            return false;
        }
        let pair = u.min(v) * n + u.max(v);
        let (byte, bit) = (pair / 8, 1u8 << (pair % 8));
        if seen[byte] & bit != 0 {
            return false;
        }
        seen[byte] |= bit;
    }
    true
}

/// Chromatic color-block partition of a CSR graph's dense node indices.
///
/// `nodes` is grouped by color; `starts`/`counts` index into it per color.
/// Same-color nodes are pairwise non-adjacent (independent set), which is
/// all the kernel's per-color parallel update requires.
///
/// Fields are public because the color-block layout is this type's contract:
/// integration tests (and kernel-side consumers) read `starts`/`counts`/
/// `nodes`/`num_colors` directly to assert coloring invariants.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::IsingGraph;
/// use quip_solver_metal::topology::SelfFeedingTopology;
///
/// let graph = IsingGraph::new(
///     vec![1.0, -1.0, 0.0, 1.0],
///     vec![1.0, -1.0, 1.0, -1.0],
///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
/// );
/// let c = &SelfFeedingTopology::build(&graph).colors;
/// assert_eq!(c.starts.len(), c.num_colors as usize);
/// assert_eq!(c.counts.len(), c.num_colors as usize);
/// assert_eq!(c.nodes.len(), 4);
/// assert_eq!(c.counts.iter().sum::<i32>(), 4);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorBlocks {
    /// Offset into [`Self::nodes`] for each color (`starts.len() == num_colors`).
    pub starts: Vec<i32>,
    /// Node count per color (`counts.len() == num_colors`).
    pub counts: Vec<i32>,
    /// Dense node indices grouped by color.
    pub nodes: Vec<i32>,
    /// Number of colors in the partition.
    pub num_colors: i32,
}

/// Greedy (Welsh-Powell) coloring of a CSR adjacency: process nodes in
/// degree-descending order, assign the smallest color unused by any
/// already-colored neighbor. Not the Zephyr-optimal 4-coloring, but valid
/// for any graph and typically close to it for sparse Ising topologies.
fn greedy_color(n: usize, row_ptr: &[i32], col_ind: &[i32]) -> ColorBlocks {
    if n == 0 {
        return ColorBlocks {
            starts: Vec::new(),
            counts: Vec::new(),
            nodes: Vec::new(),
            num_colors: 0,
        };
    }
    let degree = |i: usize| (row_ptr[i + 1] - row_ptr[i]) as usize;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(degree(i)));

    let mut color_of = vec![-1i32; n];
    let mut used = vec![false; n]; // reused scratch, cleared per node
    for &node in &order {
        let start = row_ptr[node] as usize;
        let end = row_ptr[node + 1] as usize;
        let mut touched: Vec<usize> = Vec::with_capacity(end - start);
        for &nbr in &col_ind[start..end] {
            let c = color_of[nbr as usize];
            if c >= 0 {
                used[c as usize] = true;
                touched.push(c as usize);
            }
        }
        let mut c = 0usize;
        while c < n && used[c] {
            c += 1;
        }
        color_of[node] = c as i32;
        for t in touched {
            used[t] = false;
        }
    }

    color_blocks(&color_of)
}

fn color_blocks(color_of: &[i32]) -> ColorBlocks {
    let n = color_of.len();
    let num_colors = color_of.iter().copied().max().unwrap_or(-1) + 1;
    let mut groups: Vec<Vec<i32>> = vec![Vec::new(); num_colors.max(0) as usize];
    for (i, &c) in color_of.iter().enumerate() {
        groups[c as usize].push(i as i32);
    }
    let mut starts = Vec::with_capacity(groups.len());
    let mut counts = Vec::with_capacity(groups.len());
    let mut nodes = Vec::with_capacity(n);
    let mut cur = 0i32;
    for g in &groups {
        starts.push(cur);
        counts.push(g.len() as i32);
        nodes.extend_from_slice(g);
        cur += g.len() as i32;
    }
    ColorBlocks {
        starts,
        counts,
        nodes,
        num_colors,
    }
}

// Physical labels absent from the 2026-06-09 Advantage2 System 1 snapshot.
// Compact IDs are positions in the sorted remaining labels, not Zephyr labels.
// Source: quip-coordinator fixtures/drive/advantage2-system1.spec.json.
// Source SHA-256: f72af98b2bd6c1217d6d3c6389b3ba3e2790274d9e97822945c9f65b701ff4d3.
const ADVANTAGE2_MISSING_LABELS: [usize; 223] = [
    21, 31, 76, 91, 93, 169, 181, 215, 234, 239, 249, 285, 316, 327, 328, 340, 348, 351, 354, 363,
    364, 370, 373, 376, 380, 381, 400, 406, 441, 451, 465, 484, 495, 496, 510, 518, 530, 534, 544,
    555, 556, 568, 570, 577, 585, 623, 630, 738, 753, 762, 769, 780, 790, 801, 817, 863, 880, 918,
    972, 1008, 1018, 1020, 1031, 1139, 1173, 1218, 1268, 1276, 1281, 1283, 1308, 1395, 1413, 1436,
    1537, 1549, 1590, 1598, 1674, 1702, 1842, 1844, 1845, 1866, 1871, 1895, 1913, 1979, 1993, 2002,
    2004, 2049, 2072, 2111, 2123, 2149, 2171, 2230, 2355, 2377, 2386, 2403, 2491, 2516, 2550, 2587,
    2612, 2641, 2642, 2659, 2680, 2682, 2756, 2758, 2759, 2782, 2795, 2838, 2911, 2912, 2926, 2927,
    2940, 3060, 3108, 3110, 3112, 3122, 3198, 3205, 3206, 3212, 3218, 3225, 3240, 3252, 3253, 3264,
    3265, 3266, 3267, 3268, 3277, 3278, 3280, 3281, 3289, 3290, 3298, 3309, 3312, 3325, 3382, 3434,
    3437, 3458, 3467, 3534, 3545, 3546, 3551, 3651, 3698, 3705, 3722, 3792, 3796, 3827, 3888, 3953,
    3961, 3985, 3997, 4058, 4071, 4082, 4083, 4119, 4120, 4121, 4132, 4133, 4134, 4155, 4172, 4177,
    4186, 4189, 4199, 4202, 4220, 4226, 4237, 4259, 4270, 4279, 4334, 4350, 4374, 4386, 4388, 4393,
    4412, 4424, 4447, 4450, 4476, 4508, 4575, 4579, 4604, 4608, 4641, 4650, 4684, 4686, 4720, 4723,
    4725, 4761, 4768, 4778, 4780,
];

/// A candidate partition only: acceptance proves edge independence, not that
/// an unfamiliar graph belongs to this hardware family. Coefficients do not
/// affect the validation, including edges whose current coupling is zero.
fn advantage2_color(graph: &IsingGraph) -> Option<ColorBlocks> {
    if graph.h.len() != 4577 {
        return None;
    }
    let mut missing = ADVANTAGE2_MISSING_LABELS.iter().peekable();
    let mut color_of = Vec::with_capacity(4577);
    for physical in 0..4800 {
        if missing.peek() == Some(&&physical) {
            missing.next();
            continue;
        }
        let z = physical % 12;
        let j = physical / 12 % 2;
        let w = physical / (12 * 2 * 4) % 25;
        let u = physical / (12 * 2 * 4 * 25);
        color_of.push((j + ((w + 2 * (z + u) + j) & 2)) as i32);
    }
    if graph
        .edges
        .iter()
        .any(|&(u, v)| u >= color_of.len() || v >= color_of.len() || color_of[u] == color_of[v])
    {
        return None;
    }
    Some(color_blocks(&color_of))
}

/// Fixed CSR topology shared by every nonce/slot in a self-feeding session.
///
/// Built once from the first job's graph. Subsequent jobs must supply the
/// exact same `(n, edges)` (checked by the caller via [`IsingGraph`]
/// equality) to reuse it; `edge_pos` gives each edge's two CSR positions in
/// that fixed order, so per-job `J` upload is a direct scatter with no
/// per-job graph traversal.
///
/// Fields are public because the CSR layout is this type's contract:
/// integration tests read `n`/`nnz`/`row_ptr`/`col_ind`/`edge_pos`/`colors`
/// to check shape and coloring invariants. Do not narrow to `pub(crate)`.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::IsingGraph;
/// use quip_solver_metal::topology::SelfFeedingTopology;
///
/// let graph = IsingGraph::new(
///     vec![0.0, 0.0],
///     vec![1.0],
///     vec![(0, 1)],
/// );
/// let t = SelfFeedingTopology::build(&graph);
/// assert_eq!(t.n, 2);
/// assert_eq!(t.nnz, 2); // one undirected edge → two directed halves
/// assert_eq!(t.row_ptr.len(), t.n + 1);
/// assert_eq!(t.col_ind.len(), t.nnz);
/// assert_eq!(t.edge_pos.len(), graph.edges.len());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelfFeedingTopology {
    /// Number of nodes (`graph.h.len()` of the establishing graph).
    pub n: usize,
    /// Number of directed CSR half-edges (`col_ind.len()`).
    pub nnz: usize,
    /// CSR row pointer, length `n + 1`.
    pub row_ptr: Vec<i32>,
    /// CSR column indices, length `nnz`.
    pub col_ind: Vec<i32>,
    /// Per-edge `(pos_ij, pos_ji)` into `col_ind`/`j` arrays, parallel to the
    /// establishing graph's `edges` order.
    pub edge_pos: Vec<(u32, u32)>,
    /// Chromatic color-block partition of the dense node indices.
    pub colors: ColorBlocks,
}

impl SelfFeedingTopology {
    /// Try the validated System 1 candidate, falling back to the default
    /// partition on incompatible input. CSR layout and coefficient order stay
    /// identical. A different partition changes seeded annealing trajectories.
    pub(crate) fn build_with_advantage2_coloring(graph: &IsingGraph) -> Self {
        let mut topology = Self::build(graph);
        if let Some(colors) = advantage2_color(graph) {
            topology.colors = colors;
        }
        topology
    }

    /// Build CSR + coloring from a graph. `graph.edges` fixes the canonical
    /// edge order used by `edge_pos` (and thus by [`fill_h_j`] for this and
    /// every later job sharing this topology).
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_solver_metal::IsingGraph;
    /// use quip_solver_metal::topology::SelfFeedingTopology;
    ///
    /// let graph = IsingGraph::new(
    ///     vec![1.0, -1.0, 0.0, 1.0],
    ///     vec![1.0, -1.0, 1.0, -1.0],
    ///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
    /// );
    /// let t = SelfFeedingTopology::build(&graph);
    /// assert_eq!(t.n, 4);
    /// assert_eq!(t.nnz, 8); // 4 undirected edges × 2 directed halves
    /// assert_eq!(t.row_ptr, vec![0, 2, 4, 6, 8]);
    /// ```
    pub fn build(graph: &IsingGraph) -> Self {
        let n = graph.h.len();
        // Per-node list of (neighbor, edge_index, is_forward_half): carries
        // the originating `graph.edges[edge_index]` through the sort so the
        // final CSR position can be written straight into `edge_pos` below,
        // with no post-hoc search for "where did this edge end up".
        let mut adj: Vec<Vec<(usize, usize, bool)>> = vec![Vec::new(); n];
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            adj[u].push((v, k, true));
            if u != v {
                adj[v].push((u, k, false));
            }
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable_by_key(|&(nbr, _, _)| nbr);
        }

        let mut row_ptr = vec![0i32; n + 1];
        let mut col_ind = Vec::new();
        // (0, 0) for an edge with an out-of-range endpoint: never read,
        // since `fill_h_j` skips those edges too (matches the guard above).
        let mut edge_pos = vec![(0u32, 0u32); graph.edges.len()];
        // `zip` stops at `adj` (length `n`), so the trailing `row_ptr[n]` is
        // left for the explicit terminator write below.
        for (row, nbrs) in row_ptr.iter_mut().zip(&adj) {
            *row = col_ind.len() as i32;
            for &(nbr, k, is_forward) in nbrs {
                let pos = col_ind.len() as u32;
                col_ind.push(nbr as i32);
                if is_forward {
                    edge_pos[k].0 = pos;
                } else {
                    edge_pos[k].1 = pos;
                }
            }
        }
        row_ptr[n] = col_ind.len() as i32;
        let nnz = col_ind.len();

        let colors = greedy_color(n, &row_ptr, &col_ind);

        Self {
            n,
            nnz,
            row_ptr,
            col_ind,
            edge_pos,
            colors,
        }
    }
}

/// Truncating cast to int8, saturating on overflow (Rust's `as` semantics
/// since 1.45). Matches numpy's `dtype=np.int8` cast for the in-range
/// values consensus actually produces (h in {-1,0,1}, J in {-1,1}, milli
/// units); saturates instead of wrapping for out-of-range test fixtures.
fn quantize_i8(v: f64) -> i8 {
    v as i8
}

/// Quantize one job's `h`/`J` into the topology's fixed CSR layout.
///
/// `j_csr` has length `topology.nnz`; `h_i8` has length `topology.n`.
/// Positions not touched by any edge stay `0` (matches `j_csr` being
/// allocated/cleared before this call).
///
/// # Precondition
///
/// `graph` must share `topology`'s establishing edge list. `topology.edge_pos`
/// is sized from that graph's `edges`, and this walks `edge_pos` positionally,
/// so entry `k` is only meaningful when `graph.edges[k]` is the same edge.
/// Callers ([`crate::sampler::encode_batch`]) guarantee this by batching on an
/// identical `(n, edges)` key.
///
/// # Panics
///
/// Does not panic on a short `graph.edges` or `graph.j`: both are read with
/// `.get(k)` and a missing entry skips that edge (`edges`) or quantizes as
/// `0.0` (`j`). A *mismatched* edge list is still a caller bug — it silently
/// writes the wrong couplings — but it cannot crash the miner.
///
/// # Examples
///
/// ```
/// use quip_solver_metal::IsingGraph;
/// use quip_solver_metal::topology::{fill_h_j, SelfFeedingTopology};
///
/// let graph = IsingGraph::new(
///     vec![1.0, -1.0, 0.0, 1.0],
///     vec![1.0, -1.0, 1.0, -1.0],
///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
/// );
/// let t = SelfFeedingTopology::build(&graph);
/// let (j_csr, h_i8) = fill_h_j(&t, &graph);
/// assert_eq!(h_i8, vec![1i8, -1, 0, 1]);
/// // Each undirected edge writes both directed CSR halves.
/// assert_eq!(j_csr.iter().filter(|&&v| v != 0).count(), 8);
/// ```
pub fn fill_h_j(topology: &SelfFeedingTopology, graph: &IsingGraph) -> (Vec<i8>, Vec<i8>) {
    fill_h_j_inspecting_edges(topology, graph, |_, _| {})
}

/// [`fill_h_j`] for the wide kernels: integer milli units without quantization.
///
/// The caller has classified `graph` as wide, so every coefficient converts.
pub(crate) fn fill_h_j_wide(
    topology: &SelfFeedingTopology,
    graph: &IsingGraph,
) -> (Vec<i32>, Vec<i32>) {
    use crate::representation::to_milli;

    let mut j_csr = vec![0_i32; topology.nnz];
    for (k, &(pos_ij, pos_ji)) in topology.edge_pos.iter().enumerate() {
        let Some(&(u, v)) = graph.edges.get(k) else {
            continue;
        };
        if u >= topology.n || v >= topology.n {
            continue;
        }
        let val = graph.j.get(k).copied().and_then(to_milli).unwrap_or(0);
        j_csr[pos_ij as usize] = val;
        if u != v {
            j_csr[pos_ji as usize] = val;
        }
    }
    let h = graph.h.iter().map(|&v| to_milli(v).unwrap_or(0)).collect();
    (j_csr, h)
}

/// [`fill_h_j`] for the Float32 kernels: each `f64` coefficient rounded to
/// the nearest `f32`.
///
/// The caller has classified `graph`, which refuses any value past `f32::MAX`.
pub(crate) fn fill_h_j_float(
    topology: &SelfFeedingTopology,
    graph: &IsingGraph,
) -> (Vec<f32>, Vec<f32>) {
    let mut j_csr = vec![0_f32; topology.nnz];
    for (k, &(pos_ij, pos_ji)) in topology.edge_pos.iter().enumerate() {
        let Some(&(u, v)) = graph.edges.get(k) else {
            continue;
        };
        if u >= topology.n || v >= topology.n {
            continue;
        }
        let val = graph.j.get(k).copied().unwrap_or(0.0) as f32;
        j_csr[pos_ij as usize] = val;
        if u != v {
            j_csr[pos_ji as usize] = val;
        }
    }
    let h = graph.h.iter().map(|&v| v as f32).collect();
    (j_csr, h)
}

/// [`fill_h_j`]'s couplings from whole-unit `j_units` in the establishing
/// edge order.
///
/// # Precondition
///
/// The caller verified once that the establishing edges have no self-loop and
/// no out-of-range endpoint. Under that condition the result equals
/// [`fill_h_j`]'s couplings for the same values.
pub(crate) fn fill_couplings(topology: &SelfFeedingTopology, j_units: &[i8]) -> Vec<i8> {
    debug_assert_eq!(j_units.len(), topology.edge_pos.len());
    let mut j_csr = vec![0i8; topology.nnz];
    for (&(pos_ij, pos_ji), &value) in topology.edge_pos.iter().zip(j_units) {
        j_csr[pos_ij as usize] = value;
        j_csr[pos_ji as usize] = value;
    }
    j_csr
}

/// Compare the exact ordered edges during coefficient construction, avoiding
/// a separate topology walk. No coefficients escape on a mismatch.
pub(crate) fn fill_h_j_matching(
    topology: &SelfFeedingTopology,
    edges: &[(usize, usize)],
    graph: &IsingGraph,
) -> Option<(Vec<i8>, Vec<i8>)> {
    if graph.num_nodes() != topology.n
        || graph.edges.len() != edges.len()
        || topology.edge_pos.len() != edges.len()
    {
        return None;
    }
    let mut matches = true;
    let coefficients = fill_h_j_inspecting_edges(topology, graph, |k, edge| {
        matches &= edges[k] == edge;
    });
    matches.then_some(coefficients)
}

fn fill_h_j_inspecting_edges(
    topology: &SelfFeedingTopology,
    graph: &IsingGraph,
    mut inspect: impl FnMut(usize, (usize, usize)),
) -> (Vec<i8>, Vec<i8>) {
    let mut j_csr = vec![0i8; topology.nnz];
    for (k, &(pos_ij, pos_ji)) in topology.edge_pos.iter().enumerate() {
        let Some(&(u, v)) = graph.edges.get(k) else {
            continue;
        };
        inspect(k, (u, v));
        if u >= topology.n || v >= topology.n {
            continue;
        }
        let val = quantize_i8(graph.j.get(k).copied().unwrap_or(0.0));
        j_csr[pos_ij as usize] = val;
        if u != v {
            j_csr[pos_ji as usize] = val;
        }
    }
    let h_i8: Vec<i8> = graph.h.iter().map(|&v| quantize_i8(v)).collect();
    (j_csr, h_i8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_coefficients_require_exact_topology() {
        let graph = g();
        let topology = SelfFeedingTopology::build(&graph);
        assert_eq!(
            fill_h_j_matching(&topology, &graph.edges, &graph).unwrap(),
            fill_h_j(&topology, &graph)
        );
        let mut changed = graph.clone();
        let mut swapped = changed.edges.to_vec();
        swapped.swap(0, 1);
        changed.edges = swapped.into();
        assert!(fill_h_j_matching(&topology, &graph.edges, &changed).is_none());
        changed = graph.clone();
        let mut edges = changed.edges.to_vec();
        edges[0] = (0, 2);
        changed.edges = edges.clone().into();
        assert!(fill_h_j_matching(&topology, &graph.edges, &changed).is_none());
        edges.pop();
        changed.edges = edges.into();
        assert!(fill_h_j_matching(&topology, &graph.edges, &changed).is_none());
        changed = graph.clone();
        changed.h.pop();
        assert!(fill_h_j_matching(&topology, &graph.edges, &changed).is_none());
    }

    fn g() -> IsingGraph {
        // Small ring: 0-1-2-3-0, unit J, ternary h.
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    fn advantage2_fixture() -> IsingGraph {
        let edges: Vec<_> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut ids = line
                    .split_whitespace()
                    .map(|id| id.parse::<usize>().unwrap());
                (ids.next().unwrap(), ids.next().unwrap())
            })
            .collect();
        assert_eq!(edges.len(), 41515);
        IsingGraph::new(vec![0.0; 4577], vec![1.0; edges.len()], edges)
    }

    fn labels(colors: &ColorBlocks) -> Vec<i32> {
        let mut result = vec![-1; colors.nodes.len()];
        for (color, (&start, &count)) in colors.starts.iter().zip(&colors.counts).enumerate() {
            for &node in &colors.nodes[start as usize..(start + count) as usize] {
                assert_eq!(result[node as usize], -1);
                result[node as usize] = color as i32;
            }
        }
        assert!(result.iter().all(|&color| color >= 0));
        result
    }

    #[test]
    fn advantage2_four_colors_cover_all_edges_and_preserve_csr() {
        let graph = advantage2_fixture();
        let greedy = SelfFeedingTopology::build(&graph);
        let mut candidate = SelfFeedingTopology::build_with_advantage2_coloring(&graph);
        assert_eq!(greedy.colors.num_colors, 8);
        assert_eq!(candidate.colors.num_colors, 4);
        assert_eq!(candidate.colors.counts, [1148, 1145, 1145, 1139]);
        let colors = labels(&candidate.colors);
        for &(u, v) in graph.edges.iter() {
            assert_ne!(colors[u], colors[v], "edge ({u}, {v})");
        }
        assert!(graph.edges.contains(&(880, 2695)));
        candidate.colors = greedy.colors.clone();
        assert_eq!(candidate, greedy);
    }

    #[test]
    fn advantage2_candidate_ignores_coefficients_and_edge_order() {
        let mut graph = advantage2_fixture();
        let expected = advantage2_color(&graph).unwrap();
        graph.j.fill(0.0);
        graph.h.fill(-1.0);
        let mut edges = graph.edges.to_vec();
        edges.reverse();
        for edge in &mut edges {
            *edge = (edge.1, edge.0);
        }
        graph.edges = edges.into();
        assert_eq!(advantage2_color(&graph), Some(expected));
    }

    #[test]
    fn advantage2_candidate_checks_conflicts_even_with_zero_coupling() {
        let mut graph = advantage2_fixture();
        let colors = labels(&advantage2_color(&graph).unwrap());
        let other = (1..colors.len())
            .find(|&node| colors[node] == colors[0])
            .unwrap();
        graph.edges = [&graph.edges[..], &[(0, other)]].concat().into();
        graph.j.push(0.0);
        assert_eq!(advantage2_color(&graph), None);
        assert_eq!(
            SelfFeedingTopology::build_with_advantage2_coloring(&graph),
            SelfFeedingTopology::build(&graph)
        );
    }

    #[test]
    fn advantage2_candidate_rejects_malformed_edges_and_wrong_size() {
        assert_eq!(
            SelfFeedingTopology::build_with_advantage2_coloring(&g()),
            SelfFeedingTopology::build(&g())
        );
        for edge in [(0, 0), (0, 4577), (usize::MAX, 0)] {
            let mut graph = advantage2_fixture();
            graph.edges = [&graph.edges[..], &[edge]].concat().into();
            graph.j.push(1.0);
            assert_eq!(advantage2_color(&graph), None);
            assert_eq!(
                SelfFeedingTopology::build_with_advantage2_coloring(&graph),
                SelfFeedingTopology::build(&graph)
            );
        }
    }

    #[test]
    fn csr_shape_and_symmetry() {
        let t = SelfFeedingTopology::build(&g());
        assert_eq!(t.n, 4);
        assert_eq!(t.nnz, 8); // 4 edges * 2 directed halves
        assert_eq!(t.row_ptr, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn coloring_is_proper() {
        let t = SelfFeedingTopology::build(&g());
        // Every node gets exactly one color; adjacent nodes differ.
        let mut color_of = vec![-1i32; t.n];
        for (c, (&start, &count)) in t.colors.starts.iter().zip(&t.colors.counts).enumerate() {
            for i in 0..count {
                let node = t.colors.nodes[(start + i) as usize] as usize;
                assert_eq!(color_of[node], -1, "node colored twice");
                color_of[node] = c as i32;
            }
        }
        assert!(color_of.iter().all(|&c| c >= 0), "every node colored");
        for i in 0..t.n {
            let s = t.row_ptr[i] as usize;
            let e = t.row_ptr[i + 1] as usize;
            for &nbr in &t.col_ind[s..e] {
                assert_ne!(
                    color_of[i], color_of[nbr as usize],
                    "adjacent nodes {i} and {nbr} share a color"
                );
            }
        }
    }

    #[test]
    fn quantization_is_lossless_for_consensus_range() {
        let t = SelfFeedingTopology::build(&g());
        let (j, h) = fill_h_j(&t, &g());
        assert_eq!(h, vec![1i8, -1, 0, 1]);
        // Each edge's J appears at both directed CSR positions.
        assert_eq!(j.iter().filter(|&&v| v != 0).count(), 8);
    }

    #[test]
    fn fill_h_j_wide_writes_milli_into_both_csr_halves() {
        let graph = IsingGraph::new(
            vec![0.25, -1.5, 0.0],
            vec![0.001, -2.0],
            vec![(0, 1), (1, 2)],
        );
        let t = SelfFeedingTopology::build(&graph);
        let (j, h) = fill_h_j_wide(&t, &graph);
        assert_eq!(h, vec![250, -1500, 0]);
        let mut nonzero: Vec<i32> = j.into_iter().filter(|&v| v != 0).collect();
        nonzero.sort_unstable();
        assert_eq!(nonzero, vec![-2000, -2000, 1, 1]);
    }

    #[test]
    fn fill_h_j_skips_edges_the_graph_does_not_have() {
        // `edge_pos` is sized from the establishing graph; a graph with a
        // shorter edge list must skip, not panic (the `j` read already did).
        let t = SelfFeedingTopology::build(&g());
        let short = IsingGraph::new(vec![1.0, -1.0, 0.0, 1.0], vec![1.0], vec![(0, 1)]);
        let (j, h) = fill_h_j(&t, &short);
        assert_eq!(h, vec![1i8, -1, 0, 1]);
        // Only edge 0 is present: its two directed CSR slots are written.
        assert_eq!(j.iter().filter(|&&v| v != 0).count(), 2);
    }

    #[test]
    fn empty_graph_has_no_colors() {
        let t = SelfFeedingTopology::build(&IsingGraph::new(vec![], vec![], vec![]));
        assert_eq!(t.n, 0);
        assert_eq!(t.colors.num_colors, 0);
    }
}
