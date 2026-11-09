// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Property-based tests for host CSR topology, coloring, and `fill_h_j`.
//!
//! Topology modules are macOS-gated in the library, so this file matches
//! `tests/conformance.rs` and is empty on other OSes.
//! Pure host math — no Metal device required.

#![cfg(target_os = "macos")]

use proptest::prelude::*;
use quip_solver_metal::topology::{fill_h_j, ColorBlocks, SelfFeedingTopology};
use quip_solver_metal::IsingGraph;

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// Bounded structural graphs: arbitrary `n`, edges that may be OOB or loops.
///
/// `n` stays small so shrinking is fast. Endpoints range over `0..(n+4)` so
/// out-of-range endpoints appear; self-loops arise when `u == v`.
fn arb_struct_graph() -> impl Strategy<Value = IsingGraph> {
    (0usize..=32).prop_flat_map(|n| {
        let max_edges = 64.min(if n == 0 { 8 } else { n.saturating_mul(3) });
        let endpoint_hi = n.saturating_add(4).max(1);
        prop::collection::vec((0usize..endpoint_hi, 0usize..endpoint_hi), 0..=max_edges).prop_map(
            move |edges| {
                let h = vec![0.0; n];
                let j = vec![0.0; edges.len()];
                IsingGraph::new(h, j, edges)
            },
        )
    })
}

/// Graphs with consensus-range `h`/`J` for `fill_h_j` scatter properties.
///
/// `h ∈ {-1,0,1}`, `J ∈ {-1,1}` (protocol defaults). Same edge-shape bounds
/// as [`arb_struct_graph`].
fn arb_consensus_graph() -> impl Strategy<Value = IsingGraph> {
    (0usize..=32).prop_flat_map(|n| {
        let max_edges = 64.min(if n == 0 { 8 } else { n.saturating_mul(3) });
        let endpoint_hi = n.saturating_add(4).max(1);
        let h_strat = prop::collection::vec(prop_oneof![Just(-1.0f64), Just(0.0), Just(1.0)], n);
        let edges_strat =
            prop::collection::vec((0usize..endpoint_hi, 0usize..endpoint_hi), 0..=max_edges);
        (h_strat, edges_strat).prop_flat_map(move |(h, edges)| {
            let m = edges.len();
            prop::collection::vec(prop_oneof![Just(-1.0f64), Just(1.0)], m)
                .prop_map(move |j| IsingGraph::new(h.clone(), j, edges.clone()))
        })
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map each node index to its color, asserting each node appears once.
fn color_of_nodes(n: usize, colors: &ColorBlocks) -> Result<Vec<i32>, TestCaseError> {
    prop_assert_eq!(colors.nodes.len(), n);
    let mut color_of = vec![-1i32; n];
    for (c, (&start, &count)) in colors.starts.iter().zip(&colors.counts).enumerate() {
        prop_assert!(start >= 0);
        prop_assert!(count >= 0);
        let start_u = start as usize;
        let end_u = (start + count) as usize;
        prop_assert!(end_u <= colors.nodes.len());
        for i in start_u..end_u {
            let node = colors.nodes[i] as usize;
            prop_assert!(node < n, "color node index out of range: {}", node);
            prop_assert_eq!(color_of[node], -1, "node {} colored more than once", node);
            color_of[node] = c as i32;
        }
    }
    for (i, &c) in color_of.iter().enumerate() {
        prop_assert!(c >= 0, "node {i} never colored");
    }
    Ok(color_of)
}

/// Expected CSR nonzeros: each in-range edge contributes 2 halves, loops 1.
fn expected_nnz(n: usize, edges: &[(usize, usize)]) -> usize {
    edges
        .iter()
        .filter(|&&(u, v)| u < n && v < n)
        .map(|&(u, v)| if u == v { 1 } else { 2 })
        .sum()
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// CSR shape: row_ptr / col_ind / nnz / edge_pos length and symmetry.
    #[test]
    fn csr_shape_and_symmetry(graph in arb_struct_graph()) {
        let t = SelfFeedingTopology::build(&graph);
        let n = graph.num_nodes();

        prop_assert_eq!(t.n, n);
        prop_assert_eq!(t.row_ptr.len(), n + 1);
        prop_assert_eq!(t.row_ptr[0], 0);
        prop_assert_eq!(t.row_ptr[n] as usize, t.nnz);
        prop_assert_eq!(t.col_ind.len(), t.nnz);
        prop_assert_eq!(t.edge_pos.len(), graph.edges.len());
        prop_assert_eq!(t.nnz, expected_nnz(n, &graph.edges));

        // row_ptr is non-decreasing and bounds each row of col_ind.
        for i in 0..n {
            prop_assert!(
                t.row_ptr[i] <= t.row_ptr[i + 1],
                "row_ptr not non-decreasing at {i}"
            );
        }

        // Every column index is a valid node.
        for &c in &t.col_ind {
            prop_assert!(c >= 0);
            prop_assert!((c as usize) < n, "col_ind entry {c} out of 0..{n}");
        }

        // Each in-range edge is present as both directed halves (loop once).
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            let (pos_ij, pos_ji) = t.edge_pos[k];
            let pos_ij = pos_ij as usize;
            let pos_ji = pos_ji as usize;

            prop_assert!(pos_ij < t.nnz, "pos_ij out of range for edge {k}");
            prop_assert_eq!(t.col_ind[pos_ij] as usize, v);
            prop_assert!(
                (t.row_ptr[u] as usize) <= pos_ij
                    && pos_ij < (t.row_ptr[u + 1] as usize),
                "pos_ij not in row u={u} for edge {k}"
            );

            if u != v {
                prop_assert!(pos_ji < t.nnz, "pos_ji out of range for edge {k}");
                prop_assert_eq!(t.col_ind[pos_ji] as usize, u);
                prop_assert!(
                    (t.row_ptr[v] as usize) <= pos_ji
                        && pos_ji < (t.row_ptr[v + 1] as usize),
                    "pos_ji not in row v={v} for edge {k}"
                );
                prop_assert_ne!(pos_ij, pos_ji, "halves share one CSR slot");
            }
        }
    }

    /// Coloring: every node once, blocks partition nodes, adjacent differ.
    #[test]
    fn coloring_is_proper(graph in arb_struct_graph()) {
        let t = SelfFeedingTopology::build(&graph);
        let n = t.n;
        let colors = &t.colors;

        prop_assert_eq!(colors.num_colors == 0, n == 0);
        prop_assert_eq!(colors.starts.len(), colors.num_colors as usize);
        prop_assert_eq!(colors.counts.len(), colors.num_colors as usize);
        prop_assert_eq!(colors.nodes.len(), n);

        // starts/counts partition colors.nodes exactly.
        if n == 0 {
            prop_assert!(colors.starts.is_empty());
            prop_assert!(colors.counts.is_empty());
            prop_assert!(colors.nodes.is_empty());
        } else {
            prop_assert!(!colors.starts.is_empty());
            prop_assert_eq!(colors.starts[0], 0);
            for c in 0..colors.starts.len() {
                if c + 1 < colors.starts.len() {
                    prop_assert_eq!(
                        colors.starts[c] + colors.counts[c],
                        colors.starts[c + 1],
                        "block boundary mismatch at color {}",
                        c
                    );
                } else {
                    prop_assert_eq!(
                        colors.starts[c] + colors.counts[c],
                        n as i32,
                        "last color block does not cover n"
                    );
                }
            }
        }

        let color_of = color_of_nodes(n, colors)?;

        // Proper coloring on non-loop CSR adjacencies (Gibbs parallel update).
        for i in 0..n {
            let s = t.row_ptr[i] as usize;
            let e = t.row_ptr[i + 1] as usize;
            for &nbr in &t.col_ind[s..e] {
                let j = nbr as usize;
                if i == j {
                    continue;
                }
                prop_assert_ne!(
                    color_of[i],
                    color_of[j],
                    "adjacent nodes {} and {} share color {}",
                    i,
                    j,
                    color_of[i]
                );
            }
        }
    }

    /// `fill_h_j` scatters consensus-range h/J into CSR slots correctly.
    #[test]
    fn fill_h_j_scatters_consensus_range(graph in arb_consensus_graph()) {
        let t = SelfFeedingTopology::build(&graph);
        // Same graph (same edge list) — do not call with a shorter edge list.
        let (j_csr, h_i8) = fill_h_j(&t, &graph);
        let n = t.n;

        prop_assert_eq!(h_i8.len(), n);
        prop_assert_eq!(j_csr.len(), t.nnz);

        for (&got, &want) in h_i8.iter().zip(graph.h.iter()) {
            prop_assert_eq!(got, want as i8);
        }

        // Track which CSR positions each in-range edge touches.
        let mut touched = vec![false; t.nnz];
        for (k, &(u, v)) in graph.edges.iter().enumerate() {
            if u >= n || v >= n {
                continue;
            }
            let (pos_ij, pos_ji) = t.edge_pos[k];
            let val = graph.j.get(k).copied().unwrap_or(0.0) as i8;
            let pos_ij = pos_ij as usize;

            prop_assert_eq!(
                j_csr[pos_ij],
                val,
                "J not at pos_ij for edge {}",
                k
            );
            touched[pos_ij] = true;

            if u != v {
                let pos_ji = pos_ji as usize;
                prop_assert_eq!(
                    j_csr[pos_ji],
                    val,
                    "J not at pos_ji for edge {}",
                    k
                );
                touched[pos_ji] = true;
            }
        }

        // Untouched CSR slots stay zero (j_csr allocated cleared).
        // Multi-edges each own distinct CSR positions, so "last write wins"
        // only if two edges somehow shared a slot — they do not under build.
        for (p, &was) in touched.iter().enumerate() {
            if !was {
                prop_assert_eq!(j_csr[p], 0, "untouched CSR slot {} is non-zero", p);
            }
        }
    }

    /// `build` is deterministic: same graph ⇒ identical CSR layout.
    #[test]
    fn build_is_deterministic(graph in arb_struct_graph()) {
        let a = SelfFeedingTopology::build(&graph);
        let b = SelfFeedingTopology::build(&graph);

        prop_assert_eq!(a.n, b.n);
        prop_assert_eq!(a.nnz, b.nnz);
        prop_assert_eq!(&a.row_ptr, &b.row_ptr);
        prop_assert_eq!(&a.col_ind, &b.col_ind);
        prop_assert_eq!(&a.edge_pos, &b.edge_pos);
        // Coloring is also pure given CSR; pin it too (kernel reuses blocks).
        prop_assert_eq!(a.colors.num_colors, b.colors.num_colors);
        prop_assert_eq!(&a.colors.starts, &b.colors.starts);
        prop_assert_eq!(&a.colors.counts, &b.colors.counts);
        prop_assert_eq!(&a.colors.nodes, &b.colors.nodes);
    }
}
