// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Pack validated motifs and energy layers into bounded device arrays.

use std::collections::HashMap;

use super::{key, Edge, GraphError, Model, Motifs, OtsukaGraph, Weights};

use crate::capacity::{cluster_footprint, ClusterShape, DeviceLimits, Footprint};
use quip_solver_core::families::FamilyContext;
use quip_solver_core::ising_generators::descriptor::Contract;
use quip_solver_core::ising_generators::ice::twisted_dimer::local_gauges;
use quip_solver_core::ising_generators::ice::TWISTED_DIMER_V1;
use quip_solver_core::ising_generators::instance::Instance;
use thiserror::Error;

/// A packed neighbor uses 16 bits (`gpucluster.cu:123-124,226-228`).
/// Node positions 0 through 65535 are representable.
pub const MAX_PACKED_NODES: usize = 1 << 16;

/// ELL addressing multiplies slot by node count in signed int (`gpucluster.cu:123,226`).
pub const MAX_ELL_ENTRIES: usize = i32::MAX as usize;

/// Component sums use signed int atomic addition (`gpucluster.cu:233`).
const MAX_COMPONENT_FIELD: u64 = i32::MAX as u64;

/// A rejection before device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PrepareError {
    /// The model does not satisfy the CPU Otsuka graph contract.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Only twisted-ice-dimer revision 1 is supported.
    #[error("the cluster preparation supports only quip/twisted-ice-dimer revision 1")]
    UnsupportedFamily,
    /// The canonical metadata does not contain readable tiles, gauges, and pairs.
    #[error("the family metadata does not contain valid tiles, gauges, and pairs")]
    Metadata,
    /// A device index or array length exceeds the kernel integer width.
    #[error("{array} exceeds the cluster kernel index width")]
    IndexOverflow {
        /// Array or index that failed the width check.
        array: &'static str,
    },
    /// A component could overflow its signed 32-bit residual accumulator.
    #[error("the sum of absolute residual couplings exceeds the component accumulator")]
    AccumulatorOverflow,
    /// No anneals were requested.
    #[error("the cluster launch must contain at least one anneal")]
    EmptyLaunch,
    /// Device allocations exceed the usable device memory.
    #[error("cluster device memory {requested} exceeds the usable budget {budget}")]
    DeviceMemory {
        /// Required device bytes.
        requested: usize,
        /// Usable global bytes after headroom.
        budget: usize,
    },
}

/// Why a state cannot be scored or permuted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SpinError {
    /// The spin count differs from the node count.
    #[error("the spin count differs from the node count")]
    Length,
    /// A spin is not a sign.
    #[error("spin {node} is not 1 or -1")]
    Value {
        /// Position in the supplied order.
        node: usize,
    },
}

/// Upload arrays for `Dev` in `gpucluster.cu:68-74`, in device node order.
/// All int arrays are 32-bit. Gauges upload as signed bytes, not file int32.
/// Tile, pair, and gauge arrays are flat for aligned device allocation and
/// reinterpretation as `int4`, `int2`, and `char4` respectively.
#[derive(Debug)]
pub struct DeviceLayout {
    /// `[n, tiles, links, weak_width, all_width, colors]` (`gpucluster.cu:15-18`).
    pub header: [i32; 6],
    /// Four node indices per tile.
    pub tiles: Vec<i32>,
    /// Four local gauge signs per tile, in tile-position order.
    pub gauges: Vec<i8>,
    /// Two node indices per metadata pair.
    pub pairs: Vec<i32>,
    /// Nonzero residual degree per node.
    pub weak_deg: Vec<i32>,
    /// Slot-major packed residual adjacency, padded with zero words.
    pub weak_ell: Vec<i32>,
    /// Nonzero original degree per node.
    pub all_deg: Vec<i32>,
    /// Slot-major packed original adjacency, padded with zero words.
    pub all_ell: Vec<i32>,
    /// Contiguous color offsets, including zero and the final node count.
    pub color_off: Vec<i32>,
    /// Zero fields, uploaded explicitly for the kernel field pointer.
    pub h: Vec<i32>,
    /// One positive milli weight per tile.
    pub tile_w: Vec<i32>,
    /// One nonnegative milli bond weight per pair, subtracted from energy.
    pub link_w: Vec<i32>,
}

/// A validated original graph, its permutations, and bounded device inputs.
#[derive(Debug)]
pub struct PreparedGraph {
    original: OtsukaGraph,
    layout: DeviceLayout,
    device_to_original: Vec<usize>,
    original_to_device: Vec<usize>,
    footprint: Footprint,
}

impl PreparedGraph {
    /// Validate canonical coefficients and verified motifs, then prepare arrays.
    ///
    /// This low-level entry checks the same graph rules as the CPU port.
    /// The caller verifies the family metadata before supplying motifs.
    /// `anneals` counts concurrent blocks at the fixed cluster thread count.
    ///
    /// # Errors
    /// Returns a graph, width, accumulator, launch, or capacity rejection before
    /// any device allocation. This module makes no Metal calls.
    pub fn new(
        model: &Model<'_>,
        motifs: &Motifs<'_>,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<Self, PrepareError> {
        let original = OtsukaGraph::new(model, motifs, weights)?;
        let n = original.n;
        if n > MAX_PACKED_NODES {
            return Err(PrepareError::IndexOverflow { array: "nodes" });
        }
        if anneals == 0 {
            return Err(PrepareError::EmptyLaunch);
        }
        check_residual_sum(original.weak_edges.iter().map(|edge| edge.2))?;
        let (color_off, device_to_original, original_to_device) = coloring(&original);
        let all_rows = rows(n, &original.edges, &original_to_device);
        // Keep original edge order for the residual ELL, as prep.py does.
        let residual: HashMap<_, _> = original
            .weak_edges
            .iter()
            .map(|&(u, v, j)| (key(u, v), j))
            .collect();
        let weak_edges: Vec<_> = original
            .edges
            .iter()
            .map(|&(u, v, _)| (u, v, residual.get(&key(u, v)).copied().unwrap_or(0)))
            .collect();
        let weak_rows = rows(n, &weak_edges, &original_to_device);
        let shape = ClusterShape {
            nodes: n,
            tiles: original.tiles.len(),
            links: original.pairs.len(),
            weak_width: weak_rows.iter().map(Vec::len).max().unwrap_or(0),
            all_width: all_rows.iter().map(Vec::len).max().unwrap_or(0),
            colors: color_off.len() - 1,
        };
        check_widths(shape)?;
        let footprint = cluster_footprint(shape);
        if footprint.total(anneals, 1) > limits.usable_bytes() {
            return Err(PrepareError::DeviceMemory {
                requested: footprint.total(anneals, 1),
                budget: limits.usable_bytes(),
            });
        }
        let (weak_deg, weak_ell) = ell(&weak_rows, shape.weak_width)?;
        let (all_deg, all_ell) = ell(&all_rows, shape.all_width)?;
        let id = |u: usize| int_index(original_to_device[u], "motif node");
        let tiles = original
            .tiles
            .iter()
            .flat_map(|tile| tile.nodes)
            .map(id)
            .collect::<Result<_, _>>()?;
        let gauges = original.tiles.iter().flat_map(|tile| tile.gauge).collect();
        let pairs = original
            .pairs
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .map(id)
            .collect::<Result<_, _>>()?;
        let header = [
            shape.nodes,
            shape.tiles,
            shape.links,
            shape.weak_width,
            shape.all_width,
            shape.colors,
        ]
        .map(|value| int_index(value, "header"))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let layout = DeviceLayout {
            header: [
                header[0], header[1], header[2], header[3], header[4], header[5],
            ],
            tiles,
            gauges,
            pairs,
            weak_deg,
            weak_ell,
            all_deg,
            all_ell,
            color_off: color_off
                .into_iter()
                .map(|off| int_index(off, "color offsets"))
                .collect::<Result<_, _>>()?,
            h: vec![0; n],
            tile_w: vec![weights.tile_milli; shape.tiles],
            link_w: vec![weights.pair_milli; shape.links],
        };
        Ok(Self {
            original,
            layout,
            device_to_original,
            original_to_device,
            footprint,
        })
    }

    /// Prepare a session-validated canonical instance under its family contract.
    ///
    /// The caller must validate the instance against `contract` and its declared
    /// parameters first. Prefer [`Self::from_family`] for session jobs.
    ///
    /// # Errors
    /// Returns unsupported family or unreadable metadata, then the same errors
    /// as [`Self::new`]. The instance offset stays with the session for scoring.
    pub fn from_instance(
        instance: &Instance,
        contract: &Contract,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<Self, PrepareError> {
        if contract.family_id != TWISTED_DIMER_V1.family_id
            || contract.revision != TWISTED_DIMER_V1.revision
        {
            return Err(PrepareError::UnsupportedFamily);
        }
        let (tiles, gauges) = local_gauges(instance).ok_or(PrepareError::Metadata)?;
        let pairs = metadata_pairs(instance.metadata()).ok_or(PrepareError::Metadata)?;
        Self::new(
            &Model {
                h_milli: instance.h(),
                edges: instance.edges(),
                j_milli: instance.j(),
            },
            &Motifs {
                tiles: &tiles,
                gauges: &gauges,
                pairs: &pairs,
            },
            weights,
            limits,
            anneals,
        )
    }

    /// Read the bound canonical instance supplied by a lease or explicit job.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::from_instance`].
    pub fn from_family(
        family: &FamilyContext,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<Self, PrepareError> {
        Self::from_instance(
            family.instance(),
            family.contract(),
            weights,
            limits,
            anneals,
        )
    }

    /// Immutable upload arrays in device node order.
    #[must_use]
    pub fn layout(&self) -> &DeviceLayout {
        &self.layout
    }

    /// Original coefficients and motifs for rescoring in original node order.
    #[must_use]
    pub fn original_graph(&self) -> &OtsukaGraph {
        &self.original
    }

    /// Device position to original position: `device[k] = original[perm[k]]`.
    #[must_use]
    pub fn device_to_original(&self) -> &[usize] {
        &self.device_to_original
    }

    /// Original position to device position, the explicit inverse permutation.
    #[must_use]
    pub fn original_to_device(&self) -> &[usize] {
        &self.original_to_device
    }

    /// Checked device allocation estimate for the requested concurrent anneals.
    #[must_use]
    pub fn footprint(&self) -> Footprint {
        self.footprint
    }

    fn check_spins(&self, spins: &[i8]) -> Result<(), SpinError> {
        if spins.len() != self.original.n {
            return Err(SpinError::Length);
        }
        if let Some(node) = spins.iter().position(|&s| s != 1 && s != -1) {
            return Err(SpinError::Value { node });
        }
        Ok(())
    }

    /// Convert an original-order state to device order.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn permute(&self, spins: &[i8]) -> Result<Vec<i8>, SpinError> {
        self.check_spins(spins)?;
        Ok(self
            .device_to_original
            .iter()
            .map(|&old| spins[old])
            .collect())
    }

    /// Restore a device-order state to original order before returning spins.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn unpermute(&self, spins: &[i8]) -> Result<Vec<i8>, SpinError> {
        self.check_spins(spins)?;
        Ok(self
            .original_to_device
            .iter()
            .map(|&new| spins[new])
            .collect())
    }

    /// Original energy in milli, without the session-owned offset.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn energy(&self, spins: &[i8]) -> Result<i64, SpinError> {
        self.check_spins(spins)?;
        Ok(self.original.energy(spins))
    }

    /// Weighted tile and pair energy in milli, on original-order spins.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn strong_energy(&self, spins: &[i8]) -> Result<i64, SpinError> {
        self.check_spins(spins)?;
        Ok(self.original.strong_energy(spins))
    }

    /// Non-motif energy plus softened motif residuals, on original-order spins.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn weak_energy(&self, spins: &[i8]) -> Result<i64, SpinError> {
        self.check_spins(spins)?;
        Ok(self.original.weak_energy(spins))
    }

    /// `(tile, link, residual)` energies in milli, on original-order spins.
    /// Fields are zero under the admitted contract.
    ///
    /// # Errors
    /// Returns [`SpinError`] for a wrong length or a spin that is not a sign.
    pub fn split_energies(&self, spins: &[i8]) -> Result<(i64, i64, i64), SpinError> {
        self.check_spins(spins)?;
        let link: i64 = self
            .original
            .pairs
            .iter()
            .map(|&(u, v)| {
                -i64::from(self.original.weights.pair_milli) * i64::from(spins[u] * spins[v])
            })
            .sum();
        let strong = self.original.strong_energy(spins);
        Ok((strong - link, link, self.original.energy(spins) - strong))
    }
}

fn int_index(value: usize, array: &'static str) -> Result<i32, PrepareError> {
    i32::try_from(value).map_err(|_| PrepareError::IndexOverflow { array })
}

fn check_widths(shape: ClusterShape) -> Result<(), PrepareError> {
    for (array, count) in [
        ("tile indices", shape.tiles.saturating_mul(4)),
        ("pair indices", shape.links.saturating_mul(2)),
        ("weak ELL", shape.nodes.saturating_mul(shape.weak_width)),
        ("all ELL", shape.nodes.saturating_mul(shape.all_width)),
        ("color offsets", shape.colors.saturating_add(1)),
    ] {
        if count > MAX_ELL_ENTRIES {
            return Err(PrepareError::IndexOverflow { array });
        }
    }
    Ok(())
}

fn check_residual_sum(couplings: impl Iterator<Item = i32>) -> Result<(), PrepareError> {
    let sum = couplings.fold(0_u64, |sum, j| {
        sum.saturating_add(u64::from(j.unsigned_abs()))
    });
    if sum > MAX_COMPONENT_FIELD {
        return Err(PrepareError::AccumulatorOverflow);
    }
    Ok(())
}

fn coloring(graph: &OtsukaGraph) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = graph.n;
    let mut adj = vec![Vec::new(); n];
    // Include zero edges, matching prep.py coloring and the canonical topology.
    for &(u, v, _) in &graph.edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|&u| std::cmp::Reverse(adj[u].len()));
    let mut color = vec![usize::MAX; n];
    let mut seen = vec![usize::MAX; n];
    for u in order {
        for &v in &adj[u] {
            if color[v] != usize::MAX {
                seen[color[v]] = u;
            }
        }
        color[u] = seen.iter().position(|&mark| mark != u).unwrap_or(0);
    }
    let mut perm: Vec<_> = (0..n).collect();
    perm.sort_by_key(|&u| color[u]);
    let mut off = vec![0; color.iter().max().copied().unwrap_or(0) + 2];
    let mut inverse = vec![0; n];
    for (new, &old) in perm.iter().enumerate() {
        inverse[old] = new;
        off[color[old] + 1] += 1;
    }
    for c in 1..off.len() {
        off[c] += off[c - 1];
    }
    (off, perm, inverse)
}

fn rows(n: usize, edges: &[Edge], inverse: &[usize]) -> Vec<Vec<(usize, i32)>> {
    let mut rows = vec![Vec::new(); n];
    // prep.py concatenates the forward halves before the reverse halves.
    for reverse in [false, true] {
        for &(u, v, j) in edges {
            if j == 0 {
                continue;
            }
            let (u, v) = if reverse { (v, u) } else { (u, v) };
            rows[inverse[u]].push((inverse[v], j));
        }
    }
    rows
}

fn ell(rows: &[Vec<(usize, i32)>], width: usize) -> Result<(Vec<i32>, Vec<i32>), PrepareError> {
    let mut deg = Vec::with_capacity(rows.len());
    let mut table = vec![0; width * rows.len()];
    for (u, row) in rows.iter().enumerate() {
        deg.push(int_index(row.len(), "degree")?);
        for (slot, &(v, j)) in row.iter().enumerate() {
            let neighbor =
                u16::try_from(v).map_err(|_| PrepareError::IndexOverflow { array: "neighbor" })?;
            let coupling =
                i16::try_from(j).map_err(|_| PrepareError::IndexOverflow { array: "coupling" })?;
            let word =
                u32::from(neighbor) << 16 | u32::from(u16::from_ne_bytes(coupling.to_ne_bytes()));
            table[slot * rows.len() + u] = i32::from_ne_bytes(word.to_ne_bytes());
        }
    }
    Ok((deg, table))
}

fn metadata_pairs(region: &[u8]) -> Option<Vec<(u32, u32)>> {
    let (count, mut rest) = region.split_first_chunk::<2>()?;
    for _ in 0..u16::from_le_bytes(*count) {
        let (header, tail) = rest.split_first_chunk::<6>()?;
        let len = usize::try_from(u32::from_le_bytes(header[2..6].try_into().ok()?)).ok()?;
        let (payload, next) = tail.split_at_checked(len)?;
        rest = next;
        if u16::from_le_bytes([header[0], header[1]]) != 3 {
            continue;
        }
        let (count, data) = payload.split_first_chunk::<4>()?;
        let (pairs, tail) = data.as_chunks::<8>();
        if !tail.is_empty() || usize::try_from(u32::from_le_bytes(*count)).ok()? != pairs.len() {
            return None;
        }
        return pairs
            .iter()
            .map(|p| {
                Some((
                    u32::from_le_bytes(p[..4].try_into().ok()?),
                    u32::from_le_bytes(p[4..].try_into().ok()?),
                ))
            })
            .collect();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_accumulator_rejects_the_first_overflow() {
        assert_eq!(check_residual_sum([i32::MAX].into_iter()), Ok(()));
        assert_eq!(
            check_residual_sum([i32::MAX, -1].into_iter()),
            Err(PrepareError::AccumulatorOverflow)
        );
        assert_eq!(
            check_residual_sum([i32::MIN].into_iter()),
            Err(PrepareError::AccumulatorOverflow)
        );
    }

    #[test]
    fn kernel_array_products_cannot_overflow_signed_addressing() {
        let shape = ClusterShape {
            nodes: 65_536,
            tiles: 0,
            links: 0,
            weak_width: 0,
            all_width: 32_768,
            colors: 1,
        };
        assert_eq!(
            check_widths(shape),
            Err(PrepareError::IndexOverflow { array: "all ELL" })
        );
        assert_eq!(
            check_widths(ClusterShape {
                all_width: 32_767,
                ..shape
            }),
            Ok(())
        );
        assert_eq!(
            check_widths(ClusterShape {
                tiles: usize::MAX,
                ..shape
            }),
            Err(PrepareError::IndexOverflow {
                array: "tile indices"
            })
        );
    }

    #[test]
    fn pair_payload_lengths_and_truncation_are_checked() {
        let mut region = vec![1, 0, 3, 0, 12, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0];
        assert_eq!(metadata_pairs(&region), Some(vec![(1, 3)]));
        region[8] = 2;
        assert_eq!(metadata_pairs(&region), None);
        for end in 0..region.len() {
            assert_eq!(metadata_pairs(&region[..end]), None);
        }
    }
}
