// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

use quip_solver_metal::capacity::DeviceLimits;
use quip_solver_metal::cluster::{Model, Motifs, PrepareError, PreparedGraph, Weights};

#[path = "corner_sharing.rs"]
mod corner_sharing_data;
pub(crate) use corner_sharing_data::{corner_sharing, Fixture};

impl Fixture {
    pub(crate) fn prepare_with_anneals(
        &self,
        weights: Weights,
        limits: &DeviceLimits,
        anneals: usize,
    ) -> Result<PreparedGraph, PrepareError> {
        PreparedGraph::new(
            &Model {
                h_milli: &self.h,
                edges: &self.edges,
                j_milli: &self.j,
            },
            &Motifs {
                tiles: &self.tiles,
                gauges: &self.gauges,
                pairs: &self.pairs,
            },
            weights,
            limits,
            anneals,
        )
    }
}
