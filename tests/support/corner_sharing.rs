// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#[derive(Clone)]
pub(crate) struct Fixture {
    pub(crate) h: Vec<i32>,
    pub(crate) edges: Vec<(u32, u32)>,
    pub(crate) j: Vec<i32>,
    pub(crate) tiles: Vec<[u32; 4]>,
    pub(crate) gauges: Vec<[i8; 4]>,
    pub(crate) pairs: Vec<(u32, u32)>,
}

// Source: the reviewed CPU tests/otsuka_kernel.rs corner_sharing fixture.
pub(crate) fn corner_sharing() -> Fixture {
    let mut f = Fixture {
        h: vec![0; 8],
        edges: vec![(0, 4), (5, 7), (1, 5), (2, 6), (1, 4), (6, 7)],
        j: vec![-1000, -1000, 300, -200, 700, 500],
        tiles: vec![[0, 1, 2, 3], [3, 4, 5, 6]],
        gauges: vec![[1, -1, 1, 1], [-1, 1, 1, -1]],
        pairs: vec![(0, 4), (5, 7)],
    };
    for (tile, gauge) in f.tiles.iter().zip(&f.gauges) {
        for a in 0..4 {
            for b in a + 1..4 {
                f.edges.push((tile[a], tile[b]));
                f.j.push(1000 * i32::from(gauge[a] * gauge[b]));
            }
        }
    }
    f
}
