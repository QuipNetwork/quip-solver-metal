use crate::graph::MAX_LANES;
use crate::AneError;
use quip_solver_core::beta::default_ising_beta_range;
use quip_solver_core::{IsingGraph, SampleParams};

const MAX_SWEEPS: usize = 65_536;
const GROUPS: usize = 4;
const READS_PER_GROUP: usize = 32;
const ROW_LEN: usize = 8_192;
const SPIN_STREAM_DOMAIN: u64 = 0x5350_494e_5f49_4e49;
const THRESHOLD_STREAM_DOMAIN: u64 = 0x5448_5245_5348_4f4c;
const OFFSET_STREAM_DOMAIN: u64 = 0x4f46_4653_4554_5f31;
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rung {
    pub(crate) beta: f64,
    pub(crate) sweeps: usize,
}

pub(crate) fn validate_params(params: &SampleParams) -> Result<(), AneError> {
    if !(1..=MAX_LANES).contains(&params.num_reads) {
        return Err(AneError::Capacity(format!(
            "read count must be between 1 and {MAX_LANES}"
        )));
    }
    if params.num_sweeps > MAX_SWEEPS {
        return Err(AneError::Capacity(format!(
            "sweep count exceeds {MAX_SWEEPS}"
        )));
    }
    if params.sweeps_per_beta == 0 {
        return Err(AneError::Capacity(
            "sweeps per beta must be positive".into(),
        ));
    }
    if let Some((hot, cold)) = params.beta_range {
        if !hot.is_finite() || !cold.is_finite() || hot <= 0.0 || cold <= 0.0 {
            return Err(AneError::Capacity(
                "beta range endpoints must be finite and positive".into(),
            ));
        }
        if hot > cold {
            return Err(AneError::Capacity(
                "hot beta must not exceed cold beta".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn schedule(graph: &IsingGraph, params: &SampleParams) -> Result<Vec<Rung>, AneError> {
    validate_params(params)?;
    if params.num_sweeps == 0 {
        return Ok(Vec::new());
    }

    let rung_count = params.num_sweeps.div_ceil(params.sweeps_per_beta);
    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| default_ising_beta_range(graph));
    let betas = params.schedule.build(hot, cold, rung_count);
    let rungs = betas
        .into_iter()
        .enumerate()
        .map(|(index, beta)| Rung {
            beta,
            sweeps: params
                .sweeps_per_beta
                .min(params.num_sweeps - index * params.sweeps_per_beta),
        })
        .collect();
    Ok(rungs)
}

pub(crate) fn initial_spins(nodes: usize, lanes: usize, seed: u64) -> Vec<i8> {
    let mut stream = SplitMix64(seed ^ SPIN_STREAM_DOMAIN);
    let mut spins = Vec::with_capacity(nodes * lanes);
    for _node in 0..nodes {
        // Consume the original stream stride so retained lanes keep their seed
        // identity when a narrower program replaces the padded 128-lane one.
        for read in 0..MAX_LANES {
            let spin = if stream.next_u64() & 1 == 0 { 1 } else { -1 };
            if read < lanes {
                spins.push(spin);
            }
        }
    }
    spins
}

pub(crate) struct ThresholdRows {
    seed: u64,
    streams: [SplitMix64; GROUPS],
    rows: [[u8; ROW_LEN]; GROUPS],
}

impl ThresholdRows {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            seed,
            streams: std::array::from_fn(|group| {
                SplitMix64(seed ^ THRESHOLD_STREAM_DOMAIN ^ group as u64)
            }),
            rows: [[0; ROW_LEN]; GROUPS],
        }
    }

    pub(crate) fn begin_rung(&mut self, beta: f64) {
        let cut = cuts(beta);
        for group in 0..GROUPS {
            for entry in &mut self.rows[group] {
                let uniform = (self.streams[group].next_u64() >> 32) as u32;
                *entry = draw_threshold(&cut, uniform);
            }
        }
    }

    #[must_use]
    pub(crate) fn expand(&self, nodes: usize, lanes: usize, rung: usize, sweep: usize) -> Vec<u8> {
        let mut expanded = vec![0; nodes * lanes];
        let offsets: [usize; GROUPS] = std::array::from_fn(|group| {
            let key = self.seed
                ^ OFFSET_STREAM_DOMAIN
                ^ (group as u64).wrapping_mul(GOLDEN_GAMMA)
                ^ (rung as u64).rotate_left(21)
                ^ (sweep as u64).rotate_left(42);
            (SplitMix64(key).next_u64() as usize) & (ROW_LEN - 1)
        });
        for node in 0..nodes {
            for (group, &offset) in offsets
                .iter()
                .enumerate()
                .take(lanes.div_ceil(READS_PER_GROUP))
            {
                let threshold = self.rows[group][(node + offset) & (ROW_LEN - 1)];
                let start = node * lanes + group * READS_PER_GROUP;
                expanded[start..(start + READS_PER_GROUP).min((node + 1) * lanes)].fill(threshold);
            }
        }
        expanded
    }
}

fn cuts(beta: f64) -> [u64; 64] {
    std::array::from_fn(|m| ((-2.0 * beta * m as f64).exp() * 4_294_967_296.0).floor() as u64)
}

fn draw_threshold(cut: &[u64; 64], uniform: u32) -> u8 {
    // Largest m in 1..=63 with uniform < cut[m], else 0. cut is monotone
    // non-increasing, so the predicate is a prefix of cut[1..].
    let uniform = u64::from(uniform);
    cut[1..].partition_point(|&cutoff| uniform < cutoff) as u8
}

/// SplitMix64 reference implementation: <https://prng.di.unimi.it/splitmix64.c>.
#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::{IsingGraph, SampleParams};

    #[test]
    fn narrow_lanes_preserve_full_width_spins_and_thresholds() {
        let nodes = 7;
        let full = initial_spins(nodes, MAX_LANES, 19);
        let mut rows = ThresholdRows::new(19);
        for rung in 0..3 {
            rows.begin_rung(0.05 + rung as f64 * 0.1);
            let full_thresholds = rows.expand(nodes, MAX_LANES, rung, 3);
            for lanes in [1, 7, 31, 32, 33, 64, 127, 128] {
                let narrow = initial_spins(nodes, lanes, 19);
                let thresholds = rows.expand(nodes, lanes, rung, 3);
                assert_eq!(narrow.len(), nodes * lanes);
                assert_eq!(thresholds.len(), nodes * lanes);
                for node in 0..nodes {
                    assert_eq!(
                        narrow[node * lanes..(node + 1) * lanes],
                        full[node * MAX_LANES..node * MAX_LANES + lanes]
                    );
                    assert_eq!(
                        thresholds[node * lanes..(node + 1) * lanes],
                        full_thresholds[node * MAX_LANES..node * MAX_LANES + lanes]
                    );
                }
            }
        }
    }

    #[test]
    fn splitmix64_matches_reference_outputs() {
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn initial_spins_are_physical_lane_order_and_reproducible() {
        let first = initial_spins(2, MAX_LANES, 7);
        let rerun = initial_spins(2, MAX_LANES, 7);
        let other = initial_spins(2, MAX_LANES, 8);

        assert_eq!(first.len(), 2 * 128);
        assert_eq!(first, rerun);
        assert_ne!(first, other);
        assert!(first.iter().all(|&spin| spin == -1 || spin == 1));
        assert_eq!(&first[..8], &[-1, 1, 1, 1, 1, -1, 1, 1]);
    }

    #[test]
    fn threshold_cuts_honor_every_nonzero_boundary() {
        let cut = cuts(0.5);
        for m in 1..=63 {
            if cut[m] == 0 {
                continue;
            }
            let below = u32::try_from(cut[m] - 1).unwrap();
            let at = u32::try_from(cut[m]).unwrap();
            assert!(draw_threshold(&cut, below) >= m as u8, "m={m}");
            assert!(draw_threshold(&cut, at) < m as u8, "m={m}");
        }
    }

    #[test]
    fn threshold_binary_search_matches_former_reverse_scan() {
        fn former(cut: &[u64; 64], uniform: u32) -> u8 {
            (1..=63)
                .rev()
                .find(|&m| u64::from(uniform) < cut[m])
                .map_or(0, |m| m as u8)
        }

        let mut tables: Vec<[u64; 64]> = [
            0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 10.0, 1_000.0,
        ]
        .into_iter()
        .map(cuts)
        .collect();

        tables.push([1u64 << 32; 64]);

        let mut none_accept = [0u64; 64];
        none_accept[0] = 1u64 << 32;
        tables.push(none_accept);

        let mut plateau = [0u64; 64];
        plateau[0] = 1u64 << 32;
        plateau[1..=20].fill(1_000);
        plateau[21..=40].fill(500);
        tables.push(plateau);

        for cut in &tables {
            let mut uniforms = vec![0, 1, u32::MAX / 2, u32::MAX];
            for cutoff in cut.iter().copied().skip(1) {
                let Ok(at) = u32::try_from(cutoff) else {
                    continue;
                };
                uniforms.push(at);
                if at > 0 {
                    uniforms.push(at - 1);
                }
                if at < u32::MAX {
                    uniforms.push(at + 1);
                }
            }
            for uniform in uniforms {
                assert_eq!(
                    draw_threshold(cut, uniform),
                    former(cut, uniform),
                    "cut[1]={} cut[63]={} u={uniform}",
                    cut[1],
                    cut[63]
                );
            }
        }
    }

    #[test]
    fn threshold_extremes_cover_zero_and_underflowed_beta() {
        let zero = cuts(0.0);
        assert_eq!(draw_threshold(&zero, 0), 63);
        assert_eq!(draw_threshold(&zero, u32::MAX), 63);

        let frozen = cuts(1_000.0);
        assert!(frozen[1..].iter().all(|&cut| cut == 0));
        assert_eq!(draw_threshold(&frozen, 0), 0);
        assert_eq!(draw_threshold(&frozen, u32::MAX), 0);
    }

    #[test]
    fn expansion_shares_groups_wraps_rows_and_reuses_rung_values() {
        let mut thresholds = ThresholdRows::new(19);
        thresholds.begin_rung(0.05);
        let first = thresholds.expand(8_193, MAX_LANES, 3, 5);
        let rerun = thresholds.expand(8_193, MAX_LANES, 3, 5);

        assert_eq!(first, rerun);
        assert_eq!(first.len(), 8_193 * 128);
        for node in [0, 1, 8_191, 8_192] {
            let row = &first[node * 128..(node + 1) * 128];
            for group in 0..4 {
                let group_values = &row[group * 32..(group + 1) * 32];
                assert!(group_values.iter().all(|&value| value == group_values[0]));
            }
        }
        assert_eq!(&first[..128], &first[8_192 * 128..8_193 * 128]);
        assert!((0..8_193).any(|node| { first[node * 128] != first[node * 128 + READS_PER_GROUP] }));
    }

    #[test]
    fn threshold_streams_repeat_by_seed_and_advance_by_rung() {
        let mut first = ThresholdRows::new(23);
        let mut rerun = ThresholdRows::new(23);
        first.begin_rung(0.75);
        rerun.begin_rung(0.75);
        assert_eq!(
            first.expand(64, MAX_LANES, 0, 0),
            rerun.expand(64, MAX_LANES, 0, 0)
        );

        let prior = first.expand(64, MAX_LANES, 0, 0);
        first.begin_rung(0.75);
        // Reuse the same (rung, sweep) so the expansion offsets are identical;
        // the next expansion must differ because the random streams advanced.
        let next = first.expand(64, MAX_LANES, 0, 0);
        assert_ne!(prior, next);
    }

    #[test]
    fn threshold_rows_use_seeded_group_streams_and_upper_bits() {
        let mut thresholds = ThresholdRows::new(19);
        thresholds.begin_rung(0.05);

        let expected = [
            [2, 5, 54, 11, 8, 2, 7, 0],
            [7, 19, 52, 19, 9, 3, 3, 27],
            [3, 7, 6, 4, 9, 8, 4, 3],
            [9, 0, 35, 5, 1, 6, 16, 11],
        ];
        for (row, expected_prefix) in thresholds.rows.iter().zip(expected) {
            assert_eq!(&row[..expected_prefix.len()], expected_prefix);
        }
    }

    #[test]
    fn expansion_offsets_change_by_sweep_without_mutating_rows() {
        let mut thresholds = ThresholdRows::new(29);
        thresholds.begin_rung(0.5);
        let first = thresholds.expand(128, MAX_LANES, 4, 0);
        let later = thresholds.expand(128, MAX_LANES, 4, 1);
        let repeated = thresholds.expand(128, MAX_LANES, 4, 0);
        assert_ne!(first, later);
        assert_eq!(first, repeated);
    }

    #[test]
    fn validate_params_checks_every_host_limit() {
        let valid = SampleParams {
            num_reads: 128,
            num_sweeps: 65_536,
            sweeps_per_beta: 1,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: Some((0.25, 4.0)),
            seed: 0,
        };
        assert!(validate_params(&valid).is_ok());

        for num_reads in [0, 129] {
            assert!(validate_params(&SampleParams {
                num_reads,
                ..valid.clone()
            })
            .is_err());
        }
        assert!(validate_params(&SampleParams {
            num_sweeps: 65_537,
            ..valid.clone()
        })
        .is_err());
        assert!(validate_params(&SampleParams {
            sweeps_per_beta: 0,
            ..valid.clone()
        })
        .is_err());

        for beta_range in [
            Some((0.0, 1.0)),
            Some((-1.0, 1.0)),
            Some((1.0, 0.5)),
            Some((f64::NAN, 1.0)),
            Some((1.0, f64::NAN)),
            Some((f64::INFINITY, 1.0)),
            Some((1.0, f64::INFINITY)),
        ] {
            assert!(validate_params(&SampleParams {
                beta_range,
                ..valid.clone()
            })
            .is_err());
        }
        assert!(validate_params(&SampleParams {
            beta_range: Some((1.0, 1.0)),
            ..valid
        })
        .is_ok());
    }

    #[test]
    fn schedule_keeps_the_last_partial_rung_and_exact_endpoints() {
        let graph = IsingGraph::new(vec![1.0], vec![], vec![]);
        let params = SampleParams {
            num_sweeps: 5,
            sweeps_per_beta: 2,
            beta_range: Some((0.25, 4.0)),
            ..SampleParams::default()
        };
        let rungs = schedule(&graph, &params).unwrap();
        assert_eq!(
            rungs.iter().map(|rung| rung.sweeps).collect::<Vec<_>>(),
            [2, 2, 1]
        );
        assert!((rungs[0].beta - 0.25).abs() < 1e-12);
        assert!((rungs[2].beta - 4.0).abs() < 1e-12);
    }

    #[test]
    fn a_linear_schedule_spaces_its_rungs_evenly() {
        let graph = IsingGraph::new(vec![1.0], vec![], vec![]);
        let params = SampleParams {
            num_sweeps: 5,
            beta_range: Some((1.0, 5.0)),
            schedule: quip_solver_core::BetaSchedule::Linear,
            ..SampleParams::default()
        };
        let betas: Vec<f64> = schedule(&graph, &params)
            .unwrap()
            .iter()
            .map(|rung| rung.beta)
            .collect();
        assert_eq!(betas, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn schedule_returns_every_requested_sweep_or_rejects_the_cap() {
        let graph = IsingGraph::new(vec![0.0], vec![], vec![]);
        for sweeps_per_beta in [1, 2, 7] {
            for num_sweeps in [0, 1, 2, 5, 65_536] {
                let params = SampleParams {
                    num_sweeps,
                    sweeps_per_beta,
                    ..SampleParams::default()
                };
                let rungs = schedule(&graph, &params).unwrap();
                assert_eq!(
                    rungs.iter().map(|rung| rung.sweeps).sum::<usize>(),
                    num_sweeps
                );
                assert!(rungs.iter().all(|rung| rung.sweeps <= sweeps_per_beta));
                assert_eq!(rungs.len(), num_sweeps.div_ceil(sweeps_per_beta));
            }
            let invalid = SampleParams {
                num_sweeps: 65_537,
                sweeps_per_beta,
                ..SampleParams::default()
            };
            assert!(schedule(&graph, &invalid).is_err());
        }
    }

    #[test]
    fn one_rung_uses_the_cold_endpoint() {
        let graph = IsingGraph::new(vec![1.0], vec![], vec![]);
        let params = SampleParams {
            num_sweeps: 1,
            sweeps_per_beta: 7,
            beta_range: Some((0.25, 4.0)),
            ..SampleParams::default()
        };
        assert_eq!(schedule(&graph, &params).unwrap()[0].beta, 4.0);
    }
}
