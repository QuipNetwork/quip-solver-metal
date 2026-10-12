// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Host-only decisions and schedules for the always-on MSA cascade.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::sampler::build_beta_schedule;
use crate::slots::Edges;
use crate::{IsingGraph, SampleParams};
use quip_solver_core::beta::default_ising_beta_range;
use quip_solver_core::StreamJob;

use crate::cutoff::{Cutoff, CutoffConfig};
use crate::model_checks::{Action, AuditLane, Drift, Yield};

pub(crate) const MAX_STAGES: usize = 16;

/// Stage budgets padded with zeros to `MAX_STAGES`.
pub(crate) fn stage_array(stages: &[usize]) -> [usize; MAX_STAGES] {
    let mut out = [0; MAX_STAGES];
    out[..stages.len()].copy_from_slice(stages);
    out
}

/// Arm D constants from docs/perf/2026-09-23-seeded-segments.md.
/// Times describe a 64-read job: a + b * sweeps (unchanged from G4).
pub(crate) struct Calibration {
    pub(crate) stages: &'static [usize],
    /// Audit-lane miss rate r0 for each probe-to-next-stage transition.
    pub(crate) false_negative: &'static [((usize, usize), f64)],
    pub(crate) skew: &'static [f64],
    pub(crate) excess_kurtosis: &'static [f64],
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retained calibration checked by the Arm D regression test"
        )
    )]
    pub(crate) cost_a_us: f64,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retained calibration checked by the Arm D regression test"
        )
    )]
    pub(crate) cost_b_us: f64,
    pub(crate) k0: f64,
}

pub(crate) const CALIBRATION: Calibration = Calibration {
    stages: &[32, 256],
    false_negative: &[((32, 256), 0.0353), ((256, 14_336), 0.0051)],
    skew: &[-0.123, -0.056],
    excess_kurtosis: &[0.014, 0.002],
    cost_a_us: 124.0,
    cost_b_us: 2.284,
    k0: 1.0,
};

fn calibrated_miss_rate(from: usize, to: usize) -> Option<f64> {
    CALIBRATION
        .false_negative
        .iter()
        .find_map(|&(transition, rate)| (transition == (from, to)).then_some(rate))
}

fn calibrated_rate_from(from: usize) -> Option<f64> {
    CALIBRATION
        .false_negative
        .iter()
        .find_map(|&((start, _), rate)| (start == from).then_some(rate))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CascadeSettings {
    pub(crate) stages: [usize; MAX_STAGES],
    /// Fixed best-energy gate per checkpoint. A gated stage keeps a job when
    /// its best is at or below the gate and bypasses the adaptive cutoff.
    /// Only [`CascadeSettings::effective`] sets gates, for [`CHAIN_GATES`].
    pub(crate) gates: [Option<i64>; MAX_STAGES],
    /// Keep every job at every checkpoint. For gate measurement only.
    pub(crate) open_gates: bool,
    /// Overall probe-to-full denominator, factored across configured probes.
    pub(crate) keep: f64,
    pub(crate) keep_min: f64,
    pub(crate) keep_max: f64,
    pub(crate) audit: u32,
    pub(crate) yield_per_million: Option<f64>,
}

impl Default for CascadeSettings {
    fn default() -> Self {
        Self {
            stages: stage_array(CALIBRATION.stages),
            gates: [None; MAX_STAGES],
            open_gates: false,
            keep: 2_000.0,
            keep_min: 1_000.0,
            keep_max: 30_000.0,
            audit: 200,
            yield_per_million: None,
        }
    }
}

/// Gates for the chain topology `cbec1eb4` (4,577 nodes, 41,514 edges, h = 0,
/// J = ±1), measured at 64 reads and one sweep per beta on a schedule where
/// every leg reheated the previous leg's spins to beta 0.25
/// (`tests/gate_trace.rs`, removed in 44a4126). The screening legs, up to 256
/// sweeps, still run that way. The deep legs are now fresh anneals of their
/// full budget (see [`segment_schedule`]), so the deep gates apply to a
/// schedule they were not measured on until they are measured again. The four screening gates are the
/// shallowest checkpoint best over 30 seeds of the chain's 100 deepest winners
/// (-14,742 to -14,708), plus 10. The deep gates, one per doubling from 512 to
/// 524,288, are the shallowest best among 10 seeds of the same winners that
/// had not yet reached -14,710 but did later, plus 20. Winners shallower than
/// about -14,700 can fall outside the gates.
pub(crate) struct ChainGates {
    pub(crate) fingerprint: u64,
    pub(crate) stages: &'static [usize],
    pub(crate) gates: &'static [i64],
    pub(crate) min_reads: usize,
    /// Final budget for a gated job, whatever the job asked for. Few models
    /// survive the last gate, so a deep final stage costs little.
    pub(crate) full_sweeps: usize,
}

pub(crate) const CHAIN_GATES: ChainGates = ChainGates {
    fingerprint: 0x38cd_e7d7_931d_f32f,
    stages: &[
        8, 16, 64, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
        524_288,
    ],
    gates: &[
        -13_252_000,
        -13_836_000,
        -14_276_000,
        -14_466_000,
        -14_512_000,
        -14_552_000,
        -14_584_000,
        -14_608_000,
        -14_624_000,
        -14_640_000,
        -14_648_000,
        -14_658_000,
        -14_662_000,
        -14_670_000,
        -14_678_000,
    ],
    min_reads: 64,
    full_sweeps: 1_048_576,
};

/// FNV-1a over the node count and the sorted, normalized edge list, so edge
/// order and orientation do not change the result.
pub(crate) fn topology_fingerprint(nodes: usize, edges: &[(usize, usize)]) -> u64 {
    let mut sorted: Vec<(usize, usize)> =
        edges.iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
    sorted.sort_unstable();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let words = std::iter::once(nodes).chain(sorted.into_iter().flat_map(|(u, v)| [u, v]));
    for word in words {
        for byte in (word as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

impl CascadeSettings {
    /// True when a job on a chain-topology graph runs under the conditions
    /// the chain gates were measured at.
    pub(crate) fn chain_gated(&self, chain_topology: bool, params: &SampleParams) -> bool {
        chain_topology
            && params.num_reads >= CHAIN_GATES.min_reads
            && params.sweeps_per_beta == 1
            && params.beta_range.is_none()
            && params.schedule == quip_solver_core::BetaSchedule::Geometric
    }

    /// The configured settings, or the chain-gated plan when `gated`. Open
    /// gates keep the configured stages, so a study can place checkpoints.
    pub(crate) fn effective(self, gated: bool) -> Self {
        if gated && !self.open_gates {
            let mut gates = [None; MAX_STAGES];
            for (gate, &value) in gates.iter_mut().zip(CHAIN_GATES.gates) {
                *gate = Some(value);
            }
            Self {
                stages: stage_array(CHAIN_GATES.stages),
                gates,
                ..self
            }
        } else {
            self
        }
    }
}

#[derive(serde::Deserialize, Default, Clone)]
pub(crate) struct CascadeToml {
    pub(crate) cascade_stages: Option<Vec<usize>>,
    pub(crate) cascade_open_gates: Option<bool>,
    pub(crate) cascade_keep: Option<u32>,
    pub(crate) cascade_keep_min: Option<u32>,
    pub(crate) cascade_keep_max: Option<u32>,
    pub(crate) cascade_audit: Option<u32>,
    pub(crate) cascade_yield_per_million: Option<f64>,
}

impl CascadeSettings {
    /// Merge valid keys, retaining previous values for invalid keys. The three
    /// overall keep denominators are validated and accepted as one group.
    pub(crate) fn merge(&mut self, cfg: &CascadeToml) {
        if let Some(stages) = &cfg.cascade_stages {
            if (1..=MAX_STAGES).contains(&stages.len())
                && stages[0] >= 1
                && stages.windows(2).all(|pair| pair[0] < pair[1])
            {
                self.stages = stage_array(stages);
            } else {
                tracing::warn!(
                    ?stages,
                    "invalid cascade_stages; retaining previous budgets"
                );
            }
        }
        if let Some(open) = cfg.cascade_open_gates {
            self.open_gates = open;
        }
        let keep = cfg.cascade_keep.map_or(self.keep, f64::from);
        let min = cfg.cascade_keep_min.map_or(self.keep_min, f64::from);
        let max = cfg.cascade_keep_max.map_or(self.keep_max, f64::from);
        if min >= 2.0 && min <= keep && keep <= max {
            self.keep = keep;
            self.keep_min = min;
            self.keep_max = max;
        } else {
            tracing::warn!(
                keep,
                min,
                max,
                "invalid cascade keep denominators; retaining previous values"
            );
        }
        if let Some(audit) = cfg.cascade_audit {
            if audit >= 2 {
                self.audit = audit;
            } else {
                tracing::warn!(audit, "invalid cascade_audit; retaining previous value");
            }
        }
        if let Some(value) = cfg.cascade_yield_per_million {
            if value.is_finite() && value > 0.0 {
                self.yield_per_million = Some(value);
            } else {
                tracing::warn!(
                    value,
                    "invalid cascade_yield_per_million; retaining previous value"
                );
            }
        }
    }
}

fn kept_median(kept: &VecDeque<i64>) -> Option<f64> {
    if kept.is_empty() {
        return None;
    }
    let mut sorted: Vec<_> = kept.iter().copied().collect();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some(sorted[mid - 1] as f64 / 2.0 + sorted[mid] as f64 / 2.0)
    } else {
        Some(sorted[mid] as f64)
    }
}

pub(crate) struct Ticket {
    pub(crate) gates: usize,
    pub(crate) stage: usize,
    pub(crate) audited: bool,
    pub(crate) topology_epoch: u64,
    pub(crate) yield_epoch: u64,
    /// The lease round this unit was admitted in.
    round: u64,
    /// Set by [`Controller::checkpoint`] when a target hit ends the unit early,
    /// to the sweep count of the checkpoint that hit. `None` otherwise.
    pub(crate) finish_sweeps: Option<usize>,
    // Preserve admitted transitions and controllers across settings changes.
    plan: Arc<Mutex<StagePlan>>,
    final_sweeps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
    Audit,
    Drift,
    Yield,
}

struct StagePlan {
    settings: CascadeSettings,
    cutoffs: Vec<Cutoff>,
    audit_lanes: Vec<AuditLane>,
    kept: Vec<VecDeque<i64>>,
    audit_rng: Option<u64>,
    audit_loosen: Vec<f64>,
    audit_unavailable_logged: Vec<bool>,
    drift: Drift,
}

pub(crate) struct Controller {
    settings: CascadeSettings,
    /// The lease target the yield check measures against.
    yield_target_milli: Option<i64>,
    plan: Arc<Mutex<StagePlan>>,
    yield_check: Option<Yield>,
    topology: Option<(usize, Edges)>,
    topology_epoch: u64,
    yield_epoch: u64,
    stats: Stats,
    /// The newest lease round admitted; see [`Self::enter_round`].
    round: u64,
    /// Lowest and highest final energy over this round's finished models,
    /// as `(best, worst)`.
    range: Option<(i64, i64)>,
}

/// Pass counts at one checkpoint during a report window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LevelStats {
    pub(crate) sweeps: usize,
    pub(crate) checked: u64,
    pub(crate) kept: u64,
}

/// Per-window counts for the periodic cascade report. Level `k` is
/// checkpoint `k`.
pub(crate) struct Stats {
    started: Instant,
    pub(crate) models: u64,
    pub(crate) levels: [Option<LevelStats>; MAX_STAGES],
}

impl Stats {
    fn new(now: Instant) -> Self {
        Self {
            started: now,
            models: 0,
            levels: [None; MAX_STAGES],
        }
    }

    fn record(&mut self, level: usize, sweeps: usize, kept: bool) {
        let entry = self.levels[level].get_or_insert(LevelStats {
            sweeps,
            checked: 0,
            kept: 0,
        });
        // A settings change mid-window can move a level's budget.
        entry.sweeps = sweeps;
        entry.checked += 1;
        entry.kept += u64::from(kept);
    }
}

/// Milli energy as a whole-unit string; chain energies are unit integers.
fn energy(milli: i64) -> String {
    if milli % 1000 == 0 {
        (milli / 1000).to_string()
    } else {
        format!("{:.3}", milli as f64 / 1000.0)
    }
}

impl StagePlan {
    fn new(settings: CascadeSettings) -> Self {
        let count = settings.stages.iter().filter(|&&s| s > 0).count();
        // Keep one persistent controller per configured index. Use configured
        // count for the root, even when a job has fewer effective probes, so
        // mixed budgets do not rebuild controllers and lose their observations.
        // Such jobs skip the unused transitions and have a looser overall keep.
        let root = 1.0 / count.max(1) as f64;
        let cfg = CutoffConfig {
            keep: settings.keep.powf(root),
            keep_min: settings.keep_min.powf(root),
            keep_max: settings.keep_max.powf(root),
            k0: CALIBRATION.k0,
            ..CutoffConfig::default()
        };
        Self {
            settings,
            cutoffs: (0..=count).map(|_| Cutoff::new(cfg)).collect(),
            audit_lanes: (0..count)
                .map(|s| AuditLane::new(calibrated_rate_from(settings.stages[s]).unwrap_or(0.0)))
                .collect(),
            kept: (0..count).map(|_| VecDeque::with_capacity(256)).collect(),
            audit_rng: None,
            audit_loosen: vec![1.0; count],
            audit_unavailable_logged: vec![false; count],
            drift: Drift::new(CALIBRATION.skew[0], CALIBRATION.excess_kurtosis[0]),
        }
    }

    fn audit_supported(&mut self, ticket: &Ticket, stage: usize) -> bool {
        let from = self.settings.stages[stage];
        let to = if stage + 1 < ticket.gates {
            self.settings.stages[stage + 1]
        } else {
            ticket.final_sweeps
        };
        let supported = calibrated_miss_rate(from, to).is_some();
        if !supported && !self.audit_unavailable_logged[stage] {
            tracing::debug!(stage, from, to, "cascade audit unavailable for transition");
            self.audit_unavailable_logged[stage] = true;
        }
        supported
    }

    fn audit_selected(&mut self, stage: usize) -> bool {
        let Some(state) = &mut self.audit_rng else {
            return false;
        };
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state % u64::from(self.audit_lanes[stage].denominator(self.settings.audit)) == 0
    }

    fn apply_action(
        &mut self,
        stage: usize,
        action: Action,
        check: Check,
        observed: &[f64],
        expected: &[f64],
    ) {
        let cutoff = &mut self.cutoffs[stage];
        match action {
            Action::None => return,
            Action::Loosen { raise_audit } => {
                let factor = cutoff.loosen_step();
                if check == Check::Audit {
                    self.audit_loosen[stage] *= factor;
                }
                if raise_audit {
                    self.audit_lanes[stage].raise();
                }
            }
            Action::Restore => {
                cutoff.restore_step(self.audit_loosen[stage]);
                self.audit_loosen[stage] = 1.0;
            }
            Action::LoosenAndDoubleK0 => {
                cutoff.loosen_step();
                cutoff.double_k0();
            }
        }
        tracing::warn!(
            stage,
            ?check,
            ?action,
            ?observed,
            ?expected,
            denominator = cutoff.denominator(),
            audit_denominator = self.audit_lanes[stage].denominator(self.settings.audit),
            "cascade model check"
        );
    }
}

impl Controller {
    pub(crate) fn new(settings: CascadeSettings) -> Self {
        Self {
            settings,
            yield_target_milli: None,
            plan: Arc::new(Mutex::new(StagePlan::new(settings))),
            yield_check: None,
            topology: None,
            topology_epoch: 0,
            yield_epoch: 0,
            stats: Stats::new(Instant::now()),
            round: 0,
            range: None,
        }
    }

    /// Log the window's rate and pass rates, and this round's energy range,
    /// when `period` has passed, then start a new window.
    pub(crate) fn report(&mut self, now: Instant, period: Duration) {
        let elapsed = now.saturating_duration_since(self.stats.started);
        if elapsed < period {
            return;
        }
        let stats = std::mem::replace(&mut self.stats, Stats::new(now));
        let secs = elapsed.as_secs_f64().max(f64::EPSILON);
        let pass: Vec<String> = stats
            .levels
            .iter()
            .flatten()
            .map(|l| {
                let percent = 100.0 * l.kept as f64 / l.checked.max(1) as f64;
                format!("{}sw:{percent:.2}%", l.sweeps)
            })
            .collect();
        let (best, worst) = self.range.map_or((String::new(), String::new()), |(b, w)| {
            (energy(b), energy(w))
        });
        tracing::info!(
            models_per_s = (stats.models as f64 / secs).round() as u64,
            best = %best,
            worst = %worst,
            pass = %pass.join(" "),
            "cascade report"
        );
    }

    /// Rebuild on a stages/keep change, update audit and yield independently.
    pub(crate) fn refresh(&mut self, settings: CascadeSettings) {
        if settings.stages != self.settings.stages
            || settings.keep != self.settings.keep
            || settings.keep_min != self.settings.keep_min
            || settings.keep_max != self.settings.keep_max
        {
            self.plan = Arc::new(Mutex::new(StagePlan::new(settings)));
        }
        if settings.yield_per_million != self.settings.yield_per_million {
            self.yield_epoch = self.yield_epoch.wrapping_add(1);
            self.yield_check = settings
                .yield_per_million
                .zip(self.yield_target_milli)
                .map(|(rate, target)| Yield::new(rate, target, Duration::from_secs(3600)));
        }
        self.settings = settings;
        let mut plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
        plan.settings.audit = settings.audit;
        plan.settings.open_gates = settings.open_gates;
    }

    /// The session target of an admitted lease unit. A change resets the yield check.
    pub(crate) fn set_yield_target(&mut self, target_milli: Option<i64>) {
        if target_milli == self.yield_target_milli {
            return;
        }
        self.yield_target_milli = target_milli;
        self.yield_epoch = self.yield_epoch.wrapping_add(1);
        self.yield_check = self
            .settings
            .yield_per_million
            .zip(target_milli)
            .map(|(rate, target)| Yield::new(rate, target, Duration::from_secs(3600)));
    }

    #[cfg(test)]
    pub(crate) fn matches_topology(&self, graph: &IsingGraph) -> bool {
        self.topology
            .as_ref()
            .is_some_and(|(n, edges)| *n == graph.num_nodes() && edges[..] == graph.edges[..])
    }

    /// Whether `edges` is the current topology. Storage already seen skips the
    /// edge compare. Equal edges in new storage replace the stored copy.
    fn same_topology(&mut self, nodes: usize, edges: &Edges) -> bool {
        let Some((n, current)) = &mut self.topology else {
            return false;
        };
        if *n != nodes {
            return false;
        }
        if Arc::ptr_eq(current, edges) {
            return true;
        }
        let same = current[..] == edges[..];
        if same {
            *current = Arc::clone(edges);
        }
        same
    }

    /// Reset gate state on a gated topology change. Return the job's schedule plan.
    #[cfg(test)]
    pub(crate) fn admit(&mut self, job: &StreamJob) -> (Ticket, Vec<f32>, Vec<usize>) {
        let (ticket, schedule) = self.admit_schedule(job);
        (ticket, schedule.betas.to_vec(), schedule.checkpoints)
    }

    /// [`Self::admit`] with the whole prepared schedule, for slot jobs.
    #[cfg(test)]
    pub(crate) fn admit_schedule(&mut self, job: &StreamJob) -> (Ticket, PreparedSchedule) {
        let chain = topology_fingerprint(job.graph.num_nodes(), &job.graph.edges)
            == CHAIN_GATES.fingerprint;
        let gated = self.settings.chain_gated(chain, &job.params);
        let mut schedule = PreparedSchedule::new(job, self.settings, gated, true);
        let edges = Arc::clone(&job.graph.edges);
        let ticket = self.admit_prepared(job, &mut schedule, &edges).unwrap();
        (ticket, schedule)
    }

    /// `edges` must equal the job's edges; preparation verified them.
    pub(crate) fn admit_prepared(
        &mut self,
        job: &StreamJob,
        schedule: &mut PreparedSchedule,
        edges: &Edges,
    ) -> Result<Ticket, crate::sampler::SampleError> {
        // Only settings changes rebuild on the runner. Queued jobs must use
        // the current gate plan.
        if schedule.settings.stages != self.settings.stages {
            let rebuilt =
                PreparedSchedule::new(job, self.settings, schedule.gated, schedule.screen);
            crate::slots::validate_schedule(&rebuilt.betas, &rebuilt.checkpoints)?;
            *schedule = rebuilt;
        }
        if schedule.checkpoints.len() > 1 {
            let wanted = self.settings.effective(schedule.gated);
            if !self.same_topology(job.graph.num_nodes(), edges) {
                self.plan = Arc::new(Mutex::new(StagePlan::new(wanted)));
                self.topology = Some((job.graph.num_nodes(), Arc::clone(edges)));
                self.topology_epoch = self.topology_epoch.wrapping_add(1);
            } else {
                let plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
                let stale =
                    plan.settings.stages != wanted.stages || plan.settings.gates != wanted.gates;
                drop(plan);
                if stale {
                    self.plan = Arc::new(Mutex::new(StagePlan::new(wanted)));
                }
            }
            self.plan
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .audit_rng
                .get_or_insert(job.params.seed.max(1));
        }
        if let Some(check) = &mut self.yield_check {
            check.admit();
        }
        Ok(Ticket {
            gates: schedule.checkpoints.len() - 1,
            stage: 0,
            audited: false,
            topology_epoch: self.topology_epoch,
            yield_epoch: self.yield_epoch,
            round: self.round,
            finish_sweeps: None,
            plan: Arc::clone(&self.plan),
            final_sweeps: job.params.num_sweeps,
        })
    }

    /// Called at a non-last checkpoint with the unit's target. true = keep
    /// running. A target hit strictly below the target (the chain rule) ends
    /// the unit now: `false`, with `ticket.finish_sweeps` set so the caller
    /// delivers the current reads instead of screening the unit out.
    pub(crate) fn checkpoint(
        &mut self,
        ticket: &mut Ticket,
        best: i64,
        target_milli: Option<i64>,
    ) -> bool {
        if ticket.topology_epoch != self.topology_epoch {
            return false;
        }
        let mut plan = ticket.plan.lock().unwrap_or_else(|p| p.into_inner());
        Self::observe_transition(&mut plan, ticket, best);
        let stage = ticket.stage;
        let sweeps = plan.settings.stages[stage];
        if let Some(target) = target_milli {
            if best < target {
                self.stats.record(stage, sweeps, true);
                ticket.finish_sweeps = Some(sweeps);
                return false;
            }
            if best == target {
                self.stats.record(stage, sweeps, true);
                ticket.audited = false;
                ticket.stage += 1;
                return true;
            }
        }
        if plan.settings.open_gates {
            self.stats.record(stage, sweeps, true);
            ticket.stage += 1;
            return true;
        }
        if let Some(gate) = plan.settings.gates[stage] {
            let keep = best <= gate;
            self.stats.record(stage, sweeps, keep);
            ticket.audited = false;
            ticket.stage += usize::from(keep);
            return keep;
        }
        let keep = plan.cutoffs[stage].decide(best as f64);
        ticket.audited = !keep && plan.audit_supported(ticket, stage) && plan.audit_selected(stage);
        if stage == 0 && plan.settings.stages[0] == 32 {
            let m = plan.cutoffs[0].moments().clone();
            let observed = [m.skew(), m.excess_kurtosis()];
            let action = plan.drift.check(&m);
            plan.apply_action(
                0,
                action,
                Check::Drift,
                &observed,
                &[CALIBRATION.skew[0], CALIBRATION.excess_kurtosis[0]],
            );
        }
        self.stats.record(stage, sweeps, keep || ticket.audited);
        if keep || ticket.audited {
            ticket.stage += 1;
            true
        } else {
            false
        }
    }

    fn observe_transition(plan: &mut StagePlan, ticket: &Ticket, best: i64) {
        if ticket.stage == 0 || !plan.audit_supported(ticket, ticket.stage - 1) {
            return;
        }
        let previous = ticket.stage - 1;
        if ticket.audited {
            if let Some(median) = kept_median(&plan.kept[previous]) {
                let miss = best as f64 <= median;
                let (observed, expected) = plan.audit_lanes[previous].observation(miss);
                let action = plan.audit_lanes[previous].record(miss);
                plan.apply_action(previous, action, Check::Audit, &[observed], &[expected]);
            }
        } else {
            let kept = &mut plan.kept[previous];
            if kept.len() == 256 {
                kept.pop_front();
            }
            kept.push_back(best);
        }
    }

    /// Start a new report range when a salt from a newer lease round
    /// arrives. Units from older rounds still finish, outside the range.
    pub(crate) fn enter_round(&mut self, round: u64) {
        if round > self.round {
            self.round = round;
            self.range = None;
        }
    }

    /// Called once per job with its final best (None for error or cancel).
    pub(crate) fn finish(&mut self, ticket: &Ticket, best: Option<i64>, delivered: bool) {
        if let Some(best) = best {
            self.stats.models += 1;
            if ticket.round == self.round {
                self.range = Some(
                    self.range
                        .map_or((best, best), |(b, w)| (b.min(best), w.max(best))),
                );
            }
            if ticket.topology_epoch == self.topology_epoch
                && ticket.stage == ticket.gates
                && ticket.gates > 0
            {
                let mut plan = ticket.plan.lock().unwrap_or_else(|p| p.into_inner());
                Self::observe_transition(&mut plan, ticket, best);
                let final_index = plan.cutoffs.len() - 1;
                plan.cutoffs[final_index].observe(best as f64);
            }
            if delivered && ticket.yield_epoch == self.yield_epoch {
                if let Some(check) = &mut self.yield_check {
                    check.result(best);
                }
            }
        }
    }

    fn check_yield(&mut self, now: Instant) {
        if let Some(check) = &mut self.yield_check {
            let (observed, expected) = check.observation();
            let action = check.tick(now);
            let mut plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
            for stage in 0..plan.audit_lanes.len() {
                plan.apply_action(stage, action, Check::Yield, &[observed], &[expected]);
            }
        }
    }

    /// live[k] = live slots whose next checkpoint is k; capacity = total slots.
    pub(crate) fn update_load(&mut self, busy: f64, live: &[usize], capacity: usize) {
        self.check_yield(Instant::now());
        let slack = (0.9 - busy).max(0.0) / 0.9;
        let mut plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
        for stage in 0..plan.cutoffs.len() - 1 {
            let at_next = live.get(stage + 1).copied().unwrap_or(0);
            let reference = (capacity / 2).max(1);
            let backlog = (at_next as f64 / reference as f64).max(1.0).ln();
            let cutoff = &mut plan.cutoffs[stage];
            let error = if backlog == 0.0 && slack == 0.0 {
                -cutoff.log_adjust()
            } else {
                backlog - slack
            };
            cutoff.load_update(error, 0.1);
        }
    }
}

pub(crate) struct PreparedSchedule {
    settings: CascadeSettings,
    gated: bool,
    screen: bool,
    pub(crate) betas: Arc<[f32]>,
    pub(crate) checkpoints: Vec<usize>,
    /// Index of the first leg that starts a fresh anneal.
    pub(crate) fresh_from: usize,
}

/// Everything a segment schedule depends on. The graph enters only through
/// its beta range.
#[derive(Clone, Copy, PartialEq)]
struct ScheduleKey {
    hot: u64,
    cold: u64,
    sweeps: usize,
    per_beta: usize,
    kind: quip_solver_core::BetaSchedule,
    stages: [usize; MAX_STAGES],
}

struct CachedSchedule {
    key: ScheduleKey,
    betas: Arc<[f32]>,
    checkpoints: Vec<usize>,
    fresh_from: usize,
}

/// The last validated schedule shared by preparation workers. Equal schedules
/// retain one `Arc` identity so slots can skip uploads across worker changes.
#[derive(Clone, Default)]
pub(crate) struct ScheduleCache {
    entry: Arc<Mutex<Option<CachedSchedule>>>,
}

impl ScheduleCache {
    /// `beta_range`, when given, is the caller's [`resident_beta_range`] of
    /// `job.graph`, used when the job's parameters set none.
    pub(crate) fn prepare(
        &mut self,
        job: &StreamJob,
        settings: CascadeSettings,
        gated: bool,
        screen: bool,
        beta_range: Option<(f64, f64)>,
    ) -> Result<PreparedSchedule, crate::sampler::SampleError> {
        let stages = if screen {
            settings.effective(gated).stages
        } else {
            [0; MAX_STAGES]
        };
        let (hot, cold) = job
            .params
            .beta_range
            .or(beta_range)
            .unwrap_or_else(|| resident_beta_range(&job.graph));
        let key = ScheduleKey {
            hot: hot.to_bits(),
            cold: cold.to_bits(),
            sweeps: job.params.num_sweeps,
            per_beta: job.params.sweeps_per_beta,
            kind: job.params.schedule,
            stages,
        };
        // Compute the graph-dependent range before locking. Cache hits only
        // compare the key and clone shared storage under the lock.
        let mut cached = self.entry.lock().unwrap_or_else(|p| p.into_inner());
        let entry = match cached.as_ref() {
            Some(entry) if entry.key == key => entry,
            _ => {
                let params = SampleParams {
                    beta_range: Some((hot, cold)),
                    ..job.params.clone()
                };
                let (betas, checkpoints, fresh_from) =
                    segment_schedule(&job.graph, &params, &key.stages);
                crate::slots::validate_schedule(&betas, &checkpoints)?;
                cached.insert(CachedSchedule {
                    key,
                    betas: betas.into(),
                    checkpoints,
                    fresh_from,
                })
            }
        };
        let schedule = PreparedSchedule {
            settings,
            gated,
            screen,
            betas: Arc::clone(&entry.betas),
            checkpoints: entry.checkpoints.clone(),
            fresh_from: entry.fresh_from,
        };
        Ok(schedule)
    }
}

impl PreparedSchedule {
    /// `settings` are the configured settings. `gated` selects the chain plan.
    pub(crate) fn new(
        job: &StreamJob,
        settings: CascadeSettings,
        gated: bool,
        screen: bool,
    ) -> Self {
        let stages = if screen {
            settings.effective(gated).stages
        } else {
            [0; MAX_STAGES]
        };
        let (betas, checkpoints, fresh_from) = segment_schedule(&job.graph, &job.params, &stages);
        Self {
            settings,
            gated,
            screen,
            betas: betas.into(),
            checkpoints,
            fresh_from,
        }
    }
}

// Unit-coupling draws have minimum gap one at every nonzero effective field.
// Count incident terms instead of tracking Option<f64> minima per endpoint.
// Keep the library path for every other coefficient domain.
fn resident_beta_range(graph: &IsingGraph) -> (f64, f64) {
    resident_beta_range_from(graph, &node_degrees(graph.num_nodes(), &graph.edges))
}

/// In-range edge endpoints per node, the topology part of
/// [`resident_beta_range`]. A self-loop counts twice.
pub(crate) fn node_degrees(nodes: usize, edges: &[(usize, usize)]) -> Vec<u32> {
    let mut degrees = vec![0u32; nodes];
    for &(u, v) in edges {
        if u < nodes && v < nodes {
            degrees[u] += 1;
            degrees[v] += 1;
        }
    }
    degrees
}

/// [`resident_beta_range`] with the topology's [`node_degrees`] precomputed,
/// so a job pays one pass over its nodes instead of one over its edges.
/// `degrees` must come from exactly `graph.edges`.
pub(crate) fn resident_beta_range_from(graph: &IsingGraph, degrees: &[u32]) -> (f64, f64) {
    if graph.j.len() != graph.edges.len()
        || degrees.len() != graph.num_nodes()
        || graph.j.iter().any(|j| j.abs() != 1.0)
        || graph.h.iter().any(|&h| h != 0.0 && h.abs() != 1.0)
    {
        return default_ising_beta_range(graph);
    }
    unit_beta_range(graph.h.iter().map(|&h| h != 0.0), degrees)
}

/// The beta range of a graph whose couplings are all ±1 and whose fields are
/// all 0 or ±1, from which fields are nonzero and the node degrees.
pub(crate) fn unit_beta_range(
    field_set: impl Iterator<Item = bool>,
    degrees: &[u32],
) -> (f64, f64) {
    let terms = field_set
        .zip(degrees)
        .map(|(set, &degree)| u32::from(set) + degree);
    let (max_eff, gaps) = terms.fold((0u32, 0usize), |(max, gaps), count| {
        (max.max(count), gaps + usize::from(count != 0))
    });
    if max_eff == 0 {
        return (0.1, 1.0);
    }
    let hot = std::f64::consts::LN_2 / (2.0 * f64::from(max_eff));
    let cold = (gaps as f64 / 0.01).ln() / 2.0;
    (hot, cold.max(hot))
}

/// Legs of at most this many sweeps continue one anneal: each later
/// checkpoint reheats the previous leg's spins to [`REHEAT_BETA`] and cools
/// them again. The chain's screening gates were measured on that schedule.
pub(crate) const REHEAT_THROUGH: usize = 256;

/// Beta a continuing leg reheats to (Arm D,
/// `docs/perf/2026-09-23-seeded-segments.md`).
pub(crate) const REHEAT_BETA: f64 = 0.25;

/// A job's legs laid end to end, with each checkpoint at the schedule
/// position where a leg ends. Legs with a budget of at most
/// [`REHEAT_THROUGH`] sweeps form one continuing anneal: a standard schedule
/// to the first checkpoint, then a reheat to [`REHEAT_BETA`] cooling to the
/// cold beta for each later one, so their checkpoints sit at the stage sweep
/// counts. Every longer leg, the last one included, is a complete standard
/// anneal of its own budget, which the slot runner starts from new random
/// spins and a new RNG stream (`slots::SLOT_RESTART`). Returns the betas, the
/// checkpoints, and the index of the first fresh leg.
pub(crate) fn segment_schedule(
    graph: &IsingGraph,
    params: &SampleParams,
    stages: &[usize],
) -> (Vec<f32>, Vec<usize>, usize) {
    let budgets: Vec<usize> = stages
        .iter()
        .copied()
        .take_while(|&s| s > 0 && s < params.num_sweeps)
        .chain(std::iter::once(params.num_sweeps))
        .collect();
    let fresh_from = budgets
        .iter()
        .position(|&b| b > REHEAT_THROUGH)
        .unwrap_or(budgets.len());
    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| resident_beta_range(graph));
    let standard = |schedule: &mut Vec<f32>, sweeps: usize| {
        let (betas, repeats) = build_beta_schedule(
            graph,
            sweeps,
            params.sweeps_per_beta,
            Some((hot, cold)),
            params.schedule,
        );
        let end = schedule.len() + sweeps;
        for beta in &betas {
            schedule.extend(std::iter::repeat_n(
                *beta,
                repeats.min(end - schedule.len()),
            ));
        }
        // A non-divisible sweep count holds the final standard rung for the remainder.
        schedule.resize(end, betas.last().copied().unwrap_or(cold as f32));
    };
    let mut schedule = Vec::with_capacity(budgets.iter().sum());
    let mut checkpoints = Vec::with_capacity(budgets.len());
    let continuing = &budgets[..fresh_from];
    if let (Some(&first), Some(&last)) = (continuing.first(), continuing.last()) {
        if REHEAT_BETA > hot && REHEAT_BETA < cold {
            standard(&mut schedule, first);
            for pair in continuing.windows(2) {
                schedule.extend(
                    params
                        .schedule
                        .build(REHEAT_BETA, cold, pair[1] - pair[0])
                        .into_iter()
                        .map(|b| b as f32),
                );
            }
        } else {
            tracing::warn!(
                reheat_beta = REHEAT_BETA,
                hot,
                cold,
                "cascade reheat beta outside graph range; using standard schedule"
            );
            standard(&mut schedule, last);
        }
        checkpoints.extend_from_slice(continuing);
    }
    for &sweeps in &budgets[fresh_from..] {
        standard(&mut schedule, sweeps);
        checkpoints.push(schedule.len());
    }
    (schedule, checkpoints, fresh_from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate_array(gates: &[Option<i64>]) -> [Option<i64>; MAX_STAGES] {
        let mut out = [None; MAX_STAGES];
        out[..gates.len()].copy_from_slice(gates);
        out
    }

    #[test]
    fn resident_beta_range_matches_library_bits_on_draws() {
        let edges: Vec<(usize, usize)> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut nodes = line.split_whitespace();
                (
                    nodes.next().unwrap().parse().unwrap(),
                    nodes.next().unwrap().parse().unwrap(),
                )
            })
            .collect();
        let mut rng = 12345u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for seed in 0..512 {
            let mut graph = IsingGraph::new(
                (0..4577)
                    .map(|_| [-1.0, 0.0, 1.0][(draw() % 3) as usize])
                    .collect(),
                (0..edges.len())
                    .map(|_| if draw() & 1 == 0 { -1.0 } else { 1.0 })
                    .collect(),
                edges.clone(),
            );
            let degrees = node_degrees(graph.num_nodes(), &graph.edges);
            for nonunit in [false, true] {
                if nonunit {
                    graph.j[seed] = [0.0, 0.5, -2.0, 127.0][seed % 4];
                }
                let actual = resident_beta_range(&graph);
                let cached = resident_beta_range_from(&graph, &degrees);
                assert_eq!(
                    (actual.0.to_bits(), actual.1.to_bits()),
                    (cached.0.to_bits(), cached.1.to_bits())
                );
                let expected = default_ising_beta_range(&graph);
                assert_eq!(
                    (actual.0.to_bits(), actual.1.to_bits()),
                    (expected.0.to_bits(), expected.1.to_bits())
                );
            }
        }
        for graph in [
            IsingGraph::new(vec![], vec![], vec![]),
            IsingGraph::new(vec![0.0; 3], vec![], vec![]),
            IsingGraph::new(vec![0.0, -1.0, 0.0], vec![1.0, -1.0], vec![(0, 0), (0, 8)]),
            IsingGraph::new(vec![f64::NAN], vec![], vec![]),
        ] {
            let actual = resident_beta_range(&graph);
            let expected = default_ising_beta_range(&graph);
            assert_eq!(
                (actual.0.to_bits(), actual.1.to_bits()),
                (expected.0.to_bits(), expected.1.to_bits())
            );
        }
    }

    #[test]
    fn settings_change_rejects_invalid_rebuilt_schedule() {
        // A negative range leaves multi-rung standard legs valid but makes a
        // one-sweep leg a single negative beta: two fresh legs (300, 600)
        // pass, and a one-sweep continuing leg does not.
        let old = CascadeSettings {
            stages: stage_array(&[300]),
            ..Default::default()
        };
        let mut params = params(600, 1);
        params.beta_range = Some((-2.0, -1.0));
        let job = StreamJob {
            job_id: vec![1],
            graph: ring(),
            params,
            watermark: None,
        };
        let mut schedule = PreparedSchedule::new(&job, old, false, true);
        crate::slots::validate_schedule(&schedule.betas, &schedule.checkpoints).unwrap();
        let mut controller = Controller::new(old);
        controller.refresh(CascadeSettings {
            stages: stage_array(&[1]),
            ..old
        });
        let edges = Arc::clone(&job.graph.edges);
        assert!(matches!(
            controller.admit_prepared(&job, &mut schedule, &edges),
            Err(crate::sampler::SampleError::Driver(_))
        ));
        assert!(
            controller.topology.is_none(),
            "rejection must precede ticket side effects"
        );
    }

    #[test]
    fn segment_schedule_reheats_through_256_then_runs_fresh_legs() {
        let graph = ring();
        let kinds = [
            quip_solver_core::BetaSchedule::Geometric,
            quip_solver_core::BetaSchedule::Linear,
        ];
        for (kind, range) in kinds
            .into_iter()
            .flat_map(|kind| [None, Some((0.1, 6.0))].map(|range| (kind, range)))
        {
            for sweeps in [1, 31, 32, 33, 256, 257, 1000] {
                for repeats in [1, 4, 64] {
                    let mut params = params(sweeps, repeats);
                    params.beta_range = range;
                    params.schedule = kind;
                    let (actual, checkpoints, fresh_from) =
                        segment_schedule(&graph, &params, &[32, 256]);
                    let (_, cold) = range.unwrap_or_else(|| default_ising_beta_range(&graph));
                    let budgets: Vec<usize> = [32, 256]
                        .into_iter()
                        .filter(|&s| s < sweeps)
                        .chain(std::iter::once(sweeps))
                        .collect();
                    let lone = |leg: usize| {
                        let (standard, repeat) =
                            build_beta_schedule(&graph, leg, repeats, range, kind);
                        let mut part: Vec<_> = standard
                            .iter()
                            .flat_map(|b| std::iter::repeat_n(*b, repeat))
                            .take(leg)
                            .collect();
                        part.resize(leg, standard.last().copied().unwrap_or(cold as f32));
                        part
                    };
                    // Budgets up to 256 continue one anneal with reheated
                    // legs; each longer budget is the standard schedule a
                    // lone job of that many sweeps would run.
                    let continuing = budgets.iter().filter(|&&b| b <= REHEAT_THROUGH).count();
                    let mut expected = Vec::new();
                    let mut ends = Vec::new();
                    if continuing > 0 {
                        expected.extend(lone(budgets[0]));
                        for pair in budgets[..continuing].windows(2) {
                            expected.extend(
                                kind.build(REHEAT_BETA, cold, pair[1] - pair[0])
                                    .into_iter()
                                    .map(|b| b as f32),
                            );
                        }
                        ends.extend_from_slice(&budgets[..continuing]);
                    }
                    for &leg in &budgets[continuing..] {
                        expected.extend(lone(leg));
                        ends.push(expected.len());
                    }
                    assert_eq!(fresh_from, continuing);
                    assert_eq!(checkpoints, ends);
                    assert_eq!(
                        actual.iter().map(|b| b.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|b| b.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn a_job_without_checkpoints_keeps_one_standard_schedule() {
        for sweeps in [100, 256, 1000] {
            let (s, c, _) = segment_schedule(&ring(), &params(sweeps, 1), &[]);
            assert_eq!(
                s,
                build_beta_schedule(
                    &ring(),
                    sweeps,
                    1,
                    None,
                    quip_solver_core::BetaSchedule::Geometric
                )
                .0
            );
            assert_eq!(c, vec![sweeps]);
        }
    }

    #[test]
    fn prepared_schedule_refreshes_before_admission() {
        let job = StreamJob {
            job_id: vec![1],
            graph: ring(),
            params: params(1000, 1),
            watermark: None,
        };
        let old = CascadeSettings {
            stages: stage_array(&[32, 256]),
            ..Default::default()
        };
        {
            let new = CascadeSettings {
                stages: stage_array(&[16, 128, 512]),
                ..old
            };
            let mut prepared = PreparedSchedule::new(&job, old, false, true);
            let mut controller = Controller::new(old);
            controller.refresh(new);
            let edges = Arc::clone(&job.graph.edges);
            let ticket = controller
                .admit_prepared(&job, &mut prepared, &edges)
                .unwrap();
            let mut expected = Controller::new(new);
            let (expected_ticket, betas, checkpoints) = expected.admit(&job);
            assert_eq!(&prepared.betas[..], &betas[..]);
            assert_eq!(prepared.checkpoints, checkpoints);
            assert_eq!(ticket.gates, expected_ticket.gates);
            assert_eq!(ticket.stage, expected_ticket.stage);
            assert_eq!(ticket.final_sweeps, expected_ticket.final_sweeps);
        }
    }
    use crate::sampler::build_beta_schedule;
    use crate::{IsingGraph, SampleParams};
    use quip_solver_core::beta::default_ising_beta_range;

    fn ring() -> IsingGraph {
        IsingGraph::new(
            vec![0.0; 4],
            vec![1.0; 4],
            vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        )
    }
    fn params(num_sweeps: usize, sweeps_per_beta: usize) -> SampleParams {
        SampleParams {
            num_reads: 2,
            num_sweeps,
            sweeps_per_beta,
            schedule: quip_solver_core::BetaSchedule::Geometric,
            beta_range: None,
            seed: 42,
        }
    }
    fn job(id: u64, sweeps: usize) -> StreamJob {
        StreamJob {
            job_id: id.to_le_bytes().to_vec(),
            graph: ring(),
            params: params(sweeps, 1),
            watermark: Some(1),
        }
    }

    #[test]
    fn checkpoint_decision_at_a_target() {
        let mut controller = Controller::new(CascadeSettings {
            stages: stage_array(&[16, 64]),
            ..CascadeSettings::default()
        });
        for id in 0..2_000 {
            let mut observation = job(id, 256);
            observation.params.num_reads = 64;
            let (mut ticket, _, _) = controller.admit(&observation);
            controller.checkpoint(&mut ticket, -1_000, None);
        }
        let mut candidate = job(4, 256);
        candidate.params.num_reads = 64;

        // No target: the adaptive gate screens a shallow best.
        let (mut strict_ticket, _, _) = controller.admit(&candidate);
        assert!(!controller.checkpoint(&mut strict_ticket, -100, None));

        // Above target: falls through to the same adaptive gate and is
        // screened, same as no target at all.
        let (mut above, _, _) = controller.admit(&candidate);
        assert!(!controller.checkpoint(&mut above, -100, Some(-150)));
        assert_eq!(above.finish_sweeps, None);

        // Equal to target: the chain rule keeps the unit, unscreened.
        let (mut equal, _, _) = controller.admit(&candidate);
        assert!(controller.checkpoint(&mut equal, -150, Some(-150)));
        assert_eq!(equal.finish_sweeps, None);

        // Below target: the unit finishes immediately at this checkpoint.
        let (mut below, _, _) = controller.admit(&candidate);
        assert!(!controller.checkpoint(&mut below, -200, Some(-150)));
        assert_eq!(below.finish_sweeps, Some(16));
    }

    #[test]
    fn report_range_covers_one_lease_round() {
        let mut c = Controller::new(CascadeSettings::default());
        let (old, _, _) = c.admit(&job(0, 256));
        let (late, _, _) = c.admit(&job(1, 256));
        c.finish(&old, Some(-14_702), true);
        assert_eq!(c.range, Some((-14_702, -14_702)));

        c.enter_round(3);
        assert_eq!(c.range, None, "a newer round starts an empty range");
        let (new, _, _) = c.admit(&job(2, 256));
        c.finish(&new, Some(-14_600), true);
        c.finish(&late, Some(-14_800), true);
        assert_eq!(
            c.range,
            Some((-14_600, -14_600)),
            "a unit admitted in the old round stays out of the new range"
        );

        c.enter_round(2);
        assert_eq!(
            c.range,
            Some((-14_600, -14_600)),
            "an older round is ignored"
        );
    }

    #[test]
    fn plain_units_between_lease_units_keep_the_yield_check() {
        let mut c = Controller::new(CascadeSettings {
            stages: stage_array(&[16, 64]),
            yield_per_million: Some(1.0),
            ..CascadeSettings::default()
        });
        c.set_yield_target(Some(-100));
        let (mut salt, _, _) = c.admit(&job(0, 256));
        let (mut plain, _, _) = c.admit(&job(1, 256));
        c.checkpoint(&mut plain, 0, None);
        c.finish(&plain, Some(0), true);
        assert!(!c.checkpoint(&mut salt, -200, Some(-100)));
        assert!(salt.finish_sweeps.is_some());
        c.finish(&salt, Some(-200), true);
        let hits = c.yield_check.as_ref().expect("yield check").observation().0;
        assert_eq!(hits, 1.0, "the lease unit's hit is counted");
    }

    #[test]
    fn an_unscreened_schedule_has_one_checkpoint_at_the_full_budget() {
        let mut job = job(4, 256);
        job.params.num_reads = 64;
        let schedule = PreparedSchedule::new(&job, CascadeSettings::default(), false, false);
        assert_eq!(schedule.checkpoints, vec![256]);
    }

    #[test]
    fn short_budget_has_one_segment() {
        let (s, c, _) = segment_schedule(&ring(), &params(32, 1), &[32, 256]);
        assert_eq!(s.len(), 32);
        assert_eq!(c, vec![32]);
    }

    #[test]
    fn sweeps_per_beta_expands_rungs() {
        let (s, c, fresh_from) = segment_schedule(&ring(), &params(1000, 4), &[32, 256]);
        assert_eq!(s.len(), 256 + 1000);
        assert_eq!(c, vec![32, 256, 1256]);
        assert_eq!(fresh_from, 2);
        let standard = build_beta_schedule(
            &ring(),
            32,
            4,
            None,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0;
        for (chunk, beta) in s[..32].chunks(4).zip(standard) {
            assert_eq!(chunk, &[beta; 4]);
        }
    }

    #[test]
    fn continuing_legs_reheat_and_fresh_legs_start_hot() {
        let (s, c, _) = segment_schedule(&ring(), &params(1000, 1), &[32, 256]);
        let (hot, cold) = default_ising_beta_range(&ring());
        assert_eq!(s[0], hot as f32);
        assert_eq!(s[32], REHEAT_BETA as f32);
        assert_eq!(s[256], hot as f32);
        for end in c {
            assert_eq!(s[end - 1], cold as f32);
        }
    }

    #[test]
    fn a_fresh_leg_equals_a_lone_anneal_of_its_budget() {
        let (s, _, _) = segment_schedule(&ring(), &params(1000, 1), &[32, 256, 512]);
        assert_eq!(
            s[..32],
            build_beta_schedule(
                &ring(),
                32,
                1,
                None,
                quip_solver_core::BetaSchedule::Geometric
            )
            .0
        );
        assert_eq!(
            s[256..768],
            build_beta_schedule(
                &ring(),
                512,
                1,
                None,
                quip_solver_core::BetaSchedule::Geometric
            )
            .0
        );
        assert_eq!(
            s[768..],
            build_beta_schedule(
                &ring(),
                1000,
                1,
                None,
                quip_solver_core::BetaSchedule::Geometric
            )
            .0
        );
    }

    #[test]
    fn warm_up_screens_out_and_the_keep_rate_converges() {
        let settings = CascadeSettings {
            keep: 100.0,
            keep_min: 100.0,
            keep_max: 100.0,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let template = job(0, 1000);
        let mut rng = 12345u64;
        let mut uniform = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut counts = [[0usize; 2]; 2];
        for i in 0..50_000 {
            let (mut ticket, _, _) = controller.admit(&template);
            // Disable audit draws to measure only the cutoff keep rate.
            controller.plan.lock().unwrap().audit_rng = None;
            for count in &mut counts {
                let best = (1000.0
                    * (-2.0 * uniform().ln()).sqrt()
                    * (std::f64::consts::TAU * uniform()).cos()) as i64;
                let keep = controller.checkpoint(&mut ticket, best, None);
                if i == 0 {
                    assert!(!keep);
                }
                if i >= 10_000 {
                    count[0] += 1;
                    count[1] += usize::from(keep);
                }
                if !keep {
                    break;
                }
            }
        }
        for [total, kept] in counts {
            let rate = kept as f64 / total as f64;
            assert!((rate - 0.1).abs() < 0.025, "{rate}");
        }
    }

    #[test]
    fn audited_job_records_one_observation_at_the_next_checkpoint() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut t, _, _) = c.admit(&job(0, 14_336));
        {
            let mut p = c.plan.lock().unwrap();
            p.kept[0].push_back(-100);
            p.audit_rng = Some(0);
        }
        assert!(c.checkpoint(&mut t, 0, None));
        assert!(t.audited);
        c.plan.lock().unwrap().audit_rng = None;
        assert!(!c.checkpoint(&mut t, -200, None));
        let observation = c.plan.lock().unwrap().audit_lanes[0].observation(false);
        let mut expected = AuditLane::new(CALIBRATION.false_negative[0].1);
        expected.record(true);
        assert_eq!(observation, expected.observation(false));
        c.finish(&t, Some(-200), true);
        assert_eq!(
            c.plan.lock().unwrap().audit_lanes[0].observation(false),
            observation
        );
        assert_eq!(c.plan.lock().unwrap().kept[0].len(), 1);
    }

    #[test]
    fn topology_change_resets_the_distribution() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut old, _, _) = c.admit(&job(0, 1000));
        c.checkpoint(&mut old, 0, None);
        assert!(c.matches_topology(&ring()));
        let mut same_topology = job(1, 1000);
        same_topology.graph.h[0] = 0.5;
        c.admit(&same_topology);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
        let mut changed = job(1, 1000);
        changed.graph.h.push(0.0);
        c.admit(&changed);
        assert!(!c.matches_topology(&ring()));
        assert!(!c.checkpoint(&mut old, -100, None));
        c.finish(&old, Some(-100), true);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
    }

    #[test]
    fn settings_change_to_stages_rebuilds_but_audit_change_does_not() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut t, _, _) = c.admit(&job(0, 1000));
        c.checkpoint(&mut t, 0, None);
        c.refresh(CascadeSettings {
            audit: 50,
            ..CascadeSettings::default()
        });
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
        assert_eq!(c.plan.lock().unwrap().settings.audit, 50);
        c.refresh(CascadeSettings {
            stages: stage_array(&[16, 128]),
            ..CascadeSettings::default()
        });
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
    }

    #[test]
    fn controller_and_ticket_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Controller>();
        assert_send::<Ticket>();
    }

    #[test]
    fn merge_rejects_invalid_values_and_keeps_the_previous_ones() {
        let mut settings = CascadeSettings::default();
        let previous = settings;
        settings.merge(&CascadeToml {
            cascade_stages: Some(vec![128, 32]),
            cascade_keep_min: Some(4000),
            cascade_audit: Some(0),
            cascade_yield_per_million: Some(f64::NAN),
            ..CascadeToml::default()
        });
        assert_eq!(settings, previous);
        for stages in [
            vec![],
            vec![0],
            vec![32, 32],
            (1..=MAX_STAGES + 1).collect(),
        ] {
            settings.merge(&CascadeToml {
                cascade_stages: Some(stages),
                ..CascadeToml::default()
            });
            assert_eq!(settings, previous);
        }
    }

    #[test]
    fn fixed_gates_keep_at_or_below_the_gate_and_skip_audits() {
        let mut c = Controller::new(CascadeSettings {
            stages: stage_array(&[8, 16]),
            gates: gate_array(&[Some(-10), Some(-20)]),
            audit: 2,
            ..CascadeSettings::default()
        });
        let (mut kept, _, checkpoints) = c.admit(&job(0, 100));
        assert_eq!(checkpoints, vec![8, 16, 100]);
        assert!(c.checkpoint(&mut kept, -10, None));
        assert!(c.checkpoint(&mut kept, -20, None));
        assert_eq!(kept.stage, 2);
        for _ in 0..50 {
            let (mut screened, _, _) = c.admit(&job(0, 100));
            assert!(!c.checkpoint(&mut screened, -9, None));
            assert!(!screened.audited);
            assert_eq!(screened.stage, 0);
        }
    }

    fn fixture_edges() -> Vec<(usize, usize)> {
        include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut nodes = line.split_whitespace();
                (
                    nodes.next().unwrap().parse().unwrap(),
                    nodes.next().unwrap().parse().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn chain_fingerprint_is_the_fixture_without_its_removed_edge() {
        let mut edges = fixture_edges();
        assert_ne!(topology_fingerprint(4577, &edges), CHAIN_GATES.fingerprint);
        edges.retain(|&(u, v)| (u.min(v), u.max(v)) != (880, 2695));
        assert_eq!(edges.len(), 41_514);
        assert_eq!(topology_fingerprint(4577, &edges), CHAIN_GATES.fingerprint);
        // Order and orientation do not matter; the node count does.
        edges.reverse();
        edges[0] = (edges[0].1, edges[0].0);
        assert_eq!(topology_fingerprint(4577, &edges), CHAIN_GATES.fingerprint);
        assert_ne!(topology_fingerprint(4578, &edges), CHAIN_GATES.fingerprint);
    }

    #[test]
    fn chain_gates_need_the_measured_conditions() {
        let settings = CascadeSettings::default();
        let measured = SampleParams {
            num_reads: CHAIN_GATES.min_reads,
            ..params(14_336, 1)
        };
        assert!(settings.chain_gated(true, &measured));
        assert!(!settings.chain_gated(false, &measured));
        let fewer = SampleParams {
            num_reads: 32,
            ..measured
        };
        let slower = SampleParams {
            sweeps_per_beta: 2,
            ..measured
        };
        let ranged = SampleParams {
            beta_range: Some((0.1, 4.0)),
            ..measured
        };
        for other in [fewer, slower, ranged] {
            assert!(!settings.chain_gated(true, &other));
        }
        let gated = settings.effective(true);
        assert_eq!(gated.stages, stage_array(CHAIN_GATES.stages));
        let expected: Vec<_> = CHAIN_GATES.gates.iter().copied().map(Some).collect();
        assert_eq!(gated.gates, gate_array(&expected));
        let study = CascadeSettings {
            open_gates: true,
            stages: stage_array(&[8, 512, 4096]),
            ..settings
        };
        assert_eq!(study.effective(true), study);
        assert_eq!(settings.effective(false), settings);
    }

    #[test]
    fn schedule_cache_shares_equal_schedules_and_rebuilds_on_change() {
        let settings = CascadeSettings::default();
        let mut cache = ScheduleCache::default();
        let mut other_worker = cache.clone();
        let first = cache
            .prepare(&job(0, 1000), settings, false, true, None)
            .unwrap();
        let second = other_worker
            .prepare(&job(1, 1000), settings, false, true, None)
            .unwrap();
        assert!(Arc::ptr_eq(&first.betas, &second.betas));
        let unscreened = cache
            .prepare(&job(1, 1000), settings, false, false, None)
            .unwrap();
        assert_eq!(unscreened.checkpoints, vec![1000]);
        assert!(!Arc::ptr_eq(&first.betas, &unscreened.betas));
        let reference = PreparedSchedule::new(&job(1, 1000), settings, false, true);
        assert_eq!(&second.betas[..], &reference.betas[..]);
        assert_eq!(second.checkpoints, reference.checkpoints);
        let longer = other_worker
            .prepare(&job(2, 2000), settings, false, true, None)
            .unwrap();
        assert!(!Arc::ptr_eq(&first.betas, &longer.betas));
        assert_eq!(longer.betas.len(), 256 + 2000);
        let shared_longer = cache
            .prepare(&job(3, 2000), settings, false, true, None)
            .unwrap();
        assert!(Arc::ptr_eq(&longer.betas, &shared_longer.betas));
        assert_eq!(&first.betas[..], &reference.betas[..]);
        let gated = cache
            .prepare(&job(2, 2000), settings, true, true, None)
            .unwrap();
        assert_eq!(gated.checkpoints, vec![8, 16, 64, 256, 768, 1792, 3792]);
    }

    #[test]
    fn report_counts_each_level_and_resets_the_window() {
        let mut c = Controller::new(CascadeSettings::default().effective(true));
        let start = c.stats.started;
        let (mut kept, _, _) = c.admit(&job(0, 1000));
        for best in [
            -13_300_000,
            -13_900_000,
            -14_300_000,
            -14_500_000,
            -14_550_000,
        ] {
            assert!(c.checkpoint(&mut kept, best, None));
        }
        c.finish(&kept, Some(-14_600_000), true);
        let (mut screened, _, _) = c.admit(&job(0, 1000));
        assert!(!c.checkpoint(&mut screened, -13_000_000, None));
        c.finish(&screened, Some(-13_000_000), true);
        let (cancelled, _, _) = c.admit(&job(0, 1000));
        c.finish(&cancelled, None, false);

        assert_eq!(c.stats.models, 2, "a cancelled job is not a model");
        assert_eq!(
            c.stats.levels[0].unwrap(),
            LevelStats {
                sweeps: 8,
                checked: 2,
                kept: 1,
            }
        );
        assert_eq!(c.stats.levels[1].unwrap().checked, 1);
        assert_eq!(c.range, Some((-14_600_000, -13_000_000)));

        c.report(start + Duration::from_secs(59), Duration::from_secs(60));
        assert_eq!(c.stats.models, 2);
        c.report(start + Duration::from_secs(60), Duration::from_secs(60));
        assert_eq!(c.stats.models, 0);
        assert!(c.stats.levels.iter().all(Option::is_none));
        assert_eq!(
            c.range,
            Some((-14_600_000, -13_000_000)),
            "the energy range spans every window"
        );
    }

    #[test]
    fn energy_prints_whole_units_and_fractions() {
        assert_eq!(energy(-14_272_000), "-14272");
        assert_eq!(energy(-1_500), "-1.500");
        assert_eq!(energy(250), "0.250");
        assert_eq!(energy(0), "0");
    }

    #[test]
    fn open_gates_keep_every_job_to_the_full_budget() {
        let mut settings = CascadeSettings::default();
        settings.merge(&CascadeToml {
            cascade_open_gates: Some(true),
            ..CascadeToml::default()
        });
        let mut c = Controller::new(CascadeSettings {
            gates: gate_array(&[Some(-10)]),
            ..settings
        });
        let (mut ticket, _, _) = c.admit(&job(0, 1000));
        assert!(c.checkpoint(&mut ticket, 0, None));
        assert!(c.checkpoint(&mut ticket, 0, None));
        assert_eq!(ticket.stage, ticket.gates);
    }

    #[test]
    fn removed_cascade_key_warns_and_keeps_screening() {
        #[derive(Clone)]
        struct LogWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for LogWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let output = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(Arc::clone(&output));
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let backend_toml = "cascade = false\nnum_sweeps = 1024";
        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(
                crate::resolve_governor_config(backend_toml, 73, true),
                (73, true)
            );
        });
        let log = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert_eq!(
            log.matches("unknown field 'cascade' for metal (ignored)")
                .count(),
            1
        );
        assert!(!log.contains("unknown field 'num_sweeps'"));
        let cfg: crate::MetalConfig = toml::from_str(backend_toml).unwrap();
        assert_eq!(cfg.unknown["cascade"].as_bool(), Some(false));
        let mut settings = CascadeSettings::default();
        settings.merge(&cfg.cascade);
        let mut controller = Controller::new(settings);
        let (mut ticket, _, checkpoints) = controller.admit(&job(0, 1024));
        assert_eq!(checkpoints, vec![32, 256, 1280]);
        assert!(!controller.checkpoint(&mut ticket, 0, None));
    }

    #[test]
    fn backend_toml_routes_cascade_keys_and_leaves_unknown_keys_unknown() {
        let cfg: crate::MetalConfig = toml::from_str("cascade_keep = 5000\nbogus = 1").unwrap();
        assert_eq!(cfg.cascade.cascade_keep, Some(5000));
        assert_eq!(cfg.unknown.keys().collect::<Vec<_>>(), vec!["bogus"]);
    }

    #[test]
    fn admitted_jobs_retain_their_plan_after_stage_change() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut t, _, checkpoints) = c.admit(&job(0, 14_336));
        for cutoff in &mut c.plan.lock().unwrap().cutoffs {
            for _ in 0..200 {
                cutoff.observe(0.0);
            }
        }
        let old = Arc::downgrade(&c.plan);
        c.refresh(CascadeSettings {
            stages: stage_array(&[64]),
            ..CascadeSettings::default()
        });
        assert!(!Arc::ptr_eq(&t.plan, &c.plan));
        assert_eq!(checkpoints, vec![32, 256, 14_592]);
        assert!(c.checkpoint(&mut t, -1, None));
        assert!(c.checkpoint(&mut t, -1, None));
        c.finish(&t, Some(-1), true);
        assert_eq!(t.plan.lock().unwrap().cutoffs[2].moments().count(), 201);
        drop(t);
        assert!(old.upgrade().is_none());
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
        let (t, _, checkpoints) = c.admit(&job(1, 1024));
        assert_eq!(t.gates, 1);
        assert_eq!(checkpoints, vec![64, 1088]);
    }

    #[test]
    fn live_settings_change_stages_on_one_stream() {
        let mut c = Controller::new(CascadeSettings::default());
        for (stages, expected) in [
            (stage_array(&[32, 256]), vec![32, 256, 1256]),
            (stage_array(&[64]), vec![64, 1064]),
            (stage_array(&[16, 64]), vec![16, 64, 1064]),
        ] {
            c.refresh(CascadeSettings {
                stages,
                ..CascadeSettings::default()
            });
            let (t, _, checkpoints) = c.admit(&job(0, 1000));
            assert_eq!(checkpoints, expected);
            assert_eq!(t.gates, expected.len() - 1);
        }
    }

    #[test]
    fn custom_stages_use_transition_calibration_and_skip_32_sweep_drift() {
        let ((probe, full), rate) = CALIBRATION.false_negative[1];
        let mut c = Controller::new(CascadeSettings {
            stages: stage_array(&[probe]),
            ..CascadeSettings::default()
        });
        let (mut t, _, _) = c.admit(&job(0, full));
        let before = {
            let mut p = c.plan.lock().unwrap();
            assert_eq!(
                p.audit_lanes[0].observation(true),
                AuditLane::new(rate).observation(true)
            );
            for _ in 0..10_000 {
                p.cutoffs[0].observe(0.0);
            }
            let mut expected = p.cutoffs[0].clone();
            expected.observe(-1.0);
            expected.denominator()
        };
        c.checkpoint(&mut t, -1, None);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].denominator(), before);
    }

    #[test]
    fn short_full_budget_never_trains_another_transition() {
        let mut c = Controller::new(CascadeSettings::default());
        for budget in [14_336, 1024] {
            let (mut t, _, _) = c.admit(&job(0, budget));
            for cutoff in &mut c.plan.lock().unwrap().cutoffs {
                for _ in 0..200 {
                    cutoff.observe(0.0);
                }
            }
            assert!(c.checkpoint(&mut t, -1, None));
            assert!(c.checkpoint(&mut t, -1, None));
            c.finish(&t, Some(-1), true);
        }
        assert_eq!(c.plan.lock().unwrap().kept[1].len(), 1);
        for cutoff in &mut c.plan.lock().unwrap().cutoffs {
            cutoff.reset();
        }
        let (mut t, _, _) = c.admit(&job(2, 1024));
        c.plan.lock().unwrap().audit_rng = Some(0);
        assert!(c.checkpoint(&mut t, 0, None));
        assert!(!c.checkpoint(&mut t, 0, None));
        assert!(!t.audited);
    }

    #[test]
    fn audit_restore_preserves_drift_at_the_clamp() {
        let c = Controller::new(CascadeSettings::default());
        let mut p = c.plan.lock().unwrap();
        for _ in 0..100_000 {
            p.cutoffs[0].observe(0.0);
        }
        p.cutoffs[0].load_update(100.0, 1.0);
        p.apply_action(0, Action::LoosenAndDoubleK0, Check::Drift, &[], &[]);
        let before = p.cutoffs[0].denominator();
        p.apply_action(
            0,
            Action::Loosen { raise_audit: true },
            Check::Audit,
            &[],
            &[],
        );
        p.apply_action(0, Action::Restore, Check::Audit, &[], &[]);
        assert!((p.cutoffs[0].denominator() - before).abs() < 1e-12);
    }

    #[test]
    fn reference_median_and_zero_seed_are_well_defined() {
        assert_eq!(kept_median(&VecDeque::new()), None);
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3])), Some(3.0));
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3, 7])), Some(4.0));
        let mut c = Controller::new(CascadeSettings::default());
        let mut first = job(0, 1024);
        first.params.seed = 0;
        c.admit(&first);
        let mut p = c.plan.lock().unwrap();
        let selected = (0..10_000).filter(|_| p.audit_selected(0)).count();
        assert!((20..100).contains(&selected), "{selected}");
    }

    #[test]
    fn injected_false_negative_burst_loosens_the_stage() {
        let mut c = Controller::new(CascadeSettings {
            stages: stage_array(&[32]),
            keep: 100.0,
            keep_min: 2.0,
            keep_max: 1000.0,
            audit: 2,
            ..CascadeSettings::default()
        });
        let (mut t, _, _) = c.admit(&job(0, 256));
        for _ in 0..100_000 {
            c.plan.lock().unwrap().cutoffs[0].observe(-100.0);
        }
        assert!(c.checkpoint(&mut t, -1000, None));
        c.finish(&t, Some(-1000), true);
        let before = c.plan.lock().unwrap().cutoffs[0].denominator();
        for id in 1..1000 {
            let (mut t, _, _) = c.admit(&job(id, 256));
            if c.checkpoint(&mut t, 0, None) {
                c.finish(&t, Some(-2000), true);
            } else {
                c.finish(&t, Some(0), true);
            }
            let p = c.plan.lock().unwrap();
            if p.audit_lanes[0].denominator(200) == 50 {
                assert!((p.cutoffs[0].denominator() - before / 2.0).abs() < 0.01);
                assert_eq!(p.kept[0].len(), 1);
                return;
            }
        }
        panic!("audit burst did not loosen the screen");
    }

    #[test]
    fn a_job_at_or_below_the_first_budget_passes_straight_through() {
        let mut c = Controller::new(CascadeSettings::default());
        for budget in [16, 32] {
            let (t, s, checkpoints) = c.admit(&job(0, budget));
            assert_eq!(t.gates, 0);
            assert_eq!(checkpoints, vec![budget]);
            assert_eq!(
                s,
                build_beta_schedule(
                    &ring(),
                    budget,
                    1,
                    None,
                    quip_solver_core::BetaSchedule::Geometric
                )
                .0
            );
        }
    }

    #[test]
    fn configured_stage_roots_and_effective_budgets_preserve_the_original_job() {
        let settings = CascadeSettings {
            stages: stage_array(&[32, 128, 256]),
            ..CascadeSettings::default()
        };
        let mut c = Controller::new(settings);
        let mut original = job(0, 128);
        original.params.beta_range = Some((0.1, 4.0));
        original.params.sweeps_per_beta = 2;
        let (mut t, s, checkpoints) = c.admit(&original);
        assert_eq!(t.gates, 1);
        assert_eq!(checkpoints, vec![32, 128]);
        assert_eq!(original.params.seed, 42);
        assert_eq!(original.params.num_sweeps, 128);
        assert_eq!(original.params.beta_range, Some((0.1, 4.0)));
        assert_eq!(original.watermark, Some(1));
        assert_eq!(
            s[..32],
            build_beta_schedule(
                &original.graph,
                32,
                2,
                Some((0.1, 4.0)),
                quip_solver_core::BetaSchedule::Geometric
            )
            .0
            .into_iter()
            .flat_map(|b| [b; 2])
            .collect::<Vec<_>>()
        );
        {
            let mut p = c.plan.lock().unwrap();
            assert!((p.cutoffs[0].denominator() - settings.keep_min.powf(1.0 / 3.0)).abs() < 1e-10);
            for _ in 0..200 {
                p.cutoffs[0].observe(0.0);
            }
        }
        assert!(c.checkpoint(&mut t, -1, None));
        c.finish(&t, Some(-2), true);
        let p = c.plan.lock().unwrap();
        assert_eq!(p.cutoffs[1].moments().count(), 0);
        assert_eq!(p.cutoffs[2].moments().count(), 0);
        assert_eq!(p.cutoffs[3].moments().count(), 1);
        assert!(p.kept.iter().all(VecDeque::is_empty));
    }

    #[test]
    fn load_feedback_loosens_tightens_and_decays() {
        let mut c = Controller::new(CascadeSettings::default());
        c.update_load(0.0, &[], 4);
        assert!(c.plan.lock().unwrap().cutoffs[0].log_adjust() < 0.0);
        c.plan.lock().unwrap().cutoffs[0].reset();
        c.update_load(2.0, &[1, 3, 0], 4);
        let positive = c.plan.lock().unwrap().cutoffs[0].log_adjust();
        assert!((positive - 0.1 * 1.5f64.ln()).abs() < 1e-12);
        c.update_load(2.0, &[4, 0, 0], 4);
        assert!((c.plan.lock().unwrap().cutoffs[0].log_adjust() - 0.9 * positive).abs() < 1e-12);
        c.update_load(0.9, &[], 0);
        assert!(c.plan.lock().unwrap().cutoffs[0].log_adjust().is_finite());
    }

    #[test]
    fn final_audit_and_cancel_accounting() {
        let mut c = Controller::new(CascadeSettings {
            stages: stage_array(&[32]),
            ..CascadeSettings::default()
        });
        let (mut t, _, _) = c.admit(&job(0, 256));
        {
            let mut p = c.plan.lock().unwrap();
            p.kept[0].push_back(-100);
            p.audit_rng = Some(0);
        }
        assert!(c.checkpoint(&mut t, 0, None));
        assert!(t.audited);
        c.finish(&t, Some(-200), true);
        let mut expected = AuditLane::new(CALIBRATION.false_negative[0].1);
        expected.record(true);
        assert_eq!(
            c.plan.lock().unwrap().audit_lanes[0].observation(false),
            expected.observation(false)
        );
        let (mut cancelled, _, _) = c.admit(&job(1, 256));
        assert!(c.checkpoint(&mut cancelled, 0, None));
        c.finish(&cancelled, None, true);
        let p = c.plan.lock().unwrap();
        assert_eq!(
            p.audit_lanes[0].observation(false),
            expected.observation(false)
        );
        assert_eq!(p.cutoffs[1].moments().count(), 1);
    }

    #[test]
    fn remainder_and_one_sweep_segments_have_exact_lengths() {
        let (s, checkpoints, _) = segment_schedule(&ring(), &params(34, 3), &[32, 33]);
        assert_eq!(s.len(), 34);
        assert_eq!(checkpoints, vec![32, 33, 34]);
        let cold = default_ising_beta_range(&ring()).1 as f32;
        assert_eq!(s[31..], [cold; 3]);
        let (s, checkpoints, _) = segment_schedule(&ring(), &params(0, 0), &[32, 256]);
        assert!(s.is_empty());
        assert_eq!(checkpoints, vec![0]);
    }

    #[test]
    fn arm_d_constants_match_the_report() {
        assert_eq!(CALIBRATION.stages, &[32, 256]);
        assert_eq!(CALIBRATION.skew, &[-0.123, -0.056]);
        assert_eq!(CALIBRATION.excess_kurtosis, &[0.014, 0.002]);
        assert_eq!(
            CALIBRATION.false_negative,
            &[((32, 256), 0.0353), ((256, 14_336), 0.0051)]
        );
        assert_eq!(CALIBRATION.cost_a_us, 124.0);
        assert_eq!(CALIBRATION.cost_b_us, 2.284);
        assert_eq!(CALIBRATION.k0, 1.0);
    }

    #[test]
    fn short_second_topology_preserves_inflight_training() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut ticket, _, _) = c.admit(&job(0, 1000));
        let plan = Arc::clone(&c.plan);
        let epochs = (c.topology_epoch, c.yield_epoch);
        let rng = c.plan.lock().unwrap().audit_rng;
        let mut short = job(1, 32);
        short.graph.h.push(0.0);
        let (short_ticket, _, _) = c.admit(&short);
        assert_eq!(short_ticket.gates, 0);
        assert!(Arc::ptr_eq(&plan, &c.plan));
        assert_eq!((c.topology_epoch, c.yield_epoch), epochs);
        assert!(c.matches_topology(&ring()));
        assert_eq!(c.plan.lock().unwrap().audit_rng, rng);
        c.checkpoint(&mut ticket, -100, None);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
    }

    #[test]
    fn short_admission_leaves_topology_unset_and_plan_untouched() {
        let mut c = Controller::new(CascadeSettings::default());
        let plan = Arc::clone(&c.plan);
        let (ticket, _, checkpoints) = c.admit(&job(0, 16));
        assert_eq!(ticket.gates, 0);
        assert_eq!(checkpoints, vec![16]);
        assert!(c.topology.is_none());
        assert!(Arc::ptr_eq(&plan, &c.plan));
        assert_eq!((c.topology_epoch, c.yield_epoch), (0, 0));
        assert_eq!(c.plan.lock().unwrap().audit_rng, None);
    }

    #[test]
    fn a_reheat_outside_the_range_continues_one_standard_schedule() {
        let mut params = params(1000, 2);
        params.beta_range = Some((0.5, 4.0));
        let (s, checkpoints, fresh_from) = segment_schedule(&ring(), &params, &[32, 256]);
        assert_eq!(checkpoints, vec![32, 256, 1256]);
        assert_eq!(fresh_from, 2);
        let continuing: Vec<_> = build_beta_schedule(
            &ring(),
            256,
            2,
            params.beta_range,
            quip_solver_core::BetaSchedule::Geometric,
        )
        .0
        .into_iter()
        .flat_map(|b| [b; 2])
        .collect();
        assert_eq!(s[..256], continuing[..]);
        assert_eq!(s[256], 0.5);
        assert_eq!(s[1255], 4.0);
    }
}
