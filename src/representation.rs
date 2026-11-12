// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Which coefficient path can run a model exactly, decided before any
//! quantization.
//!
//! The int8 kernels hold `h` and `J` as whole units in `int8_t`, and the SA
//! delta `2·s·field` is an `int8_t`. A model qualifies when every coefficient
//! is a whole number and every local field `|h_i| + Σ_j |J_ij|` is at most 63.
//! The wide kernels hold milli units in `int`. A model qualifies when every
//! coefficient is a whole number of milli units and every local field is at
//! most [`WIDE_MAX_FIELD_MILLI`]. Any other finite model needs `float`
//! coefficients. Only a model that no tier can hold is refused.

use quip_solver_core::IsingGraph;

use crate::capacity::Precision;

/// Largest local field the int8 path takes, in milli: 63 whole units.
pub(crate) const INT8_MAX_FIELD_MILLI: u32 = 63_000;

/// Largest local field the wide path takes, in milli. The SA neighbour update
/// adds `4·J`, which stays inside `int` because `|J| ≤ field ≤ 268_435_000`.
pub(crate) const WIDE_MAX_FIELD_MILLI: u32 = 268_435_000;

/// The smallest coefficient tier that represents a model exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Representation {
    /// Every `h` and `J` is -1, 0 or 1. For MSA, the unit kernel. For SA and
    /// Gibbs, a unit model whose local field fits the int8 path.
    Binary,
    /// Whole units in `int8_t` with a local field of at most
    /// [`INT8_MAX_FIELD_MILLI`], and not [`Representation::Binary`]. SA and
    /// Gibbs only: MSA has no int8 kernel.
    Int8,
    /// `int` milli units: the `QUIP_WIDE` SA and Gibbs builds, or
    /// `msa_anneal_weighted`.
    Wide,
    /// Every other finite model: a value off the milli grid, or a local field
    /// past [`WIDE_MAX_FIELD_MILLI`]. No kernel runs it yet.
    Float32,
}

impl Representation {
    /// The capacity-model precision of this tier.
    pub(crate) fn precision(self) -> Precision {
        match self {
            Self::Binary => Precision::Binary,
            Self::Int8 => Precision::Int8,
            Self::Wide => Precision::Int32,
            Self::Float32 => Precision::Float32,
        }
    }
}

/// `v` in milli units, when `v` is exactly `m / 1000.0` for an integer `m`
/// within the wide bound. The round trip refuses values that only round to a
/// milli step, such as `0.1 + 0.2`.
pub(crate) fn to_milli(v: f64) -> Option<i32> {
    if !v.is_finite() {
        return None;
    }
    let m = (v * 1000.0).round();
    if m.abs() > f64::from(WIDE_MAX_FIELD_MILLI) || m / 1000.0 != v {
        return None;
    }
    // In range by the check above.
    Some(m as i32)
}

/// The milli value one `int8_t` unit stands for when `graph` is a wide model
/// whose coefficients all fit `i8` after division by their greatest common
/// divisor, or `None`. The scale is the gcd of the absolute nonzero milli
/// values of `h` and `J`. A coefficient off the milli grid, a model with no
/// nonzero coefficient, and a quotient outside `-128..=127` give `None`.
#[cfg(test)]
pub(crate) fn wide_i8_scale(graph: &IsingGraph) -> Option<i32> {
    batch_wide_i8_scale(&[graph])
}

/// The scale for the graphs of one batch, which share one scale: the gcd over
/// every graph, and every quotient must fit `i8`. [`wide_i8_scale`] is the
/// one-graph case.
pub(crate) fn batch_wide_i8_scale(graphs: &[&IsingGraph]) -> Option<i32> {
    let mut milli = Vec::new();
    for graph in graphs {
        for &v in graph.h.iter().chain(&graph.j) {
            milli.push(to_milli(v)?);
        }
    }
    let scale = milli.iter().fold(0_i32, |g, &m| gcd(g, m.abs()));
    if scale == 0 {
        return None;
    }
    milli
        .iter()
        .all(|&m| i8::try_from(m / scale).is_ok())
        .then_some(scale)
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// What the shared checks learn about a model.
struct Survey {
    /// The largest local field in milli, when every value is on the milli
    /// grid and the field fits the wide bound. `None` means Float32.
    max_field_milli: Option<i64>,
    /// Every `h` and `J` is -1, 0 or 1. `-0.0 == 0.0`, so a negative zero
    /// counts as zero.
    unit: bool,
}

/// The refusals both classifiers share, and the facts they sort on.
fn survey(graph: &IsingGraph) -> Result<Survey, String> {
    let n = graph.h.len();
    if graph.j.len() != graph.edges.len() {
        return Err(format!(
            "{} couplings for {} edges",
            graph.j.len(),
            graph.edges.len()
        ));
    }
    for (k, &(u, v)) in graph.edges.iter().enumerate() {
        if u == v {
            return Err(format!("edge {k} is a self-loop at node {u}"));
        }
        if u >= n || v >= n {
            return Err(format!("edge {k} has an endpoint outside 0..{n}"));
        }
    }
    let float_max = f64::from(f32::MAX);
    for (name, values) in [("h", &graph.h), ("J", &graph.j)] {
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() || v.abs() > float_max {
                return Err(format!("{name}[{i}] = {v} is not a finite float32 value"));
            }
        }
    }
    let unit = graph
        .h
        .iter()
        .chain(&graph.j)
        .all(|v| *v == -1.0 || *v == 0.0 || *v == 1.0);
    let mut field = vec![0_i64; n];
    let mut on_grid = true;
    for (i, &h) in graph.h.iter().enumerate() {
        match to_milli(h) {
            Some(m) => field[i] = i64::from(m).abs(),
            None => on_grid = false,
        }
    }
    for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
        match to_milli(j) {
            Some(m) => {
                field[u] += i64::from(m).abs();
                field[v] += i64::from(m).abs();
            }
            None => on_grid = false,
        }
    }
    let max_field = field.iter().copied().max().unwrap_or(0);
    let max_field_milli =
        (on_grid && max_field <= i64::from(WIDE_MAX_FIELD_MILLI)).then_some(max_field);
    Ok(Survey {
        max_field_milli,
        unit,
    })
}

/// Sort `graph` onto the smallest SA or Gibbs tier that runs it exactly.
///
/// # Errors
///
/// A message naming the first node, edge or coefficient that no tier
/// represents: a non-finite value, a value past `f32::MAX`, a self-loop, a
/// coupling count that differs from the edge count, or an endpoint outside
/// the node range. The caller reports it as a capacity refusal.
pub(crate) fn classify(graph: &IsingGraph) -> Result<Representation, String> {
    let survey = survey(graph)?;
    let Some(max_field) = survey.max_field_milli else {
        return Ok(Representation::Float32);
    };
    let whole_units = graph.h.iter().chain(&graph.j).all(|v| v.fract() == 0.0);
    let fits_int8 = whole_units && max_field <= i64::from(INT8_MAX_FIELD_MILLI);
    Ok(match (fits_int8, survey.unit) {
        (true, true) => Representation::Binary,
        (true, false) => Representation::Int8,
        (false, _) => Representation::Wide,
    })
}

/// Sort `graph` onto the smallest MSA tier that runs it exactly.
///
/// [`Representation::Binary`] selects the unit kernel, which reads every `h`
/// and `J` by sign and never sees its magnitude. [`Representation::Wide`]
/// selects `msa_anneal_weighted`, which holds milli `int` coefficients under
/// the same local-field bound as the wide SA kernel. MSA has no int8 kernel,
/// so this never returns [`Representation::Int8`]. The degree bound is
/// checked separately, on the CSR, by `sampler::validate_batch`.
///
/// # Errors
///
/// The same refusals as [`classify`].
pub(crate) fn classify_msa(graph: &IsingGraph) -> Result<Representation, String> {
    let survey = survey(graph)?;
    Ok(match (survey.max_field_milli, survey.unit) {
        (None, _) => Representation::Float32,
        (Some(_), true) => Representation::Binary,
        (Some(_), false) => Representation::Wide,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::IsingGraph;

    fn pair(h: [f64; 2], j: f64) -> IsingGraph {
        IsingGraph::new(h.to_vec(), vec![j], vec![(0, 1)])
    }

    fn star(j: f64, leaves: usize) -> IsingGraph {
        IsingGraph::new(
            vec![0.0; leaves + 1],
            vec![j; leaves],
            (1..=leaves).map(|v| (0, v)).collect(),
        )
    }

    #[test]
    fn unit_models_classify_as_binary() {
        assert_eq!(
            classify(&pair([1.0, -1.0], -1.0)),
            Ok(Representation::Binary)
        );
        assert_eq!(classify(&pair([0.0, 0.0], 0.0)), Ok(Representation::Binary));
        assert_eq!(
            classify(&pair([-0.0, 0.0], -0.0)),
            Ok(Representation::Binary)
        );
        assert_eq!(
            classify_msa(&pair([1.0, -1.0], -1.0)),
            Ok(Representation::Binary)
        );
        // A unit star past the int8 field bound still runs wide today.
        assert_eq!(classify(&star(1.0, 63)), Ok(Representation::Binary));
        assert_eq!(classify(&star(1.0, 64)), Ok(Representation::Wide));
        assert_eq!(classify_msa(&star(1.0, 64)), Ok(Representation::Binary));
    }

    #[test]
    fn whole_unit_models_in_i8_range_classify_as_int8() {
        assert_eq!(classify(&pair([2.0, 0.0], 3.0)), Ok(Representation::Int8));
        assert_eq!(classify(&star(21.0, 3)), Ok(Representation::Int8));
        assert_eq!(classify(&star(22.0, 3)), Ok(Representation::Wide));
    }

    #[test]
    fn msa_never_returns_int8() {
        let graph = pair([2.0, 0.0], 3.0);
        assert_eq!(classify_msa(&graph), Ok(Representation::Wide));
        assert_eq!(classify(&graph), Ok(Representation::Int8));
    }

    #[test]
    fn whole_milli_models_classify_as_wide() {
        assert_eq!(
            classify(&pair([0.25, -0.5], 0.001)),
            Ok(Representation::Wide)
        );
        assert_eq!(
            classify(&pair([1000.0, 0.0], 1.0)),
            Ok(Representation::Wide)
        );
        assert_eq!(
            classify_msa(&pair([0.25, -0.5], 0.001)),
            Ok(Representation::Wide)
        );
        assert_eq!(
            classify_msa(&pair([0.0, 0.0], 1000.0)),
            Ok(Representation::Wide)
        );
    }

    #[test]
    fn off_grid_models_classify_as_float32() {
        assert_eq!(
            classify(&pair([0.1 + 0.2, 0.0], 1.0)),
            Ok(Representation::Float32)
        );
        assert_eq!(
            classify(&pair([0.0005, 0.0], 1.0)),
            Ok(Representation::Float32)
        );
        assert_eq!(
            classify_msa(&pair([0.1 + 0.2, 0.0], 1.0)),
            Ok(Representation::Float32)
        );
    }

    #[test]
    fn a_field_past_the_wide_bound_is_float32() {
        let limit_units = f64::from(WIDE_MAX_FIELD_MILLI) / 1000.0;
        assert_eq!(
            classify(&pair([limit_units - 1.0, 0.0], 1.0)),
            Ok(Representation::Wide)
        );
        assert_eq!(
            classify(&pair([limit_units, 0.0], 1.0)),
            Ok(Representation::Float32)
        );
        assert_eq!(
            classify_msa(&pair([limit_units, 0.0], 1.0)),
            Ok(Representation::Float32)
        );
    }

    #[test]
    fn non_finite_coefficients_are_refused() {
        classify(&pair([f64::NAN, 0.0], 1.0)).unwrap_err();
        classify(&pair([0.0, 0.0], f64::INFINITY)).unwrap_err();
        classify(&pair([0.0, f64::NEG_INFINITY], 1.0)).unwrap_err();
        classify_msa(&pair([f64::NAN, 0.0], 1.0)).unwrap_err();
        classify(&pair([1e300, 0.0], 1.0)).unwrap_err();
    }

    #[test]
    fn wide_bound_matches_the_advertised_milli_limit() {
        assert_eq!(WIDE_MAX_FIELD_MILLI, 268_435_000);
        assert_eq!(to_milli(268_435.0), Some(268_435_000));
        assert_eq!(to_milli(268_435.001), None);
    }

    #[test]
    fn a_coupling_list_shorter_than_the_edges_is_refused() {
        let graph = IsingGraph::new(vec![0.0; 3], vec![1.0], vec![(0, 1), (1, 2)]);
        classify(&graph).unwrap_err();
        classify_msa(&graph).unwrap_err();
    }

    #[test]
    fn an_endpoint_outside_the_nodes_is_refused() {
        let graph = IsingGraph::new(vec![0.0; 2], vec![0.0005], vec![(0, 5)]);
        let err = classify(&graph).unwrap_err();
        assert!(err.contains("endpoint"), "{err}");
    }

    #[test]
    fn self_loop_is_refused_before_coefficient_conversion() {
        let graph = IsingGraph::new(vec![0.0], vec![0.0005], vec![(0, 0)]);
        let err = classify(&graph).unwrap_err();
        assert!(err.contains("self-loop"), "{err}");
        let err = classify_msa(&graph).unwrap_err();
        assert!(err.contains("self-loop"), "{err}");
    }

    #[test]
    fn a_wide_model_of_small_multiples_has_a_scale() {
        // 250 milli times {-2, 3}, and a zero.
        let graph = IsingGraph::new(
            vec![0.5, 0.0, -0.75],
            vec![0.25, -0.5],
            vec![(0, 1), (1, 2)],
        );
        assert_eq!(wide_i8_scale(&graph), Some(250));
        // The scale is the gcd, not the smallest value.
        let graph = pair([0.6, 0.0], 0.4);
        assert_eq!(wide_i8_scale(&graph), Some(200));
    }

    #[test]
    fn a_quotient_outside_i8_has_no_scale() {
        // 127 and -128 units fit; 128 does not.
        assert_eq!(wide_i8_scale(&pair([0.127, -0.128], 0.001)), Some(1));
        assert_eq!(wide_i8_scale(&pair([0.128, 0.0], 0.001)), None);
        assert_eq!(wide_i8_scale(&pair([-0.129, 0.0], 0.001)), None);
        assert_eq!(wide_i8_scale(&pair([0.0, 0.0], 300.0)), Some(300_000));
    }

    #[test]
    fn mixed_signs_share_one_scale() {
        let graph = pair([-0.25, 0.5], -0.75);
        assert_eq!(wide_i8_scale(&graph), Some(250));
    }

    #[test]
    fn whole_unit_wide_models_scale_by_1000() {
        // 64 unit bonds exceed the int8 field bound, so the model is Wide, and
        // every coefficient is one multiple of 1000 milli.
        let graph = star(1.0, 64);
        assert_eq!(classify(&graph), Ok(Representation::Wide));
        assert_eq!(wide_i8_scale(&graph), Some(1000));
    }

    #[test]
    fn a_model_without_nonzero_coefficients_or_off_grid_has_no_scale() {
        assert_eq!(wide_i8_scale(&pair([0.0, 0.0], 0.0)), None);
        assert_eq!(wide_i8_scale(&pair([0.1 + 0.2, 0.0], 1.0)), None);
        assert_eq!(wide_i8_scale(&pair([f64::NAN, 0.0], 1.0)), None);
    }

    #[test]
    fn a_batch_shares_the_gcd_and_the_i8_bound() {
        let a = pair([0.5, 0.0], 0.5);
        let b = pair([0.75, 0.0], 0.25);
        assert_eq!(batch_wide_i8_scale(&[&a, &b]), Some(250));
        // Alone, `c` has scale 10; with `d` the shared scale is 1 and 1270 does not fit.
        let c = pair([1.27, 0.0], 0.01);
        let d = pair([0.001, 0.0], 0.0);
        assert_eq!(wide_i8_scale(&c), Some(10));
        assert_eq!(batch_wide_i8_scale(&[&c, &d]), None);
    }

    #[test]
    fn precision_maps_each_tier() {
        assert_eq!(Representation::Binary.precision(), Precision::Binary);
        assert_eq!(Representation::Int8.precision(), Precision::Int8);
        assert_eq!(Representation::Wide.precision(), Precision::Int32);
        assert_eq!(Representation::Float32.precision(), Precision::Float32);
    }
}
