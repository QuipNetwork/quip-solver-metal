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
//! most [`WIDE_MAX_FIELD_MILLI`]. Any other model is refused, so a caller can
//! route it to a solver that takes it.

use quip_solver_core::IsingGraph;

/// Largest local field the int8 path takes, in milli: 63 whole units.
pub(crate) const INT8_MAX_FIELD_MILLI: u32 = 63_000;

/// Largest local field the wide path takes, in milli. The SA neighbour update
/// adds `4·J`, which stays inside `int` because `|J| ≤ field ≤ 2^28`.
pub(crate) const WIDE_MAX_FIELD_MILLI: u32 = 1 << 28;

/// The coefficient representation a model needs for exact execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Representation {
    /// Today's `int8_t` kernels, whole units.
    Int8,
    /// Reserved for the `QUIP_WIDE` kernels, `int` milli units.
    Wide,
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

/// Sort `graph` onto a path, or say why no path runs it exactly.
///
/// # Errors
///
/// A message naming the first coefficient or node that no path represents.
/// The caller reports it as a capacity refusal.
pub(crate) fn classify(graph: &IsingGraph) -> Result<Representation, String> {
    let n = graph.h.len();
    if graph.j.len() != graph.edges.len() {
        return Err(format!(
            "{} couplings for {} edges",
            graph.j.len(),
            graph.edges.len()
        ));
    }
    let mut field = vec![0_i64; n];
    for (i, &h) in graph.h.iter().enumerate() {
        let m = to_milli(h).ok_or_else(|| {
            format!("h[{i}] = {h} is not a whole number of milli units within the wide bound")
        })?;
        field[i] = i64::from(m).abs();
    }
    for (k, (&(u, v), &j)) in graph.edges.iter().zip(&graph.j).enumerate() {
        let m = to_milli(j).ok_or_else(|| {
            format!("J[{k}] = {j} is not a whole number of milli units within the wide bound")
        })?;
        if u >= n || v >= n {
            return Err(format!("edge {k} has an endpoint outside 0..{n}"));
        }
        // Both endpoints, also for a self-loop: an over-count only refuses
        // sooner.
        field[u] += i64::from(m).abs();
        field[v] += i64::from(m).abs();
    }
    let max_field = field.iter().copied().max().unwrap_or(0);
    if max_field > i64::from(WIDE_MAX_FIELD_MILLI) {
        return Err(format!(
            "local field {max_field} milli exceeds the wide path's {WIDE_MAX_FIELD_MILLI} milli"
        ));
    }
    let whole_units = graph.h.iter().chain(&graph.j).all(|v| v.fract() == 0.0);
    if whole_units && max_field <= i64::from(INT8_MAX_FIELD_MILLI) {
        Ok(Representation::Int8)
    } else {
        Ok(Representation::Wide)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::IsingGraph;

    fn pair(h: [f64; 2], j: f64) -> IsingGraph {
        IsingGraph::new(h.to_vec(), vec![j], vec![(0, 1)])
    }

    #[test]
    fn pm1_models_take_the_int8_path() {
        assert_eq!(classify(&pair([1.0, -1.0], -1.0)), Ok(Representation::Int8));
        assert_eq!(classify(&pair([0.0, 0.0], 0.0)), Ok(Representation::Int8));
    }

    #[test]
    fn a_whole_unit_field_of_63_is_int8_and_64_is_wide() {
        let star = |j: f64, leaves: usize| {
            IsingGraph::new(
                vec![0.0; leaves + 1],
                vec![j; leaves],
                (1..=leaves).map(|v| (0, v)).collect(),
            )
        };
        assert_eq!(classify(&star(1.0, 63)), Ok(Representation::Int8));
        assert_eq!(classify(&star(1.0, 64)), Ok(Representation::Wide));
        assert_eq!(classify(&star(21.0, 3)), Ok(Representation::Int8));
        assert_eq!(classify(&star(22.0, 3)), Ok(Representation::Wide));
    }

    #[test]
    fn fractional_milli_values_take_the_wide_path() {
        assert_eq!(
            classify(&pair([0.25, -0.5], 0.001)),
            Ok(Representation::Wide)
        );
        assert_eq!(
            classify(&pair([1000.0, 0.0], 1.0)),
            Ok(Representation::Wide)
        );
    }

    #[test]
    fn classify_refuses_values_that_are_not_exact_milli() {
        classify(&pair([0.0005, 0.0], 1.0)).unwrap_err();
        classify(&pair([0.1 + 0.2, 0.0], 1.0)).unwrap_err();
    }

    #[test]
    fn classify_handles_nonfinite_and_negative_zero() {
        classify(&pair([f64::NAN, 0.0], 1.0)).unwrap_err();
        classify(&pair([0.0, 0.0], f64::INFINITY)).unwrap_err();
        assert_eq!(classify(&pair([-0.0, 0.0], -0.0)), Ok(Representation::Int8));
    }

    #[test]
    fn a_field_past_the_wide_bound_is_refused() {
        let limit_units = f64::from(WIDE_MAX_FIELD_MILLI) / 1000.0;
        assert_eq!(
            classify(&pair([limit_units - 1.0, 0.0], 1.0)),
            Ok(Representation::Wide)
        );
        let err = classify(&pair([limit_units, 0.0], 1.0)).unwrap_err();
        assert!(err.contains("local field"), "{err}");
    }

    #[test]
    fn a_coupling_list_shorter_than_the_edges_is_refused() {
        let graph = IsingGraph::new(vec![0.0; 3], vec![1.0], vec![(0, 1), (1, 2)]);
        classify(&graph).unwrap_err();
    }
}
