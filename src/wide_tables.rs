// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Integer acceptance thresholds for the wide kernels, built on the host.
//!
//! Each beta rung gets one row of [`WIDE_ROW`] thresholds in milli units.
//! The kernel picks an entry with its own integer random stream and compares
//! an integer energy change or field against it, so the spins it produces do
//! not depend on the GPU's floating-point unit. Rows come from f64 on the
//! host, from the f32 schedule the kernels already receive.

/// Entries per rung. Must equal `WIDE_ROW` in `sa.metal` and `gibbs.metal`.
pub(crate) const WIDE_ROW: usize = 2048;
const _: () = assert!(WIDE_ROW.is_power_of_two());

/// Largest threshold table one batch uploads: 8,192 rungs.
// Task 6 uses this cap when preparing wide batches.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) const WIDE_TABLE_MAX_BYTES: usize = 64 << 20;

/// Distinct from every other stream seeded from the job seed.
const ROW_SALT: u64 = 0x5749_4445_5441_424C;

/// Whether a schedule of `rungs` rungs fits [`WIDE_TABLE_MAX_BYTES`].
// Task 6 checks the table cap before upload.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn wide_table_fits(rungs: usize) -> bool {
    rungs.saturating_mul(WIDE_ROW * 4) <= WIDE_TABLE_MAX_BYTES
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in the open interval (0, 1).
fn open_unit(state: &mut u64) -> f64 {
    let u = ((splitmix64(state) >> 11) as f64 + 0.5) * (1.0 / (1_u64 << 53) as f64);
    // The top draw rounds to 1.0 in f64; keep both logarithms finite.
    u.min(1.0 - f64::EPSILON / 2.0)
}

/// An integral threshold converted to `i32`, saturating.
fn saturate(x: f64) -> i32 {
    // `as` saturates and maps NaN to 0. Callers never pass NaN.
    x as i32
}

fn rows(betas: &[f32], seed: u64, mut entry: impl FnMut(f64, &mut u64) -> i32) -> Vec<i32> {
    let mut state = seed ^ ROW_SALT;
    let mut out = Vec::with_capacity(betas.len() * WIDE_ROW);
    for &beta in betas {
        let per_milli = f64::from(beta) / 1000.0;
        for _ in 0..WIDE_ROW {
            out.push(entry(per_milli, &mut state));
        }
    }
    out
}

/// Metropolis rows: accept an uphill `Δ` (milli) when `Δ ≤ entry`.
// Task 6 passes these rows to the wide Metropolis kernel.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn metropolis_rows(betas: &[f32], seed: u64) -> Vec<i32> {
    rows(betas, seed, |b, state| {
        let e = -open_unit(state).ln();
        if b <= 0.0 {
            i32::MAX
        } else {
            saturate((e / b).floor())
        }
    })
}

/// Heat-bath rows: set `s = +1` when the local field `f` (milli) is `≤ entry`.
// Task 6 passes these rows to the wide heat-bath kernel.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn heat_bath_rows(betas: &[f32], seed: u64) -> Vec<i32> {
    rows(betas, seed, |b, state| {
        let u = open_unit(state);
        let logistic = (u / (1.0 - u)).ln();
        if b <= 0.0 {
            // P = 1/2: `+1` exactly when the draw falls below zero.
            return if logistic < 0.0 { i32::MAX } else { i32::MIN };
        }
        saturate((-logistic / (2.0 * b)).ceil() - 1.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fraction(row: &[i32], pred: impl Fn(i32) -> bool) -> f64 {
        row.iter().filter(|&&t| pred(t)).count() as f64 / row.len() as f64
    }

    #[test]
    fn rows_are_deterministic_per_seed() {
        let betas = [0.1_f32, 1.0, 5.0];
        assert_eq!(metropolis_rows(&betas, 7), metropolis_rows(&betas, 7));
        assert_ne!(metropolis_rows(&betas, 7), metropolis_rows(&betas, 8));
        assert_eq!(heat_bath_rows(&betas, 7).len(), 3 * WIDE_ROW);
    }

    #[test]
    fn metropolis_rows_accept_at_the_boltzmann_rate() {
        // beta 0.5 per unit, an uphill step of 1 unit = 1000 milli.
        let row = metropolis_rows(&[0.5], 1);
        let p = fraction(&row, |t| 1000 <= t);
        assert!((p - (-0.5_f64).exp()).abs() < 0.04, "{p}");
    }

    #[test]
    fn heat_bath_rows_match_the_logistic_probability() {
        // P(s = +1) at field +1 unit and beta 0.5 is sigma(-1).
        let row = heat_bath_rows(&[0.5], 2);
        let p = fraction(&row, |g| 1000 <= g);
        let want = 1.0 / (1.0 + 1.0_f64.exp());
        assert!((p - want).abs() < 0.04, "{p}");
        // At field 0 the probability is one half.
        let p0 = fraction(&row, |g| 0 <= g);
        assert!((p0 - 0.5).abs() < 0.04, "{p0}");
    }

    #[test]
    fn rows_clamp_extreme_betas() {
        let betas = [0.0_f32, f32::MIN_POSITIVE, f32::MAX];
        let m = metropolis_rows(&betas, 3);
        assert!(m[..WIDE_ROW].iter().all(|&t| t == i32::MAX));
        assert!(m[2 * WIDE_ROW..].iter().all(|&t| t == 0));
        let h = heat_bath_rows(&betas, 3);
        assert!(h[..WIDE_ROW]
            .iter()
            .all(|&g| g == i32::MAX || g == i32::MIN));
        // At f32::MAX per unit the field threshold collapses to -1 or 0.
        assert!(h[2 * WIDE_ROW..].iter().all(|&g| (-1..=0).contains(&g)));
    }

    #[test]
    fn the_table_cap_counts_rows() {
        assert!(wide_table_fits(WIDE_TABLE_MAX_BYTES / (WIDE_ROW * 4)));
        assert!(!wide_table_fits(WIDE_TABLE_MAX_BYTES / (WIDE_ROW * 4) + 1));
    }
}
