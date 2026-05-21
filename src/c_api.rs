//! C ABI layer for HFF.
//!
//! Exposes the same mathematical surface as the PyO3 bindings, but through
//! `extern "C"` functions that speak the C calling convention. This makes the
//! library callable from Go (cgo / purego), C, C++, Swift, Zig, Julia, and any
//! other language with a C FFI.
//!
//! All functions are pure: there is no global state, no opaque handles, and no
//! hidden allocations other than the Rayon thread pool used internally. Callers
//! own all buffers and their lifetimes.
//!
//! # Error codes
//!
//! Every function returns an `i32`:
//!   *  `0` — success
//!   * `-1` — null pointer passed for a required argument
//!   * `-2` — invalid argument (e.g. zero-length dimension, unknown method)
//!   * `-3` — internal error (shouldn't happen; indicates a bug)
//!
//! # Memory layout
//!
//! Objective matrices are **row-major**, `n_individuals * n_objectives`
//! contiguous `f64`s. Output buffers must be pre-allocated by the caller with
//! the documented length.

use ndarray::Array1;
use rayon::prelude::*;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::slice;

use crate::core_functions;
use crate::higd;

pub const HFF_OK: c_int = 0;
pub const HFF_ERR_NULL: c_int = -1;
pub const HFF_ERR_INVALID: c_int = -2;
pub const HFF_ERR_INTERNAL: c_int = -3;

/// Column-wise min-max normalise a row-major `(n, m)` matrix in place.
///
/// Matches the behaviour embedded in `calculate_hyperspherical_fitness_hf1_enhanced`
/// so the C and Python paths agree bit-for-bit.
fn min_max_normalize_rowmajor(data: &mut [f64], n: usize, m: usize) {
    if n <= 1 || m == 0 {
        return;
    }
    for j in 0..m {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            let v = data[i * m + j];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = hi - lo;
        let r = if range < f64::EPSILON { 1.0 } else { range };
        for i in 0..n {
            data[i * m + j] = (data[i * m + j] - lo) / r;
        }
    }
}

/// Compute per-objective (mean, std_dev) across the population, used for
/// decrowding. Mirrors the current PyO3 logic: collapses to scalar (avg_mean,
/// avg_std) because `core_functions` accepts a single tuple.
fn population_stats_for_decrowding(
    data: &[f64],
    n: usize,
    m: usize,
) -> Option<(f64, f64)> {
    if n <= 1 || m == 0 {
        return None;
    }
    let mut sum_mean = 0.0;
    let mut sum_std = 0.0;
    for j in 0..m {
        let mut s = 0.0;
        for i in 0..n {
            s += data[i * m + j];
        }
        let mean = s / n as f64;
        let mut var = 0.0;
        for i in 0..n {
            let d = data[i * m + j] - mean;
            var += d * d;
        }
        var /= n as f64;
        sum_mean += mean;
        sum_std += var.sqrt();
    }
    Some((sum_mean / m as f64, sum_std / m as f64))
}

/// HF1 (plain f64 variant): caller must pre-normalize.
///
/// # Safety
/// * `objectives` must point to `n_individuals * n_objectives` valid `f64`s.
/// * `out_fitness` must point to `n_individuals` writable `f64`s.
#[cfg(feature = "c-api")]
#[no_mangle]
pub unsafe extern "C" fn hff_hf1_f64(
    objectives: *const f64,
    n_individuals: usize,
    n_objectives: usize,
    decrowding: c_int,
    out_fitness: *mut f64,
) -> c_int {
    if objectives.is_null() || out_fitness.is_null() {
        return HFF_ERR_NULL;
    }
    if n_individuals == 0 {
        return HFF_OK;
    }
    if n_objectives == 0 {
        return HFF_ERR_INVALID;
    }

    let n = n_individuals;
    let m = n_objectives;
    let input = slice::from_raw_parts(objectives, n * m);
    let output = slice::from_raw_parts_mut(out_fitness, n);
    let decrowding = decrowding != 0;
    let stats = if decrowding {
        population_stats_for_decrowding(input, n, m)
    } else {
        None
    };

    let results: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = Array1::from_iter(input[i * m..(i + 1) * m].iter().copied());
            core_functions::calculate_single_hyperspherical_fitness_f64(
                &row, m, decrowding, stats,
            )
        })
        .collect();

    output.copy_from_slice(&results);
    HFF_OK
}

/// HF1 Enhanced: performs column-wise min-max normalisation and supports the
/// `"balanced"` / `"truenorth"` north-pole methods.
///
/// `north_pole_method` must be a null-terminated UTF-8 string. A null pointer
/// is treated as `"balanced"`.
///
/// # Safety
/// * `objectives` must point to `n_individuals * n_objectives` valid `f64`s.
/// * `out_fitness` must point to `n_individuals` writable `f64`s.
/// * `north_pole_method`, if non-null, must be a valid null-terminated C string.
#[cfg(feature = "c-api")]
#[no_mangle]
pub unsafe extern "C" fn hff_hf1_enhanced(
    objectives: *const f64,
    n_individuals: usize,
    n_objectives: usize,
    decrowding: c_int,
    north_pole_method: *const c_char,
    normalize: c_int,
    out_fitness: *mut f64,
) -> c_int {
    if objectives.is_null() || out_fitness.is_null() {
        return HFF_ERR_NULL;
    }
    if n_individuals == 0 {
        return HFF_OK;
    }
    if n_objectives == 0 {
        return HFF_ERR_INVALID;
    }

    let method: &str = if north_pole_method.is_null() {
        "balanced"
    } else {
        match CStr::from_ptr(north_pole_method).to_str() {
            Ok(s) => s,
            Err(_) => return HFF_ERR_INVALID,
        }
    };
    match method {
        "balanced" | "truenorth" => {}
        _ => return HFF_ERR_INVALID,
    }

    let n = n_individuals;
    let m = n_objectives;
    let input = slice::from_raw_parts(objectives, n * m);
    let output = slice::from_raw_parts_mut(out_fitness, n);
    let decrowding = decrowding != 0;
    let normalize = normalize != 0;

    let stats = if decrowding {
        population_stats_for_decrowding(input, n, m)
    } else {
        None
    };

    // Optional column-wise min-max normalisation. Callers with already-bounded
    // objectives should pass normalize=0 to skip — otherwise the column-best
    // individual maps to all-ones and collapses onto the reference pole.
    let normalized: Vec<f64> = if normalize {
        let mut buf = input.to_vec();
        min_max_normalize_rowmajor(&mut buf, n, m);
        buf
    } else {
        input.to_vec()
    };

    let results: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = Array1::from_iter(normalized[i * m..(i + 1) * m].iter().copied());
            core_functions::calculate_single_hyperspherical_fitness_f64_with_method(
                &row, m, decrowding, stats, method,
            )
        })
        .collect();

    output.copy_from_slice(&results);
    HFF_OK
}

/// CDF-corrected angular IGD (HIGD).
///
/// # Safety
/// * `solutions` must point to `n_solutions * dimensions` valid `f64`s.
/// * `out_value` must point to one writable `f64`.
#[cfg(feature = "c-api")]
#[no_mangle]
pub unsafe extern "C" fn hff_higd(
    solutions: *const f64,
    n_solutions: usize,
    dimensions: usize,
    n_reference_points: usize,
    seed: u64,
    positive_orthant: c_int,
    out_value: *mut f64,
) -> c_int {
    if solutions.is_null() || out_value.is_null() {
        return HFF_ERR_NULL;
    }
    if dimensions == 0 {
        return HFF_ERR_INVALID;
    }
    if n_solutions == 0 {
        *out_value = 1.0;
        return HFF_OK;
    }

    let input = slice::from_raw_parts(solutions, n_solutions * dimensions);
    let sols: Vec<Vec<f64>> = (0..n_solutions)
        .map(|i| input[i * dimensions..(i + 1) * dimensions].to_vec())
        .collect();

    *out_value = higd::calculate_higd(
        &sols,
        n_reference_points,
        dimensions,
        seed,
        positive_orthant != 0,
    );
    HFF_OK
}

/// Apply the Beta-CDF correction to a raw angular distance. Transforms
/// raw theta into a dimension-independent percentile rank in [0, 1], using
/// I_{sin²θ}((d−1)/2, 1/2) — the CDF of angular distance between uniformly
/// random points on an (m−1)-sphere (Cai, Fan, Jiang 2013). This is what
/// makes fitness comparable across runs with different objective counts.
///
/// Returns the corrected value directly; no error code, no buffer.
#[cfg(feature = "c-api")]
#[no_mangle]
pub extern "C" fn hff_cdf_correction(theta: f64, dimensions: usize) -> f64 {
    higd::cdf_beta_correction(theta, dimensions)
}

/// Log of the Beta-CDF correction. Returns ln(CDF) directly; useful when the
/// raw CDF underflows f64 (large dimensions with small theta — the regime
/// typical of image-reconstruction-style problems). Always representable in
/// f64 since ln is well-behaved all the way to the left tail.
#[cfg(feature = "c-api")]
#[no_mangle]
pub extern "C" fn hff_log_cdf_correction(theta: f64, dimensions: usize) -> f64 {
    higd::log_cdf_beta_correction(theta, dimensions)
}

/// Raw angular IGD (no CDF correction).
///
/// # Safety
/// Same invariants as [`hff_higd`].
#[cfg(feature = "c-api")]
#[no_mangle]
pub unsafe extern "C" fn hff_angular_igd(
    solutions: *const f64,
    n_solutions: usize,
    dimensions: usize,
    n_reference_points: usize,
    seed: u64,
    positive_orthant: c_int,
    out_value: *mut f64,
) -> c_int {
    if solutions.is_null() || out_value.is_null() {
        return HFF_ERR_NULL;
    }
    if dimensions == 0 {
        return HFF_ERR_INVALID;
    }
    if n_solutions == 0 {
        *out_value = std::f64::consts::PI;
        return HFF_OK;
    }

    let input = slice::from_raw_parts(solutions, n_solutions * dimensions);
    let sols: Vec<Vec<f64>> = (0..n_solutions)
        .map(|i| input[i * dimensions..(i + 1) * dimensions].to_vec())
        .collect();

    *out_value = higd::calculate_angular_igd(
        &sols,
        n_reference_points,
        dimensions,
        seed,
        positive_orthant != 0,
    );
    HFF_OK
}

#[cfg(all(test, feature = "c-api"))]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn hf1_enhanced_matches_plain_on_prenormalized_input() {
        // Pre-normalised (col-wise 0..1) input so both entry points see the
        // same values post-normalisation.
        let obj: Vec<f64> = vec![
            0.0, 1.0,
            1.0, 0.0,
            0.5, 0.5,
        ];
        let n = 3;
        let m = 2;
        let mut out_plain = vec![0.0; n];
        let mut out_enh = vec![0.0; n];
        let method = CString::new("balanced").unwrap();

        unsafe {
            let r1 = hff_hf1_f64(obj.as_ptr(), n, m, 0, out_plain.as_mut_ptr());
            let r2 = hff_hf1_enhanced(
                obj.as_ptr(),
                n,
                m,
                0,
                method.as_ptr(),
                1,                        // normalize
                out_enh.as_mut_ptr(),
            );
            assert_eq!(r1, HFF_OK);
            assert_eq!(r2, HFF_OK);
        }
        for i in 0..n {
            assert!(
                (out_plain[i] - out_enh[i]).abs() < 1e-10,
                "mismatch at {}: {} vs {}",
                i,
                out_plain[i],
                out_enh[i]
            );
        }
    }

    #[test]
    fn null_pointer_is_rejected() {
        let mut out = vec![0.0; 1];
        unsafe {
            let r = hff_hf1_f64(std::ptr::null(), 1, 1, 0, out.as_mut_ptr());
            assert_eq!(r, HFF_ERR_NULL);
        }
    }

    #[test]
    fn unknown_method_is_rejected() {
        let obj = [0.1, 0.2];
        let mut out = vec![0.0];
        let bad = CString::new("sideways").unwrap();
        unsafe {
            let r = hff_hf1_enhanced(
                obj.as_ptr(),
                1,
                2,
                0,
                bad.as_ptr(),
                1,                        // normalize
                out.as_mut_ptr(),
            );
            assert_eq!(r, HFF_ERR_INVALID);
        }
    }

    #[test]
    fn higd_runs() {
        let sols: Vec<f64> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut out = 0.0;
        unsafe {
            let r = hff_higd(sols.as_ptr(), 3, 3, 100, 42, 1, &mut out);
            assert_eq!(r, HFF_OK);
        }
        assert!((0.0..=1.0).contains(&out));
    }
}
