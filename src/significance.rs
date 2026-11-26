//! Angle significance: turn an HFF angle into a dimension-free log p-value.
//!
//! An angle on its own is not comparable across objective counts. Concentration
//! of measure moves the typical angle as `m` grows, so the same θ means
//! something different at m = 10 and m = 40. A significance score fixes this by
//! asking how likely a *random* solution is to be at least this close to the
//! pole, i.e. P(Θ ≤ θ) under a null distribution. The answer has the same
//! meaning at every `m`.
//!
//! The null must match the geometry of the angle being scored:
//!
//! * [`AngleNull::UniformSphere`]: Θ is the angle from a fixed pole to a
//!   direction drawn uniformly on S^{m-1}. P(Θ ≤ θ) = I_{sin²θ}((m−1)/2, ½)
//!   (Cai & Fang 2013), the GECCO 2026 poster's eq. 3. This is the right
//!   null for angles between uniformly spread directions, as in HIGD.
//!
//! * [`AngleNull::TrueNorth`]: Θ is the HF1 TrueNorth angle of a solution
//!   whose min-max normalised objectives are independent Uniform(0, 1).
//!   TrueNorth sets cos θ = 1 − ‖x‖²/m, so 1 − cos θ is the mean of m iid
//!   squared uniforms, with mean 1/3 and variance 4/(45m). By the central
//!   limit theorem, P(Θ ≤ θ) ≈ Φ((1 − cos θ − 1/3) / √(4/(45m))).
//!
//! # Why TrueNorth needs its own null
//!
//! The TrueNorth angle of a random solution concentrates near
//! arccos(2/3) ≈ 48°, not near 90°. Scoring it under the uniform-sphere null
//! therefore drifts with `m`: the median random solution scores
//! p ≈ 0.1 at m = 5 but p ≈ 1e-11 at m = 80, so random solutions look more
//! "significant" as objectives are added. Under the TrueNorth null, the
//! median random solution scores p ≈ 0.5 at every m, and the best of N
//! random solutions scores p ≈ 1/(N + 1). The unit tests below check this
//! calibration.
//!
//! The ranking of solutions within one population is unchanged by either
//! null, because both are monotone in θ. The null only matters when scores
//! are compared across different objective counts: rolling re-planning,
//! stopping rules, and quality tracking over time.
//!
//! # Assumptions and accuracy
//!
//! The TrueNorth null assumes normalised objectives in [0, 1] and no
//! decrowding transform. It is a CLT approximation: well calibrated from
//! roughly m ≥ 5 and slightly conservative below that. Correlated objectives
//! are genuine structure, not noise, and correctly produce small p-values.
//!
//! All functions return the natural log of p, which stays representable in
//! the deep left tail where p itself underflows f64.

use statrs::function::erf::erfc;
use std::f64::consts::{PI, SQRT_2};

/// Null distribution used to calibrate an angle into a p-value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleNull {
    /// Uniform direction on S^{m-1}; Beta-CDF correction (poster eq. 3).
    UniformSphere,
    /// Random normalised objectives scored by HF1 TrueNorth.
    TrueNorth,
}

impl AngleNull {
    /// Parse `"truenorth"` or `"uniform_sphere"`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "truenorth" => Some(AngleNull::TrueNorth),
            "uniform_sphere" => Some(AngleNull::UniformSphere),
            _ => None,
        }
    }
}

/// Natural log of the standard normal CDF, accurate into the far left tail.
///
/// Uses erfc down to z = −37 (where Φ(z) is still a normal f64), then the
/// asymptotic Mills-ratio series, whose truncation error is below 1e-15 there.
pub fn log_normal_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z >= -37.0 {
        return (0.5 * erfc(-z / SQRT_2)).ln();
    }
    let z2 = z * z;
    let inv = 1.0 / z2;
    // 1 − 1/z² + 3/z⁴ − 15/z⁶ + 105/z⁸ − 945/z¹⁰
    let series = 1.0 + inv * (-1.0 + inv * (3.0 + inv * (-15.0 + inv * (105.0 - 945.0 * inv))));
    -0.5 * z2 - (-z).ln() - 0.5 * (2.0 * PI).ln() + series.ln()
}

/// ln P(Θ ≤ θ) for an HF1 TrueNorth angle under the TrueNorth null.
///
/// `theta` is in radians (as returned by HF1 TrueNorth, in [0, π/2]);
/// `n_objectives` is m, the number of objectives (not m + 1).
pub fn truenorth_log_pvalue(theta: f64, n_objectives: usize) -> f64 {
    if !theta.is_finite() || n_objectives == 0 {
        return f64::NAN;
    }
    let m = n_objectives as f64;
    let mean_sq = 1.0 - theta.cos(); // = ‖x‖² / m
    let z = (mean_sq - 1.0 / 3.0) / (4.0 / (45.0 * m)).sqrt();
    log_normal_cdf(z)
}

/// ln P(Θ ≤ θ) under the chosen null.
///
/// `UniformSphere` delegates to [`crate::higd::log_cdf_beta_correction`]
/// with `dimensions = n_objectives`, matching the existing CDF correction.
pub fn angle_log_pvalue(theta: f64, n_objectives: usize, null: AngleNull) -> f64 {
    match null {
        AngleNull::TrueNorth => truenorth_log_pvalue(theta, n_objectives),
        AngleNull::UniformSphere => crate::higd::log_cdf_beta_correction(theta, n_objectives),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_functions::calculate_single_hyperspherical_fitness_f64_with_method;
    use ndarray::Array1;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    // Reference values from scipy.stats.norm.logcdf.
    const REFERENCE: [(f64, f64); 8] = [
        (0.0, -0.6931471805599453),
        (-1.0, -1.8410216450092634),
        (-5.0, -15.064998393988727),
        (-10.0, -53.23128515051248),
        (-36.9, -685.3328831653507),
        (-37.1, -692.7382807156233),
        (-40.0, -804.6084420137538),
        (3.0, -0.0013508099647481938),
    ];

    #[test]
    fn log_normal_cdf_matches_reference() {
        for (z, expected) in REFERENCE {
            let got = log_normal_cdf(z);
            assert!(
                (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "z={z}: got {got}, expected {expected}"
            );
        }
        assert!(log_normal_cdf(-200.0).is_finite());
        assert!(log_normal_cdf(-200.0) < log_normal_cdf(-40.0));
    }

    fn random_truenorth_thetas(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let x: Array1<f64> = (0..m).map(|_| rng.gen::<f64>()).collect();
                calculate_single_hyperspherical_fitness_f64_with_method(&x, m, false, None, "truenorth")
            })
            .collect()
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    #[test]
    fn truenorth_null_is_calibrated_across_objective_counts() {
        for (k, m) in [10usize, 20, 40, 80].into_iter().enumerate() {
            let thetas = random_truenorth_thetas(m, 4000, 7 + k as u64);
            let p: Vec<f64> = thetas.iter().map(|&t| truenorth_log_pvalue(t, m).exp()).collect();
            let med = median(p.clone());
            let below_5pct = p.iter().filter(|&&v| v < 0.05).count() as f64 / p.len() as f64;
            assert!((med - 0.5).abs() < 0.06, "m={m}: median p {med}");
            assert!((below_5pct - 0.05).abs() < 0.02, "m={m}: P(p<0.05) = {below_5pct}");
        }
    }

    #[test]
    fn uniform_sphere_null_drifts_for_truenorth_angles() {
        // Documents why the TrueNorth null exists: for random solutions the
        // Beta-CDF p-value collapses as m grows.
        let med_low = median(random_truenorth_thetas(5, 2000, 1))
            .pipe(|t| angle_log_pvalue(t, 5, AngleNull::UniformSphere));
        let med_high = median(random_truenorth_thetas(80, 2000, 2))
            .pipe(|t| angle_log_pvalue(t, 80, AngleNull::UniformSphere));
        assert!(med_low > (0.01f64).ln());
        assert!(med_high < (1e-8f64).ln());
    }

    #[test]
    fn uniform_sphere_matches_existing_log_cdf_correction() {
        for &(t, d) in &[(0.3, 5usize), (0.8, 20), (1.2, 100)] {
            assert_eq!(
                angle_log_pvalue(t, d, AngleNull::UniformSphere),
                crate::higd::log_cdf_beta_correction(t, d)
            );
        }
    }

    #[test]
    fn truenorth_pvalue_is_monotone_in_theta() {
        let mut last = f64::NEG_INFINITY;
        for i in 0..=90 {
            let v = truenorth_log_pvalue((i as f64).to_radians(), 25);
            assert!(v >= last);
            last = v;
        }
    }

    #[test]
    fn parse_names() {
        assert_eq!(AngleNull::parse("truenorth"), Some(AngleNull::TrueNorth));
        assert_eq!(AngleNull::parse("uniform_sphere"), Some(AngleNull::UniformSphere));
        assert_eq!(AngleNull::parse("balanced"), None);
    }

    trait Pipe: Sized {
        fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
            f(self)
        }
    }
    impl Pipe for f64 {}
}
