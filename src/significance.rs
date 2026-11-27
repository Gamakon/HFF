//! Angle significance: express an HFF angle as ln P(Θ ≤ θ) under a chosen null.
//!
//! * [`AngleNull::UniformSphere`]: the Beta-CDF correction
//!   P(Θ ≤ θ) = I_{sin²θ}((m−1)/2, ½) (Cai & Fang 2013; GECCO 2026 poster,
//!   eq. 3), as used by HIGD. Identical to
//!   [`crate::higd::log_cdf_beta_correction`].
//!
//! * [`AngleNull::TrueNorth`]: an optional null for HF1 TrueNorth angles.
//!   TrueNorth sets cos θ = 1 − ‖x‖²/m, so with normalised objectives
//!   modelled as independent Uniform(0, 1), 1 − cos θ has mean 1/3 and
//!   variance 4/(45m), giving P(Θ ≤ θ) ≈ Φ((1 − cos θ − 1/3) / √(4/(45m))).
//!
//! Both are monotone in θ, so ranking within a population is unchanged.
//! All functions return natural logs, which stay representable in the deep
//! left tail.

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
    fn truenorth_null_matches_uniform_objective_model() {
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
}
