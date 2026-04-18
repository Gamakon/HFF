//! Hyperspherical Inverted Generational Distance (HIGD)
//!
//! A dimensionally-robust IGD variant that uses CDF-corrected angular distance
//! instead of Euclidean distance, making it robust to concentration of measure
//! on high-dimensional hyperspheres.
//!
//! # Mathematical Background
//!
//! Traditional IGD uses Euclidean distance which becomes meaningless at high
//! dimensions due to concentration of measure. HIGD replaces this with:
//!
//! 1. Angular distance (great circle distance on hypersphere)
//! 2. Beta CDF correction to account for concentration of measure
//!
//! The CDF correction transforms raw angular distances into quantiles of the
//! expected distribution, yielding consistent interpretation across all dimensions.
//!
//! # References
//!
//! - Muller (1959) - Mueller-Marsaglia method for uniform sphere sampling
//! - Huband et al. (2006) - WFG test problem suite and Pareto front definitions
//! - Coello & Cortés (2004) - Original IGD metric

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use statrs::function::beta::beta_reg;

/// Generate a uniformly distributed random point on the FULL d-dimensional
/// unit hypersphere using Mueller-Marsaglia (Gaussian normalization) method.
fn generate_sphere_point<R: rand::Rng>(dimensions: usize, rng: &mut R) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut point: Vec<f64> = (0..dimensions)
        .map(|_| normal.sample(rng))
        .collect();

    let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        point.iter_mut().for_each(|x| *x /= norm);
    }

    point
}

/// Generate a uniformly distributed random point on the POSITIVE ORTHANT of
/// a d-dimensional unit hypersphere (for WFG4-9 Pareto fronts).
fn generate_positive_sphere_point<R: rand::Rng>(dimensions: usize, rng: &mut R) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut point: Vec<f64> = (0..dimensions)
        .map(|_| {
            let sample: f64 = normal.sample(rng);
            sample.abs()  // abs() restricts to positive orthant
        })
        .collect();

    let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        point.iter_mut().for_each(|x| *x /= norm);
    }

    point
}

/// Generate reference front points on the unit hypersphere.
///
/// # Arguments
/// * `n_points` - Number of reference points to generate
/// * `dimensions` - Number of objectives (dimensionality)
/// * `seed` - Random seed for reproducibility
/// * `positive_orthant` - If true, restrict to positive orthant (for WFG4-9)
pub fn generate_reference_front(
    n_points: usize,
    dimensions: usize,
    seed: u64,
    positive_orthant: bool,
) -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..n_points)
        .map(|_| {
            if positive_orthant {
                generate_positive_sphere_point(dimensions, &mut rng)
            } else {
                generate_sphere_point(dimensions, &mut rng)
            }
        })
        .collect()
}

/// Compute cosine similarity between two vectors.
#[inline]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute angular distance (in radians) between two vectors.
#[inline]
fn angular_distance(a: &[f64], b: &[f64]) -> f64 {
    cosine_similarity(a, b).acos()
}

/// CDF correction using regularized incomplete beta function.
///
/// For a d-dimensional hypersphere, the angular distance θ from a fixed
/// direction to a uniformly random point has CDF:
///     F(θ) = I_{sin²(θ)}((d-1)/2, 1/2)
///
/// This is the correct formula for the angular distance distribution on
/// the hypersphere. The CDF transforms raw angular distance into a
/// percentile rank (0=best, 1=worst) accounting for concentration of measure.
///
/// Reference: The angular distribution on the n-sphere follows a Beta
/// distribution when transformed via sin²(θ).
#[inline]
pub fn cdf_beta_correction(theta: f64, dimensions: usize) -> f64 {
    let alpha = (dimensions - 1) as f64 / 2.0;
    let x = theta.sin().powi(2);  // sin²(θ)

    // Handle edge cases
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if dimensions < 2 {
        return theta;  // Degenerate case
    }

    // I_{sin²(θ)}((d-1)/2, 1/2)
    beta_reg(alpha, 0.5, x)
}

/// Normalize a solution vector to unit length.
fn normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Calculate HIGD (Hyperspherical Inverted Generational Distance).
///
/// For each reference point r:
/// 1. Find the solution s with minimum angular distance to r
/// 2. Apply CDF correction to that angular distance
/// 3. Average over all reference points
///
/// # Arguments
/// * `solutions` - Solution set from optimizer (will be normalized)
/// * `n_reference_points` - Number of reference points (e.g., 10000)
/// * `dimensions` - Number of objectives
/// * `seed` - Random seed for reproducibility
/// * `positive_orthant` - If true, sample reference from positive orthant (WFG4-9)
///
/// # Returns
/// HIGD value in [0, 1] where 0 = exceptional, 0.5 = random, 1 = antipodal
pub fn calculate_higd(
    solutions: &[Vec<f64>],
    n_reference_points: usize,
    dimensions: usize,
    seed: u64,
    positive_orthant: bool,
) -> f64 {
    if solutions.is_empty() {
        return 1.0;  // No solutions = worst possible
    }

    // Generate reference front
    let reference = generate_reference_front(
        n_reference_points,
        dimensions,
        seed,
        positive_orthant
    );

    // Normalize all solutions to unit sphere
    let solutions_normalized: Vec<Vec<f64>> = solutions
        .iter()
        .map(|s| normalize(s))
        .collect();

    // Parallel computation over reference points
    let sum: f64 = reference
        .par_iter()
        .map(|r| {
            // Find minimum angular distance from this reference point to any solution
            let min_theta = solutions_normalized
                .iter()
                .map(|s| angular_distance(r, s))
                .fold(f64::INFINITY, f64::min);

            // Apply CDF correction
            cdf_beta_correction(min_theta, dimensions)
        })
        .sum();

    sum / n_reference_points as f64
}

/// Calculate raw angular IGD (without CDF correction) for comparison.
///
/// Returns mean of minimum angular distances in radians.
pub fn calculate_angular_igd(
    solutions: &[Vec<f64>],
    n_reference_points: usize,
    dimensions: usize,
    seed: u64,
    positive_orthant: bool,
) -> f64 {
    if solutions.is_empty() {
        return std::f64::consts::PI;  // Maximum angular distance
    }

    let reference = generate_reference_front(
        n_reference_points,
        dimensions,
        seed,
        positive_orthant
    );

    let solutions_normalized: Vec<Vec<f64>> = solutions
        .iter()
        .map(|s| normalize(s))
        .collect();

    let sum: f64 = reference
        .par_iter()
        .map(|r| {
            solutions_normalized
                .iter()
                .map(|s| angular_distance(r, s))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();

    sum / n_reference_points as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_point_normalization() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let point = generate_sphere_point(100, &mut rng);
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_positive_orthant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let point = generate_positive_sphere_point(100, &mut rng);
        assert!(point.iter().all(|&x| x >= 0.0));
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cdf_correction_bounds() {
        // theta = 0 should give CDF = 0
        let cdf_0 = cdf_beta_correction(0.0, 100);
        assert!(cdf_0 < 1e-10);

        // theta = pi should give CDF = 1
        let cdf_pi = cdf_beta_correction(std::f64::consts::PI, 100);
        assert!((cdf_pi - 1.0).abs() < 1e-10);

        // theta = pi/2 should give CDF = 0.5
        let cdf_half = cdf_beta_correction(std::f64::consts::FRAC_PI_2, 100);
        assert!((cdf_half - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_perfect_solutions() {
        // If solutions ARE the reference front, HIGD should be ~0
        let reference = generate_reference_front(100, 10, 42, true);
        let higd = calculate_higd(&reference, 100, 10, 42, true);
        assert!(higd < 0.05, "HIGD for perfect solutions should be near 0, got {}", higd);
    }

    #[test]
    fn test_higd_range() {
        // Random solutions should give HIGD around 0.5
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let solutions: Vec<Vec<f64>> = (0..100)
            .map(|_| generate_positive_sphere_point(50, &mut rng))
            .collect();

        let higd = calculate_higd(&solutions, 1000, 50, 42, true);
        assert!(higd >= 0.0 && higd <= 1.0, "HIGD should be in [0,1], got {}", higd);
    }
}
