// HFF -- Hyperspherical Fitness Functions
//
// Pure Rust core with optional Python (pyo3) and C (extern "C") binding layers.
//
// Binding layers (opt-in via Cargo features):
//   - feature "python" (default): PyO3 module `hff_core`
//       * calculate_hyperspherical_fitness_hf1_f64
//       * calculate_hyperspherical_fitness_hf1_enhanced
//       * calculate_higd
//       * calculate_angular_igd
//   - feature "c-api": C ABI symbols exported from the cdylib
//       * hff_hf1_f64
//       * hff_hf1_enhanced
//       * hff_higd
//       * hff_angular_igd

/// Core mathematical functions for Hyperspherical Fitness calculations
pub mod core_functions;

/// Hyperspherical Inverted Generational Distance (HIGD) - dimensionally-robust IGD variant
pub mod higd;

/// C ABI bindings for use from Go (cgo), C/C++, and any other C-FFI-capable language.
#[cfg(feature = "c-api")]
pub mod c_api;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use ndarray::{Array1, s};
#[cfg(feature = "python")]
use rayon::prelude::*;

/// Calculate HF1 geometric fitness with standard f64 precision
///
/// PyO3 wrapper for the core HF1 algorithm. This function provides a Python interface
/// to the high-performance Rust implementation of single-reference hyperspherical fitness.
///
/// **IMPORTANT**: This function expects objectives to already be column-wise normalized!
/// For automatic normalization, use `calculate_hyperspherical_fitness_hf1_enhanced` instead.
///
/// # Arguments
///
/// * `objectives` - 2D numpy array of shape (n_individuals, n_objectives)
///                 containing PRE-NORMALIZED objective values
///
/// # Returns
///
/// 1D numpy array of hyperspherical fitness values for each individual.
/// Values are in radians, range [0, pi], where lower values indicate better fitness.
///
/// # Performance Notes
///
/// - Uses Rayon for automatic parallelization across individuals
/// - Optimized for large populations (scales well with n_individuals)
/// - Memory usage: O(n_individuals) temporary storage
#[cfg(feature = "python")]
#[pyfunction]
fn calculate_hyperspherical_fitness_hf1_f64(
    py: Python,
    objectives: PyReadonlyArray2<f64>,
    decrowding: Option<bool>,
) -> PyResult<Py<PyArray1<f64>>> {
    let objectives = objectives.as_array();
    let (n_individuals, n_objectives) = objectives.dim();
    let decrowding = decrowding.unwrap_or(false);

    if n_individuals == 0 {
        return Ok(Array1::zeros(0).into_pyarray(py).to_owned());
    }

    // Calculate population statistics for decrowding if needed
    let population_stats = if decrowding && n_individuals > 1 {
        // Calculate mean and std dev across all individuals for each objective
        let mut means = Vec::with_capacity(n_objectives);
        let mut std_devs = Vec::with_capacity(n_objectives);

        for obj_idx in 0..n_objectives {
            let obj_column: Vec<f64> = (0..n_individuals)
                .map(|i| objectives[[i, obj_idx]])
                .collect();

            let mean = obj_column.iter().sum::<f64>() / n_individuals as f64;
            let variance = obj_column
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / n_individuals as f64;
            let std_dev = variance.sqrt();

            means.push(mean);
            std_devs.push(std_dev);
        }

        // For simplicity, use average of all objective means and stds
        // In practice, each objective should be handled separately
        let avg_mean = means.iter().sum::<f64>() / n_objectives as f64;
        let avg_std = std_devs.iter().sum::<f64>() / n_objectives as f64;

        Some((avg_mean, avg_std))
    } else {
        None
    };

    // Delegate to core implementation with parallel processing
    let hyperspherical_fitness_values: Vec<f64> = (0..n_individuals)
        .into_par_iter()
        .map(|i| {
            let individual = objectives.slice(s![i, ..]);
            core_functions::calculate_single_hyperspherical_fitness_f64(
                &individual.to_owned(),
                n_objectives,
                decrowding,
                population_stats,
            )
        })
        .collect();

    Ok(Array1::from_vec(hyperspherical_fitness_values)
        .into_pyarray(py)
        .to_owned())
}

/// Enhanced HF1 with TrueNorth vs BalancedNorth method selection
///
/// PyO3 wrapper for enhanced HF1 algorithm supporting both north pole methods.
/// This is the main API function that external teams should use to get both
/// TrueNorth and BalancedNorth fitness calculation options.
///
/// # Arguments
///
/// * `objectives` - 2D numpy array of shape (n_individuals, n_objectives)
/// * `decrowding` - Optional boolean to enable/disable decrowding transformation
/// * `north_pole_method` - String specifying north pole method:
///   - "balanced": BalancedNorth Fitness - equal objective trade-offs (default)
///   - "truenorth": TrueNorth Fitness - direct minimization convergence
///
/// # Returns
///
/// * `PyArray1<f64>` - Array of angular distances (lower values = better fitness)
///
/// # North Pole Methods
///
/// **BalancedNorth ("balanced")**:
/// - Philosophy: Equal trade-off optimization
/// - Reference: (1/sqrt(m), 1/sqrt(m), ..., 1/sqrt(m)) in R^m
/// - Use case: Multi-criteria decision making, equal importance
///
/// **TrueNorth ("truenorth")**:
/// - Philosophy: Direct minimization convergence
/// - Reference: (0, 0, ..., 0, 1) in R^(m+1) - augmented space
/// - Use case: Benchmark comparisons, absolute optimization
#[cfg(feature = "python")]
#[pyfunction]
fn calculate_hyperspherical_fitness_hf1_enhanced(
    py: Python,
    objectives: PyReadonlyArray2<f64>,
    decrowding: Option<bool>,
    north_pole_method: Option<&str>,
) -> PyResult<Py<PyArray1<f64>>> {
    let objectives = objectives.as_array();
    let (n_individuals, n_objectives) = objectives.dim();
    let decrowding = decrowding.unwrap_or(false);
    let north_pole_method = north_pole_method.unwrap_or("balanced");

    if n_individuals == 0 {
        return Ok(Array1::zeros(0).into_pyarray(py).to_owned());
    }

    // Validate north pole method
    match north_pole_method {
        "balanced" | "truenorth" => {}
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid north_pole_method: '{}'. Must be 'balanced' or 'truenorth'",
                north_pole_method
            )))
        }
    }

    // Calculate population statistics for decrowding if needed
    let population_stats = if decrowding && n_individuals > 1 {
        // Calculate mean and std dev across all individuals for each objective
        let mut means = Vec::with_capacity(n_objectives);
        let mut std_devs = Vec::with_capacity(n_objectives);

        for obj_idx in 0..n_objectives {
            let obj_column: Vec<f64> = (0..n_individuals)
                .map(|i| objectives[[i, obj_idx]])
                .collect();

            let mean = obj_column.iter().sum::<f64>() / n_individuals as f64;
            let variance = obj_column
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / n_individuals as f64;
            let std_dev = variance.sqrt();

            means.push(mean);
            std_devs.push(std_dev);
        }

        // Use average of all objective means and stds
        let avg_mean = means.iter().sum::<f64>() / n_objectives as f64;
        let avg_std = std_devs.iter().sum::<f64>() / n_objectives as f64;

        Some((avg_mean, avg_std))
    } else {
        None
    };

    // CRITICAL: Column-wise min-max normalization before core calculation
    let normalized_objectives = if n_individuals > 1 {
        // Column-wise min-max normalization
        let mut normalized = objectives.to_owned();
        for j in 0..n_objectives {
            let column: Vec<f64> = (0..n_individuals)
                .map(|i| objectives[[i, j]])
                .collect();

            let col_min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let col_max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let col_range = col_max - col_min;

            // Handle constant columns (avoid division by zero)
            let range = if col_range < f64::EPSILON {
                1.0
            } else {
                col_range
            };

            for i in 0..n_individuals {
                normalized[[i, j]] = (objectives[[i, j]] - col_min) / range;
            }
        }
        normalized
    } else {
        // Single individual - no normalization possible
        objectives.to_owned()
    };

    // Delegate to enhanced core implementation with parallel processing
    let hyperspherical_fitness_values: Vec<f64> = (0..n_individuals)
        .into_par_iter()
        .map(|i| {
            let individual = normalized_objectives.slice(s![i, ..]);
            core_functions::calculate_single_hyperspherical_fitness_f64_with_method(
                &individual.to_owned(),
                n_objectives,
                decrowding,
                population_stats,
                north_pole_method,
            )
        })
        .collect();

    Ok(Array1::from_vec(hyperspherical_fitness_values)
        .into_pyarray(py)
        .to_owned())
}

// =============================================================================
// HIGD PyO3 wrappers
// =============================================================================
// Thin PyO3-compatible shims for the pure-Rust functions in higd.rs.
// These match the signatures expected by demo/nsga3_nsga2balcrowd.py
// (hff_core.calculate_higd / calculate_angular_igd).

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (solutions, n_reference_points, dimensions, seed, positive_orthant=true))]
fn calculate_higd(
    solutions: Vec<Vec<f64>>,
    n_reference_points: usize,
    dimensions: usize,
    seed: u64,
    positive_orthant: bool,
) -> PyResult<f64> {
    Ok(higd::calculate_higd(
        &solutions,
        n_reference_points,
        dimensions,
        seed,
        positive_orthant,
    ))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (solutions, n_reference_points, dimensions, seed, positive_orthant=true))]
fn calculate_angular_igd(
    solutions: Vec<Vec<f64>>,
    n_reference_points: usize,
    dimensions: usize,
    seed: u64,
    positive_orthant: bool,
) -> PyResult<f64> {
    Ok(higd::calculate_angular_igd(
        &solutions,
        n_reference_points,
        dimensions,
        seed,
        positive_orthant,
    ))
}

#[cfg(feature = "python")]
#[pymodule]
fn hff_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core HF1 fitness functions
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_f64, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_enhanced, m)?)?;

    // HIGD (Hyperspherical Inverted Generational Distance) - paper's reference quality indicator
    m.add_function(wrap_pyfunction!(calculate_higd, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_angular_igd, m)?)?;

    Ok(())
}
