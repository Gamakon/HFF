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

/// GPU-accelerated batch HFF via wgpu compute shader.
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
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

/// Validate the north pole method string; return Err message on unknown method.
#[cfg(feature = "python")]
fn validate_north_pole_method(name: &str) -> Result<(), String> {
    match name {
        "balanced" | "truenorth" => Ok(()),
        _ => Err(format!(
            "Invalid north_pole_method: '{}'. Must be 'balanced' or 'truenorth'",
            name
        )),
    }
}

/// Compute per-column min and max across `objectives` (shape n x m).
/// Returns (col_min, col_max), each length m.
#[cfg(feature = "python")]
fn compute_col_ranges(objectives: &ndarray::ArrayView2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (n_individuals, n_objectives) = objectives.dim();
    let mut col_min = vec![f64::INFINITY; n_objectives];
    let mut col_max = vec![f64::NEG_INFINITY; n_objectives];
    for i in 0..n_individuals {
        for j in 0..n_objectives {
            let v = objectives[[i, j]];
            if v < col_min[j] {
                col_min[j] = v;
            }
            if v > col_max[j] {
                col_max[j] = v;
            }
        }
    }
    (col_min, col_max)
}

/// Apply column-wise min-max normalisation using the supplied ranges.
/// Constant columns (range < EPSILON) are scaled with range=1.0, leaving them
/// effectively at zero distance from col_min — the value is then `value - col_min`.
#[cfg(feature = "python")]
fn apply_minmax(
    objectives: &ndarray::ArrayView2<f64>,
    col_min: &[f64],
    col_max: &[f64],
) -> ndarray::Array2<f64> {
    let (n_individuals, n_objectives) = objectives.dim();
    let mut normalized = objectives.to_owned();
    for j in 0..n_objectives {
        let col_range = col_max[j] - col_min[j];
        let range = if col_range.abs() < f64::EPSILON {
            1.0
        } else {
            col_range
        };
        for i in 0..n_individuals {
            normalized[[i, j]] = (objectives[[i, j]] - col_min[j]) / range;
        }
    }
    normalized
}

/// Compute decrowding population stats (avg mean, avg std) — needed when
/// `decrowding=true` and population has more than one individual.
#[cfg(feature = "python")]
fn compute_decrowding_stats(objectives: &ndarray::ArrayView2<f64>) -> (f64, f64) {
    let (n_individuals, n_objectives) = objectives.dim();
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
        means.push(mean);
        std_devs.push(variance.sqrt());
    }
    let avg_mean = means.iter().sum::<f64>() / n_objectives as f64;
    let avg_std = std_devs.iter().sum::<f64>() / n_objectives as f64;
    (avg_mean, avg_std)
}

/// Score the normalised objective matrix row-wise via the core HF1 function.
#[cfg(feature = "python")]
fn score_normalised(
    normalised: &ndarray::Array2<f64>,
    decrowding: bool,
    population_stats: Option<(f64, f64)>,
    north_pole_method: &str,
) -> Vec<f64> {
    let (n_individuals, n_objectives) = normalised.dim();
    (0..n_individuals)
        .into_par_iter()
        .map(|i| {
            let individual = normalised.slice(s![i, ..]);
            core_functions::calculate_single_hyperspherical_fitness_f64_with_method(
                &individual.to_owned(),
                n_objectives,
                decrowding,
                population_stats,
                north_pole_method,
            )
        })
        .collect()
}

/// Enhanced HF1 with TrueNorth vs BalancedNorth method selection and
/// optional per-batch column-wise min-max normalisation.
///
/// * `normalize=true` (default): per-batch min/max — backwards compatible.
/// * `normalize=false`: caller has pre-normalised, or objectives are
///   already bounded (e.g. classification metrics in [0,1]).
///
/// For evolutionary loops where the HFF pole must stay stable across
/// generations, use `calculate_hyperspherical_fitness_hf1_with_ranges`
/// on gen 0 to capture (col_min, col_max), then
/// `calculate_hyperspherical_fitness_hf1_fixed` on subsequent gens.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (objectives, decrowding=None, north_pole_method=None, normalize=None))]
fn calculate_hyperspherical_fitness_hf1_enhanced(
    py: Python,
    objectives: PyReadonlyArray2<f64>,
    decrowding: Option<bool>,
    north_pole_method: Option<&str>,
    normalize: Option<bool>,
) -> PyResult<Py<PyArray1<f64>>> {
    let objectives = objectives.as_array();
    let (n_individuals, _n_objectives) = objectives.dim();
    let decrowding = decrowding.unwrap_or(false);
    let north_pole_method = north_pole_method.unwrap_or("balanced");
    let normalize = normalize.unwrap_or(true);

    if n_individuals == 0 {
        return Ok(Array1::zeros(0).into_pyarray(py).to_owned());
    }
    validate_north_pole_method(north_pole_method).map_err(PyValueError::new_err)?;

    let population_stats = if decrowding && n_individuals > 1 {
        Some(compute_decrowding_stats(&objectives))
    } else {
        None
    };

    let normalised = if normalize && n_individuals > 1 {
        let (col_min, col_max) = compute_col_ranges(&objectives);
        apply_minmax(&objectives, &col_min, &col_max)
    } else {
        objectives.to_owned()
    };

    let fitness = score_normalised(
        &normalised,
        decrowding,
        population_stats,
        north_pole_method,
    );

    Ok(Array1::from_vec(fitness).into_pyarray(py).to_owned())
}

/// Same as `calculate_hyperspherical_fitness_hf1_enhanced` but also returns the
/// per-column (min, max) ranges actually used for normalisation. Use this on
/// generation 0 to freeze the scale; subsequent generations should call
/// `calculate_hyperspherical_fitness_hf1_fixed` with those ranges so the HFF
/// pole stays geometrically meaningful as the population improves.
///
/// Returns `(fitness, col_min, col_max)`. When `normalize=false` or there is a
/// single individual, the returned ranges are the raw per-column min/max of
/// the input matrix.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (objectives, decrowding=None, north_pole_method=None, normalize=None))]
fn calculate_hyperspherical_fitness_hf1_with_ranges(
    py: Python,
    objectives: PyReadonlyArray2<f64>,
    decrowding: Option<bool>,
    north_pole_method: Option<&str>,
    normalize: Option<bool>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let objectives = objectives.as_array();
    let (n_individuals, n_objectives) = objectives.dim();
    let decrowding = decrowding.unwrap_or(false);
    let north_pole_method = north_pole_method.unwrap_or("balanced");
    let normalize = normalize.unwrap_or(true);

    if n_individuals == 0 {
        let zero_fit = Array1::<f64>::zeros(0).into_pyarray(py).to_owned();
        let zero_min = Array1::<f64>::zeros(n_objectives).into_pyarray(py).to_owned();
        let zero_max = Array1::<f64>::zeros(n_objectives).into_pyarray(py).to_owned();
        return Ok((zero_fit, zero_min, zero_max));
    }
    validate_north_pole_method(north_pole_method).map_err(PyValueError::new_err)?;

    let population_stats = if decrowding && n_individuals > 1 {
        Some(compute_decrowding_stats(&objectives))
    } else {
        None
    };

    let (col_min, col_max) = compute_col_ranges(&objectives);
    let normalised = if normalize && n_individuals > 1 {
        apply_minmax(&objectives, &col_min, &col_max)
    } else {
        objectives.to_owned()
    };

    let fitness = score_normalised(
        &normalised,
        decrowding,
        population_stats,
        north_pole_method,
    );

    Ok((
        Array1::from_vec(fitness).into_pyarray(py).to_owned(),
        Array1::from_vec(col_min).into_pyarray(py).to_owned(),
        Array1::from_vec(col_max).into_pyarray(py).to_owned(),
    ))
}

/// Score an objective matrix using caller-supplied per-column ranges. Use
/// after generation 0 so the HFF pole is anchored on the same scale as the
/// initial population — solutions that genuinely improve over time can then
/// approach distance zero meaningfully.
///
/// `col_min` and `col_max` must each be length `n_objectives`. Both must be
/// supplied together; mismatched lengths raise ValueError.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (objectives, col_min, col_max, decrowding=None, north_pole_method=None))]
fn calculate_hyperspherical_fitness_hf1_fixed(
    py: Python,
    objectives: PyReadonlyArray2<f64>,
    col_min: PyReadonlyArray1<f64>,
    col_max: PyReadonlyArray1<f64>,
    decrowding: Option<bool>,
    north_pole_method: Option<&str>,
) -> PyResult<Py<PyArray1<f64>>> {
    let objectives = objectives.as_array();
    let col_min = col_min.as_array();
    let col_max = col_max.as_array();
    let (n_individuals, n_objectives) = objectives.dim();
    let decrowding = decrowding.unwrap_or(false);
    let north_pole_method = north_pole_method.unwrap_or("balanced");

    if n_individuals == 0 {
        return Ok(Array1::zeros(0).into_pyarray(py).to_owned());
    }
    validate_north_pole_method(north_pole_method).map_err(PyValueError::new_err)?;

    if col_min.len() != n_objectives || col_max.len() != n_objectives {
        return Err(PyValueError::new_err(format!(
            "col_min and col_max must have length n_objectives={}, got col_min={}, col_max={}",
            n_objectives,
            col_min.len(),
            col_max.len()
        )));
    }

    let population_stats = if decrowding && n_individuals > 1 {
        Some(compute_decrowding_stats(&objectives))
    } else {
        None
    };

    let col_min_vec: Vec<f64> = col_min.iter().copied().collect();
    let col_max_vec: Vec<f64> = col_max.iter().copied().collect();
    let normalised = apply_minmax(&objectives, &col_min_vec, &col_max_vec);

    let fitness = score_normalised(
        &normalised,
        decrowding,
        population_stats,
        north_pole_method,
    );

    Ok(Array1::from_vec(fitness).into_pyarray(py).to_owned())
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

// =============================================================================
// GPU entry point (PyO3, feature = "gpu")
// =============================================================================
// Same API shape as `calculate_hyperspherical_fitness_hf1_enhanced` but runs
// on a wgpu compute pipeline. north_pole_method="truenorth" only for now.
// A single GPU context is lazy-initialized on first call.

#[cfg(all(feature = "python", feature = "gpu"))]
use std::sync::OnceLock;

#[cfg(all(feature = "python", feature = "gpu"))]
static HFF_GPU_CTX: OnceLock<Result<gpu::HffGpuContext, String>> = OnceLock::new();

#[cfg(all(feature = "python", feature = "gpu"))]
#[pyfunction]
#[pyo3(signature = (objectives, north_pole_method="truenorth", normalize=true))]
fn calculate_hyperspherical_fitness_hf1_enhanced_gpu(
    py: Python,
    objectives: PyReadonlyArray2<f64>,
    north_pole_method: &str,
    normalize: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    if north_pole_method != "truenorth" {
        return Err(PyValueError::new_err(
            "GPU path supports only north_pole_method='truenorth' for now",
        ));
    }
    let objectives = objectives.as_array().to_owned();
    let ctx = HFF_GPU_CTX.get_or_init(gpu::HffGpuContext::new);
    let ctx = match ctx {
        Ok(c) => c,
        Err(e) => return Err(PyValueError::new_err(format!("GPU init failed: {e}"))),
    };
    let out = ctx
        .calculate_hf1_truenorth_batch(&objectives, normalize)
        .map_err(|e| PyValueError::new_err(format!("GPU compute failed: {e}")))?;
    Ok(Array1::from(out).into_pyarray(py).to_owned())
}

#[cfg(feature = "python")]
#[pymodule]
fn hff_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core HF1 fitness functions
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_f64, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_enhanced, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_with_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_fixed, m)?)?;

    // HIGD (Hyperspherical Inverted Generational Distance) - paper's reference quality indicator
    m.add_function(wrap_pyfunction!(calculate_higd, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_angular_igd, m)?)?;

    #[cfg(feature = "gpu")]
    m.add_function(wrap_pyfunction!(calculate_hyperspherical_fitness_hf1_enhanced_gpu, m)?)?;

    Ok(())
}
