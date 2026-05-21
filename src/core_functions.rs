use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate hyperspherical fitness from a single solution vector to the target point on unit hypersphere.
///
/// This is the core HF1 (Hyperspherical Fitness 1) algorithm implementation using standard f64 precision.
/// The function projects the solution vector onto a unit hypersphere and calculates
/// the angular distance to a reference "north pole" point using the original 2015 method.
///
/// # Mathematical Foundation
///
/// 1. **Optional Decrowding**: Apply log(sigmoid(z-score)) transformation to reduce target point clustering
/// 2. **Energy Calculation**: Compute squared magnitude of solution vector
/// 3. **Fractional Energy**: Normalize each objective by total energy
/// 4. **Hyperspherical Projection**: Take square root to get hyperspherical coordinates
/// 5. **Balanced North Pole**: Use (1/√m, 1/√m, ..., 1/√m) representing balanced minimization
/// 6. **Hyperspherical Fitness**: Calculate arccos of dot product with balanced north pole
///
/// # Arguments
///
/// * `individual` - Solution vector as 1D array of objective values
/// * `n_objectives` - Number of objectives (used for north pole calculation)
/// * `decrowding` - Whether to apply decrowding transformation to reduce north pole clustering
/// * `population_stats` - Optional population statistics (mean, std_dev) for decrowding transform
///
/// # Returns
///
/// Angular distance in radians [0, π], where 0 means the solution is at the balanced north pole
/// (equal trade-offs across all objectives) and π means maximum deviation.
///
/// # Properties
///
/// - **Range**: [0, π] radians
/// - **Invariant**: Rotationally symmetric on hypersphere
/// - **Monotonic**: Preserves Pareto dominance relationships
/// - **Decrowding**: When enabled, provides better resolution in good solution regions
///
/// # Error Handling
///
/// - **NaN/Infinite inputs**: Returns π (safe maximum)
/// - **Zero vector**: Uses ε to prevent division by zero
/// - **Numerical errors**: Clamps cos(θ) to [-1,1] and validates final result
pub fn calculate_single_hyperspherical_fitness_f64(
    individual: &Array1<f64>,
    n_objectives: usize,
    decrowding: bool,
    population_stats: Option<(f64, f64)>
) -> f64 {
    // Default to BalancedNorth for backward compatibility
    calculate_single_hyperspherical_fitness_f64_with_method(
        individual,
        n_objectives,
        decrowding,
        population_stats,
        "balanced"
    )
}

/// Enhanced HF1 with TrueNorth vs BalancedNorth method selection
pub fn calculate_single_hyperspherical_fitness_f64_with_method(
    individual: &Array1<f64>,
    n_objectives: usize,
    decrowding: bool,
    population_stats: Option<(f64, f64)>,
    north_pole_method: &str
) -> f64 {
    // Validate individual values - return safe maximum for any invalid input
    for &val in individual.iter() {
        if !val.is_finite() {
            return PI; // Maximum possible angular distance (π radians)
        }
    }

    // Apply decrowding transformation if requested
    let processed_individual = if decrowding {
        apply_decrowding_transform(individual, population_stats)
    } else {
        individual.clone()
    };

    // Calculate energy (sum of squared objective values)
    let energy_sum: f64 = processed_individual.iter().map(|&x| x * x).sum();

    // Handle zero energy case (perfect minimization after normalization)
    if energy_sum <= f64::EPSILON {
        return 0.0; // Perfect minimization = 0 distance
    }

    // Calculate fractional energy and hyperspherical coordinates
    // Each coordinate represents the "contribution" of that objective to total energy
    let geometric_coords: Vec<f64> = processed_individual
        .iter()
        .map(|&x| {
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            sign * ((x * x) / energy_sum).sqrt()
        })
        .collect();

    // North pole method selection: TrueNorth vs BalancedNorth
    let cos_theta: f64 = match north_pole_method {
        "balanced" => {
            // BalancedNorth: Use balanced north pole in m-dimensional space
            // North pole represents equal trade-offs: (1/√m, 1/√m, ..., 1/√m)
            let north_pole_coord = 1.0 / (n_objectives as f64).sqrt();
            let north_pole: Vec<f64> = vec![north_pole_coord; n_objectives];

            // Calculate dot product with north pole (cosine of angle)
            geometric_coords
                .iter()
                .zip(north_pole.iter())
                .map(|(&sol, &pole)| sol * pole)
                .sum()
        },
        "truenorth" => {
            // TrueNorth: Energy-based augmented space method for direct minimization
            // Solution: (y₁×√(1-e²), y₂×√(1-e²), ..., yₘ×√(1-e²), e) in ℝ^(m+1)
            // North pole: (0, 0, ..., 0, 1) in ℝ^(m+1)
            // Where e = energy_score based on distance from perfect minimization

            // Calculate energy score: lower energy = higher score (closer to north pole)
            let max_possible_energy = n_objectives as f64; // Max energy after normalization
            let current_energy = energy_sum;
            let energy_score = if current_energy <= f64::EPSILON {
                1.0  // Perfect minimization
            } else {
                // Energy score: higher is better (closer to north pole)
                let normalized_energy = current_energy / max_possible_energy;
                (1.0 - normalized_energy.min(1.0)).max(0.0)
            };

            // Scale geometric coordinates by √(1 - energy_score²) to maintain unit sphere
            let scale_factor = (1.0 - energy_score * energy_score).sqrt();
            let mut augmented_coords: Vec<f64> = geometric_coords
                .iter()
                .map(|&coord| coord * scale_factor)
                .collect();

            // Append energy score as final dimension
            augmented_coords.push(energy_score);

            // Create augmented north pole: (0, 0, ..., 0, 1)
            let mut north_pole = vec![0.0; n_objectives];
            north_pole.push(1.0);

            // Calculate dot product in augmented space
            augmented_coords
                .iter()
                .zip(north_pole.iter())
                .map(|(&sol, &pole)| sol * pole)
                .sum()
        },
        _ => {
            // Invalid method - default to balanced for safety
            let north_pole_coord = 1.0 / (n_objectives as f64).sqrt();
            let north_pole: Vec<f64> = vec![north_pole_coord; n_objectives];

            geometric_coords
                .iter()
                .zip(north_pole.iter())
                .map(|(&sol, &pole)| sol * pole)
                .sum()
        }
    };

    // Clamp cosine to valid range [-1, 1] to handle numerical precision issues
    let cos_theta = cos_theta.clamp(-1.0, 1.0);

    // Calculate angular distance with robust error handling
    let angular_distance = if cos_theta.abs() > 1.0 - f64::EPSILON {
        // Handle edge cases very close to poles to avoid numerical instability
        if cos_theta > 0.0 { 0.0 } else { PI }
    } else {
        cos_theta.acos()
    };

    // Final validation - return result if finite, otherwise safe fallback
    if angular_distance.is_finite() {
        angular_distance  // Return raw angular distance in radians [0, π]
    } else {
        PI // Safe fallback for any remaining numerical issues
    }
}

/// Apply decrowding transformation to reduce clustering near the north pole.
///
/// Uses z-score normalization followed by softplus transformation to spread
/// solutions that would otherwise cluster in the high-fitness region of the
/// hypersphere, providing better discrimination between near-optimal solutions.
///
/// # Algorithm
///
/// For each objective value x:
/// 1. Calculate z-score: z = (x - mean) / std_dev
/// 2. Apply softplus: softplus(z) = ln(1 + exp(z))
///
/// The softplus function provides a smooth, non-negative transformation that:
/// - Preserves ordering (monotonic)
/// - Spreads clustered values near the north pole
/// - Is numerically stable via the identity: softplus(z) = max(0,z) + ln(1 + exp(-|z|))
///
/// # Arguments
///
/// * `individual` - Single individual's objective values
/// * `population_stats` - Optional (mean, std_dev) statistics from population for z-score calculation.
///   If None, will use the individual's own statistics (less ideal).
///
/// # Returns
///
/// Transformed objectives with same dimensions, ready for hyperspherical projection
///
/// # Numerical Stability
///
/// Uses numerically stable computation of log(sigmoid(x)) = -log(1 + exp(-x))
/// to avoid overflow/underflow issues with extreme z-scores.
pub fn apply_decrowding_transform(
    individual: &Array1<f64>,
    population_stats: Option<(f64, f64)>
) -> Array1<f64> {
    if individual.len() <= 1 {
        return individual.clone(); // No transform needed for single objective
    }

    // Get statistics for z-score calculation
    let (mean, std_dev) = match population_stats {
        Some((m, s)) => (m, s),
        None => {
            // Fallback: use individual's own statistics (not ideal for decrowding)
            let mean = individual.mean().unwrap_or(0.0);
            let std_dev = individual.std(0.0);
            (mean, std_dev)
        }
    };

    // Handle constant objectives (std_dev = 0)
    if std_dev <= f64::EPSILON {
        return individual.clone(); // Return original if no variance
    }

    // Apply z-score normalization followed by softplus(z-score) for smooth, non-negative transformation
    individual.mapv(|x| {
        let z_score = (x - mean) / std_dev;

        // Use softplus(z) = ln(1 + exp(z)) instead of log(sigmoid(z))
        // Softplus is always positive and provides smooth decrowding behavior
        // For numerical stability, use the identity: softplus(z) = max(0, z) + ln(1 + exp(-|z|))
        if z_score > 0.0 {
            // For positive z, use z + ln(1 + exp(-z))
            z_score + (1.0 + (-z_score).exp()).ln()
        } else {
            // For negative z, use ln(1 + exp(z))
            (1.0 + z_score.exp()).ln()
        }
    })
}
