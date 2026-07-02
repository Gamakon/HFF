# Hyperspherical Fitness: A Unified Framework for Multi-Objective Algorithm Comparison

## Abstract

We present Hyperspherical Fitness (HF1), a novel approach for comparing multi-objective optimization algorithms when absolute objective minima are unknown. By projecting objective vectors onto a unit hypersphere and measuring angular distances, HF1 provides a scale-invariant, unified ranking system that enables fair comparison across different algorithms. We demonstrate how HF1 can be used both as an optimization operator and as a post-hoc analysis tool for benchmarking algorithm performance.

## 1. Introduction

Multi-objective optimization benchmarking faces a fundamental challenge: how do we compare algorithms when the true Pareto front is unknown? Traditional metrics like hypervolume and IGD require reference points or true Pareto sets that may not be available for novel problems. We propose Hyperspherical Fitness (HF1) as a solution that provides:

1. **Scale invariance**: Objectives with different ranges are normalized naturally
2. **Unified comparison**: All solutions mapped to the same geometric space
3. **Interpretable metric**: Angular distance has clear geometric meaning
4. **No reference required**: Works without knowing the true optima

## 2. Mathematical Foundation

### 2.1 Hyperspherical Projection

Given an objective vector $\mathbf{f} = (f_1, f_2, ..., f_m) \in \mathbb{R}^m$ for $m$ objectives, we first apply column-wise normalization across the population to ensure comparable scales:

$$\tilde{f}_{ij} = \frac{f_{ij} - \min_k f_{kj}}{\max_k f_{kj} - \min_k f_{kj}}$$

where $i$ indexes individuals and $j$ indexes objectives.

### 2.2 Fractional Energy Decomposition

For each normalized objective vector $\tilde{\mathbf{f}}$, we compute the total energy:

$$E = \sum_{j=1}^{m} \tilde{f}_j^2$$

The fractional energy contribution of each objective is:

$$\phi_j = \frac{\tilde{f}_j^2}{E}$$

Note that $\sum_{j=1}^m \phi_j = 1$, ensuring the values lie on a simplex.

### 2.3 Hyperspherical Mapping

We project each solution onto the unit hypersphere using:

$$y_j = \text{sgn}(\tilde{f}_j) \sqrt{\phi_j}$$

This preserves the sign information while mapping to the hypersphere. The resulting vector $\mathbf{y} = (y_1, y_2, ..., y_m)$ satisfies:

$$\|\mathbf{y}\|_2 = \sqrt{\sum_{j=1}^m y_j^2} = \sqrt{\sum_{j=1}^m \phi_j} = 1$$

### 2.4 Angular Distance Metric and the North Pole

The fitness of a solution is defined as its angular distance to the ideal point (north pole). For minimization problems, the north pole represents the point where all objectives reach their minimum (zero after normalization).

**Critical clarification**: The north pole is **NOT** a balanced point across objectives. For minimization problems, it represents the ideal state where all fractional energies approach zero. To avoid the singularity of a zero vector, we augment with an additional dimension:

$$\mathbf{n} = (0, 0, ..., 0, 1) \in \mathbb{R}^{m+1}$$

where the first $m$ components are zeros (representing perfect minimization of all objectives) and the final component is 1 to ensure the vector lies on the unit hypersphere.

The angular distance is then:

$$\theta = \arccos\left(\mathbf{y}_{aug} \cdot \mathbf{n}\right)$$

where $\mathbf{y}_{aug} = (y_1, y_2, ..., y_m, 0)$ is the augmented solution vector.

**Key insight**: When all objectives are minimized toward zero, their fractional energies approach zero, placing the solution near the north pole. This is fundamentally different from a "balanced" point - it represents the optimal state where all objectives achieve their best possible values.

## 3. The HF1 Algorithm

### 3.1 Pseudocode

```
Algorithm: Hyperspherical Fitness (HF1)
Input: Population objectives F ∈ ℝ^(n×m) (n individuals, m objectives)
Output: Angular fitness scores θ ∈ ℝ^n

1: // Column-wise normalization
2: for j = 1 to m do
3:     f_min[j] ← min(F[:, j])
4:     f_max[j] ← max(F[:, j])
5:     F_norm[:, j] ← (F[:, j] - f_min[j]) / (f_max[j] - f_min[j])
6: end for

7: // Initialize north pole reference (zeros with augmented dimension)
8: n ← (0, 0, ..., 0, 1) ∈ ℝ^(m+1)

9: // Process each individual
10: for i = 1 to n do
11:     // Compute total energy
12:     E ← sum(F_norm[i, :]²)
13:     
14:     // Compute fractional energies
15:     for j = 1 to m do
16:         φ[j] ← F_norm[i, j]² / E
17:     end for
18:     
19:     // Project to hypersphere
20:     for j = 1 to m do
21:         y[j] ← sign(F_norm[i, j]) × √φ[j]
22:     end for
23:     
24:     // Augment solution vector
25:     y_aug ← (y[1], y[2], ..., y[m], 0)
26:     
27:     // Calculate angular distance
28:     dot_product ← sum(y_aug × n)
29:     θ[i] ← arccos(clamp(dot_product, -1, 1))
30: end for

31: return θ
```

### 3.2 Rust Implementation

Our high-performance Rust implementation leverages SIMD operations and parallel processing:

```rust
pub fn calculate_hyperspherical_fitness_hf1(
    objectives: &Array2<f64>
) -> Array1<f64> {
    let (n_individuals, n_objectives) = objectives.dim();
    
    // Column-wise normalization
    let normalized = column_normalize(objectives);
    
    // North pole reference vector (zeros with augmented 1)
    let mut north_pole = Array1::zeros(n_objectives + 1);
    north_pole[n_objectives] = 1.0;
    
    // Process individuals in parallel
    let fitness_values: Vec<f64> = (0..n_individuals)
        .into_par_iter()
        .map(|i| {
            let individual = normalized.row(i);
            
            // Compute total energy
            let total_energy: f64 = individual
                .iter()
                .map(|&x| x * x)
                .sum();
            
            // Project to hypersphere
            let mut y_aug = vec![0.0; n_objectives + 1];
            for (j, &x) in individual.iter().enumerate() {
                y_aug[j] = x.signum() * (x * x / total_energy).sqrt();
            }
            // Last component remains 0 for solution vector
            
            // Angular distance to north pole
            let dot_product: f64 = y_aug.iter()
                .zip(north_pole.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            dot_product.clamp(-1.0, 1.0).acos()
        })
        .collect();
    
    Array1::from_vec(fitness_values)
}
```

## 4. HF1 as a Survival Operator

When used within an evolutionary algorithm, HF1 serves as a selection mechanism:

```python
class HypersphericalFitnessSurvival(Survival):
    def _do(self, problem, pop, n_survive):
        # Extract objectives
        F = pop.get("F")
        
        # Calculate HF1 scores
        hf1_scores = calculate_hyperspherical_fitness_hf1(F)
        
        # Select best individuals (lowest angular distance)
        selected_indices = np.argsort(hf1_scores)[:n_survive]
        
        return pop[selected_indices]
```

## 5. HF1 as a Unified Ranking Metric

### 5.1 The Challenge of Algorithm Comparison

When comparing multiple algorithms (e.g., NSGA-II, NSGA-III, HF1-based), each may produce different solution sets with varying characteristics. Without knowing the true Pareto front, how do we determine which algorithm performs better?

### 5.2 Post-Hoc Unified Ranking

We propose using HF1 as a post-processing step to create a unified ranking across all algorithms:

1. **Collect all solutions**: Gather final populations from all runs of all algorithms
2. **Global normalization**: Apply normalization using the entire solution set
3. **Hyperspherical projection**: Map all solutions to the unit hypersphere
4. **Unified ranking**: Sort by angular distance to create a global ranking

### 5.3 Mathematical Justification

This approach is justified because:

1. **Common space**: All solutions are mapped to the same geometric space (unit hypersphere)
2. **Scale invariance**: The fractional energy formulation handles objectives with different scales
3. **Relative comparison**: We measure relative performance without needing absolute optima
4. **Statistical validity**: Large sample sizes (31 runs × multiple algorithms) ensure robust statistics

### 5.4 Analysis Framework

Given solutions from algorithms $A_1, A_2, ..., A_k$, we compute:

$$\mathcal{R} = \text{sort}(\{(\theta_i, a_i, r_i) : i \in \text{all solutions}\})$$

where $\theta_i$ is the HF1 score, $a_i$ is the algorithm identifier, and $r_i$ is the run number.

From this unified ranking $\mathcal{R}$, we can compute:

1. **Top-k performance**: Proportion of each algorithm in the top k% of solutions
2. **Mean rank**: Average ranking position for each algorithm
3. **Rank distribution**: Statistical properties of rank distributions
4. **Dominance probability**: Likelihood that algorithm $A_i$ outranks $A_j$

## 6. Statistical Analysis

### 6.1 Rank-Based Comparisons

For algorithms $A_i$ and $A_j$, we define the dominance probability:

$$P(A_i \succ A_j) = \frac{|\{(s_i, s_j) : \text{rank}(s_i) < \text{rank}(s_j), s_i \in A_i, s_j \in A_j\}|}{|A_i| \times |A_j|}$$

### 6.2 Distribution Analysis

The empirical cumulative distribution function for algorithm $A_i$:

$$F_{A_i}(r) = \frac{|\{s \in A_i : \text{rank}(s) \leq r\}|}{|A_i|}$$

This allows us to compute percentile-based metrics and confidence intervals.

### 6.3 Statistical Significance

We employ non-parametric tests suitable for rank data:

1. **Kruskal-Wallis test**: Overall difference between algorithms
2. **Mann-Whitney U test**: Pairwise comparisons
3. **Friedman test**: When solutions are paired by problem instance

## 7. Implementation Example

```python
def unified_hyperspherical_ranking(results_dict):
    """
    Create unified ranking across all algorithms
    
    Args:
        results_dict: {algorithm: [solutions]}
        
    Returns:
        Unified ranking with statistics
    """
    # Collect all solutions
    all_solutions = []
    for algorithm, solutions in results_dict.items():
        for run_idx, solution_set in enumerate(solutions):
            for sol in solution_set:
                all_solutions.append({
                    'objectives': sol,
                    'algorithm': algorithm,
                    'run': run_idx
                })
    
    # Extract objective matrix
    F = np.array([s['objectives'] for s in all_solutions])
    
    # Apply HF1 to entire population
    hf1_scores = calculate_hyperspherical_fitness_hf1(F)
    
    # Create unified ranking
    for i, score in enumerate(hf1_scores):
        all_solutions[i]['hf1_score'] = score
    
    # Sort by HF1 score
    ranked_solutions = sorted(all_solutions, 
                            key=lambda x: x['hf1_score'])
    
    # Assign ranks
    for rank, sol in enumerate(ranked_solutions):
        sol['rank'] = rank + 1
    
    return analyze_rankings(ranked_solutions)
```

## 8. Important Note on North Pole Interpretation

It is crucial to understand that the north pole in this framework is **not** about balancing objectives or finding a compromise solution. Instead:

- **North pole = (0, 0, ..., 0, 1)**: Represents the ideal state where all objectives are minimized to their optimal values
- **The zeros**: Indicate that all fractional energies approach zero (perfect minimization)
- **The augmented 1**: Avoids the mathematical singularity of a zero vector while maintaining the geometric interpretation
- **Angular distance**: Measures how far a solution is from achieving perfect minimization across all objectives

This is fundamentally different from approaches that seek to balance trade-offs between objectives. HF1 identifies solutions that come closest to optimizing all objectives simultaneously.

## 9. Advantages of This Approach

1. **No ground truth required**: Works without knowing true Pareto front
2. **Fair comparison**: All algorithms evaluated in the same space
3. **Interpretable**: Angular distance has clear geometric meaning
4. **Robust**: Large sample sizes provide statistical power
5. **Generalizable**: Works for any number of objectives
6. **Handles minimization naturally**: North pole correctly represents optimal minimization

## 10. Conclusion

Hyperspherical Fitness provides both an effective optimization operator and a principled framework for comparing multi-objective algorithms. By mapping all solutions to a unit hypersphere and measuring angular distances to a north pole that represents perfect minimization (zeros), we create a unified comparison space that is scale-invariant and geometrically interpretable. This approach is particularly valuable for benchmarking on novel problems where true optima are unknown, providing researchers with a fair and robust comparison methodology.

## References

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.

2. Emmerich, M. T., & Deutz, A. H. (2018). A tutorial on multiobjective optimization: fundamentals and evolutionary methods. Natural computing, 17(3), 585-609.

3. Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015). Modified distance calculation in generational distance and inverted generational distance. In International conference on evolutionary multi-criterion optimization (pp. 110-125).

---

*Note: This framework enables fair comparison of optimization algorithms without requiring knowledge of true optima, making it ideal for real-world problems and novel benchmarks where ground truth is unavailable.*
