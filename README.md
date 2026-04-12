# HFF — Hyperspherical Fitness Functions

A Rust library with Python bindings for many-objective optimization.

HFF projects objective vectors onto a unit hypersphere and uses angular distance
to a reference pole as a scalar fitness measure. This provides a single fitness
value that scales naturally with the number of objectives, eliminating the need
for Pareto dominance in many-objective settings.

## Installation

```bash
pip install hff
```

Or build from source:

```bash
pip install maturin
maturin develop --release
```

### Requirements

- Python >= 3.9
- Rust toolchain (for building from source)
- pymoo (for experiment integration, not required by the library itself)

## Quick start

```python
import numpy as np
import hff

# Random 100-individual, 50-objective problem
objectives = np.random.random((100, 50))

# HF1 Balanced — equal trade-off reference pole
fitness_balanced = hff.calculate_fitness_hf1(objectives, normalize=True)

# HF1 TrueNorth — direct minimization via augmented space
fitness_truenorth = hff.calculate_fitness_hf1_enhanced(
    objectives, normalize=True, north_pole_method="truenorth"
)

# HIGD — CDF-corrected angular IGD (set-level quality metric)
higd_score = hff.calculate_higd(
    objectives.tolist(), n_ref=10000, n_dims=50, seed=42, positive_orthant=True
)
```

## API

| Function | Purpose |
|----------|---------|
| `calculate_fitness_hf1(F)` | HF1 Balanced fitness (angular distance to diagonal pole) |
| `calculate_fitness_hf1_enhanced(F, north_pole_method=)` | HF1 with method selection: `"balanced"` or `"truenorth"` |
| `calculate_higd(solutions, n_ref, n_dims, seed, positive_orthant)` | CDF-corrected angular IGD |
| `calculate_angular_igd(solutions, n_ref, n_dims, seed, positive_orthant)` | Raw angular IGD |

## Citation

If you use HFF in your research, please cite:

```bibtex
@inproceedings{morgan2026hff,
  author    = {Morgan, Andrew},
  title     = {Hyperspherical Fitness Functions for Many-Objective Optimization},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year      = {2026}
}
```

## License

MIT
