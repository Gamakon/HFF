# HFF — Hyperspherical Fitness Functions

A Rust library (with Python bindings via PyO3 and a C ABI) for many-objective
optimisation. HFF projects an objective vector onto a unit hypersphere and uses
**angular distance to a reference pole** as a scalar fitness measure.

The dimensionality of Pareto-dominance front degrades as the number of
objectives grows — at high dimensionality nearly every solution is
non-dominated and selection pressure collapses. HFF replaces the dominance
relation with a single scalar that scales naturally with objective count, and
remains useful at low dimensions (2–3 objectives) as a principled alternative
to weighted sums.

This repository contains the library, two demonstration notebooks (regression
and binary classification), and the as-submitted GECCO 2026 poster.

---

## Repository contents

```
hff/
├── src/                       Rust core (HF1 Balanced/TrueNorth, HIGD)
├── python/hff/                PyO3 module + Python convenience wrappers
├── include/hff.h              C header for the optional c-api feature
├── notebooks/
│   ├── hff_geppy_helpers.py   shared helpers (primitives, LSM, rerankers, HIGD)
│   ├── v1.0.4_Multidemic_SymbolicLinearRegression.ipynb    UCI PowerPlant
│   ├── v1.0.4_Multidemic_SymbolicLogisticReg.ipynb         UCI Heart Disease
│   └── data/                  UCI PowerPlant CSV + dictionary
├── papers/
│   ├── GECCO_..._Poster_SUBMITTED.pdf
│   └── hff-gecco2026-poster_Submitted.tex
├── CLAUDE.md                  notes for AI-assisted contributors
└── README.md                  this file
```

---

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

- Python ≥ 3.9
- Rust toolchain (for building from source)
- `geppy`, `deap`, `multiprocess`, `scikit-learn`, `pandas`, `matplotlib`,
  `seaborn`, `graphviz`, `sympy` — for running the demonstration notebooks
  (not required to use the library itself)

---

## Quick start

```python
import numpy as np
import hff

# Random 100-individual, 50-objective problem
objectives = np.random.random((100, 50))

# HF1 Balanced — equal trade-off reference pole (1/√m, ..., 1/√m)
fitness_balanced = hff.calculate_fitness_hf1(objectives)

# HF1 TrueNorth — direct minimisation via augmented space (0, ..., 0, 1)
fitness_truenorth = hff.calculate_fitness_hf1_enhanced(
    objectives, normalize=True, north_pole_method="truenorth"
)

# HIGD — CDF-corrected angular IGD (set-level quality metric)
higd_score = hff.calculate_higd(
    objectives.tolist(),
    n_reference_points=10000,
    dimensions=50,
    seed=42,
    positive_orthant=True,
)
```

### Choosing `normalize`

| Input style | Setting | Reason |
|-|-|-|
| Unbounded objectives (e.g. MSE, raw cost) | `normalize=True` (default) | HFF rescales each column to [0, 1] before projection. |
| Already-bounded objectives in [0, 1] (e.g. AUC, F1, accuracy) | `normalize=False` | Otherwise the column-best individual is mapped to all-ones and collapses onto the reference pole. |

---

## API

| Function | Purpose |
|---|---|
| `calculate_fitness_hf1(F)` | HF1 Balanced — angular distance to the diagonal pole `(1/√m, …, 1/√m)`. |
| `calculate_fitness_hf1_enhanced(F, normalize=, north_pole_method=)` | HF1 with method selection: `"balanced"` or `"truenorth"`, plus an optional `normalize` flag. |
| `calculate_higd(solutions, n_reference_points, dimensions, seed, positive_orthant)` | Set-level quality indicator. CDF-corrected angular IGD that is dimensionally robust. |
| `calculate_angular_igd(solutions, n_reference_points, dimensions, seed, positive_orthant)` | Raw angular IGD (no CDF correction). |

The same surface is also exposed via a C ABI when built with
`--features c-api`; see `include/hff.h`.

---

## Demonstration notebooks

The two practical takeaways live in `notebooks/`:

| Notebook | Task | Default dataset |
|---|---|---|
| `v1.0.4_Multidemic_SymbolicLinearRegression.ipynb` | Symbolic regression (continuous target) | UCI Combined Cycle Power Plant |
| `v1.0.4_Multidemic_SymbolicLogisticReg.ipynb` | Symbolic logistic regression (binary classification) | UCI Heart Disease (Cleveland) |

Both notebooks use the same template:

1. Load data + dictionary.
2. Three-way **train / validation / holdout** split.
3. Configure geppy GEP-RNC genes (head length, number of genes, linker).
4. Define a multi-objective fitness vector built from train *and* validation
   metrics, project it through `hff.calculate_fitness_hf1_enhanced`, and
   evolve under a multidemic island model.
5. After evolution: sympy simplification + graphviz tree, holdout metrics,
   error histograms, Pareto-marked Hall-of-Fame report, and the set-level
   HIGD diagnostic.

The validation-aware fitness selects directly for *generalisation*: a model
that's good on train but mediocre on validation is penalised as imbalanced
across objectives, so **parsimony emerges without any explicit complexity
constraint**.

To run them:

```bash
maturin develop --release
cd notebooks
jupyter notebook v1.0.4_Multidemic_SymbolicLinearRegression.ipynb
```

---

## Citing this work

Please cite the GECCO 2026 poster when using HFF in published research:

```bibtex
@inproceedings{morgan2026hff,
  author    = {Andrew James Morgan},
  title     = {Hyperspherical Fitness Functions for Many-Objective Optimization},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation
               Conference Companion (GECCO Companion '26)},
  series    = {GECCO Companion '26},
  year      = {2026},
  month     = jul,
  location  = {San Jose, Costa Rica},
  publisher = {ACM},
  address   = {New York, NY, USA},
  isbn      = {979-8-4007-2488-6/2026/07},
  doi       = {} % add when assigned
}
```

The PDF and LaTeX source of the submitted poster are in
[`papers/`](papers/).

### Citing the code repository specifically

If you reference the library implementation (Rust core, Python wrappers, C
ABI, or the demonstration notebooks) rather than the underlying method, also
cite the repository:

```bibtex
@software{morgan2026hff_repo,
  author  = {Andrew James Morgan},
  title   = {{HFF}: Hyperspherical Fitness Functions
             (Rust + Python + C library, demonstration notebooks)},
  year    = {2026},
  url     = {https://github.com/Gamakon/HFF},
  version = {0.1.0}
}
```

### Citing the demonstration notebooks

If your work builds on the symbolic regression / classification templates in
`notebooks/`, please attribute both the method and the templates:

```bibtex
@misc{morgan2026hff_notebooks,
  author = {Andrew James Morgan},
  title  = {Symbolic Regression and Classification with Hyperspherical
            Fitness Functions ({HFF}): geppy demonstration notebooks},
  year   = {2026},
  url    = {https://github.com/Gamakon/HFF/tree/main/notebooks},
  note   = {v1.0.4: Multidemic GEP-RNC with HF1 TrueNorth fitness and
            train/validation/holdout splits.}
}
```

### Datasets used in the demonstration notebooks

The notebooks use publicly available UCI datasets. If you use the notebooks
in published work, also cite the underlying datasets:

```bibtex
@misc{uci_powerplant,
  author       = {Tüfekci, Pınar and Kaya, Heysem},
  title        = {Combined Cycle Power Plant Data Set},
  howpublished = {UCI Machine Learning Repository},
  year         = {2014},
  url          = {https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant}
}

@misc{uci_heart_cleveland,
  author       = {Janosi, Andras and Steinbrunn, William and
                  Pfisterer, Matthias and Detrano, Robert},
  title        = {Heart Disease Data Set (Cleveland)},
  howpublished = {UCI Machine Learning Repository},
  year         = {1988},
  url          = {https://archive.ics.uci.edu/ml/datasets/Heart+Disease}
}
```

### Dependencies worth acknowledging

The notebooks rely heavily on:

- [`geppy`](https://github.com/ShuhuaGao/geppy) — Gene Expression Programming
  on top of DEAP.
- [`DEAP`](https://github.com/DEAP/deap) — Distributed Evolutionary Algorithms
  in Python.
- [`PyO3`](https://github.com/PyO3/pyo3) and
  [`maturin`](https://github.com/PyO3/maturin) — Rust ↔ Python bridge and
  build tooling.

Their respective citations are listed in the linked repositories.

---

## License

MIT.
