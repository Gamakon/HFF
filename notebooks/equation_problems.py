"""Problem registry for the v1.0.4 SymbolicEquationRecovery notebook.

Each entry encodes a known equation, the input variables and their ranges
(separate ranges for train and extrapolation), default sample sizes,
optional Gaussian noise, and the sympy ground-truth expression used for
post-hoc recovery scoring.

Data is generated on demand and cached under ``data/equations/<id>/``
with a manifest. Re-running with the same config hits the cache; changing
any input invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import sympy as sp


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

@dataclass
class EquationProblem:
    """A known equation to attempt to recover symbolically."""
    name: str                                  # short id used in cache paths
    description: str                           # human-readable label
    variables: list[str]                       # input variable names, in order
    train_ranges: dict[str, tuple[float, float]]   # uniform ranges for train + val
    extrap_ranges: dict[str, tuple[float, float]]  # ranges for the extrapolation slice
    callable: Callable                         # numpy-vectorised: y = f(**inputs)
    truth_expr: str                            # sympy-parseable string
    constants_used: list[str]                  # names matching the snap library
    n_train: int = 800
    n_val: int = 200
    n_holdout: int = 200
    n_extrap: int = 200
    noise_std: float = 0.0
    seed: int = 0


# Global library of constants the snap step can pull from. Notebooks may
# extend this in-place for BYO problems.
#
# Composite entries (2π, 4π², 2π/√g, …) are explicitly listed because the
# snap's candidate-form generator only handles ±c, ±1/c, ±c², ±√c and
# shallow rationals × c — it does NOT search products of two library
# constants. Adding the most useful composites by hand keeps that
# combinatorial explosion under control.
KNOWN_CONSTANTS = {
    "pi":           sp.pi,
    "E":            sp.E,
    "sqrt2":        sp.sqrt(2),
    "G":            6.6743e-11,            # gravitational constant
    "k_e":          8.9875517923e9,        # Coulomb's constant
    "c_light":      299792458.0,           # speed of light
    "h":            6.62607015e-34,        # Planck's constant
    "k_B":          1.380649e-23,          # Boltzmann
    "R":            8.314462618,           # ideal gas constant
    "g":            9.80665,               # standard gravity

    # Composite constants commonly appearing in physics laws:
    "2pi":          2 * sp.pi,
    "4pi":          4 * sp.pi,
    "4pi_sq":       4 * sp.pi ** 2,
    "2pi_over_sqg": 2 * sp.pi / sp.sqrt(sp.Float(9.80665)),
}


REGISTRY: dict[str, EquationProblem] = {

    "circle_area": EquationProblem(
        name="circle_area",
        description="Area of a circle:  A = π·r²",
        variables=["r"],
        train_ranges={"r": (0.1, 5.0)},
        extrap_ranges={"r": (5.0, 10.0)},
        callable=lambda r: math.pi * r * r,
        truth_expr="pi * r**2",
        constants_used=["pi"],
    ),

    "gravity": EquationProblem(
        name="gravity",
        description="Newton's gravitation:  F = G·m1·m2 / r²",
        variables=["m1", "m2", "r"],
        train_ranges={"m1": (1.0, 100.0), "m2": (1.0, 100.0), "r": (0.5, 5.0)},
        extrap_ranges={"m1": (1.0, 100.0), "m2": (1.0, 100.0), "r": (5.0, 10.0)},
        callable=lambda m1, m2, r: 6.6743e-11 * m1 * m2 / (r * r),
        truth_expr="G * m1 * m2 / r**2",
        constants_used=["G"],
    ),

    "coulomb": EquationProblem(
        name="coulomb",
        description="Coulomb's law:  F = k_e·q1·q2 / r²",
        # Charges scaled to O(1) so evolution sees a well-conditioned problem.
        # The truth's k_e is unchanged; only the input domain is rescaled.
        variables=["q1", "q2", "r"],
        train_ranges={"q1": (0.1, 1.0), "q2": (0.1, 1.0), "r": (0.5, 5.0)},
        extrap_ranges={"q1": (0.1, 1.0), "q2": (0.1, 1.0), "r": (5.0, 10.0)},
        callable=lambda q1, q2, r: 8.9875517923e9 * q1 * q2 / (r * r),
        truth_expr="k_e * q1 * q2 / r**2",
        constants_used=["k_e"],
    ),

    "pendulum": EquationProblem(
        name="pendulum",
        description="Simple pendulum period:  T = 2π·√(L/g)",
        variables=["L"],
        train_ranges={"L": (0.05, 2.0)},
        extrap_ranges={"L": (2.0, 5.0)},
        callable=lambda L: 2.0 * math.pi * np.sqrt(L / 9.80665),
        truth_expr="2 * pi * sqrt(L / g)",
        constants_used=["pi", "g"],
    ),

    "keplers3": EquationProblem(
        name="keplers3",
        description="Kepler's third law (period from semi-major axis, fixed M):  T² = (4π²/GM)·a³",
        variables=["a"],
        # Use a fixed mass M (solar mass) baked into the callable;
        # evolution should rediscover the composite coefficient.
        train_ranges={"a": (1.0e10, 1.0e11)},
        extrap_ranges={"a": (1.0e11, 5.0e11)},
        callable=lambda a: np.sqrt((4.0 * math.pi**2 / (6.6743e-11 * 1.989e30)) * a**3),
        truth_expr="sqrt((4 * pi**2 / (G * 1.989e30)) * a**3)",
        constants_used=["pi", "G"],
    ),

    "ideal_gas": EquationProblem(
        name="ideal_gas",
        description="Ideal gas (pressure from n, T, V):  P = n·R·T / V",
        variables=["n", "T", "V"],
        train_ranges={"n": (0.1, 5.0), "T": (200.0, 400.0), "V": (0.01, 1.0)},
        extrap_ranges={"n": (0.1, 5.0), "T": (200.0, 400.0), "V": (1.0, 5.0)},
        callable=lambda n, T, V: 8.314462618 * n * T / V,
        truth_expr="R * n * T / V",
        constants_used=["R"],
    ),
}


# -----------------------------------------------------------------------------
# On-demand cached data generation
# -----------------------------------------------------------------------------

def _cache_key(problem: EquationProblem) -> str:
    """Stable hash of the inputs that determine the dataset."""
    payload = {
        "name":          problem.name,
        "variables":     problem.variables,
        "train_ranges":  problem.train_ranges,
        "extrap_ranges": problem.extrap_ranges,
        "n_train":       problem.n_train,
        "n_val":         problem.n_val,
        "n_holdout":     problem.n_holdout,
        "n_extrap":      problem.n_extrap,
        "noise_std":     problem.noise_std,
        "seed":          problem.seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _sample(problem: EquationProblem, ranges: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    cols = {v: rng.uniform(*ranges[v], size=n) for v in problem.variables}
    df = pd.DataFrame(cols)
    df["target"] = problem.callable(**cols)
    if problem.noise_std > 0:
        df["target"] = df["target"] + rng.normal(0.0, problem.noise_std * np.std(df["target"]), size=n)
    return df


def generate_data(
    problem: EquationProblem,
    cache_dir: str = "data/equations",
    force: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate (or load from cache) train/val/holdout/extrapolation splits.

    Returns ``{"train": df, "val": df, "holdout": df, "extrapolation": df}``.
    The holdout slice is drawn from the *train* ranges (so it's an honest
    in-distribution holdout); the extrapolation slice is drawn from
    ``extrap_ranges`` (a different region of the input space).
    """
    cache_path = os.path.join(cache_dir, problem.name)
    manifest_path = os.path.join(cache_path, "manifest.json")
    key = _cache_key(problem)

    if not force and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("cache_key") == key:
                splits = {}
                for split in ("train", "val", "holdout", "extrapolation"):
                    splits[split] = pd.read_csv(os.path.join(cache_path, f"{split}.csv"))
                if verbose:
                    print(f"  ✓ cache hit  ({cache_path}, key={key})")
                return splits
        except Exception:
            pass

    if verbose:
        print(f"  • generating  ({cache_path}, key={key})")

    os.makedirs(cache_path, exist_ok=True)
    rng = np.random.default_rng(problem.seed)

    splits = {
        "train":         _sample(problem, problem.train_ranges,  problem.n_train,   rng),
        "val":           _sample(problem, problem.train_ranges,  problem.n_val,     rng),
        "holdout":       _sample(problem, problem.train_ranges,  problem.n_holdout, rng),
        "extrapolation": _sample(problem, problem.extrap_ranges, problem.n_extrap,  rng),
    }
    for name, df in splits.items():
        df.to_csv(os.path.join(cache_path, f"{name}.csv"), index=False)

    manifest = {
        "cache_key": key,
        "problem": problem.name,
        "description": problem.description,
        "variables": problem.variables,
        "train_ranges": problem.train_ranges,
        "extrap_ranges": problem.extrap_ranges,
        "n_train": problem.n_train,
        "n_val": problem.n_val,
        "n_holdout": problem.n_holdout,
        "n_extrap": problem.n_extrap,
        "noise_std": problem.noise_std,
        "seed": problem.seed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return splits


# -----------------------------------------------------------------------------
# BYO equation
# -----------------------------------------------------------------------------

def make_custom_problem(
    name: str,
    callable: Callable,
    variables: Sequence[str],
    train_ranges: dict,
    extrap_ranges: dict,
    truth_expr: str | None = None,
    constants_used: Sequence[str] | None = None,
    **kwargs,
) -> EquationProblem:
    """Build a one-off problem definition for the user's own equation.

    ``truth_expr`` is optional — without it, the notebook still scores
    R²/MSE but skips structural equation-recovery checking.
    """
    return EquationProblem(
        name=name,
        description=kwargs.pop("description", f"Custom equation: {name}"),
        variables=list(variables),
        train_ranges=dict(train_ranges),
        extrap_ranges=dict(extrap_ranges),
        callable=callable,
        truth_expr=truth_expr or "0",   # placeholder; recovery skipped if 0
        constants_used=list(constants_used or []),
        **kwargs,
    )
