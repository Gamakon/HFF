"""Hyperspherical Fitness Functions for many-objective optimization.

Public API:
    calculate_fitness_hf1()          — HF1 balanced fitness
    calculate_fitness_hf1_enhanced() — HF1 with method selection (balanced/truenorth)
    calculate_higd()                 — CDF-corrected angular IGD
    calculate_angular_igd()          — Raw angular IGD
"""

from hff.core import (
    calculate_fitness_hf1,
    calculate_fitness_hf1_enhanced,
    calculate_fitness_hf1_with_ranges,
    calculate_fitness_hf1_fixed,
)
from hff.hff_core import (
    calculate_higd,
    calculate_angular_igd,
)

__version__ = "0.1.0"

__all__ = [
    "calculate_fitness_hf1",
    "calculate_fitness_hf1_enhanced",
    "calculate_fitness_hf1_with_ranges",
    "calculate_fitness_hf1_fixed",
    "calculate_higd",
    "calculate_angular_igd",
]
