"""HFF Python API — column normalization and convenience wrappers over Rust."""

import os

import numpy as np

from hff import hff_core


# Opt-in GPU path. Set HFF_GPU=1 to route truenorth batches to the wgpu
# compute pipeline (when the Rust core was built with --features gpu).
# Falls back to CPU silently if the symbol or device is missing.
_HFF_GPU_ENABLED = bool(os.environ.get("HFF_GPU"))
_HFF_GPU_SYMBOL = getattr(
    hff_core, "calculate_hyperspherical_fitness_hf1_enhanced_gpu", None
)


def calculate_fitness_hf1(
    objectives: np.ndarray,
    normalize: bool = True,
    decrowding: bool = False,
) -> np.ndarray:
    """Calculate HF1 (Balanced) hyperspherical fitness.

    Projects solutions onto a unit hypersphere and measures angular distance
    to the balanced north pole (1/sqrt(m), ..., 1/sqrt(m)).

    Args:
        objectives: (n_individuals, n_objectives) array.
        normalize: Apply column-wise min-max normalization.
        decrowding: Apply decrowding transform (log-sigmoid z-score).

    Returns:
        1-D array of fitness values (lower is better).
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)
    if objectives.shape[0] == 0:
        return np.array([])
    if normalize:
        objectives = _column_normalize(objectives)
    return hff_core.calculate_hyperspherical_fitness_hf1_f64(objectives, decrowding)


def calculate_fitness_hf1_enhanced(
    objectives: np.ndarray,
    normalize: bool = True,
    decrowding: bool = False,
    north_pole_method: str = "balanced",
) -> np.ndarray:
    """Calculate HF1 fitness with selectable north-pole method.

    Args:
        objectives: (n_individuals, n_objectives) array.
        normalize: Apply column-wise min-max normalisation inside the Rust
            core. Pass False when objectives are already bounded (e.g.
            classification metrics in [0, 1]) — otherwise the column-best
            individual is mapped to all-ones and collapses onto the pole.
        decrowding: Apply decrowding transform.
        north_pole_method: "balanced" or "truenorth".

    Returns:
        1-D array of fitness values (lower is better).
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)
    if objectives.shape[0] == 0:
        return np.array([])
    # GPU path — opt-in via HFF_GPU=1, only for truenorth (the GPU shader's
    # only supported pole method today). Decrowding stays on CPU.
    if (_HFF_GPU_ENABLED and _HFF_GPU_SYMBOL is not None
            and north_pole_method == "truenorth" and not decrowding):
        try:
            return _HFF_GPU_SYMBOL(objectives, north_pole_method, normalize)
        except Exception:
            pass  # fall through to CPU on any error
    # The Rust core handles its own normalisation when `normalize=True` —
    # we no longer double-normalise in Python.
    return hff_core.calculate_hyperspherical_fitness_hf1_enhanced(
        objectives, decrowding, north_pole_method, normalize
    )


def calculate_fitness_hf1_with_ranges(
    objectives: np.ndarray,
    normalize: bool = True,
    decrowding: bool = False,
    north_pole_method: str = "balanced",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as :func:`calculate_fitness_hf1_enhanced` but ALSO returns the
    per-column ``(col_min, col_max)`` used for normalisation. Call this once
    on generation 0 to capture the scale, then feed the ranges to
    :func:`calculate_fitness_hf1_fixed` on subsequent generations so the
    HFF pole stays stable across the run.

    Returns:
        ``(fitness, col_min, col_max)`` — fitness is length n_individuals,
        col_min/col_max are length n_objectives.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)
    if objectives.shape[0] == 0:
        m = objectives.shape[1]
        return np.array([]), np.zeros(m), np.zeros(m)
    return hff_core.calculate_hyperspherical_fitness_hf1_with_ranges(
        objectives, decrowding, north_pole_method, normalize,
    )


def calculate_fitness_hf1_fixed(
    objectives: np.ndarray,
    col_min: np.ndarray,
    col_max: np.ndarray,
    decrowding: bool = False,
    north_pole_method: str = "balanced",
) -> np.ndarray:
    """Score ``objectives`` using caller-supplied per-column min/max.
    Use this on every generation > 0 with the ``(col_min, col_max)``
    captured from :func:`calculate_fitness_hf1_with_ranges` on gen 0.

    The pole stays geometrically meaningful as the population improves —
    later good solutions can genuinely approach HFF=0.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    col_min = np.asarray(col_min, dtype=np.float64)
    col_max = np.asarray(col_max, dtype=np.float64)
    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)
    if objectives.shape[0] == 0:
        return np.array([])
    return hff_core.calculate_hyperspherical_fitness_hf1_fixed(
        objectives, col_min, col_max, decrowding, north_pole_method,
    )


def _column_normalize(objectives: np.ndarray) -> np.ndarray:
    """Column-wise min-max normalization to [0, 1]."""
    if objectives.shape[0] <= 1:
        return objectives
    min_vals = np.min(objectives, axis=0)
    max_vals = np.max(objectives, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1.0, ranges)
    return (objectives - min_vals) / ranges
