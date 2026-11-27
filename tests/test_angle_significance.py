"""Angle significance: ln P(angle <= theta) under a chosen null."""

import numpy as np
import pytest

import hff


def _random_truenorth(m, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return hff.calculate_fitness_hf1_enhanced(
        rng.random((n, m)), normalize=True, north_pole_method="truenorth"
    )


def test_truenorth_null_matches_uniform_objective_model():
    # Best of 40 random solutions should score p ~ 1/41 at every m.
    for m in (10, 20, 40, 80):
        best = [
            hff.calculate_angle_significance(_random_truenorth(m, seed=s).min(), m, log=False)
            for s in range(150)
        ]
        assert 0.012 < np.median(best) < 0.045, m


def test_significance_parameter_matches_manual_and_preserves_ranking():
    F = np.random.default_rng(3).random((30, 12))
    theta = hff.calculate_fitness_hf1_enhanced(F, north_pole_method="truenorth")
    logp = hff.calculate_fitness_hf1_enhanced(F, north_pole_method="truenorth",
                                              significance="truenorth")
    np.testing.assert_allclose(logp, hff.calculate_angle_significance(theta, 12))
    assert (np.argsort(theta, kind="stable") == np.argsort(logp, kind="stable")).all()


def test_fixed_ranges_support_significance():
    F = np.random.default_rng(4).random((20, 6))
    _, lo, hi = hff.calculate_fitness_hf1_with_ranges(F, north_pole_method="truenorth")
    a = hff.calculate_fitness_hf1_fixed(F, lo, hi, north_pole_method="truenorth",
                                        significance="truenorth")
    b = hff.calculate_fitness_hf1_enhanced(F, north_pole_method="truenorth",
                                           significance="truenorth")
    np.testing.assert_allclose(a, b)


def test_shapes_log_and_errors():
    th = np.array([[0.1, 0.5], [0.9, 1.2]])
    assert hff.calculate_angle_significance(th, 8).shape == (2, 2)
    np.testing.assert_allclose(hff.calculate_angle_significance(th, 8, log=False),
                               np.exp(hff.calculate_angle_significance(th, 8)))
    assert np.isfinite(hff.calculate_angle_significance(np.array([0.0]), 500)).all()
    with pytest.raises(ValueError):
        hff.calculate_angle_significance(th, 8, null="balanced")
    with pytest.raises(ValueError):
        hff.calculate_fitness_hf1_enhanced(np.ones((3, 3)), north_pole_method="balanced",
                                           significance="truenorth")
