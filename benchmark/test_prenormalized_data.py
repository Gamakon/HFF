#!/usr/bin/env python3
"""
Test with pre-normalized data to see if Z3 works when data is properly normalized
"""

import numpy as np
import sys
import hff

def preprocess_objectives(objectives):
    """Pre-normalize objectives columnwise as expected by hff."""
    normalized = objectives.copy()
    n_individuals, n_objectives = objectives.shape
    
    for j in range(n_objectives):
        column = normalized[:, j]
        norm = np.sqrt(np.sum(column ** 2))
        if norm > 1e-10:  # Avoid division by zero
            normalized[:, j] = column / norm
    
    return normalized

def test_prenormalized_data():
    """Test Z3 with properly pre-normalized data."""
    
    # Create test population
    np.random.seed(42)
    objectives_raw = np.random.rand(20, 6).astype(np.float64)
    objectives_raw = np.ascontiguousarray(objectives_raw)
    
    # Pre-normalize the data manually
    objectives = preprocess_objectives(objectives_raw)
    objectives = np.ascontiguousarray(objectives)
    
    print(f"Original data range: [{objectives_raw.min():.3f}, {objectives_raw.max():.3f}]")
    print(f"Normalized data range: [{objectives.min():.3f}, {objectives.max():.3f}]")
    
    # Verify normalization
    for j in range(objectives.shape[1]):
        column_norm = np.sqrt(np.sum(objectives[:, j] ** 2))
        print(f"Column {j} norm: {column_norm:.6f}")
    
    # Test Z1 with pre-normalized data
    print(f"\n🔬 Testing Z1 with pre-normalized data")
    try:
        z1_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z1",
            normalize_input=False,
            validate_input=True
        )
        z1_fitness = np.array(z1_result['fitness'])
        print(f"Z1 - mean: {z1_fitness.mean():.6f}, range: [{z1_fitness.min():.6f}, {z1_fitness.max():.6f}]")
        print("✅ Z1 works with pre-normalized data")
    except Exception as e:
        print(f"Z1 failed: {e}")
    
    # Test Z3 with pre-normalized data
    print(f"\n🔬 Testing Z3 with pre-normalized data")
    try:
        z3_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            normalize_input=False,
            validate_input=True,
            parallel=False
        )
        z3_fitness = np.array(z3_result['fitness'])
        print(f"Z3 - mean: {z3_fitness.mean():.6f}, range: [{z3_fitness.min():.6f}, {z3_fitness.max():.6f}]")
        
        if np.allclose(z3_fitness, 0.0):
            print("❌ Z3 still returns zeros even with pre-normalized data")
        else:
            print("✅ Z3 WORKS with pre-normalized data!")
            
    except Exception as e:
        print(f"Z3 failed: {e}")
    
    # Test with raw data and auto-normalization for comparison
    print(f"\n🔬 Testing Z3 with raw data and auto-normalization")
    try:
        z3_auto = hff.calculate_hff_fitness_batch(
            objectives_raw,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            normalize_input=True,  # Let hff do the normalization
            parallel=False
        )
        z3_auto_fitness = np.array(z3_auto['fitness'])
        print(f"Z3 auto-normalized - mean: {z3_auto_fitness.mean():.6f}, range: [{z3_auto_fitness.min():.6f}, {z3_auto_fitness.max():.6f}]")
        
        if np.allclose(z3_auto_fitness, 0.0):
            print("❌ Z3 with auto-normalization returns zeros")
        else:
            print("✅ Z3 with auto-normalization works!")
            
    except Exception as e:
        print(f"Z3 auto-normalization failed: {e}")

if __name__ == "__main__":
    print("🐛 Testing Z3 with properly pre-normalized data...")
    test_prenormalized_data()