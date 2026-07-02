#!/usr/bin/env python3
"""
Debug the core Z1 function to see if it handles normalized data correctly
"""

import numpy as np
import sys
import hff

def debug_z1_core():
    """Test Z1 with different data scenarios to understand the normalization issue."""
    
    # Test 1: Simple unnormalized data
    print("🔬 Test 1: Simple unnormalized data")
    data1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    try:
        result1 = hff.calculate_hff_fitness_batch(
            data1, algorithm="z1", normalize_input=True
        )
        print(f"Unnormalized [1,2,3] -> Z1 fitness: {result1['fitness'][0]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 2: Same data but different scale
    print("\n🔬 Test 2: Same data, different scale")  
    data2 = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    try:
        result2 = hff.calculate_hff_fitness_batch(
            data2, algorithm="z1", normalize_input=True
        )
        print(f"Scaled [10,20,30] -> Z1 fitness: {result2['fitness'][0]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
        
    # Test 3: Pre-normalized version
    print("\n🔬 Test 3: Pre-normalized version")
    # Manually normalize [1,2,3] columnwise
    data3_raw = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    data3 = data3_raw.copy()
    for j in range(data3.shape[1]):
        norm = np.sqrt(np.sum(data3[:, j] ** 2))
        if norm > 1e-10:
            data3[:, j] /= norm
    print(f"Pre-normalized: {data3[0]}")
    try:
        result3 = hff.calculate_hff_fitness_batch(
            data3, algorithm="z1", normalize_input=False, validate_input=True
        )
        print(f"Pre-normalized -> Z1 fitness: {result3['fitness'][0]:.6f}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 4: Test the actual problematic data from our test
    print("\n🔬 Test 4: Actual problematic data (single individual)")
    np.random.seed(42)
    raw_objectives = np.random.rand(1, 6).astype(np.float64)
    
    # Test Z1 on raw data with auto-normalization
    try:
        z1_auto = hff.calculate_hff_fitness_batch(
            raw_objectives, algorithm="z1", normalize_input=True
        )
        print(f"Raw data with Z1 auto-norm: {z1_auto['fitness'][0]:.6f}")
    except Exception as e:
        print(f"Z1 auto-norm failed: {e}")
    
    # Test Z3 on the same single individual (should work according to our previous tests)
    try:
        z3_single = hff.calculate_hff_fitness_batch(
            raw_objectives, 
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            normalize_input=True
        )
        print(f"Same data with Z3 auto-norm: {z3_single['fitness'][0]:.6f}")
    except Exception as e:
        print(f"Z3 single auto-norm failed: {e}")
    
    # Test Z3 on population (this is what fails)
    print("\n🔬 Test 5: Population test (this should fail)")
    pop_objectives = np.random.rand(5, 6).astype(np.float64)
    try:
        z3_pop = hff.calculate_hff_fitness_batch(
            pop_objectives,
            algorithm="z3", 
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            normalize_input=True
        )
        z3_pop_fitness = np.array(z3_pop['fitness'])
        print(f"Population Z3: mean={z3_pop_fitness.mean():.6f}, std={z3_pop_fitness.std():.6f}")
        
        if np.allclose(z3_pop_fitness, 0.0):
            print("❌ Population Z3 returns zeros")
        else:
            print("✅ Population Z3 works!")
            
    except Exception as e:
        print(f"Population Z3 failed: {e}")

if __name__ == "__main__":
    print("🐛 Debugging Z1 core function with normalized data...")
    debug_z1_core()