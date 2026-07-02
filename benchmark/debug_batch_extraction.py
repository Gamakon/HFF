#!/usr/bin/env python3
"""
Debug batch extraction to understand why Z3 fails with population data
"""

import numpy as np
import sys
import os

# Add the project root to Python path
try:
    import hff
    print("✅ hff imported successfully")
except ImportError as e:
    print(f"❌ Failed to import hff: {e}")
    sys.exit(1)

def debug_batch_extraction():
    """Debug batch data extraction issues."""
    
    # Create population data exactly like the failing test
    np.random.seed(42)
    n_individuals = 20
    n_objectives = 6
    objectives = np.random.rand(n_individuals, n_objectives).astype(np.float64)
    objectives = np.ascontiguousarray(objectives)
    
    print(f"Population shape: {objectives.shape}")
    print(f"Population dtype: {objectives.dtype}")
    print(f"Population C-contiguous: {objectives.flags['C_CONTIGUOUS']}")
    print(f"First individual: {objectives[0]}")
    
    # Test parameters
    num_groups = 2
    overlap_factor = 4
    random_seed = 42
    
    # Test 1: Full population with Z1 (this works)
    print(f"\n🔬 Test 1: Full population with Z1 (should work)")
    try:
        z1_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z1"
        )
        z1_fitness = np.array(z1_result['fitness'])
        print(f"Z1 population - mean: {z1_fitness.mean():.6f}, range: [{z1_fitness.min():.6f}, {z1_fitness.max():.6f}]")
        print(f"Z1 first individual: {z1_fitness[0]:.6f}")
    except Exception as e:
        print(f"Z1 population failed: {e}")
    
    # Test 2: Full population with Z3 (this fails)
    print(f"\n🔬 Test 2: Full population with Z3 (currently fails)")
    try:
        z3_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=num_groups,
            overlap_factor=overlap_factor,
            random_seed=random_seed
        )
        z3_fitness = np.array(z3_result['fitness'])
        print(f"Z3 population - mean: {z3_fitness.mean():.6f}, range: [{z3_fitness.min():.6f}, {z3_fitness.max():.6f}]")
        print(f"Z3 first individual: {z3_fitness[0]:.6f}")
        
        # Check if all are zeros
        if np.allclose(z3_fitness, 0.0):
            print("❌ Z3 returned all zeros for population")
        else:
            print("✅ Z3 returned non-zero values for population")
            
    except Exception as e:
        print(f"Z3 population failed: {e}")
    
    # Test 3: Individual extraction - test first individual only
    print(f"\n🔬 Test 3: First individual extracted from population")
    first_individual = objectives[0:1, :]  # Keep 2D shape but single row
    print(f"Extracted shape: {first_individual.shape}")
    print(f"Extracted data: {first_individual[0]}")
    
    try:
        z3_single = hff.calculate_hff_fitness_batch(
            first_individual,
            algorithm="z3",
            num_groups=num_groups,
            overlap_factor=overlap_factor,
            random_seed=random_seed
        )
        single_fitness = z3_single['fitness'][0]
        print(f"Z3 single individual: {single_fitness:.6f}")
        
        if single_fitness == 0.0:
            print("❌ Single individual also returns zero")
        else:
            print("✅ Single individual works correctly")
    except Exception as e:
        print(f"Z3 single failed: {e}")
    
    # Test 4: Manual iteration through population
    print(f"\n🔬 Test 4: Manual iteration through each individual")
    manual_results = []
    for i in range(min(5, n_individuals)):  # Test first 5 individuals
        individual_2d = objectives[i:i+1, :]  # Extract as 2D
        try:
            result = hff.calculate_hff_fitness_batch(
                individual_2d,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=random_seed
            )
            fitness = result['fitness'][0]
            manual_results.append(fitness)
            print(f"Individual {i}: {fitness:.6f}")
        except Exception as e:
            print(f"Individual {i} failed: {e}")
            manual_results.append(0.0)
    
    print(f"Manual results mean: {np.mean(manual_results):.6f}")
    
    return len(manual_results) > 0 and not np.allclose(manual_results, 0.0)

if __name__ == "__main__":
    print("🐛 Debugging Z3 batch extraction...")
    success = debug_batch_extraction()
    if success:
        print("\n✅ Found working approach for individuals")
    else:
        print("\n❌ All approaches failed")