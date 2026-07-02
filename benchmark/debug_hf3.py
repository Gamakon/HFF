#!/usr/bin/env python3
"""
Debug the z3 algorithm to understand why it's returning all zeros
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

def debug_single_individual():
    """Debug z3 with a single individual to understand the issue."""
    
    # Create a simple test individual
    individual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    print(f"Individual before reshape: {individual}")
    print(f"Individual sum: {np.sum(individual)}")
    print(f"Individual norm: {np.linalg.norm(individual)}")
    
    # Also test random data like the original test
    print(f"\n--- Testing with random data ---")
    np.random.seed(42)
    random_individual = np.random.rand(6).astype(np.float64)
    print(f"Random individual: {random_individual}")
    print(f"Random sum: {np.sum(random_individual)}")
    print(f"Random norm: {np.linalg.norm(random_individual)}")
    individual = individual.reshape(1, -1)  # Make it 2D for batch function
    random_individual = random_individual.reshape(1, -1)  # Make it 2D for batch function
    
    n_objectives = 6
    num_groups = 2  # floor(sqrt(6)) = 2
    overlap_factor = 4  # ceil(6 * 0.66) = 4
    
    print(f"\nBatch individual: {individual.flatten()}")
    print(f"N objectives: {n_objectives}")
    print(f"Num groups: {num_groups}")
    print(f"Overlap factor: {overlap_factor}")
    
    # Test both individuals
    for name, test_individual in [("Sequential", individual), ("Random", random_individual)]:
        print(f"\n🔬 Testing {name} individual...")
        print(f"Data: {test_individual.flatten()}")
        
        # Test z1 first
        try:
            z1_result = hff.calculate_hff_fitness_batch(
                test_individual,
                algorithm="z1"
            )
            z1_fitness = z1_result['fitness'][0]
            print(f"  Z1 fitness: {z1_fitness:.6f}")
        except Exception as e:
            print(f"  Z1 failed: {e}")
            continue
        
        # Test z3
        try:
            z3_result = hff.calculate_hff_fitness_batch(
                test_individual,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=42
            )
            z3_fitness = z3_result['fitness'][0]
            print(f"  Z3 fitness: {z3_fitness:.6f}")
            
            if z3_fitness == 0.0:
                print("  ❌ Z3 returned zero - the bug is confirmed!")
            else:
                print("  ✅ Z3 returned non-zero value - bug may be fixed!")
                
        except Exception as e:
            print(f"  Z3 failed: {e}")
            continue
        
    # Test different parameters
    print(f"\n🔬 Testing Z3 with different parameters...")
    
    # Try with overlap_factor = num_groups (the old constraint)
    try:
        z3_constrained = hff.calculate_hff_fitness_batch(
            individual,
            algorithm="z3",
            num_groups=num_groups,
            overlap_factor=num_groups,  # 2 instead of 4
            random_seed=42
        )
        constrained_fitness = z3_constrained['fitness'][0]
        print(f"Z3 with overlap_factor={num_groups}: {constrained_fitness:.6f}")
        
    except Exception as e:
        print(f"Z3 constrained failed: {e}")
        
    # Try with more groups
    try:
        z3_more_groups = hff.calculate_hff_fitness_batch(
            individual,
            algorithm="z3",
            num_groups=3,  # More groups
            overlap_factor=4,
            random_seed=42
        )
        more_groups_fitness = z3_more_groups['fitness'][0]
        print(f"Z3 with num_groups=3: {more_groups_fitness:.6f}")
        
    except Exception as e:
        print(f"Z3 more groups failed: {e}")
        
    return True

if __name__ == "__main__":
    print("🐛 Debugging Z3 algorithm zero-return issue...")
    debug_single_individual()