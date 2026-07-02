#!/usr/bin/env python3
"""
Test the new overlap formula for HF3 to verify HF3 vs rHF3 produce different results
"""

import numpy as np
import sys
import hff

def test_new_overlap_formula():
    """Test that the new overlap formula enables dynamic groups to work."""
    
    print("🧪 Testing new overlap formula: overlap = num_groups - 1\n")
    
    # Test with different problem sizes
    test_cases = [
        (6, 8, "Small problem"),
        (10, 15, "Medium problem"),  
        (50, 100, "Large problem")
    ]
    
    for n_individuals, n_objectives, desc in test_cases:
        print(f"📊 {desc}: {n_individuals} individuals, {n_objectives} objectives")
        
        # Create test population
        np.random.seed(42)
        objectives = np.random.rand(n_individuals, n_objectives).astype(np.float64)
        objectives = np.ascontiguousarray(objectives)
        
        # Calculate HF3 parameters with new formula
        import math
        num_groups = int(math.floor(math.sqrt(n_objectives)))
        if num_groups <= 1:
            overlap_factor = 1
        else:
            overlap_factor = num_groups - 1
            
        print(f"  Parameters: {num_groups} groups, overlap_factor={overlap_factor}")
        
        # Test HF3 (fixed groups)
        try:
            hf3_result = hff.calculate_hff_fitness_batch(
                objectives,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=42
            )
            hf3_fitness = np.array(hf3_result['fitness'])
        except Exception as e:
            print(f"  ❌ HF3 failed: {e}")
            continue
        
        # Test rHF3 (dynamic groups with arctic_generation)
        try:
            rhf3_result = hff.calculate_hff_fitness_batch(
                objectives,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=42,
                arctic_generation=1  # Should add 1000 to seed
            )
            rhf3_fitness = np.array(rhf3_result['fitness'])
        except Exception as e:
            print(f"  ❌ rHF3 failed: {e}")
            continue
        
        # Compare results
        print(f"  HF3 range:  [{hf3_fitness.min():.8f}, {hf3_fitness.max():.8f}]")
        print(f"  rHF3 range: [{rhf3_fitness.min():.8f}, {rhf3_fitness.max():.8f}]")
        
        # Check if they're different
        identical = np.allclose(hf3_fitness, rhf3_fitness, rtol=1e-10)
        mean_diff = np.mean(np.abs(hf3_fitness - rhf3_fitness))
        
        print(f"  Results identical: {identical}")
        print(f"  Mean absolute diff: {mean_diff:.10f}")
        
        if identical:
            print("  ❌ Still identical - dynamic groups not working")
        else:
            print("  ✅ Different results - dynamic groups WORKING!")
        
        # Check if results are reasonable (non-zero)
        hf3_all_zero = np.allclose(hf3_fitness, 0.0, atol=1e-10)
        rhf3_all_zero = np.allclose(rhf3_fitness, 0.0, atol=1e-10)
        
        if hf3_all_zero or rhf3_all_zero:
            print("  ⚠️  One algorithm returning near-zero values")
        else:
            print("  ✅ Both algorithms returning reasonable values")
            
        print()

def test_single_individual_diversity():
    """Test that single individuals get different results with different seeds."""
    
    print("🔬 Testing single individual with different seeds...")
    
    # Create single individual
    np.random.seed(42)
    individual = np.random.rand(1, 10).astype(np.float64)
    
    # Parameters with new formula
    num_groups = 3  # floor(sqrt(10)) = 3
    overlap_factor = 2  # num_groups - 1 = 2
    
    print(f"Individual shape: {individual.shape}")
    print(f"Parameters: {num_groups} groups, overlap_factor={overlap_factor}")
    
    seeds_to_test = [42, 1042, 2042, 3042]
    results = []
    
    for seed in seeds_to_test:
        try:
            result = hff.calculate_hff_fitness_batch(
                individual,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=seed
            )
            fitness = result['fitness'][0]
            results.append((seed, fitness))
            print(f"  Seed {seed}: {fitness:.10f}")
        except Exception as e:
            print(f"  Seed {seed}: FAILED - {e}")
    
    # Check diversity
    unique_values = len(set(fitness for _, fitness in results))
    print(f"\nUnique fitness values: {unique_values}/{len(results)}")
    
    if unique_values == 1:
        print("❌ All seeds produce identical results")
    elif unique_values == len(results):
        print("✅ All seeds produce different results - perfect diversity!")
    else:
        print("⚠️  Some seeds produce identical results")

if __name__ == "__main__":
    print("🧪 Testing improved HF3 overlap formula...\n")
    
    # Test the new formula with different problem sizes
    test_new_overlap_formula()
    
    # Test single individual diversity
    test_single_individual_diversity()
    
    print("\n🎯 Summary:")
    print("The new formula: overlap_factor = max(1, num_groups - 1)")
    print("Should enable:")
    print("- Each objective appears in (num_groups - 1) groups")
    print("- Each group is missing different objectives") 
    print("- Maximum diversity between different seeds")
    print("- HF3 and rHF3 produce different results")