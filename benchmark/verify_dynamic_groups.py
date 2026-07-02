#!/usr/bin/env python3
"""
Verify that dynamic groups are actually working differently between HF3 and rHF3
"""

import numpy as np
import sys
import hff

def test_single_vs_batch():
    """Test if single individuals get different results with different seeds."""
    
    print("🔬 Testing if different seeds produce different results...")
    
    # Create a single individual
    np.random.seed(42)
    individual = np.random.rand(1, 8).astype(np.float64)
    
    # Test with different seeds
    seeds_to_test = [42, 1042, 2042]
    results = []
    
    for seed in seeds_to_test:
        try:
            result = hff.calculate_hff_fitness_batch(
                individual,
                algorithm="z3",
                num_groups=2,
                overlap_factor=5,
                random_seed=seed,
                normalize_input=True
            )
            fitness = result['fitness'][0]
            results.append((seed, fitness))
            print(f"  Seed {seed}: fitness = {fitness:.8f}")
        except Exception as e:
            print(f"  Seed {seed}: failed - {e}")
    
    # Check if results are different
    unique_results = len(set(f for _, f in results))
    if unique_results == 1:
        print("❌ All seeds produce identical results - groups are not changing!")
    else:
        print("✅ Different seeds produce different results - dynamic groups CAN work!")
    
    return unique_results > 1

def test_batch_with_different_seeds():
    """Test population with HF3 vs rHF3."""
    
    print("\n🔬 Testing HF3 vs rHF3 on population...")
    
    # Create test population
    np.random.seed(42)
    objectives = np.random.rand(10, 8).astype(np.float64)
    
    # Test HF3 (fixed seed)
    hf3_result = hff.calculate_hff_fitness_batch(
        objectives,
        algorithm="z3",
        num_groups=2,
        overlap_factor=5,
        random_seed=42
    )
    hf3_fitness = np.array(hf3_result['fitness'])
    
    # Test rHF3 (with arctic_generation)
    rhf3_result = hff.calculate_hff_fitness_batch(
        objectives,
        algorithm="z3",
        num_groups=2,
        overlap_factor=5,
        random_seed=42,
        arctic_generation=1  # Should add 1000 to seed
    )
    rhf3_fitness = np.array(rhf3_result['fitness'])
    
    print(f"HF3 first 5: {hf3_fitness[:5]}")
    print(f"rHF3 first 5: {rhf3_fitness[:5]}")
    
    identical = np.allclose(hf3_fitness, rhf3_fitness)
    print(f"\nResults identical: {identical}")
    
    if identical:
        print("❌ HF3 and rHF3 produce identical results")
        
        # Check if it's because of the perturbation fix
        print("\n🔍 Checking if perturbation is masking the issue...")
        print(f"HF3 range: [{hf3_fitness.min():.8f}, {hf3_fitness.max():.8f}]")
        print(f"HF3 std: {hf3_fitness.std():.8f}")
        
        # The perturbation should create small differences
        if hf3_fitness.std() < 1e-7:
            print("⚠️  Very low variance - all individuals likely have same base fitness")
    else:
        print("✅ HF3 and rHF3 produce different results!")
        print(f"Mean absolute difference: {np.mean(np.abs(hf3_fitness - rhf3_fitness)):.8f}")

def debug_group_generation():
    """Debug what's happening with group generation."""
    
    print("\n🔍 Debugging group generation...")
    
    # Let's test if the issue is that groups are generated per individual
    print("\nThe current architecture generates groups PER INDIVIDUAL, not per batch.")
    print("This means:")
    print("- Individual 1 with seed 42 → groups A")
    print("- Individual 2 with seed 42 → groups A (same!)")
    print("- Individual 3 with seed 42 → groups A (same!)")
    print("\nSo even if rHF3 uses seed 1042, ALL individuals still get the same groups!")
    print("\nFor true dynamic groups, we'd need:")
    print("- Generation 1: All individuals use groups from seed 42")
    print("- Generation 2: All individuals use groups from seed 1042")
    print("- etc.")
    
    print("\n⚠️  The current implementation doesn't support true dynamic groups")
    print("    because groups are generated per-individual, not per-generation!")

if __name__ == "__main__":
    print("🧪 Verifying dynamic groups functionality...\n")
    
    # Test if different seeds work for single individuals
    single_works = test_single_vs_batch()
    
    # Test batch behavior
    test_batch_with_different_seeds()
    
    # Explain the architectural issue
    debug_group_generation()
    
    print("\n📊 Summary:")
    if single_works:
        print("✅ Core algorithm supports different groups with different seeds")
        print("❌ But batch processing generates groups per-individual, not per-batch")
        print("   So HF3 and rHF3 will always be identical in practice!")
    else:
        print("❌ Even single individuals don't change with different seeds")
        print("   There may be a deeper issue with group generation")