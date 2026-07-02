#!/usr/bin/env python3
"""
Test if disabling normalization fixes the Z3 batch processing issue
"""

import numpy as np
import sys
import hff

def test_normalization_fix():
    """Test Z3 with normalize_input=False to see if it fixes the batch issue."""
    
    # Create test population
    np.random.seed(42)
    objectives = np.random.rand(20, 6).astype(np.float64)
    objectives = np.ascontiguousarray(objectives)
    
    print(f"Population shape: {objectives.shape}")
    print(f"Population range: [{objectives.min():.3f}, {objectives.max():.3f}]")
    
    # Test with normalize_input=True (default, currently broken)
    print(f"\n🔬 Testing Z3 with normalize_input=True (default)")
    try:
        result_normalized = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            normalize_input=True,
            parallel=False  # Use sequential for cleaner debug output
        )
        fitness_normalized = np.array(result_normalized['fitness'])
        print(f"With normalization - mean: {fitness_normalized.mean():.6f}, range: [{fitness_normalized.min():.6f}, {fitness_normalized.max():.6f}]")
        
        if np.allclose(fitness_normalized, 0.0):
            print("❌ With normalization returns zeros")
        else:
            print("✅ With normalization works")
    except Exception as e:
        print(f"With normalization failed: {e}")
    
    # Test with normalize_input=False (hypothesis: this should work)
    print(f"\n🔬 Testing Z3 with normalize_input=False")
    try:
        result_unnormalized = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            normalize_input=False,
            parallel=False
        )
        fitness_unnormalized = np.array(result_unnormalized['fitness'])
        print(f"Without normalization - mean: {fitness_unnormalized.mean():.6f}, range: [{fitness_unnormalized.min():.6f}, {fitness_unnormalized.max():.6f}]")
        
        if np.allclose(fitness_unnormalized, 0.0):
            print("❌ Without normalization still returns zeros")
        else:
            print("✅ WITHOUT NORMALIZATION WORKS!")
            
            # Test different seeds to see if we get diversity
            print(f"\nTesting different seeds for diversity:")
            for seed in [42, 43, 44]:
                result_seed = hff.calculate_hff_fitness_batch(
                    objectives,
                    algorithm="z3",
                    num_groups=2,
                    overlap_factor=4,
                    random_seed=seed,
                    normalize_input=False,
                    parallel=False
                )
                fitness_seed = np.array(result_seed['fitness'])
                print(f"  Seed {seed}: mean={fitness_seed.mean():.6f}, std={fitness_seed.std():.6f}")
                
    except Exception as e:
        print(f"Without normalization failed: {e}")
        
    # Test comparison with Z1 without normalization
    print(f"\n🔬 Testing Z1 with normalize_input=False for comparison")
    try:
        z1_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z1",
            normalize_input=False
        )
        z1_fitness = np.array(z1_result['fitness'])
        print(f"Z1 without normalization - mean: {z1_fitness.mean():.6f}, range: [{z1_fitness.min():.6f}, {z1_fitness.max():.6f}]")
    except Exception as e:
        print(f"Z1 without normalization failed: {e}")

if __name__ == "__main__":
    print("🐛 Testing if disabling normalization fixes Z3 batch processing...")
    test_normalization_fix()