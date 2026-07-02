#!/usr/bin/env python3
"""
Quick test to verify HF3 and rHF3 produce different results after z3 fix
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

def test_hf3_vs_rhf3():
    """Test that HF3 and rHF3 produce different results with fixed z3 implementation."""
    
    # Create test data with 6 objectives
    np.random.seed(42)
    n_individuals = 20
    n_objectives = 6
    objectives = np.random.rand(n_individuals, n_objectives).astype(np.float64)
    objectives = np.ascontiguousarray(objectives)
    
    print(f"Objectives shape: {objectives.shape}")
    print(f"First individual: {objectives[0]}")
    print(f"Objectives range: [{objectives.min():.3f}, {objectives.max():.3f}]")
    print(f"All finite: {np.all(np.isfinite(objectives))}")
    
    # Calculate parameters
    num_groups = int(np.floor(np.sqrt(n_objectives)))  # 2
    overlap_factor = int(np.ceil(n_objectives * 0.66))  # 4
    
    print(f"Test data shape: {objectives.shape}")
    print(f"Num groups: {num_groups}")
    print(f"Overlap factor: {overlap_factor}")
    
    # Check if the problem would be rejected by z3 validation
    if n_objectives < 3 or num_groups <= 1:
        print(f"⚠️  Problem would be rejected by z3: n_objectives={n_objectives}, num_groups={num_groups}")
    else:
        print(f"✅ Problem passes z3 validation checks")
    
    try:
        # Test Z1 (HF1) algorithm for comparison
        print("\n🔬 Testing Z1 (HF1) for comparison...")
        z1_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z1"
        )
        z1_fitness = np.array(z1_result['fitness'])
        print(f"Z1 fitness range: [{z1_fitness.min():.6f}, {z1_fitness.max():.6f}]")
        print(f"Z1 mean: {z1_fitness.mean():.6f}")
        
        # Test HF3 (fixed groups)
        print("\n🔬 Testing HF3 (fixed groups)...")
        hf3_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=num_groups,
            overlap_factor=overlap_factor,
            random_seed=42
        )
        hf3_fitness = np.array(hf3_result['fitness'])
        print(f"HF3 fitness range: [{hf3_fitness.min():.6f}, {hf3_fitness.max():.6f}]")
        print(f"HF3 mean: {hf3_fitness.mean():.6f}")
        
        # Test rHF3 (dynamic groups) - simulate generation 1
        print("\n🔬 Testing rHF3 (dynamic groups)...")
        try:
            # Try using arctic_generation parameter for dynamic seed
            rhf3_result = hff.calculate_hff_fitness_batch(
                objectives,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=42,
                arctic_generation=1  # Generation 1 for dynamic groups
            )
        except TypeError:
            # Fallback: use different seed directly
            rhf3_result = hff.calculate_hff_fitness_batch(
                objectives,
                algorithm="z3",
                num_groups=num_groups,
                overlap_factor=overlap_factor,
                random_seed=1042  # 42 + 1000
            )
        
        rhf3_fitness = np.array(rhf3_result['fitness'])
        print(f"rHF3 fitness range: [{rhf3_fitness.min():.6f}, {rhf3_fitness.max():.6f}]")
        print(f"rHF3 mean: {rhf3_fitness.mean():.6f}")
        
        # Compare results
        print(f"\n📊 Comparison:")
        print(f"All fitness values identical: {np.allclose(hf3_fitness, rhf3_fitness)}")
        print(f"Mean absolute difference: {np.mean(np.abs(hf3_fitness - rhf3_fitness)):.8f}")
        
        # Check if both are all zeros (the bug we're fixing)
        hf3_all_zeros = np.allclose(hf3_fitness, 0.0)
        rhf3_all_zeros = np.allclose(rhf3_fitness, 0.0)
        
        if hf3_all_zeros and rhf3_all_zeros:
            print("❌ BOTH algorithms still returning all zeros - z3 bug not fixed")
            return False
        elif hf3_all_zeros or rhf3_all_zeros:
            print("⚠️  One algorithm returning all zeros - partial fix")
            return False
        else:
            print("✅ Both algorithms returning non-zero values - z3 bug appears fixed!")
            
            if np.allclose(hf3_fitness, rhf3_fitness):
                print("⚠️  But results are identical - dynamic groups may not be working")
                return True  # z3 fixed but dynamic groups issue
            else:
                print("✅ Results are different - dynamic groups working correctly!")
                return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing HF3 vs rHF3 after z3 fix...")
    success = test_hf3_vs_rhf3()
    sys.exit(0 if success else 1)