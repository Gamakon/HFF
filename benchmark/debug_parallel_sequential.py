#!/usr/bin/env python3
"""
Test parallel vs sequential processing in Z3 batch
"""

import numpy as np
import sys
import hff

def test_parallel_vs_sequential():
    """Test if the issue is with parallel vs sequential processing."""
    
    # Create test population
    np.random.seed(42)
    objectives = np.random.rand(20, 6).astype(np.float64)
    objectives = np.ascontiguousarray(objectives)
    
    print(f"Population shape: {objectives.shape}")
    
    # Test with parallel=True (default)
    print(f"\n🔬 Testing Z3 with parallel=True")
    try:
        result_parallel = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            parallel=True
        )
        fitness_parallel = np.array(result_parallel['fitness'])
        print(f"Parallel - mean: {fitness_parallel.mean():.6f}, range: [{fitness_parallel.min():.6f}, {fitness_parallel.max():.6f}]")
        
        if np.allclose(fitness_parallel, 0.0):
            print("❌ Parallel processing returns zeros")
        else:
            print("✅ Parallel processing works")
    except Exception as e:
        print(f"Parallel processing failed: {e}")
    
    # Test with parallel=False
    print(f"\n🔬 Testing Z3 with parallel=False")
    try:
        result_sequential = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            parallel=False
        )
        fitness_sequential = np.array(result_sequential['fitness'])
        print(f"Sequential - mean: {fitness_sequential.mean():.6f}, range: [{fitness_sequential.min():.6f}, {fitness_sequential.max():.6f}]")
        
        if np.allclose(fitness_sequential, 0.0):
            print("❌ Sequential processing returns zeros")
        else:
            print("✅ Sequential processing works")
    except Exception as e:
        print(f"Sequential processing failed: {e}")
    
    # Test smaller population
    print(f"\n🔬 Testing Z3 with smaller population (5 individuals)")
    small_objectives = objectives[:5, :]
    try:
        result_small = hff.calculate_hff_fitness_batch(
            small_objectives,
            algorithm="z3",
            num_groups=2,
            overlap_factor=4,
            random_seed=42,
            parallel=True
        )
        fitness_small = np.array(result_small['fitness'])
        print(f"Small population - mean: {fitness_small.mean():.6f}, range: [{fitness_small.min():.6f}, {fitness_small.max():.6f}]")
        
        if np.allclose(fitness_small, 0.0):
            print("❌ Small population returns zeros")
        else:
            print("✅ Small population works")
    except Exception as e:
        print(f"Small population failed: {e}")

if __name__ == "__main__":
    print("🐛 Testing parallel vs sequential Z3 processing...")
    test_parallel_vs_sequential()