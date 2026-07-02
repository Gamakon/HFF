#!/usr/bin/env python3
"""
Test GPU acceleration with the fixed Z3 implementation
"""

import numpy as np
import sys
import time
import hff

def test_gpu_acceleration():
    """Test if GPU acceleration works with the fixed Z3 implementation."""
    
    print("🚀 Testing GPU acceleration with fixed Z3...")
    
    # Initialize GPU
    print("\n🔧 Initializing GPU...")
    try:
        gpu_ready = hff.init_gpu()
        print(f"GPU initialization: {gpu_ready}")
        if gpu_ready:
            print("✅ GPU is ready for acceleration")
        else:
            print("❌ GPU initialization failed")
    except Exception as e:
        print(f"❌ GPU initialization failed: {e}")
        gpu_ready = False
    
    # Create test data
    np.random.seed(42)
    objectives = np.random.rand(100, 10).astype(np.float64)  # Larger test for GPU benefit
    objectives = np.ascontiguousarray(objectives)
    
    print(f"\nTest data shape: {objectives.shape}")
    
    # Test Z1 with and without GPU (if available)
    algorithms_to_test = [
        ("Z1", {"algorithm": "z1"}),
        ("Z3", {"algorithm": "z3", "num_groups": 3, "overlap_factor": 7, "random_seed": 42})
    ]
    
    for alg_name, alg_params in algorithms_to_test:
        print(f"\n🔬 Testing {alg_name}...")
        
        # Test without explicit GPU parameter (uses default)
        start_time = time.time()
        try:
            result = hff.calculate_hff_fitness_batch(
                objectives,
                **alg_params
            )
            end_time = time.time()
            
            fitness = np.array(result['fitness'])
            elapsed = end_time - start_time
            
            print(f"  {alg_name} fitness range: [{fitness.min():.6f}, {fitness.max():.6f}]")
            print(f"  {alg_name} mean: {fitness.mean():.6f}")
            print(f"  {alg_name} time: {elapsed:.3f}s")
            
            # Check if results are valid
            if np.allclose(fitness, 0.0):
                print(f"  ❌ {alg_name} returned all zeros")
            elif not np.all(np.isfinite(fitness)):
                print(f"  ❌ {alg_name} returned invalid values")
            else:
                print(f"  ✅ {alg_name} returned valid non-zero results")
                
        except Exception as e:
            print(f"  ❌ {alg_name} failed: {e}")
    
    # Test specific GPU features if available
    if gpu_ready:
        print(f"\n🎯 Testing GPU-specific features...")
        
        # Test larger population that would benefit from GPU
        large_objectives = np.random.rand(1000, 15).astype(np.float64)
        large_objectives = np.ascontiguousarray(large_objectives)
        
        print(f"Large test data shape: {large_objectives.shape}")
        
        try:
            start_time = time.time()
            large_result = hff.calculate_hff_fitness_batch(
                large_objectives,
                algorithm="z3",
                num_groups=4,
                overlap_factor=10,
                random_seed=42,
                parallel=True  # Ensure parallel processing
            )
            end_time = time.time()
            
            large_fitness = np.array(large_result['fitness'])
            large_elapsed = end_time - start_time
            
            print(f"  Large Z3 fitness range: [{large_fitness.min():.6f}, {large_fitness.max():.6f}]")
            print(f"  Large Z3 mean: {large_fitness.mean():.6f}")
            print(f"  Large Z3 time: {large_elapsed:.3f}s")
            print(f"  Throughput: {len(large_objectives)/large_elapsed:.1f} individuals/sec")
            
            if np.allclose(large_fitness, 0.0):
                print("  ❌ Large Z3 test returned all zeros")
            else:
                print("  ✅ Large Z3 test successful with GPU acceleration")
                
        except Exception as e:
            print(f"  ❌ Large Z3 test failed: {e}")
    
    return True

def test_hf3_hf3r_with_gpu():
    """Test HF3 vs rHF3 with GPU acceleration."""
    
    print(f"\n🔬 Testing HF3 vs rHF3 with GPU...")
    
    # Initialize GPU first
    try:
        gpu_ready = hff.init_gpu()
        print(f"GPU ready: {gpu_ready}")
    except:
        gpu_ready = False
    
    # Create test population 
    np.random.seed(42)
    objectives = np.random.rand(50, 8).astype(np.float64) 
    objectives = np.ascontiguousarray(objectives)
    
    # Calculate parameters
    n_objectives = 8
    num_groups = int(np.floor(np.sqrt(n_objectives)))  # 2
    overlap_factor = int(np.ceil(n_objectives * 0.66))  # 6
    
    print(f"Population shape: {objectives.shape}")
    print(f"Num groups: {num_groups}, Overlap factor: {overlap_factor}")
    
    # Test HF3 (fixed groups)
    print(f"\n🔬 Testing HF3 (fixed groups)...")
    try:
        hf3_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3",
            num_groups=num_groups,
            overlap_factor=overlap_factor,
            random_seed=42,
            parallel=True
        )
        hf3_fitness = np.array(hf3_result['fitness'])
        print(f"HF3 range: [{hf3_fitness.min():.6f}, {hf3_fitness.max():.6f}]")
        print(f"HF3 mean: {hf3_fitness.mean():.6f}")
    except Exception as e:
        print(f"HF3 failed: {e}")
        return False
    
    # Test rHF3 (dynamic groups) 
    print(f"\n🔬 Testing rHF3 (dynamic groups)...")
    try:
        rhf3_result = hff.calculate_hff_fitness_batch(
            objectives,
            algorithm="z3", 
            num_groups=num_groups,
            overlap_factor=overlap_factor,
            random_seed=42,
            arctic_generation=1,  # This should create different groups
            parallel=True
        )
        rhf3_fitness = np.array(rhf3_result['fitness'])
        print(f"rHF3 range: [{rhf3_fitness.min():.6f}, {rhf3_fitness.max():.6f}]")  
        print(f"rHF3 mean: {rhf3_fitness.mean():.6f}")
    except Exception as e:
        print(f"rHF3 failed: {e}")
        return False
    
    # Compare results
    print(f"\n📊 Comparison:")
    identical = np.allclose(hf3_fitness, rhf3_fitness)
    print(f"Results identical: {identical}")
    print(f"Mean difference: {np.mean(np.abs(hf3_fitness - rhf3_fitness)):.8f}")
    
    if identical:
        print("⚠️  HF3 and rHF3 produce identical results - dynamic groups not working yet")
    else:
        print("✅ HF3 and rHF3 produce different results - dynamic groups working!")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing GPU acceleration and fixed Z3 implementation...")
    
    # Test basic GPU acceleration
    test_gpu_acceleration()
    
    # Test HF3 vs rHF3 with GPU  
    test_hf3_hf3r_with_gpu()
    
    print(f"\n🎉 GPU acceleration testing complete!")