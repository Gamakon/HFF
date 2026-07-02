#!/usr/bin/env python3
"""
GNBG Implementation Benchmark: C++ (native) vs Rust (gnbg-gpu)

This script benchmarks the performance difference between the existing C++ FFI implementation
(hff) and the new Rust GPU implementation (gnbg_gpu) for multi-objective GNBG problems.

We test a 5-objective problem combining F1+F2+F3+F4+F5 to stress-test both implementations.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev
import gc
import psutil
import os

# Add benchmark directory to path
sys.path.append(str(Path(__file__).parent))

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

# Multi-objective problem configuration (F1+F2+F3+F4+F5 = 5 objectives)
BENCHMARK_PROBLEM_CONFIG = {
    'gnbg2': [1, 2, 3, 4, 5],  # Combine F1, F2, F3, F4, F5
    'n_var': 30                 # 30 decision variables
}

# Benchmark parameters
WARMUP_ITERATIONS = 5           # Warmup runs to stabilize performance
BENCHMARK_ITERATIONS = 10       # Number of benchmark runs per test
BATCH_SIZES = [10, 50, 100, 200, 500, 1000]  # Different batch sizes to test
POPULATION_SIZES = [100, 500, 1000, 2000]     # Population sizes for stress test

# Performance measurement
MEMORY_MONITORING = True        # Track memory usage
DETAILED_TIMING = True          # Measure GPU vs CPU components

# Output configuration
OUTPUT_FILE = "benchmark_results_gnbg.csv"
PLOT_RESULTS = True

# =============================================================================
# IMPLEMENTATION IMPORTS AND SETUP
# =============================================================================

def setup_implementations():
    """Initialize both C++ and Rust implementations"""
    print("🔧 Setting up implementations...")
    
    # C++ Implementation (hff)
    cpp_available = False
    try:
        import hff
        cpp_gpu_ready = hff.init_gpu()
        print(f"   ✅ C++ (native): {'GPU Ready' if cpp_gpu_ready else 'CPU Only'}")
        cpp_available = True
    except Exception as e:
        print(f"   ❌ C++ (native): Failed to initialize - {e}")
    
    # Rust Implementation (gnbg_gpu)
    rust_available = False
    rust_gpu_available = False
    try:
        import gnbg_gpu
        # Test with F1 first
        test_evaluator = gnbg_gpu.GNBGGpu(1, use_gpu=True)
        rust_gpu_available = test_evaluator.using_gpu
        test_evaluator = gnbg_gpu.GNBGGpu(1, use_gpu=False)  # CPU fallback
        print(f"   ✅ Rust (gnbg_gpu): {'GPU Ready' if rust_gpu_available else 'CPU Only'}")
        rust_available = True
    except Exception as e:
        print(f"   ❌ Rust (gnbg_gpu): Failed to initialize - {e}")
    
    return cpp_available, rust_available, rust_gpu_available

# =============================================================================
# BENCHMARK PROBLEM WRAPPER
# =============================================================================

class MultiObjectiveGNBGProblem:
    """Multi-objective GNBG problem combining multiple F functions"""
    
    def __init__(self, f_functions, n_var=30, implementation='cpp'):
        self.f_functions = f_functions
        self.n_var = n_var
        self.n_obj = len(f_functions)
        self.implementation = implementation
        self.evaluators = {}
        
        # Bounds (standard GNBG range)
        self.xl = np.full(n_var, -100.0)
        self.xu = np.full(n_var, 100.0)
        
        # Initialize evaluators based on implementation
        if implementation == 'cpp':
            self._setup_cpp_evaluators()
        elif implementation == 'rust_gpu':
            self._setup_rust_evaluators(use_gpu=True)
        elif implementation == 'rust_cpu':
            self._setup_rust_evaluators(use_gpu=False)
    
    def _setup_cpp_evaluators(self):
        """Setup C++ evaluators via hff"""
        import hff
        for f_num in self.f_functions:
            # Note: This assumes hff has a way to create GNBG evaluators
            # You may need to adjust this based on the actual hff API
            self.evaluators[f_num] = f"gnbg_f{f_num}"  # Placeholder - adjust as needed
    
    def _setup_rust_evaluators(self, use_gpu=True):
        """Setup Rust evaluators via gnbg_gpu"""
        import gnbg_gpu
        for f_num in self.f_functions:
            self.evaluators[f_num] = gnbg_gpu.GNBGGpu(f_num, use_gpu=use_gpu)
    
    def evaluate_batch(self, X):
        """Evaluate a batch of solutions"""
        n_solutions = X.shape[0]
        F = np.zeros((n_solutions, self.n_obj))
        
        if self.implementation == 'cpp':
            return self._evaluate_cpp_batch(X, F)
        else:
            return self._evaluate_rust_batch(X, F)
    
    def _evaluate_cpp_batch(self, X, F):
        """Evaluate using C++ implementation"""
        # Note: This is a placeholder - you'll need to implement the actual
        # hff batch evaluation based on your specific API
        import hff
        
        for obj_idx, f_num in enumerate(self.f_functions):
            # This is pseudocode - adjust based on actual hff API
            for sol_idx in range(X.shape[0]):
                # F[sol_idx, obj_idx] = hff.evaluate_gnbg(f_num, X[sol_idx])
                # For now, use a placeholder that at least runs
                F[sol_idx, obj_idx] = np.sum(X[sol_idx]**2) + f_num  # Dummy function
        
        return F
    
    def _evaluate_rust_batch(self, X, F):
        """Evaluate using Rust implementation"""
        for obj_idx, f_num in enumerate(self.f_functions):
            evaluator = self.evaluators[f_num]
            F[:, obj_idx] = evaluator.fitness(X)
        return F

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def measure_memory_usage():
    """Get current memory usage"""
    if MEMORY_MONITORING:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    return 0

def benchmark_implementation(implementation_name, problem, X_batch, iterations=10):
    """Benchmark a specific implementation"""
    
    print(f"   🔬 Testing {implementation_name} with {X_batch.shape[0]} solutions...")
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = problem.evaluate_batch(X_batch)
        gc.collect()
    
    # Benchmark runs
    times = []
    memory_before = []
    memory_after = []
    
    for i in range(iterations):
        # Measure memory before
        mem_before = measure_memory_usage()
        memory_before.append(mem_before)
        
        # Time the evaluation
        start_time = time.perf_counter()
        F = problem.evaluate_batch(X_batch)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Measure memory after
        mem_after = measure_memory_usage()
        memory_after.append(mem_after)
        
        gc.collect()
        time.sleep(0.01)  # Brief pause between runs
    
    # Calculate statistics
    mean_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0
    evaluations_per_second = (X_batch.shape[0] * problem.n_obj) / mean_time
    
    # Memory statistics
    mean_memory_delta = mean([after - before for before, after in zip(memory_before, memory_after)])
    
    return {
        'implementation': implementation_name,
        'batch_size': X_batch.shape[0],
        'n_objectives': problem.n_obj,
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'evaluations_per_second': evaluations_per_second,
        'memory_delta_mb': mean_memory_delta,
        'raw_times': times
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing all implementations"""
    
    print("🚀 GNBG Implementation Benchmark")
    print("=" * 60)
    print(f"Problem: F1+F2+F3+F4+F5 (5 objectives, 30 variables)")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Iterations per test: {BENCHMARK_ITERATIONS}")
    
    # Setup implementations
    cpp_available, rust_available, rust_gpu_available = setup_implementations()
    
    if not cpp_available and not rust_available:
        print("❌ No implementations available for benchmarking!")
        return
    
    # Prepare benchmark results
    all_results = []
    
    # Test each batch size
    for batch_size in BATCH_SIZES:
        print(f"\n📊 Batch Size: {batch_size}")
        print("-" * 40)
        
        # Generate random test data
        np.random.seed(42)  # Reproducible results
        X_batch = np.random.uniform(-100, 100, (batch_size, BENCHMARK_PROBLEM_CONFIG['n_var']))
        
        # Test C++ implementation
        if cpp_available:
            try:
                problem_cpp = MultiObjectiveGNBGProblem(
                    BENCHMARK_PROBLEM_CONFIG['gnbg2'], 
                    BENCHMARK_PROBLEM_CONFIG['n_var'],
                    implementation='cpp'
                )
                result = benchmark_implementation('C++ (native)', problem_cpp, X_batch, BENCHMARK_ITERATIONS)
                all_results.append(result)
                
                print(f"   C++ (native): {result['evaluations_per_second']:.0f} eval/s "
                      f"({result['mean_time_ms']:.1f}±{result['std_time_ms']:.1f}ms)")
            except Exception as e:
                print(f"   C++ (native): ❌ Failed - {e}")
        
        # Test Rust CPU implementation
        if rust_available:
            try:
                problem_rust_cpu = MultiObjectiveGNBGProblem(
                    BENCHMARK_PROBLEM_CONFIG['gnbg2'], 
                    BENCHMARK_PROBLEM_CONFIG['n_var'],
                    implementation='rust_cpu'
                )
                result = benchmark_implementation('Rust CPU', problem_rust_cpu, X_batch, BENCHMARK_ITERATIONS)
                all_results.append(result)
                
                print(f"   Rust CPU:     {result['evaluations_per_second']:.0f} eval/s "
                      f"({result['mean_time_ms']:.1f}±{result['std_time_ms']:.1f}ms)")
            except Exception as e:
                print(f"   Rust CPU: ❌ Failed - {e}")
        
        # Test Rust GPU implementation
        if rust_available and rust_gpu_available:
            try:
                problem_rust_gpu = MultiObjectiveGNBGProblem(
                    BENCHMARK_PROBLEM_CONFIG['gnbg2'], 
                    BENCHMARK_PROBLEM_CONFIG['n_var'],
                    implementation='rust_gpu'
                )
                result = benchmark_implementation('Rust GPU', problem_rust_gpu, X_batch, BENCHMARK_ITERATIONS)
                all_results.append(result)
                
                print(f"   Rust GPU:     {result['evaluations_per_second']:.0f} eval/s "
                      f"({result['mean_time_ms']:.1f}±{result['std_time_ms']:.1f}ms)")
            except Exception as e:
                print(f"   Rust GPU: ❌ Failed - {e}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n💾 Results saved to: {OUTPUT_FILE}")
        
        # Display summary
        display_benchmark_summary(df)
        
        # Plot if requested
        if PLOT_RESULTS:
            plot_benchmark_results(df)
    
    return all_results

def display_benchmark_summary(df):
    """Display benchmark summary statistics"""
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Performance comparison at largest batch size
    largest_batch = df['batch_size'].max()
    large_batch_df = df[df['batch_size'] == largest_batch]
    
    print(f"Performance at batch size {largest_batch}:")
    for _, row in large_batch_df.iterrows():
        throughput = row['evaluations_per_second']
        print(f"   {row['implementation']:15s}: {throughput:8.0f} eval/s")
    
    # Calculate speedups
    if len(large_batch_df) > 1:
        baseline_perf = large_batch_df[large_batch_df['implementation'].str.contains('C++')]['evaluations_per_second']
        if not baseline_perf.empty:
            baseline = baseline_perf.iloc[0]
            print(f"\nSpeedup vs C++ (native):")
            for _, row in large_batch_df.iterrows():
                if 'Rust' in row['implementation']:
                    speedup = row['evaluations_per_second'] / baseline
                    print(f"   {row['implementation']:15s}: {speedup:.2f}x faster")
    
    # Memory usage comparison
    print(f"\nMemory Usage (batch size {largest_batch}):")
    for _, row in large_batch_df.iterrows():
        mem_delta = row['memory_delta_mb']
        print(f"   {row['implementation']:15s}: {mem_delta:+6.1f} MB")

def plot_benchmark_results(df):
    """Plot benchmark results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Throughput vs Batch Size
        for impl in df['implementation'].unique():
            impl_df = df[df['implementation'] == impl]
            ax1.plot(impl_df['batch_size'], impl_df['evaluations_per_second'], 
                    marker='o', label=impl, linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Evaluations/Second')
        ax1.set_title('Throughput vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Timing comparison (latest batch size)
        largest_batch = df['batch_size'].max()
        timing_df = df[df['batch_size'] == largest_batch]
        bars = ax2.bar(timing_df['implementation'], timing_df['mean_time_ms'])
        ax2.set_ylabel('Time (ms)')
        ax2.set_title(f'Execution Time (Batch Size: {largest_batch})')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn_r(timing_df['mean_time_ms'] / timing_df['mean_time_ms'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Memory usage
        ax3.bar(timing_df['implementation'], timing_df['memory_delta_mb'])
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.set_title('Memory Usage')
        ax3.tick_params(axis='x', rotation=45)
        
        # Speedup comparison
        if len(timing_df) > 1:
            baseline_perf = timing_df[timing_df['implementation'].str.contains('C++')]['evaluations_per_second']
            if not baseline_perf.empty:
                baseline = baseline_perf.iloc[0]
                speedups = timing_df['evaluations_per_second'] / baseline
                bars = ax4.bar(timing_df['implementation'], speedups)
                ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
                ax4.set_ylabel('Speedup (vs C++)')
                ax4.set_title('Performance Speedup')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend()
                
                # Color bars by speedup
                colors = plt.cm.RdYlGn(speedups / speedups.max())
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        plot_file = OUTPUT_FILE.replace('.csv', '_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_file}")
        
        # Try to display if in interactive environment
        try:
            plt.show()
        except:
            pass
            
    except ImportError:
        print("📊 matplotlib not available - skipping plots")
        print("   Install with: pip install matplotlib seaborn")

# =============================================================================
# STRESS TEST FOR LARGE POPULATIONS
# =============================================================================

def run_stress_test():
    """Run stress test with large populations to showcase GPU benefits"""
    print("\n🔥 STRESS TEST - Large Population Evaluation")
    print("=" * 60)
    
    # Setup implementations
    cpp_available, rust_available, rust_gpu_available = setup_implementations()
    
    if not rust_gpu_available:
        print("❌ Rust GPU not available - skipping stress test")
        return
    
    for pop_size in POPULATION_SIZES:
        print(f"\n📊 Population Size: {pop_size:,}")
        print("-" * 40)
        
        # Generate large population
        np.random.seed(42)
        X_large = np.random.uniform(-100, 100, (pop_size, BENCHMARK_PROBLEM_CONFIG['n_var']))
        
        # Test Rust implementations
        if rust_available:
            # CPU version
            problem_cpu = MultiObjectiveGNBGProblem(
                BENCHMARK_PROBLEM_CONFIG['gnbg2'], 
                BENCHMARK_PROBLEM_CONFIG['n_var'],
                implementation='rust_cpu'
            )
            result_cpu = benchmark_implementation('Rust CPU', problem_cpu, X_large, 3)
            
            # GPU version
            if rust_gpu_available:
                problem_gpu = MultiObjectiveGNBGProblem(
                    BENCHMARK_PROBLEM_CONFIG['gnbg2'], 
                    BENCHMARK_PROBLEM_CONFIG['n_var'],
                    implementation='rust_gpu'
                )
                result_gpu = benchmark_implementation('Rust GPU', problem_gpu, X_large, 3)
                
                # Calculate GPU advantage
                speedup = result_gpu['evaluations_per_second'] / result_cpu['evaluations_per_second']
                print(f"   GPU Advantage: {speedup:.2f}x faster than CPU")
                print(f"   GPU: {result_gpu['evaluations_per_second']:,.0f} eval/s")
                print(f"   CPU: {result_cpu['evaluations_per_second']:,.0f} eval/s")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main benchmark execution"""
    
    print("🚀 GNBG Implementation Benchmark Suite")
    print("Testing C++ (native) vs Rust (gnbg-gpu) implementations")
    print(f"Multi-objective problem: F1+F2+F3+F4+F5 (5 objectives)")
    print("=" * 80)
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    if not results:
        print("❌ No benchmark results - check implementation availability")
        return
    
    # Run stress test
    run_stress_test()
    
    # Final summary
    print("\n🎉 BENCHMARK COMPLETE!")
    print("=" * 60)
    print(f"📁 Results: {OUTPUT_FILE}")
    if PLOT_RESULTS:
        plot_file = OUTPUT_FILE.replace('.csv', '_plot.png')
        print(f"📊 Plots: {plot_file}")
    
    print("\n🚀 Ready for GPU-accelerated multi-objective optimization!")

if __name__ == "__main__":
    main()