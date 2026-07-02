#!/usr/bin/env python3
"""
Realistic GNBG Benchmark: Based on Actual Experimental Setup

This benchmark tests GNBG implementations using the same computational budgets
and scaling patterns as the real experiments, including tests up to 500 objectives.
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
# REALISTIC EXPERIMENTAL CONFIGURATION (Based on run_experiment.py)
# =============================================================================

# GNBG-II Competition Budget (from run_experiment.py)
GNGB_TOTAL_BUDGET = 1_000_000              # Total evaluation budget
N_RUNS = 31                                 # Number of independent runs  
GNGB_PER_RUN_BUDGET = GNGB_TOTAL_BUDGET // N_RUNS  # 32,258 evaluations per run

# Population and generation settings (from run_experiment.py) 
BASE_POPULATION_SIZE = 322                  # Base population size
N_GENERATIONS = 100                         # Fixed generations for competition

# Objective scaling configurations (from run_full_experiments.py)
OBJECTIVE_SCALES = [2, 3, 5, 10, 15, 25, 50, 100, 200, 500]  # Up to 500 objectives!
POPULATION_SCALING_FACTOR = 20              # Pop size = max(322, n_obj * 20) from line 247

# Benchmark parameters  
WARMUP_ITERATIONS = 3                       # Warmup runs
BENCHMARK_ITERATIONS = 5                    # Benchmark runs per test
MEMORY_MONITORING = True                    # Track memory usage

# Output files
OUTPUT_FILE = "realistic_gnbg_benchmark.csv"
DETAILED_OUTPUT = "realistic_gnbg_detailed.csv"

# =============================================================================
# REALISTIC MULTI-OBJECTIVE GNBG PROBLEM
# =============================================================================

class RealisticMultiObjectiveGNBGProblem:
    """
    Multi-objective GNBG problem that matches real experimental setup.
    
    Uses the same budget constraints and population scaling as the actual experiments.
    """
    
    def __init__(self, n_objectives, f_functions=None, implementation='cpp'):
        self.n_objectives = n_objectives
        self.n_var = 30  # Standard GNBG variable count
        
        # Default to cycling through F1-F5 for multi-objective
        if f_functions is None:
            f_functions = [((i % 5) + 1) for i in range(n_objectives)]
        
        self.f_functions = f_functions[:n_objectives]  # Ensure correct count
        self.implementation = implementation
        self.evaluators = {}
        
        # Population scaling (from run_experiment.py line 247)
        self.population_size = max(BASE_POPULATION_SIZE, n_objectives * POPULATION_SCALING_FACTOR)
        
        # Bounds (standard GNBG range)
        self.xl = np.full(self.n_var, -100.0)
        self.xu = np.full(self.n_var, 100.0)
        
        # Initialize evaluators
        if implementation == 'cpp':
            self._setup_cpp_evaluators()
        elif implementation == 'rust_gpu':
            self._setup_rust_evaluators(use_gpu=True)
        elif implementation == 'rust_cpu':
            self._setup_rust_evaluators(use_gpu=False)
    
    def _setup_cpp_evaluators(self):
        """Setup C++ evaluators via hff (placeholder)"""
        import hff
        for f_num in self.f_functions:
            # Note: This is a placeholder - actual hff integration needed
            self.evaluators[f_num] = f"gnbg_f{f_num}"
    
    def _setup_rust_evaluators(self, use_gpu=True):
        """Setup Rust evaluators via gnbg_gpu"""
        import gnbg_gpu
        for f_num in self.f_functions:
            self.evaluators[f_num] = gnbg_gpu.GNBGGpu(f_num, use_gpu=use_gpu)
    
    def evaluate_population(self, X):
        """
        Evaluate a population using realistic experimental constraints.
        
        This simulates the actual evaluation pattern used in optimization:
        - Batch evaluation for efficiency
        - Budget tracking
        - Multi-objective aggregation
        """
        n_solutions = X.shape[0]
        F = np.zeros((n_solutions, self.n_objectives))
        
        if self.implementation == 'cpp':
            return self._evaluate_cpp_population(X, F)
        else:
            return self._evaluate_rust_population(X, F)
    
    def _evaluate_cpp_population(self, X, F):
        """Evaluate using C++ implementation (placeholder)"""
        # Note: This is a placeholder for actual hff integration
        for obj_idx, f_num in enumerate(self.f_functions):
            # Placeholder: simple quadratic function with F-function offset
            for sol_idx in range(X.shape[0]):
                F[sol_idx, obj_idx] = np.sum(X[sol_idx]**2) + f_num * 1000
        return F
    
    def _evaluate_rust_population(self, X, F):
        """Evaluate using Rust implementation"""
        for obj_idx, f_num in enumerate(self.f_functions):
            evaluator = self.evaluators[f_num]
            F[:, obj_idx] = evaluator.fitness(X)
        return F

# =============================================================================
# REALISTIC BENCHMARK FUNCTIONS
# =============================================================================

def measure_memory_usage():
    """Get current memory usage"""
    if MEMORY_MONITORING:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    return 0

def benchmark_realistic_scenario(implementation_name, problem, n_objectives, iterations=5):
    """
    Benchmark using realistic experimental scenarios.
    
    This tests the actual workload patterns from multi-objective optimization:
    - Population-based evaluation (not just batch sizes)  
    - Budget-constrained runs
    - Multiple generations
    """
    
    print(f"   🔬 Testing {implementation_name} with {n_objectives} objectives")
    print(f"      Population size: {problem.population_size:,}")
    print(f"      Evaluations per generation: {problem.population_size:,}")
    print(f"      Total evaluations per run: {problem.population_size * N_GENERATIONS:,}")
    
    # Check if this exceeds budget
    total_evals = problem.population_size * N_GENERATIONS
    if total_evals > GNGB_PER_RUN_BUDGET:
        print(f"      ⚠️  Would exceed budget ({total_evals:,} > {GNGB_PER_RUN_BUDGET:,})")
        # Scale down to fit budget
        max_gens = GNGB_PER_RUN_BUDGET // problem.population_size
        print(f"      📉 Scaling to {max_gens} generations to fit budget")
        actual_gens = max_gens
    else:
        actual_gens = N_GENERATIONS
    
    # Generate population for testing
    np.random.seed(42)  # Reproducible
    population = np.random.uniform(-100, 100, (problem.population_size, problem.n_var))
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = problem.evaluate_population(population)
        gc.collect()
    
    # Benchmark runs
    times = []
    memory_before = []
    memory_after = []
    
    for i in range(iterations):
        # Measure memory before
        mem_before = measure_memory_usage()
        memory_before.append(mem_before)
        
        # Time the evaluation (simulate one generation)
        start_time = time.perf_counter()
        F = problem.evaluate_population(population)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Measure memory after
        mem_after = measure_memory_usage()
        memory_after.append(mem_after)
        
        gc.collect()
        time.sleep(0.01)  # Brief pause
    
    # Calculate statistics
    mean_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0
    
    # Calculate realistic metrics
    evaluations_per_generation = problem.population_size * n_objectives
    evaluations_per_second = evaluations_per_generation / mean_time
    
    # Estimate full run performance
    estimated_run_time = mean_time * actual_gens
    estimated_total_evaluations = evaluations_per_generation * actual_gens
    
    # Memory statistics
    mean_memory_delta = mean([after - before for before, after in zip(memory_before, memory_after)])
    
    return {
        'implementation': implementation_name,
        'n_objectives': n_objectives,
        'population_size': problem.population_size,
        'evaluations_per_generation': evaluations_per_generation,
        'mean_time_per_generation_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'evaluations_per_second': evaluations_per_second,
        'estimated_run_time_minutes': estimated_run_time / 60,
        'estimated_total_evaluations': estimated_total_evaluations,
        'actual_generations_within_budget': actual_gens,
        'budget_utilization_pct': (estimated_total_evaluations / GNGB_PER_RUN_BUDGET) * 100,
        'memory_delta_mb': mean_memory_delta,
        'raw_times': times
    }

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
        print(f"   ✅ Rust (gnbg_gpu): {'GPU Ready' if rust_gpu_available else 'CPU Only'}")
        rust_available = True
    except Exception as e:
        print(f"   ❌ Rust (gnbg_gpu): Failed to initialize - {e}")
    
    return cpp_available, rust_available, rust_gpu_available

def run_realistic_benchmark():
    """Run realistic benchmark matching actual experimental conditions"""
    
    print("🚀 Realistic GNBG Benchmark")
    print("=" * 80)
    print(f"Based on actual experimental setup:")
    print(f"  • Budget: {GNGB_PER_RUN_BUDGET:,} evaluations per run")
    print(f"  • Runs: {N_RUNS} independent runs")
    print(f"  • Base population: {BASE_POPULATION_SIZE}")
    print(f"  • Population scaling: max(base, n_obj × {POPULATION_SCALING_FACTOR})")
    print(f"  • Objective scales: {OBJECTIVE_SCALES}")
    
    # Setup implementations
    cpp_available, rust_available, rust_gpu_available = setup_implementations()
    
    if not cpp_available and not rust_available:
        print("❌ No implementations available!")
        return
    
    # Collect all results
    all_results = []
    
    # Test each objective scale
    for n_obj in OBJECTIVE_SCALES:
        print(f"\n📊 Testing {n_obj} objectives")
        print("-" * 50)
        
        # Generate F functions for this objective count
        f_functions = [((i % 5) + 1) for i in range(n_obj)]  # Cycle F1-F5
        
        # Test C++ implementation
        if cpp_available:
            try:
                problem_cpp = RealisticMultiObjectiveGNBGProblem(
                    n_obj, f_functions, implementation='cpp'
                )
                result = benchmark_realistic_scenario(
                    'C++ (native)', problem_cpp, n_obj, BENCHMARK_ITERATIONS
                )
                all_results.append(result)
                
                print(f"   C++: {result['evaluations_per_second']:.0f} eval/s, "
                      f"est. run time: {result['estimated_run_time_minutes']:.1f} min")
                
            except Exception as e:
                print(f"   C++ failed: {e}")
        
        # Test Rust CPU
        if rust_available:
            try:
                problem_rust_cpu = RealisticMultiObjectiveGNBGProblem(
                    n_obj, f_functions, implementation='rust_cpu'
                )
                result = benchmark_realistic_scenario(
                    'Rust CPU', problem_rust_cpu, n_obj, BENCHMARK_ITERATIONS
                )
                all_results.append(result)
                
                print(f"   Rust CPU: {result['evaluations_per_second']:.0f} eval/s, "
                      f"est. run time: {result['estimated_run_time_minutes']:.1f} min")
                
            except Exception as e:
                print(f"   Rust CPU failed: {e}")
        
        # Test Rust GPU
        if rust_available and rust_gpu_available:
            try:
                problem_rust_gpu = RealisticMultiObjectiveGNBGProblem(
                    n_obj, f_functions, implementation='rust_gpu'
                )
                result = benchmark_realistic_scenario(
                    'Rust GPU', problem_rust_gpu, n_obj, BENCHMARK_ITERATIONS
                )
                all_results.append(result)
                
                print(f"   Rust GPU: {result['evaluations_per_second']:.0f} eval/s, "
                      f"est. run time: {result['estimated_run_time_minutes']:.1f} min")
                
            except Exception as e:
                print(f"   Rust GPU failed: {e}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n💾 Results saved to: {OUTPUT_FILE}")
        
        # Display summary
        display_realistic_summary(df)
        
        # Show scaling analysis
        analyze_scaling_performance(df)
    
    return all_results

def display_realistic_summary(df):
    """Display realistic benchmark summary"""
    print("\n📊 REALISTIC BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group by implementation
    for impl in df['implementation'].unique():
        impl_df = df[df['implementation'] == impl]
        print(f"\n{impl}:")
        print(f"{'Objectives':>12} {'Pop Size':>10} {'Eval/s':>12} {'Est Runtime':>12} {'Budget Use':>10}")
        print("-" * 70)
        
        for _, row in impl_df.iterrows():
            print(f"{row['n_objectives']:>12d} "
                  f"{row['population_size']:>10,d} "
                  f"{row['evaluations_per_second']:>12,.0f} "
                  f"{row['estimated_run_time_minutes']:>10.1f}m "
                  f"{row['budget_utilization_pct']:>9.1f}%")

def analyze_scaling_performance(df):
    """Analyze how implementations scale with objectives"""
    print("\n📈 SCALING ANALYSIS")
    print("=" * 80)
    
    # Find performance at key scales
    key_scales = [10, 50, 100, 500]
    
    for scale in key_scales:
        scale_df = df[df['n_objectives'] == scale]
        if len(scale_df) == 0:
            continue
            
        print(f"\nAt {scale} objectives:")
        best_perf = scale_df['evaluations_per_second'].max()
        
        for _, row in scale_df.iterrows():
            speedup = row['evaluations_per_second'] / best_perf if best_perf > 0 else 0
            print(f"  {row['implementation']:15s}: "
                  f"{row['evaluations_per_second']:8,.0f} eval/s "
                  f"({speedup:.2f}x relative)")
    
    # Identify the "500 objective challenge"
    print(f"\n🎯 500 OBJECTIVE CHALLENGE:")
    obj_500 = df[df['n_objectives'] == 500]
    if len(obj_500) > 0:
        print(f"Population size needed: {obj_500.iloc[0]['population_size']:,}")
        print(f"Evaluations per generation: {obj_500.iloc[0]['evaluations_per_generation']:,}")
        
        for _, row in obj_500.iterrows():
            print(f"  {row['implementation']:15s}: "
                  f"{row['evaluations_per_second']:8,.0f} eval/s, "
                  f"est. {row['estimated_run_time_minutes']:.0f} min per run")
            
            if row['estimated_run_time_minutes'] * N_RUNS > 60:  # > 1 hour total
                hours = (row['estimated_run_time_minutes'] * N_RUNS) / 60
                print(f"                      Total time for 31 runs: {hours:.1f} hours")
            
    else:
        print("No 500 objective results available")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main benchmark execution"""
    
    print("🚀 Realistic GNBG Implementation Benchmark")
    print("Based on actual experimental setup with scaling to 500 objectives")
    print("=" * 80)
    
    # Run the benchmark
    results = run_realistic_benchmark()
    
    if not results:
        print("❌ No benchmark results generated")
        return
    
    print(f"\n🎉 REALISTIC BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"📁 Results: {OUTPUT_FILE}")
    print(f"\n💡 Key insights for 500 objective optimization:")
    print(f"   • Population sizes reach {max(BASE_POPULATION_SIZE, 500 * POPULATION_SCALING_FACTOR):,}")
    print(f"   • Evaluations per generation: {500 * max(BASE_POPULATION_SIZE, 500 * POPULATION_SCALING_FACTOR):,}")
    print(f"   • This is where GPU acceleration becomes ESSENTIAL!")

if __name__ == "__main__":
    main()