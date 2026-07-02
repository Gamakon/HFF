#!/usr/bin/env python3
"""
Quick GNBG Realistic Benchmark - Focuses on key scaling points
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add benchmark directory to path
sys.path.append(str(Path(__file__).parent))

# =============================================================================
# QUICK BENCHMARK CONFIGURATION
# =============================================================================

# GNBG-II Competition Budget (from run_experiment.py)
GNGB_PER_RUN_BUDGET = 32_258               # Per-run budget from 1M/31

# Key objective scaling points to test
OBJECTIVE_SCALES = [2, 5, 10, 25, 50, 100, 500]  # Focus on key scales
BASE_POPULATION_SIZE = 322
POPULATION_SCALING_FACTOR = 20

BENCHMARK_ITERATIONS = 3                    # Quick benchmark

# =============================================================================
# QUICK BENCHMARK PROBLEM
# =============================================================================

class QuickMultiObjectiveGNBGProblem:
    """Quick multi-objective GNBG problem for benchmarking"""
    
    def __init__(self, n_objectives, implementation='rust_gpu'):
        self.n_objectives = n_objectives
        self.n_var = 30
        
        # F function cycling (F1-F5)
        self.f_functions = [((i % 5) + 1) for i in range(min(n_objectives, 5))]
        
        # Population scaling (from run_experiment.py)
        self.population_size = max(BASE_POPULATION_SIZE, n_objectives * POPULATION_SCALING_FACTOR)
        
        # For very large objectives, use smaller test population to avoid timeout
        if n_objectives >= 100:
            self.test_population_size = min(1000, self.population_size)
        else:
            self.test_population_size = min(2000, self.population_size)
        
        self.implementation = implementation
        self.evaluators = {}
        
        if implementation in ['rust_gpu', 'rust_cpu']:
            self._setup_rust_evaluators(use_gpu=(implementation == 'rust_gpu'))
    
    def _setup_rust_evaluators(self, use_gpu=True):
        """Setup Rust evaluators"""
        import gnbg_gpu
        for f_num in self.f_functions:
            self.evaluators[f_num] = gnbg_gpu.GNBGGpu(f_num, use_gpu=use_gpu)
    
    def evaluate_test_population(self, X):
        """Evaluate using available F functions"""
        n_solutions = X.shape[0]
        
        # For large objective counts, simulate by repeating F1-F5 evaluations
        if self.n_objectives <= 5:
            # Direct evaluation  
            F = np.zeros((n_solutions, self.n_objectives))
            for obj_idx, f_num in enumerate(self.f_functions):
                evaluator = self.evaluators[f_num]
                F[:, obj_idx] = evaluator.fitness(X)
        else:
            # Simulate large objective evaluation by repeating base functions
            base_results = np.zeros((n_solutions, len(self.f_functions)))
            for f_idx, f_num in enumerate(self.f_functions):
                evaluator = self.evaluators[f_num]
                base_results[:, f_idx] = evaluator.fitness(X)
            
            # Repeat and tile to match objective count
            F = np.tile(base_results, (1, (self.n_objectives + 4) // 5))[:, :self.n_objectives]
        
        return F

# =============================================================================
# QUICK BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_quick_scenario(implementation_name, problem, n_objectives):
    """Quick benchmark focusing on key metrics"""
    
    print(f"   🔬 {implementation_name}: {n_objectives} objectives")
    print(f"      Real pop size: {problem.population_size:,}")
    print(f"      Test pop size: {problem.test_population_size:,}")
    
    # Check budget feasibility
    evals_per_gen = problem.population_size * n_objectives
    max_gens_in_budget = GNGB_PER_RUN_BUDGET // evals_per_gen if evals_per_gen > 0 else 0
    
    print(f"      Evals/gen: {evals_per_gen:,}")
    print(f"      Max gens in budget: {max_gens_in_budget}")
    
    # Generate test population
    np.random.seed(42)
    population = np.random.uniform(-100, 100, (problem.test_population_size, problem.n_var))
    
    # Warmup
    _ = problem.evaluate_test_population(population)
    
    # Benchmark
    times = []
    for i in range(BENCHMARK_ITERATIONS):
        start_time = time.perf_counter()
        F = problem.evaluate_test_population(population)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    mean_time = np.mean(times)
    
    # Calculate realistic metrics
    test_evaluations = problem.test_population_size * n_objectives
    evaluations_per_second = test_evaluations / mean_time
    
    # Scale to real population performance estimate
    scaling_factor = problem.population_size / problem.test_population_size
    estimated_real_time = mean_time * scaling_factor
    estimated_real_eval_rate = test_evaluations * scaling_factor / mean_time
    
    # Runtime estimates
    if max_gens_in_budget > 0:
        estimated_run_time_minutes = (estimated_real_time * max_gens_in_budget) / 60
    else:
        estimated_run_time_minutes = float('inf')
    
    return {
        'implementation': implementation_name,
        'n_objectives': n_objectives,
        'real_population_size': problem.population_size,
        'test_population_size': problem.test_population_size,
        'evaluations_per_second': evaluations_per_second,
        'estimated_real_eval_rate': estimated_real_eval_rate,
        'max_generations_in_budget': max_gens_in_budget,
        'estimated_run_time_minutes': estimated_run_time_minutes,
        'feasible_within_budget': max_gens_in_budget > 0,
        'test_time_ms': mean_time * 1000
    }

def run_quick_benchmark():
    """Run quick benchmark"""
    
    print("🚀 Quick GNBG Realistic Benchmark")
    print("=" * 60)
    print(f"Budget per run: {GNGB_PER_RUN_BUDGET:,} evaluations")
    
    # Setup Rust implementation
    try:
        import gnbg_gpu
        test_eval = gnbg_gpu.GNBGGpu(1, use_gpu=True)
        gpu_available = test_eval.using_gpu
        print(f"Rust GPU: {'✅ Available' if gpu_available else '❌ Not available'}")
    except:
        print("❌ Rust implementation not available")
        return
    
    all_results = []
    
    # Test each objective scale
    for n_obj in OBJECTIVE_SCALES:
        print(f"\n📊 {n_obj} objectives")
        print("-" * 30)
        
        # Test Rust GPU
        if gpu_available:
            try:
                problem_gpu = QuickMultiObjectiveGNBGProblem(n_obj, 'rust_gpu')
                result = benchmark_quick_scenario('Rust GPU', problem_gpu, n_obj)
                all_results.append(result)
                
                print(f"      GPU: {result['evaluations_per_second']:,.0f} eval/s")
                print(f"      Est real rate: {result['estimated_real_eval_rate']:,.0f} eval/s")
                if result['feasible_within_budget']:
                    print(f"      Est run time: {result['estimated_run_time_minutes']:.1f} min")
                else:
                    print(f"      ⚠️  Exceeds budget - not feasible")
                    
            except Exception as e:
                print(f"      GPU failed: {e}")
        
        # Test Rust CPU for comparison on smaller scales
        if n_obj <= 50:  # Only test CPU on smaller scales
            try:
                problem_cpu = QuickMultiObjectiveGNBGProblem(n_obj, 'rust_cpu')
                result = benchmark_quick_scenario('Rust CPU', problem_cpu, n_obj)
                all_results.append(result)
                
                print(f"      CPU: {result['evaluations_per_second']:,.0f} eval/s")
                if gpu_available:
                    gpu_result = [r for r in all_results if r['implementation'] == 'Rust GPU' and r['n_objectives'] == n_obj]
                    if gpu_result:
                        speedup = result['estimated_real_eval_rate'] / gpu_result[0]['estimated_real_eval_rate']
                        print(f"      GPU speedup: {1/speedup:.1f}x faster than CPU")
                        
            except Exception as e:
                print(f"      CPU failed: {e}")
    
    # Save and display results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("quick_gnbg_benchmark.csv", index=False)
        
        display_quick_summary(df)
    
    return all_results

def display_quick_summary(df):
    """Display quick summary"""
    print("\n📊 QUICK BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Focus on GPU results
    gpu_df = df[df['implementation'] == 'Rust GPU']
    
    print(f"{'Objectives':>12} {'Real Pop':>10} {'Est Eval/s':>12} {'Est Runtime':>12} {'Feasible':>10}")
    print("-" * 70)
    
    for _, row in gpu_df.iterrows():
        feasible = "✅ Yes" if row['feasible_within_budget'] else "❌ No"
        runtime = f"{row['estimated_run_time_minutes']:.1f}m" if row['feasible_within_budget'] else "∞"
        
        print(f"{row['n_objectives']:>12d} "
              f"{row['real_population_size']:>10,d} "
              f"{row['estimated_real_eval_rate']:>12,.0f} "
              f"{runtime:>12s} "
              f"{feasible:>10s}")
    
    # Highlight the 500 objective challenge
    obj_500 = gpu_df[gpu_df['n_objectives'] == 500]
    if len(obj_500) > 0:
        row = obj_500.iloc[0]
        print(f"\n🎯 500 OBJECTIVE CHALLENGE:")
        print(f"   Population size: {row['real_population_size']:,}")
        print(f"   Evaluations per generation: {row['real_population_size'] * 500:,}")
        print(f"   Estimated performance: {row['estimated_real_eval_rate']:,.0f} eval/s")
        
        if row['feasible_within_budget']:
            total_time = row['estimated_run_time_minutes'] * 31  # 31 runs
            print(f"   Estimated time per run: {row['estimated_run_time_minutes']:.1f} minutes")
            print(f"   Total time for 31 runs: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
        else:
            print(f"   ❌ NOT FEASIBLE with current budget")
            print(f"   💡 Would need GPU acceleration or budget increase")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    run_quick_benchmark()

if __name__ == "__main__":
    main()