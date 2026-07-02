#!/usr/bin/env python3
"""
WFG4-9 Benchmark with HIGD
==========================
Runs WFG4-9 problems which have unit hypersphere Pareto fronts.
Uses HIGD (Hyperspherical IGD) instead of traditional IGD.

These problems use Mueller-Marsaglia sampling for reference front
generation, avoiding pymoo's memory explosion at high dimensions.
"""

import sys
import os

# Add demo directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corrected_igd_benchmark import (
    run_benchmark_suite,
    STUDY_OBJECTIVES,
    STUDY_ALGORITHMS,
    STUDY_N_RUNS,
)

# WFG4-9: Unit hypersphere Pareto front (HIGD supported)
WFG_HIGD_PROBLEMS = ['WFG4', 'WFG5', 'WFG6', 'WFG7', 'WFG8', 'WFG9']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run WFG4-9 benchmarks with HIGD')
    parser.add_argument('--problem', type=str, choices=WFG_HIGD_PROBLEMS,
                        help='Run specific problem only')
    parser.add_argument('--objectives', type=str, default=None,
                        help='Comma-separated objective counts (default: 10-100 step 10)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str,
                        default='analysis/data/igd_corrected_results.csv',
                        help='Output CSV file')

    args = parser.parse_args()

    # Determine problems
    if args.problem:
        problems = [args.problem]
    else:
        problems = WFG_HIGD_PROBLEMS

    # Determine objectives (default: 10-100 step 10 for WFG)
    if args.objectives:
        objectives = [int(x.strip()) for x in args.objectives.split(',')]
    else:
        # WFG typically runs 10-100 objectives
        objectives = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print("=" * 70)
    print("WFG4-9 BENCHMARK WITH HIGD")
    print("=" * 70)
    print(f"Problems: {problems}")
    print(f"Objectives: {objectives}")
    print(f"Algorithms: {STUDY_ALGORITHMS}")
    print(f"Runs per config: {STUDY_N_RUNS}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # Run benchmarks
    df = run_benchmark_suite(
        problems=problems,
        objectives=objectives,
        algorithms=STUDY_ALGORITHMS,
        n_runs=STUDY_N_RUNS,
        output_file=args.output,
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    if len(df) > 0 and 'higd' in df.columns:
        higd_stats = df[df['higd'].notna()].groupby('problem')['higd'].agg(['mean', 'min', 'max'])
        print("\nHIGD Summary:")
        print(higd_stats)
