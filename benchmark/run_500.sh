rm results/hf1_benchmark_results.parquet
#python3 run_full_experiments.py
python3 run_experiments_parallel.py
python3 hf1_unified_ranking.py
quarto render analysis_report.qmd
open analysis_report.html
