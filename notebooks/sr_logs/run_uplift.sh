#!/bin/zsh
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
echo "######## start $(date '+%H:%M:%S')"
S=$(date +%s)
HFF_GPU=1 HFF_RULES=none HFF_FULLER=lint HFF_LINT_EXACTNESS=finite HFF_PARSIMONY=0 HFF_GRAFT_MODE=improve python3 _sweep_equation_recovery.py --all --parallel 6 --audit-mode sr_logs/uplift
echo "######## sweep done, wall=$(( $(date +%s)-S ))s"
